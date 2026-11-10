"""Regression tests for strict, reproducible evaluation handoffs."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asd_eeg.evaluation import run_experiments
from asd_eeg.evaluation.metrics import fold_auc, summarize_oof
from asd_eeg.evaluation.splits import make_subject_folds
from asd_eeg.features import cache


@pytest.fixture
def oof():
    rows = pd.DataFrame({
        "subject_id": ["a", "b", "c", "d"], "fold": [0, 0, 1, 1],
        "model": ["rbf"] * 4, "true_label": [0, 1, 0, 1],
        "predicted_label": [0, 1, 1, 0], "decision_score": [-1.0, 1.0, 0.8, -0.4],
    })
    return pd.concat([rows, rows.assign(model="quantum")], ignore_index=True)


@pytest.mark.parametrize("column", ["subject_id", "fold", "model", "true_label",
                                    "predicted_label", "decision_score"])
def test_oof_requires_complete_columns(oof, column):
    with pytest.raises(ValueError, match="need columns"):
        summarize_oof(oof.drop(columns=column))


def test_duplicate_subject_prediction_is_rejected(oof):
    with pytest.raises(ValueError, match="exactly one OOF prediction"):
        summarize_oof(pd.concat([oof, oof.iloc[:1]], ignore_index=True))


@pytest.mark.parametrize("problem", ["missing_subject", "changed_fold", "conflicting_label"])
def test_models_must_share_cohort_folds_and_labels(oof, problem):
    if problem == "missing_subject":
        oof = oof.iloc[:-1]
    elif problem == "changed_fold":
        oof.loc[4, "fold"] = 1
    else:
        oof.loc[4, "true_label"] = 1
    with pytest.raises(ValueError, match="identical OOF|conflicting subject labels"):
        summarize_oof(oof)


@pytest.mark.parametrize("column,value", [
    ("decision_score", np.nan), ("decision_score", np.inf),
    ("true_label", 0.2), ("predicted_label", 2),
    ("fold", -1), ("fold", 0.5), ("subject_id", " "), ("model", None),
])
def test_invalid_oof_values_are_rejected(oof, column, value):
    oof[column] = oof[column].astype(object)
    oof.loc[0, column] = value
    with pytest.raises(ValueError):
        summarize_oof(oof)


def test_auc_spread_is_recomputed_without_pooling_fold_margins(oof):
    summary = summarize_oof(oof)
    assert summary["rbf"]["mean_fold_auc"] == pytest.approx(0.5)
    assert summary["rbf"]["std_fold_auc"] == pytest.approx(0.5)
    assert summary["rbf"]["fold_auc_count"] == 2
    # Fold-wise positive affine score changes must not change AUC summaries.
    modified = oof.copy()
    selected = modified.fold == 1
    modified.loc[selected, "decision_score"] = modified.loc[selected, "decision_score"] * 12 + 400
    assert summarize_oof(modified) == summary


def test_undefined_fold_auc_is_not_turned_into_a_result(oof):
    oof["fold"] = oof["true_label"]
    summary = summarize_oof(oof)
    for model in summary.values():
        assert model["mean_fold_auc"] is None
        assert model["std_fold_auc"] is None
        assert model["fold_auc_count"] == 0
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("labels,scores", [([], []), ([0, 1], [0.5]),
                                         ([0.2, 1], [0.1, 0.9]), ([0, 0], [np.nan, 0.1])])
def test_fold_auc_validates_inputs_even_when_undefined(labels, scores):
    with pytest.raises(ValueError):
        fold_auc(labels, scores)


def test_subject_splits_are_deterministic_and_minority_class_limited():
    subjects = np.array([f"subject-{i}" for i in range(8)], dtype=object)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    first = make_subject_folds(subjects, labels, n_splits=5, seed=17)
    second = make_subject_folds(subjects, labels, n_splits=5, seed=17)
    pd.testing.assert_frame_equal(first, second)
    assert first.fold.nunique() == 3
    assert first.subject_id.tolist() == subjects.tolist()
    assert not first.subject_id.duplicated().any()
    assert first.groupby("fold").label.nunique().eq(2).all()


@pytest.mark.parametrize("problem", ["fractional_labels", "missing_class", "too_few",
                                   "duplicate_id", "blank_id", "null_id", "unaligned",
                                   "two_dimensions", "fractional_folds"])
def test_invalid_split_inputs_are_rejected(problem):
    subjects = np.array(list("abcdefgh"), dtype=object)
    labels = np.array([0, 1] * 4)
    folds = 5
    if problem == "fractional_labels":
        labels = np.array([0.2, 1.2] * 4)
    elif problem == "missing_class":
        labels[:] = 0
    elif problem == "too_few":
        labels[:] = 0
        labels[0] = 1
    elif problem == "duplicate_id":
        subjects[0] = subjects[1]
    elif problem == "blank_id":
        subjects[0] = " "
    elif problem == "null_id":
        subjects[0] = None
    elif problem == "unaligned":
        labels = labels[:-1]
    elif problem == "two_dimensions":
        labels = labels.reshape(2, 4)
        subjects = subjects.reshape(2, 4)
    else:
        folds = 2.5
    with pytest.raises(ValueError):
        make_subject_folds(subjects, labels, n_splits=folds)


def test_source_commit_is_independent_of_caller_directory(monkeypatch, tmp_path):
    expected = run_experiments.git_commit()
    # Source archives/wheels intentionally omit .git and must report unknown,
    # rather than borrowing a commit from the caller's current directory.
    assert expected == "unknown" or (len(expected) == 40 and set(expected) <= set("0123456789abcdef"))
    monkeypatch.chdir(tmp_path)
    assert run_experiments.git_commit() == expected


def test_git_timeout_does_not_abort_an_experiment(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("git", 10)

    monkeypatch.setattr(run_experiments.subprocess, "run", timeout)
    assert run_experiments.git_commit() == "unknown"


def test_non_git_installation_does_not_claim_the_callers_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(run_experiments, "__file__",
                        str(tmp_path / "src/asd_eeg/evaluation/run_experiments.py"))
    assert run_experiments.git_commit() == "unknown"


def test_module_runner_uses_the_shared_feature_default(monkeypatch, tmp_path, capsys):
    requested = []
    sentinel = object()

    def load(path):
        requested.append(path)
        return sentinel

    def evaluate(dataset, output):
        assert dataset is sentinel
        return tmp_path / "run-test"

    monkeypatch.setattr(cache, "load_features", load)
    monkeypatch.setattr(run_experiments, "run_evaluation", evaluate)
    assert run_experiments.main([]) == 0
    assert requested == ["artifacts/features/subject_features.npz"]
    assert json.loads(capsys.readouterr().out)["run"] == str(tmp_path / "run-test")


def test_module_runner_errors_go_to_stderr(tmp_path, capsys):
    assert run_experiments.main(["--features", str(tmp_path / "absent.npz")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err


def test_failed_evaluation_preserves_an_explicit_failed_summary(completed_check, tmp_path,
                                                                monkeypatch):
    dataset = cache.load_features(Path(completed_check["directory"]) / "features.npz")

    def fail(*args, **kwargs):
        raise ValueError("Deliberate regression-test fitting failure")

    monkeypatch.setattr(run_experiments, "fit_model", fail)
    with pytest.raises(ValueError, match="Deliberate regression-test"):
        run_experiments.run_evaluation(dataset, tmp_path)
    summaries = list(tmp_path.glob("run-*/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert "Deliberate regression-test" in summary["error"]
