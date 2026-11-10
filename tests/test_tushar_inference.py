"""Application/inference hardening using labelled synthetic software fixtures only."""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from asd_eeg.cli import main
from asd_eeg.inference.artifacts import (
    check_run_workspace,
    load_run_bundle,
    read_results,
    read_run,
    run_file,
)
from asd_eeg.inference.compatibility import load_bundle
from asd_eeg.inference.export import export_prediction, prediction_json
from asd_eeg.inference.pipeline import analyze_subject


@pytest.mark.parametrize("problem", [
    "missing_metadata", "duplicate_test", "empty_train", "model_name", "data_kind",
    "fold", "angles", "reference", "missing_states", "state_width", "state_norm",
    "scaler", "scaler_width", "scaler_scale", "class_order", "svm_width", "kernel",
])
def test_invalid_bundles_have_actionable_restore_errors(completed_check, tmp_path, problem):
    bundle = load_bundle(Path(completed_check["run"]) / "fold_0/quantum.joblib")
    model, meta = bundle["model"], bundle["metadata"]
    if problem == "missing_metadata":
        del bundle["metadata"]
    elif problem == "duplicate_test":
        meta["test_subject_ids"].append(meta["test_subject_ids"][0])
    elif problem == "empty_train":
        meta["train_subject_ids"] = []
    elif problem == "model_name":
        meta["model"] = "rbf"
    elif problem == "data_kind":
        meta["data_kind"] = "unknown"
    elif problem == "fold":
        meta["fold"] = True
    elif problem == "angles":
        model.train_angles[0, 0] = np.nan
    elif problem == "reference":
        model.feature_reference = None
    elif problem == "missing_states":
        model.train_states = None
    elif problem == "state_width":
        model.train_states = model.train_states[:, :-1]
    elif problem == "state_norm":
        model.train_states[0] *= 2
    elif problem == "scaler":
        del model.scaler.mean_
    elif problem == "scaler_width":
        model.scaler.var_ = model.scaler.var_[:-1]
    elif problem == "scaler_scale":
        model.scaler.scale_[0] = 0
    elif problem == "class_order":
        model.estimator.classes_ = np.array([1, 0])
    elif problem == "svm_width":
        model.estimator.n_features_in_ -= 1
    else:
        model.estimator.kernel = "rbf"
    path = tmp_path / "damaged.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="Bundle|bundle|Reference|reference"):
        load_bundle(path)


def test_corrupt_bundle_cli_does_not_traceback_or_write_output(tmp_path, capsys):
    bundle = tmp_path / "broken.joblib"
    bundle.write_bytes(b"not a joblib model")
    output = tmp_path / "prediction.json"
    assert main(["predict", "--subject", "software-subject", "--bundle", str(bundle),
                 "--output", str(output)]) == 1
    assert not output.exists()
    captured = capsys.readouterr()
    assert "restore" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("model_name", ["rbf", "amplitude", "quantum", "reassigned"])
def test_restores_full_reference_order_for_every_model(completed_check, model_name):
    run = Path(completed_check["run"])
    path, bundle = load_run_bundle(run, 0, model_name)
    original = joblib.load(path)
    np.testing.assert_array_equal(bundle["model"].train_angles, original["model"].train_angles)
    if original["model"].train_states is not None:
        np.testing.assert_array_equal(bundle["model"].train_states, original["model"].train_states)
    result = analyze_subject(completed_check["manifest"], completed_check["held_out_subject"], path)
    assert result["provenance"] == "held-out"
    assert result["window_count"] <= result["accepted_window_count"]
    assert result["window_count"] <= result["max_windows"]
    assert len(result["feature_names"]) == len(result["feature_vector"])
    assert "probability" in result["score_type"]
    assert "label" not in result and "true_label" not in result


def test_new_subject_option_never_bypasses_training_guard(completed_check):
    path = Path(completed_check["run"]) / "fold_0/quantum.joblib"
    trained = load_bundle(path)["metadata"]["train_subject_ids"][0]
    with pytest.raises(ValueError, match="used for training"):
        analyze_subject(completed_check["manifest"], trained, path, require_held_out=False)


def test_prediction_json_is_strict_utf8_and_failed_export_preserves_previous_work(tmp_path):
    path = tmp_path / "prediction.json"
    prediction = {"subject_id": "software-é", "decision_score": -0.5}
    assert export_prediction(prediction, path) == path
    assert "é" in prediction_json(prediction)
    with pytest.raises(ValueError, match="JSON compliant"):
        export_prediction({"decision_score": float("nan")}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == prediction
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match=".json extension"):
        export_prediction(prediction, path.with_suffix(".joblib"))


def test_failed_atomic_replace_leaves_existing_json_and_cleans_temporary_file(tmp_path, monkeypatch):
    from asd_eeg.inference import export

    path = tmp_path / "prediction.json"
    export_prediction({"version": "existing"}, path)

    def denied(*args):
        raise PermissionError("Deliberate software-test write rejection")

    monkeypatch.setattr(export.os, "replace", denied)
    with pytest.raises(PermissionError):
        export_prediction({"version": "replacement"}, path)
    assert json.loads(path.read_text()) == {"version": "existing"}
    assert list(tmp_path.iterdir()) == [path]


def copy_run(completed_check, tmp_path):
    source = Path(completed_check["run"])
    target = tmp_path / source.name
    shutil.copytree(source, target)
    return target


@pytest.mark.parametrize("problem", ["config", "fold_label", "duplicate_subject", "missing_file"])
def test_inconsistent_completed_run_is_rejected(completed_check, tmp_path, problem):
    run = copy_run(completed_check, tmp_path)
    if problem == "config":
        path = run / "config.json"
        config = json.loads(path.read_text())
        config["preprocessing"]["max_windows"] += 1
        path.write_text(json.dumps(config))
    elif problem == "missing_file":
        (run / "metrics.json").unlink()
    else:
        path = run / "folds.csv"
        folds = pd.read_csv(path)
        if problem == "duplicate_subject":
            folds.loc[1, "subject_id"] = folds.subject_id.iloc[0]
        else:
            folds.loc[0, "label"] = 2
        folds.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_run(run)


def test_run_bundle_rejects_changed_reference_ids(completed_check, tmp_path):
    run = copy_run(completed_check, tmp_path)
    path = run / "fold_0/quantum.joblib"
    bundle = load_bundle(path)
    bundle["metadata"]["train_subject_ids"].reverse()
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="training-reference order"):
        load_run_bundle(run, 0, "quantum")


def test_run_bundle_rejects_substitution_from_another_fold(completed_check, tmp_path):
    run = copy_run(completed_check, tmp_path)
    shutil.copyfile(run / "fold_1/quantum.joblib", run / "fold_0/quantum.joblib")
    with pytest.raises(ValueError, match="model/fold"):
        load_run_bundle(run, 0, "quantum")


def test_run_file_rejects_traversal(tmp_path):
    run = tmp_path / "run-local"
    run.mkdir()
    (tmp_path / "outside.joblib").write_bytes(b"software fixture")
    with pytest.raises(ValueError, match="outside"):
        run_file(run, "../outside.joblib")


@pytest.mark.parametrize("problem", ["dataset", "config", "subjects", "labels"])
def test_saved_run_requires_matching_active_workspace(completed_check, problem):
    from asd_eeg.config import Config
    from asd_eeg.data.manifest import read_manifest

    summary, folds = read_run(completed_check["run"])
    table = read_manifest(completed_check["manifest"])
    config = Config.from_dict(json.loads((Path(completed_check["run"]) / "config.json").read_text()))
    if problem == "dataset":
        summary["dataset"] = "another-software-dataset"
    elif problem == "config":
        summary["config_id"] = "wrong-config"
    elif problem == "subjects":
        table = table.iloc[1:]
    else:
        table.loc[0, "label"] = 1 - table.label.iloc[0]
    with pytest.raises(ValueError, match="differs|differ"):
        check_run_workspace(summary, folds, table, config)


def test_saved_metrics_regenerate_without_modification(completed_check):
    metrics, comparison = read_results(completed_check["run"])
    assert set(metrics) == set(comparison.model)
    assert len(comparison) == 4


@pytest.mark.parametrize("problem", ["metrics", "comparison", "oof"])
def test_tampered_result_scores_are_not_displayed(completed_check, tmp_path, problem):
    run = copy_run(completed_check, tmp_path)
    if problem == "metrics":
        path = run / "metrics.json"
        values = json.loads(path.read_text())
        values["quantum"]["balanced_accuracy"] = 0.123456
        path.write_text(json.dumps(values))
    elif problem == "comparison":
        path = run / "model_comparison.csv"
        values = pd.read_csv(path)
        values.loc[0, "balanced_accuracy"] = 0.123456
        values.to_csv(path, index=False)
    else:
        path = run / "oof_predictions.csv"
        values = pd.read_csv(path)
        values.loc[0, "decision_score"] = np.inf
        values.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_results(run)


def test_nonfinite_model_prediction_is_rejected(completed_check, monkeypatch):
    from asd_eeg.training.model import FittedModel

    monkeypatch.setattr(FittedModel, "predict", lambda self, X: (np.array([1]), np.array([np.nan])))
    with pytest.raises(ValueError, match="nonfinite decision score"):
        analyze_subject(completed_check["manifest"], completed_check["held_out_subject"],
                        Path(completed_check["run"]) / "fold_0/quantum.joblib")
