"""Evaluation integrity: leakage-free folds, identical partitions, recomputable metrics."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from asd_eeg.evaluation.metrics import summarize_oof
from asd_eeg.features.cache import load_features
from asd_eeg.inference.compatibility import load_bundle
from asd_eeg.training.model import MODELS


def _oof(run: Path) -> pd.DataFrame:
    return pd.read_csv(run / "oof_predictions.csv")


def test_folds_disjoint_and_exhaustive(completed_check):
    run = Path(completed_check["run"])
    folds = pd.read_csv(run / "folds.csv", dtype={"subject_id": str})
    assert folds.groupby("subject_id").size().eq(1).all()
    dataset = load_features(Path(completed_check["directory"]) / "features.npz")
    assert set(folds.subject_id) == set(dataset.subject_ids.tolist())
    for fold in sorted(folds.fold.unique()):
        bundle = load_bundle(run / f"fold_{fold}/quantum.joblib")
        meta = bundle["metadata"]
        assert not set(meta["train_subject_ids"]) & set(meta["test_subject_ids"])
        assert set(meta["test_subject_ids"]) == set(
            folds[folds.fold == fold].subject_id
        )


def test_identical_partitions_across_models(completed_check):
    frame = _oof(Path(completed_check["run"]))
    assert set(frame.model) == set(MODELS)
    by_model = {
        name: rows.sort_values("subject_id").subject_id.tolist()
        for name, rows in frame.groupby("model")
    }
    assert len({tuple(ids) for ids in by_model.values()}) == 1
    by_fold = {
        (name, fold): rows.sort_values("subject_id").subject_id.tolist()
        for (name, fold), rows in frame.groupby(["model", "fold"])
    }
    folds_per_model = {}
    for (name, fold), ids in by_fold.items():
        folds_per_model.setdefault(name, {})[fold] = tuple(ids)
    first = next(iter(folds_per_model.values()))
    assert all(mapping == first for mapping in folds_per_model.values())
    assert frame.groupby(["subject_id", "model"]).size().eq(1).all()


def test_scaler_fit_on_training_subjects_only(completed_check):
    run = Path(completed_check["run"])
    dataset = load_features(Path(completed_check["directory"]) / "features.npz")
    for bundle_path in sorted(run.glob("fold_*/rbf.joblib")):
        bundle = load_bundle(bundle_path)
        train = np.isin(dataset.subject_ids, bundle["metadata"]["train_subject_ids"])
        assert train.sum() > 0
        np.testing.assert_allclose(
            bundle["model"].scaler.mean_, dataset.X[train].mean(axis=0)
        )


def test_metrics_recomputed_from_oof_match_saved(completed_check):
    import json

    run = Path(completed_check["run"])
    frame = _oof(run)
    saved = json.loads((run / "metrics.json").read_text())
    recomputed = summarize_oof(frame)
    assert set(recomputed) == set(saved) == set(MODELS)
    for name in MODELS:
        for key in (
            "subjects", "accuracy", "balanced_accuracy", "macro_f1",
            "sensitivity", "specificity", "mean_fold_auc", "std_fold_auc", "fold_auc_count",
        ):
            assert recomputed[name][key] == pytest.approx(saved[name][key]), (name, key)
        assert recomputed[name]["confusion_matrix"] == saved[name]["confusion_matrix"]


def test_features_carry_no_label_columns(completed_check):
    dataset = load_features(Path(completed_check["directory"]) / "features.npz")
    assert "label" not in [n.lower() for n in dataset.metadata["feature_names"]]
    assert np.isfinite(dataset.X).all()
    assert dataset.X.shape[1] == 28


def test_class_mapping_and_bundle_reload(completed_check):
    run = Path(completed_check["run"])
    bundle = load_bundle(run / "fold_0/quantum.joblib")
    assert {int(k): v for k, v in bundle["metadata"]["class_mapping"].items()} == {
        0: "Control", 1: "ASD",
    }
    dataset = load_features(Path(completed_check["directory"]) / "features.npz")
    again = joblib.load(run / "fold_0/quantum.joblib")["model"]
    expected = _oof(run)
    expected = expected[
        (expected.model == "quantum") & (expected.fold == 0)
    ].sort_values("subject_id")
    order = {sid: pos for pos, sid in enumerate(dataset.subject_ids.tolist())}
    rows = np.array([order[sid] for sid in expected.subject_id])
    predicted, scores = again.predict(dataset.X[rows])
    np.testing.assert_array_equal(predicted, expected.predicted_label.to_numpy())
    np.testing.assert_allclose(scores, expected.decision_score.to_numpy(), atol=0)
