"""
Artifact compatibility checks — Pranav Karhale's submission-verification suite.

Covers:
  - Bundle schema and class-mapping enforcement.
  - Feature-shape rejection on bad inputs.
  - Run summary completeness (status, subject counts, fold counts).
  - OOF predictions agree with saved metrics (re-derivation from CSV).
  - Submission archive portability (paths are relative or survive relocation).
  - Missing-file and incompatible-input errors produce actionable messages.
  - CLI and pipeline predictions agree.
  - Version metadata is recorded in every bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from asd_eeg.evaluation.metrics import summarize_oof
from asd_eeg.features.cache import load_features
from asd_eeg.inference.compatibility import load_bundle
from asd_eeg.training.model import MODELS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_path(completed_check: dict) -> Path:
    return Path(completed_check["run"])


def _oof(run: Path) -> pd.DataFrame:
    return pd.read_csv(run / "oof_predictions.csv")


def _summary(run: Path) -> dict:
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Bundle schema and provenance
# ---------------------------------------------------------------------------

class TestBundleSchema:
    """Model bundles must pass load_bundle's integrity checks."""

    def test_all_fold_bundles_load_cleanly(self, completed_check):
        run = _run_path(completed_check)
        for model_name in MODELS:
            for bundle_path in sorted(run.glob(f"fold_*/{model_name}.joblib")):
                bundle = load_bundle(bundle_path)
                assert bundle["schema_version"] == 1
                assert "model" in bundle and "metadata" in bundle

    def test_class_mapping_is_correct_in_all_bundles(self, completed_check):
        run = _run_path(completed_check)
        for bundle_path in sorted(run.glob("fold_*/quantum.joblib")):
            bundle = load_bundle(bundle_path)
            mapping = bundle["metadata"]["class_mapping"]
            assert mapping[0] == "Control"
            assert mapping[1] == "ASD"

    def test_config_id_matches_stored_model(self, completed_check):
        run = _run_path(completed_check)
        for bundle_path in sorted(run.glob("fold_*/rbf.joblib")):
            bundle = load_bundle(bundle_path)
            meta = bundle["metadata"]
            assert meta["config_id"] == bundle["model"].config.id

    def test_training_and_test_ids_are_disjoint_in_all_bundles(self, completed_check):
        run = _run_path(completed_check)
        for model_name in MODELS:
            for bundle_path in sorted(run.glob(f"fold_*/{model_name}.joblib")):
                bundle = load_bundle(bundle_path)
                meta = bundle["metadata"]
                train = set(meta["train_subject_ids"])
                test = set(meta["test_subject_ids"])
                assert not train & test, (
                    f"Bundle {bundle_path.name}: training and test sets overlap."
                )

    def test_reference_state_count_matches_training_subjects(self, completed_check):
        run = _run_path(completed_check)
        for bundle_path in sorted(run.glob("fold_*/quantum.joblib")):
            bundle = load_bundle(bundle_path)
            model = bundle["model"]
            meta = bundle["metadata"]
            # Quantum and reassigned bundles cache training states.
            if model.train_states is not None:
                assert len(model.train_states) == len(meta["train_subject_ids"])
            assert len(model.train_angles) == len(meta["train_subject_ids"])

    def test_versions_recorded_in_bundle_metadata(self, completed_check):
        run = _run_path(completed_check)
        bundle = load_bundle(run / "fold_0/quantum.joblib")
        versions = bundle["metadata"]["versions"]
        assert "python" in versions
        assert "numpy" in versions
        assert "scikit-learn" in versions
        assert "pennylane" in versions

    def test_absent_bundle_raises_clear_file_not_found(self, completed_check):
        run = _run_path(completed_check)
        with pytest.raises(FileNotFoundError, match="[Bb]undle not found"):
            load_bundle(run / "fold_0/absent.joblib")

    def test_tampered_bundle_config_id_is_rejected(self, completed_check, tmp_path):
        bundle = load_bundle(_run_path(completed_check) / "fold_0/quantum.joblib")
        bundle["metadata"]["config_id"] = "tampered"
        path = tmp_path / "bad.joblib"
        joblib.dump(bundle, path)
        with pytest.raises(ValueError, match="inconsistent"):
            load_bundle(path)

    def test_wrong_schema_version_is_rejected(self, completed_check, tmp_path):
        bundle = load_bundle(_run_path(completed_check) / "fold_0/quantum.joblib")
        bundle["schema_version"] = 99
        path = tmp_path / "wrong_version.joblib"
        joblib.dump(bundle, path)
        with pytest.raises(ValueError):
            load_bundle(path)


# ---------------------------------------------------------------------------
# Feature-shape compatibility
# ---------------------------------------------------------------------------

class TestFeatureShapeCompatibility:
    """Models reject features that do not match the trained shape."""

    def test_feature_too_wide_raises_value_error(self, completed_check):
        bundle = load_bundle(_run_path(completed_check) / "fold_0/quantum.joblib")
        model = bundle["model"]
        bad_width = model.config.n_features + 1
        with pytest.raises(ValueError, match="do not match|match the fitted"):
            model.predict(np.zeros((1, bad_width)))

    def test_feature_too_narrow_raises_value_error(self, completed_check):
        bundle = load_bundle(_run_path(completed_check) / "fold_0/quantum.joblib")
        model = bundle["model"]
        bad_width = max(1, model.config.n_features - 1)
        with pytest.raises(ValueError):
            model.predict(np.zeros((1, bad_width)))

    def test_nonfinite_features_raise_value_error(self, completed_check):
        bundle = load_bundle(_run_path(completed_check) / "fold_0/quantum.joblib")
        model = bundle["model"]
        bad = np.zeros((1, model.config.n_features))
        bad[0, 0] = float("nan")
        with pytest.raises(ValueError):
            model.predict(bad)

    def test_all_model_types_reject_shape_mismatch(self, completed_check):
        run = _run_path(completed_check)
        for model_name in MODELS:
            bundle = load_bundle(run / f"fold_0/{model_name}.joblib")
            model = bundle["model"]
            bad = np.zeros((1, model.config.n_features + 2))
            with pytest.raises(ValueError):
                model.predict(bad)


# ---------------------------------------------------------------------------
# Run summary and result-file completeness
# ---------------------------------------------------------------------------

class TestRunSummaryCompleteness:
    """The run folder must have a complete, consistent summary."""

    def test_summary_status_is_complete(self, completed_check):
        summary = _summary(_run_path(completed_check))
        assert summary["status"] == "complete", (
            f"Run status is '{summary['status']}', not 'complete'."
        )

    def test_summary_has_required_fields(self, completed_check):
        summary = _summary(_run_path(completed_check))
        required = {
            "schema_version", "run_id", "status", "created_at",
            "dataset", "data_kind", "subjects", "class_counts",
            "folds", "config_id", "feature_fingerprint",
            "protocol", "versions",
        }
        missing = required - set(summary)
        assert not missing, f"summary.json is missing fields: {missing}"

    def test_subject_count_in_summary_matches_oof(self, completed_check):
        run = _run_path(completed_check)
        summary = _summary(run)
        oof = _oof(run)
        # Every model must have exactly summary["subjects"] distinct subjects.
        for model_name, rows in oof.groupby("model"):
            n = rows.subject_id.nunique()
            assert n == summary["subjects"], (
                f"Model '{model_name}' has {n} subjects in OOF; "
                f"summary says {summary['subjects']}."
            )

    def test_all_model_bundles_exist(self, completed_check):
        run = _run_path(completed_check)
        summary = _summary(run)
        for fold in range(summary["folds"]):
            for model_name in MODELS:
                path = run / f"fold_{fold}" / f"{model_name}.joblib"
                assert path.exists(), f"Missing bundle: {path}"

    def test_folds_csv_covers_all_subjects(self, completed_check):
        run = _run_path(completed_check)
        dataset = load_features(Path(completed_check["directory"]) / "features.npz")
        folds = pd.read_csv(run / "folds.csv", dtype={"subject_id": str})
        assert set(folds.subject_id) == set(dataset.subject_ids.tolist()), (
            "folds.csv subject set does not match the feature archive."
        )

    def test_model_comparison_csv_has_all_models(self, completed_check):
        run = _run_path(completed_check)
        comparison = pd.read_csv(run / "model_comparison.csv")
        assert set(comparison.model) == set(MODELS)

    def test_fold_auc_csv_exists_and_covers_all_models(self, completed_check):
        run = _run_path(completed_check)
        auc_csv = run / "fold_auc.csv"
        assert auc_csv.exists(), "fold_auc.csv is missing from the run folder."
        auc = pd.read_csv(auc_csv)
        assert set(auc.model) == set(MODELS)


# ---------------------------------------------------------------------------
# OOF–metrics consistency
# ---------------------------------------------------------------------------

class TestOOFMetricsConsistency:
    """Metrics in metrics.json must be re-derivable from oof_predictions.csv."""

    def test_metrics_regenerate_from_oof(self, completed_check):
        run = _run_path(completed_check)
        frame = _oof(run)
        saved = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        recomputed = summarize_oof(frame)
        assert set(recomputed) == set(saved), "Metrics keys differ."
        for name in MODELS:
            for key in (
                "subjects", "accuracy", "balanced_accuracy", "macro_f1",
                "sensitivity", "specificity",
            ):
                assert recomputed[name][key] == pytest.approx(saved[name][key]), (
                    f"Metric mismatch for model={name}, key={key}."
                )
            assert recomputed[name]["confusion_matrix"] == saved[name]["confusion_matrix"]

    def test_every_subject_has_exactly_one_oof_prediction_per_model(self, completed_check):
        oof = _oof(_run_path(completed_check))
        duplicates = oof.groupby(["subject_id", "model"]).size()
        assert duplicates.eq(1).all(), "Some subjects have duplicate OOF predictions."

    def test_oof_labels_are_binary(self, completed_check):
        oof = _oof(_run_path(completed_check))
        for col in ("true_label", "predicted_label"):
            assert oof[col].isin([0, 1]).all(), (
                f"Non-binary values found in column '{col}'."
            )

    def test_oof_decision_scores_are_finite(self, completed_check):
        oof = _oof(_run_path(completed_check))
        assert np.isfinite(oof.decision_score.to_numpy()).all(), (
            "Non-finite decision scores found in oof_predictions.csv."
        )

    def test_metrics_json_serialisable_without_nan(self, completed_check):
        run = _run_path(completed_check)
        raw = (run / "metrics.json").read_text(encoding="utf-8")
        # Strict check: re-serialise to ensure no NaN slipped through.
        metrics = json.loads(raw)
        json.dumps(metrics, allow_nan=False)  # raises ValueError on NaN

    def test_true_labels_consistent_across_models(self, completed_check):
        oof = _oof(_run_path(completed_check))
        label_check = oof.groupby("subject_id")["true_label"].nunique()
        assert label_check.eq(1).all(), (
            "Subject true_label differs across models in oof_predictions.csv."
        )


# ---------------------------------------------------------------------------
# Submission archive portability
# ---------------------------------------------------------------------------

class TestSubmissionArchivePortability:
    """Results must be relocatable: no absolute paths embedded in key artifacts."""

    def test_summary_json_has_no_absolute_paths(self, completed_check):
        summary = _summary(_run_path(completed_check))
        # config_id and feature_fingerprint are hashes, not paths.
        for field in ("config_id", "feature_fingerprint", "run_id"):
            value = summary.get(field, "")
            assert not str(value).startswith(("/", "C:\\", "c:\\")), (
                f"summary.json field '{field}' looks like an absolute path."
            )

    def test_prediction_json_is_valid_and_complete(self, completed_check):
        pred_path = Path(completed_check["directory"]) / "prediction.json"
        pred_path = pred_path.resolve()
        assert pred_path.exists(), (
            "prediction.json was not saved by run_self_check."
        )
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        for field in ("subject_id", "predicted_label", "predicted_class",
                      "decision_score", "provenance", "model", "fold"):
            assert field in pred, f"prediction.json is missing field '{field}'."

    def test_prediction_provenance_is_held_out(self, completed_check):
        pred_path = Path(completed_check["directory"]) / "prediction.json"
        pred_path = pred_path.resolve()
        assert pred_path.exists(), (
            "prediction.json was not saved by run_self_check."
        )
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        assert pred["provenance"] == "held-out", (
            "The example prediction was not from a held-out subject."
        )

    def test_feature_archive_has_no_object_arrays(self, completed_check):
        """NPZ archives must not contain Python objects (unpickling risk)."""
        npz = np.load(Path(completed_check["directory"]) / "features.npz",
                      allow_pickle=False)
        # 'metadata' is stored as a 0-d string array — verify it is a string dtype.
        assert npz["metadata"].dtype.kind in ("U", "S", "O"), (
            "metadata array has unexpected dtype."
        )
        # Feature matrix and labels must be numeric.
        assert npz["X"].dtype.kind == "f"
        assert npz["y"].dtype.kind in ("i", "u", "f")


# ---------------------------------------------------------------------------
# Error-state clarity
# ---------------------------------------------------------------------------

class TestErrorMessageClarity:
    """Missing files and bad inputs must produce actionable error messages."""

    def test_missing_manifest_subject_raises_clear_error(self, completed_check):
        bundle_path = _run_path(completed_check) / "fold_0/quantum.joblib"
        from asd_eeg.inference.pipeline import analyze_subject
        with pytest.raises(ValueError, match="not present in this manifest"):
            analyze_subject(completed_check["manifest"], "NONEXISTENT-SUBJECT", bundle_path)

    def test_missing_bundle_raises_file_not_found(self, completed_check):
        bundle_path = _run_path(completed_check) / "fold_0/absent.joblib"
        with pytest.raises(FileNotFoundError, match="[Bb]undle"):
            load_bundle(bundle_path)

    def test_training_subject_blocked_from_inference(self, completed_check):
        bundle = load_bundle(_run_path(completed_check) / "fold_0/quantum.joblib")
        train_id = bundle["metadata"]["train_subject_ids"][0]
        from asd_eeg.inference.compatibility import check_subject
        with pytest.raises(ValueError, match="used for training"):
            check_subject(bundle, train_id, bundle["metadata"]["dataset"])

    def test_cross_dataset_inference_blocked(self, completed_check):
        bundle = load_bundle(_run_path(completed_check) / "fold_0/quantum.joblib")
        held_out = bundle["metadata"]["test_subject_ids"][0]
        from asd_eeg.inference.compatibility import check_subject
        with pytest.raises(ValueError, match="dataset differs"):
            check_subject(bundle, held_out, "different-dataset")
