"""Read-only application artifact checks. Discovery never deserializes models."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from asd_eeg.config import Config
from asd_eeg.evaluation.metrics import summarize_oof
from asd_eeg.features.cache import load_features
from asd_eeg.inference.compatibility import load_bundle
from asd_eeg.training.model import MODELS


def run_file(run: str | Path, name: str) -> Path:
    """Require artifacts to stay within the selected local run, including symlinks."""
    root = Path(run).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Saved run artifact is missing or outside its run: {name}.")
    return path


def read_run(run: str | Path) -> tuple[dict, pd.DataFrame]:
    summary = json.loads(run_file(run, "summary.json").read_text(encoding="utf-8"))
    required = {"run_id", "status", "dataset", "data_kind", "subjects", "folds", "seed",
                "config_id", "class_counts", "protocol", "feature_fingerprint"}
    if not isinstance(summary, dict) or not required.issubset(summary):
        raise ValueError("Saved run summary is incomplete.")
    if summary["status"] != "complete":
        raise ValueError("Only completed runs can supply results or inference bundles.")
    if summary["run_id"] != Path(run).name:
        raise ValueError("Saved run ID differs from its folder.")
    for name in ("subjects", "folds"):
        if (not isinstance(summary[name], int) or isinstance(summary[name], bool)
                or summary[name] < (2 if name == "folds" else 1)):
            raise ValueError(f"Saved run {name} count is invalid.")
    config = Config.from_dict(json.loads(run_file(run, "config.json").read_text(encoding="utf-8")))
    if summary["config_id"] != config.id:
        raise ValueError("Saved run configuration differs from its summary.")
    folds = pd.read_csv(run_file(run, "folds.csv"), dtype={"subject_id": str}, keep_default_na=False)
    if not {"subject_id", "fold", "label"}.issubset(folds.columns):
        raise ValueError("Saved folds need subject_id, fold, and label columns.")
    if (len(folds) != summary["subjects"] or folds.subject_id.duplicated().any()
            or folds.subject_id.str.strip().eq("").any()
            or not folds.label.isin([0, 1]).all()
            or not folds.fold.isin(range(summary["folds"])).all()
            or set(folds.fold) != set(range(summary["folds"]))):
        raise ValueError("Saved folds have invalid subjects, labels, or fold assignments.")
    counts = {"Control": int(folds.label.eq(0).sum()), "ASD": int(folds.label.eq(1).sum())}
    if summary["class_counts"] != counts:
        raise ValueError("Saved run class counts differ from its folds.")
    for name in ("metrics.json", "model_comparison.csv", "oof_predictions.csv"):
        run_file(run, name)
    return summary, folds


def _recording_membership(metadata: dict, subjects: set[str]) -> dict[str, set[str]]:
    """Quality records cover every input session, including windows lost to a cap."""
    details = metadata.get("subjects") if isinstance(metadata, dict) else None
    if not isinstance(details, list) or not details:
        raise ValueError("Saved run recording provenance is missing for the whole cohort.")
    membership = {}
    seen_recordings = set()
    for subject in details:
        if not isinstance(subject, dict):
            raise ValueError("Saved run recording provenance has invalid subject records.")
        sid, quality = subject.get("subject_id"), subject.get("quality")
        if (not isinstance(sid, str) or sid not in subjects or sid in membership
                or not isinstance(quality, list) or not quality):
            raise ValueError("Saved run recording provenance is incomplete or duplicates subjects.")
        recordings = set()
        for record in quality:
            rid = record.get("recording_id") if isinstance(record, dict) else None
            if not isinstance(rid, str) or not rid.strip() or rid in seen_recordings:
                raise ValueError("Saved run recording provenance has invalid or duplicate recording IDs.")
            recordings.add(rid)
            seen_recordings.add(rid)
        membership[sid] = recordings
    if set(membership) != subjects:
        raise ValueError("Saved run recording provenance is missing subjects from the whole cohort.")
    return membership


def _run_feature_metadata(summary: dict, folds: pd.DataFrame, run: str | Path | None,
                          provenance_root: str | Path | None) -> dict:
    if "feature_metadata" in summary:
        metadata = summary["feature_metadata"]
        if (not isinstance(metadata, dict)
                or any(metadata.get(key) != summary[key]
                       for key in ("dataset", "data_kind", "config_id"))):
            raise ValueError("Saved run embedded feature provenance differs from its dataset or configuration.")
        try:
            saved_config = Config.from_dict(metadata["config"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Saved run embedded feature provenance has no valid configuration.") from exc
        if saved_config.id != summary["config_id"]:
            raise ValueError("Saved run embedded feature provenance configuration differs from its summary.")
        return metadata
    # Older evaluator runs did not embed session provenance. The existing recorded
    # acceptance workflow keeps its original archive separately, bound by archive
    # SHA-256 and run ID. Never substitute the workspace's mutable feature archive.
    if run is not None:
        root = (Path(provenance_root) if provenance_root is not None
                else Path(run).resolve().parent.parent / "artifacts")
        for path in sorted(root.rglob("acceptance.json")):
            try:
                evidence = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(evidence, dict) or not isinstance(evidence.get("run"), str):
                continue
            # Acceptance records contain the original local path; match the stable
            # run ID after relocation, also on a different operating system.
            if Path(evidence["run"].replace("\\", "/")).name != summary["run_id"]:
                continue
            reproduction = evidence.get("feature_reproduction")
            if (evidence.get("status") != "complete" or evidence.get("config_id") != summary["config_id"]
                    or not isinstance(reproduction, dict)
                    or reproduction.get("feature_fingerprint") != summary["feature_fingerprint"]):
                raise ValueError("Saved run acceptance provenance differs from the completed run.")
            archive = run_file(path.parent, "features/subject_features.npz")
            if hashlib.sha256(archive.read_bytes()).hexdigest() != reproduction.get("archive_sha256"):
                raise ValueError("Saved run original feature archive differs from its frozen provenance hash.")
            dataset = load_features(archive)
            fingerprint = hashlib.sha256(dataset.X.tobytes() + dataset.y.tobytes()
                                         + "\n".join(dataset.subject_ids).encode()).hexdigest()
            if (fingerprint != summary["feature_fingerprint"]
                    or any(dataset.metadata.get(key) != summary[key]
                           for key in ("dataset", "data_kind", "config_id"))
                    or dict(zip(dataset.subject_ids, dataset.y, strict=True))
                    != folds.set_index("subject_id").label.to_dict()):
                raise ValueError("Saved run original feature provenance differs from its cohort or configuration.")
            return dataset.metadata
    raise ValueError(
        "Legacy saved run has no frozen whole-cohort recording provenance; session compatibility "
        "cannot be verified for reproduction. Restore its original acceptance evidence and "
        "hash-verified feature archive; the active feature archive is not a substitute."
    )


def check_run_workspace(summary: dict, folds: pd.DataFrame, table: pd.DataFrame,
                        config: Config, run: str | Path | None = None, *,
                        provenance_root: str | Path | None = None) -> None:
    """Require identical subjects, labels and all sessions before saved-run reproduction.

    The optional run/provenance root locate existing frozen acceptance artifacts for
    legacy runs. Calls without frozen provenance fail closed, not with an API error.
    """
    if (summary["dataset"] != str(table.dataset.iloc[0])
            or summary["data_kind"] != str(table.data_kind.iloc[0])
            or summary["config_id"] != config.id):
        raise ValueError("Saved run dataset or configuration differs from the active workspace.")
    active = table.groupby("subject_id", sort=False).label.first().to_dict()
    if folds.set_index("subject_id").label.to_dict() != active:
        raise ValueError("Saved run subjects or labels differ from the active cohort.")
    metadata = _run_feature_metadata(summary, folds, run, provenance_root)
    saved = _recording_membership(metadata, set(active))
    current = table.groupby("subject_id", sort=False).recording_id.agg(set).to_dict()
    if saved != current:
        raise ValueError("Saved run recording sessions differ from the active cohort. Restore the original manifest.")


def load_run_bundle(run: str | Path, fold: int, model_name: str) -> tuple[Path, dict]:
    """Verify the selected bundle against the complete run's original fold order."""
    summary, folds = read_run(run)
    if model_name not in MODELS or fold not in range(summary["folds"]):
        raise ValueError("Select a model and held-out fold from the saved run.")
    path = run_file(run, f"fold_{fold}/{model_name}.joblib")
    bundle = load_bundle(path)
    meta = bundle["metadata"]
    for key in ("run_id", "dataset", "data_kind", "config_id", "feature_fingerprint"):
        if meta[key] != summary[key]:
            raise ValueError(f"Selected model bundle {key} differs from the saved run.")
    if meta["fold"] != fold or meta["model"] != model_name:
        raise ValueError("Selected model bundle does not belong to this model/fold.")
    if (meta["train_subject_ids"] != folds.loc[folds.fold != fold, "subject_id"].tolist()
            or meta["test_subject_ids"] != folds.loc[folds.fold == fold, "subject_id"].tolist()):
        raise ValueError("Selected bundle training-reference order or held-out subjects differ from saved folds.")
    return path, bundle


def read_results(run: str | Path) -> tuple[dict, pd.DataFrame]:
    """Reject damaged/stale result tables rather than displaying invented scores."""
    _, folds = read_run(run)
    predictions = pd.read_csv(run_file(run, "oof_predictions.csv"),
                              dtype={"subject_id": str}, keep_default_na=False)
    regenerated = summarize_oof(predictions)
    if set(regenerated) != set(MODELS):
        raise ValueError("Saved predictions must contain all four model comparisons.")
    for _, rows in predictions.groupby("model", sort=False):
        actual = rows.set_index("subject_id")[["fold", "true_label"]].rename(columns={"true_label": "label"})
        expected = folds.set_index("subject_id")[["fold", "label"]]
        if not actual.sort_index().equals(expected.sort_index()):
            raise ValueError("Saved predictions do not match the saved subject folds/labels.")
    metrics = json.loads(run_file(run, "metrics.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(run_file(run, "model_comparison.csv"))
    if (not isinstance(metrics, dict) or set(metrics) != set(MODELS)
            or "model" not in comparison or comparison.model.duplicated().any()
            or set(comparison.model) != set(MODELS)):
        raise ValueError("Saved metrics/comparison do not contain the four expected models.")
    for name, values in regenerated.items():
        if not isinstance(metrics[name], dict):
            raise ValueError("Saved model metrics must be JSON objects.")
        for key, expected in values.items():
            stored = metrics[name].get(key)
            if expected is None:
                matches = stored is None
            else:
                matches = stored is not None and np.allclose(stored, expected, rtol=1e-9, atol=1e-12)
            if not matches:
                raise ValueError("Saved metrics differ from authoritative out-of-fold predictions.")
            if key != "confusion_matrix":
                if key not in comparison:
                    raise ValueError(f"Saved comparison is missing {key}.")
                cell = comparison.loc[comparison.model == name, key].iloc[0]
                if not (pd.isna(cell) if expected is None else np.isclose(cell, expected, rtol=1e-9, atol=1e-12)):
                    raise ValueError("Saved comparison differs from authoritative out-of-fold predictions.")
    return metrics, comparison
