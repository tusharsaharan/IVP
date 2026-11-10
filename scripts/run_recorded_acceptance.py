"""Reproduce local recorded QC, features, evaluation and held-out exports.

Run from an installed checkout. Outputs contain participant-level research evidence
and must stay local. No configuration tuning or cloud upload is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml

from asd_eeg.config import load_config
from asd_eeg.data.manifest import read_manifest
from asd_eeg.data.screening import screen_manifest
from asd_eeg.data.sfari import sha256_file
from asd_eeg.evaluation.metrics import summarize_oof
from asd_eeg.evaluation.run_experiments import git_commit, run_evaluation, software_versions
from asd_eeg.explainability.perturbation import connection_sensitivity
from asd_eeg.features.cache import build_features, load_features
from asd_eeg.inference.compatibility import load_bundle
from asd_eeg.inference.pipeline import analyze_subject
from asd_eeg.models.quantum_circuit import make_qnode
from asd_eeg.training.model import MODELS
from asd_eeg.visualization.plots import confusion_figure, connectivity_figure
from asd_eeg.workspace import check_feature_workspace


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def verify_source(source: Path) -> dict:
    """Rehash every acquired payload and sidecar against the preserved audit."""
    source = source.resolve()
    report = json.loads((source / "download_report.json").read_text(encoding="utf-8"))
    provenance = json.loads((source / "source_provenance.json").read_text(encoding="utf-8"))
    if report.get("status") != "downloaded_verified" or not report.get("files"):
        raise ValueError("Source needs a complete hash-verified acquisition report.")
    if report.get("snapshot") != provenance.get("snapshot"):
        raise ValueError("Acquisition and metadata snapshots differ.")
    total = 0
    for item in report["files"]:
        path = (source / item["relative_path"]).resolve()
        if source not in path.parents:
            raise ValueError("Acquisition audit path escapes the source directory.")
        if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Payload no longer matches acquisition: {path.name}")
        total += path.stat().st_size
    for name, expected in provenance["metadata_sha256"].items():
        path = (source / name).resolve()
        if source not in path.parents or sha256_file(path) != expected:
            raise ValueError("Source metadata no longer matches acquisition.")
    return {"snapshot": report["snapshot"], "verified_recordings": len(report["files"]),
            "verified_bytes": total, "verified_metadata_files": len(provenance["metadata_sha256"]),
            "download_report_sha256": sha256_file(source / "download_report.json"),
            "source_provenance_sha256": sha256_file(source / "source_provenance.json")}


def verify_candidate_labels(table: pd.DataFrame, source: Path) -> dict:
    """Reconcile manifest labels and payloads with explicit participant metadata."""
    source = source.resolve()
    participants = pd.read_csv(source / "participants.tsv", sep="\t", dtype=str,
                               keep_default_na=False).set_index("participant_id")
    if participants.index.duplicated().any():
        raise ValueError("Source participant identifiers must be unique.")
    definitions = json.loads((source / "participants.json").read_text(encoding="utf-8"))
    levels = definitions.get("group", {}).get("Levels", {})
    if levels.get("ASD") != "autism spectrum disorder" or levels.get("TD") != "typically developing":
        raise ValueError("Source group definitions require scientific review.")
    report = json.loads((source / "download_report.json").read_text(encoding="utf-8"))
    files = report["files"]
    paths = {(source / item["relative_path"]).resolve(): item for item in files}
    if len(paths) != len(files) or len(table) != len(files):
        raise ValueError("Candidate manifest does not contain every acquired recording exactly once.")
    if set(Path(p).resolve() for p in table.eeg_path) != set(paths):
        raise ValueError("Candidate payload paths differ from the acquisition report.")
    for row in table.itertuples():
        item = paths[Path(row.eeg_path).resolve()]
        if row.subject_id not in participants.index:
            raise ValueError("Candidate subject absent from source metadata.")
        participant = participants.loc[row.subject_id]
        group = participant["group"]
        if group not in {"ASD", "TD"} or participant["completed_Resting_state"].lower() != "yes":
            raise ValueError("Candidate does not satisfy explicit source eligibility.")
        if (int(row.label) != int(group == "ASD") or item["subject_id"] != row.subject_id
                or int(item["label"]) != int(row.label)
                or Path(item["relative_path"]).stem != row.recording_id):
            raise ValueError("Candidate labels/grouping differ from explicit source metadata.")
    hashes = pd.DataFrame(files).groupby("sha256").subject_id.nunique()
    if hashes.gt(1).any():
        raise ValueError("Identical payload occurs across subjects; review potential leakage.")
    return {"explicit_source_labels_match": True, "all_acquired_payloads_in_candidates": True,
            "cross_subject_duplicate_payloads": 0, "candidate_recordings": len(table),
            "candidate_subjects": int(table.subject_id.nunique())}


def compare_cohorts(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    columns = ["dataset", "subject_id", "recording_id", "label", "data_kind", "label_source"]
    a = left[columns].sort_values("recording_id").reset_index(drop=True)
    b = right[columns].sort_values("recording_id").reset_index(drop=True)
    if not a.equals(b):
        raise ValueError("Reproduced QC differs from the previously frozen cohort; do not evaluate.")
    return {"identical_recording_metadata": True, "recordings": len(a),
            "subjects": int(a.subject_id.nunique())}


def validate_archive_provenance(dataset, manifest: Path, config) -> None:
    """Require full protocol and session provenance, not only matching feature values."""
    dataset.validate()
    # Compare the entire serialized configuration: model/circuit/fold settings can
    # change without affecting X. JSON normalizes tuples from freshly built archives.
    if json.dumps(dataset.metadata["config"], sort_keys=True) != json.dumps(
        config.to_dict(), sort_keys=True,
    ):
        raise ValueError("Feature archive full configuration differs from the frozen protocol.")
    check_feature_workspace(dataset, manifest, config)
    table = read_manifest(manifest)
    details = dataset.metadata.get("subjects")
    if (not isinstance(details, list) or any(not isinstance(item, dict) for item in details)
            or sorted(item.get("subject_id", "") for item in details)
            != sorted(dataset.subject_ids.tolist())):
        raise ValueError("Feature archive needs complete, unique subject recording provenance.")
    if dataset.metadata.get("label_sources") != sorted(set(table.label_source)):
        raise ValueError("Feature archive label sources differ from the frozen cohort.")
    for item in details:
        quality = item.get("quality")
        active = set(table[table.subject_id.eq(item["subject_id"])].recording_id)
        if (not isinstance(quality, list) or any(not isinstance(record, dict) for record in quality)
                or sorted(record.get("recording_id", "") for record in quality) != sorted(active)):
            raise ValueError("Feature archive needs complete, unique recording provenance.")
        if dataset.window_recording_ids is not None:
            index = dataset.subject_ids.tolist().index(item["subject_id"])
            used = set(dataset.window_recording_ids[dataset.window_subject_index == index])
            if not used <= active:
                raise ValueError("Feature archive window recordings differ from the frozen cohort.")


def verify_feature_reproduction(previous, rebuilt, manifest: Path, config) -> None:
    """Fail before evaluation if either archive departs from the requested frozen protocol."""
    validate_archive_provenance(previous, manifest, config)
    validate_archive_provenance(rebuilt, manifest, config)
    same = (np.array_equal(previous.subject_ids, rebuilt.subject_ids)
            and np.array_equal(previous.y, rebuilt.y)
            and previous.X.shape == rebuilt.X.shape
            and np.allclose(previous.X, rebuilt.X, atol=1e-10, rtol=1e-10))
    if not same:
        raise ValueError("Rebuilt features differ from previously frozen archive; review before evaluation.")


def select_cases(table: pd.DataFrame) -> list[str]:
    """Choose one lexicographically first participant per class, before evaluation."""
    subjects = table.groupby("subject_id", sort=True).label.first()
    if set(subjects) != {0, 1}:
        raise ValueError("Recorded acceptance needs both classes.")
    return [str(subjects[subjects.eq(label)].index[0]) for label in (0, 1)]


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def export_cases(manifest: Path, dataset, run: Path, output: Path, cases: list[str]) -> dict:
    folds = pd.read_csv(run / "folds.csv", dtype={"subject_id": str})
    oof = pd.read_csv(run / "oof_predictions.csv", dtype={"subject_id": str})
    figures = output / "figures"
    figures.mkdir()
    checks = []
    for case_index, subject in enumerate(cases):
        index = dataset.subject_ids.tolist().index(subject)
        fold = int(folds.loc[folds.subject_id.eq(subject), "fold"].item())
        for name in MODELS:
            bundle_path = run / f"fold_{fold}" / f"{name}.joblib"
            result = analyze_subject(manifest, subject, bundle_path)
            expected = oof[oof.subject_id.eq(subject) & oof.model.eq(name)].iloc[0]
            if result["predicted_label"] != int(expected.predicted_label):
                raise ValueError("Raw-input inference label differs from saved OOF.")
            if not np.isclose(result["decision_score"], expected.decision_score, atol=1e-9, rtol=1e-9):
                raise ValueError("Raw-input inference score differs from saved OOF.")
            if not np.allclose(result["feature_vector"], dataset.X[index], atol=1e-10, rtol=1e-10):
                raise ValueError("Raw-input inference features differ from the rebuilt archive.")
            bundle = load_bundle(bundle_path)
            result["sensitivity"] = connection_sensitivity(bundle, dataset.X[index])
            result["evaluation_truth_after_prediction"] = int(expected.true_label)
            write_json(output / f"case_{case_index}_{name}.json", result)
            checks.append({"case": case_index, "subject_id": subject, "fold": fold, "model": name,
                           "matches_oof": True, "matches_rebuilt_features": True,
                           "predicted_label": result["predicted_label"],
                           "true_label": int(expected.true_label),
                           "decision_score": result["decision_score"]})
            if name == "quantum":
                save_figure(connectivity_figure(result["connectivity"], result["node_names"]),
                            figures / f"case_{case_index}_connectivity.png")
                angles = bundle["model"].angles(dataset.X[index][None, :])[0]
                fig, _ = qml.draw_mpl(make_qnode(bundle["model"].config), decimals=2)(angles)
                fig.suptitle(f"Actual fixed graph circuit / held-out case {case_index + 1}")
                save_figure(fig, figures / f"case_{case_index}_circuit.png")
                sensitivity = result["sensitivity"]
                fig, ax = plt.subplots(figsize=(9, 4.5))
                ax.barh([s["connection"] for s in sensitivity], [s["score_change"] for s in sensitivity])
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Decision score change (perturbed minus original)")
                ax.set_title("Training-reference edge sensitivity; not causal or clinical")
                fig.tight_layout()
                save_figure(fig, figures / f"case_{case_index}_sensitivity.png")
    metrics = summarize_oof(oof)
    for name in MODELS:
        save_figure(confusion_figure(metrics[name]["confusion_matrix"]), figures / f"{name}_confusion.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODELS))
    for offset, key in enumerate(("accuracy", "balanced_accuracy", "mean_fold_auc")):
        ax.bar(x + (offset - 1) * .24, [metrics[m][key] for m in MODELS], .24, label=key)
    ax.set_xticks(x, MODELS)
    ax.set_ylim(0, 1)
    ax.axhline(.5, linestyle="--", color="gray", linewidth=.8)
    ax.set_ylabel("Score")
    ax.set_title("Fixed SFARI subject-wise OOF evaluation / AUC averaged within folds")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figures / "recorded_model_comparison.png")
    return {"selection": "Lexicographically first participant per class, selected before evaluation",
            "checks": checks, "all_match": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--previous-manifest", required=True, type=Path)
    parser.add_argument("--previous-features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--results", default="results", type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "scope": "Independent unchanged SFARI protocol; not TUOS or a clinical validation",
                "git_commit": git_commit(), "versions": software_versions()}
    try:
        config = load_config(args.config)
        table = read_manifest(args.manifest)
        if set(table.data_kind) != {"recorded"} or set(table.dataset) != {"ds006780"}:
            raise ValueError("This acceptance workflow requires recorded ds006780 only.")
        evidence["source"] = verify_source(args.source)
        evidence["candidate_provenance"] = verify_candidate_labels(table, args.source)
        evidence["config_id"] = config.id
        evidence["config"] = config.to_dict()
        print("Source hashes verified; reproducing fixed recording QC.", flush=True)
        screen = screen_manifest(args.manifest, config, output / "qc")
        manifest = Path(screen["manifest"])
        screened = read_manifest(manifest)
        evidence["screen"] = screen
        evidence["qc_reproduction"] = compare_cohorts(read_manifest(args.previous_manifest), screened)
        cases = select_cases(screened)
        evidence["cases_selected_before_evaluation"] = cases
        print("QC reproduced; rebuilding all subject features from raw recordings.", flush=True)
        features = output / "features" / "subject_features.npz"
        dataset = build_features(manifest, config, features)
        old = load_features(args.previous_features)
        verify_feature_reproduction(old, dataset, manifest, config)
        evidence["feature_reproduction"] = {"matches_previous": True, "shape": list(dataset.X.shape),
            "archive_sha256": sha256_file(features), "windows_min": int(dataset.window_counts.min()),
            "windows_max": int(dataset.window_counts.max()),
            "feature_fingerprint": hashlib.sha256(dataset.X.tobytes() + dataset.y.tobytes()
                + "\n".join(dataset.subject_ids).encode()).hexdigest()}
        write_json(output / "protocol_freeze.json", evidence)
        print("Protocol and demo cases frozen; evaluating all four models.", flush=True)
        run = run_evaluation(dataset, args.results)
        evidence["run"] = str(run)
        print(f"Evaluation complete: {run.name}; checking real held-out inference.", flush=True)
        evidence["held_out"] = export_cases(manifest, dataset, run, output, cases)
        evidence["status"] = "complete"
    except Exception as exc:
        evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "acceptance.json", evidence)
    print(json.dumps({"run": str(run), "evidence": str(output / "acceptance.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
