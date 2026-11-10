"""Verify trusted local recorded bundles, OOF regeneration and real app/CLI agreement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.runtime.memory_media_file_storage import _calculate_file_id
from streamlit.testing.v1 import AppTest

from asd_eeg.config import Config
from asd_eeg.data.manifest import read_manifest
from asd_eeg.evaluation.metrics import summarize_oof
from asd_eeg.evaluation.splits import make_subject_folds
from asd_eeg.features.cache import load_features
from asd_eeg.inference.artifacts import (
    check_run_workspace,
    load_run_bundle,
    read_results,
    read_run,
    run_file,
)
from asd_eeg.inference.export import prediction_json
from asd_eeg.training.model import MODELS
from asd_eeg.workspace import check_feature_workspace


def _read_workspace(run: Path, features: Path, manifest: Path,
                    provenance_root: Path | None = None):
    """Match the frozen run, ordered feature rows, and selected manifest read-only."""
    summary, folds = read_run(run)
    if summary["data_kind"] != "recorded":
        raise ValueError("Verification requires a complete recorded-data run.")
    dataset = load_features(features)
    config = Config.from_dict(dataset.metadata["config"])
    check_feature_workspace(dataset, manifest, config)
    check_run_workspace(summary, folds, read_manifest(manifest), config, run,
                        provenance_root=(provenance_root if provenance_root is not None
                                         else Path(__file__).resolve().parents[1] / "artifacts"))
    fingerprint = hashlib.sha256(dataset.X.tobytes() + dataset.y.tobytes()
                                 + "\n".join(dataset.subject_ids).encode()).hexdigest()
    if summary["feature_fingerprint"] != fingerprint:
        raise ValueError("Run feature fingerprint differs from the selected archive.")
    expected_folds = make_subject_folds(dataset.subject_ids, dataset.y,
                                       config.evaluation.folds, config.evaluation.seed)
    if not folds[["subject_id", "fold", "label"]].equals(expected_folds):
        raise ValueError("Saved folds differ from the frozen configuration or feature-row order.")
    return summary, folds, dataset, config


def verify_run(run: Path, features: Path, manifest: Path) -> dict:
    run, features, manifest = (path.resolve() for path in (run, features, manifest))
    summary, folds, dataset, _ = _read_workspace(run, features, manifest)
    read_results(run)
    oof = pd.read_csv(run_file(run, "oof_predictions.csv"), dtype={"subject_id": str},
                      keep_default_na=False)
    regenerated = summarize_oof(oof)
    indices = {str(sid): i for i, sid in enumerate(dataset.subject_ids)}
    checks = []
    for fold in sorted(folds.fold.unique()):
        for name in MODELS:
            _, bundle = load_run_bundle(run, int(fold), name)
            meta, model = bundle["metadata"], bundle["model"]
            train = np.array([indices[s] for s in meta["train_subject_ids"]])
            test = np.array([indices[s] for s in meta["test_subject_ids"]])
            assert np.allclose(model.scaler.mean_, dataset.X[train].mean(axis=0)), "Scaler leakage/order."
            assert np.allclose(model.scaler.var_, dataset.X[train].var(axis=0)), "Scaler variance differs."
            assert np.allclose(model.feature_reference, np.median(dataset.X[train], axis=0))
            assert np.allclose(model.train_angles, model.angles(dataset.X[train]))
            if model.train_states is not None:
                assert model.train_states.shape[0] == len(train), "Training reference columns truncated."
            labels, scores = model.predict(dataset.X[test])
            expected = oof[oof.model.eq(name) & oof.fold.eq(fold)].set_index("subject_id").loc[
                meta["test_subject_ids"]]
            assert np.array_equal(labels, expected.predicted_label.to_numpy())
            assert np.allclose(scores, expected.decision_score.to_numpy(), rtol=1e-9, atol=1e-9)
            checks.append({"fold": int(fold), "model": name, "training_subjects": len(train),
                           "held_out_subjects": len(test), "training_only_fit": True,
                           "reference_order_verified": True, "saved_oof_matches": True})
    return {"run_id": summary["run_id"], "metrics_regenerate": True,
            "frozen_workspace_matches": True, "feature_fingerprint": summary["feature_fingerprint"],
             "all_bundle_checks": checks, "models": regenerated}


def verify_app(root: Path, run: Path, features: Path, manifest: Path, cases: list[str]) -> dict:
    root, run, features, manifest = (path.resolve() for path in (root, run, features, manifest))
    summary, folds, _, config = _read_workspace(run, features, manifest, root / "artifacts")
    app = AppTest.from_file(str(root / "app/main.py"), default_timeout=120)
    # Override this test session only, never activate/write a global workspace profile.
    selected_paths = {"manifest_path": manifest, "config_path": run_file(run, "config.json"),
                      "features_path": features, "results_path": run.parent}
    for key, path in selected_paths.items():
        app.session_state[key] = str(path)
    app.run()
    assert not app.exception, "App startup failed."
    assert not app.error, "App startup displayed a workspace error."
    for key, path in selected_paths.items():
        assert Path(app.session_state[key]).resolve() == path, "App ignored the selected workspace."
    app.sidebar.radio[0].set_value("Analysis").run()
    assert not app.exception, "Recorded Analysis page failed."
    experiment = next(e for e in app.selectbox if e.label == "Experiment")
    experiment.set_value(experiment.options.index(summary["run_id"])).run()
    assert not app.exception and not app.error, "Selected run is unavailable in the app."
    checks = []
    oof = pd.read_csv(run_file(run, "oof_predictions.csv"), dtype={"subject_id": str},
                      keep_default_na=False)
    for subject in cases:
        fold = int(folds.loc[folds.subject_id.eq(subject), "fold"].iloc[0])
        bundle_path, bundle = load_run_bundle(run, fold, "quantum")
        next(e for e in app.selectbox if e.label == "Held-out subject").set_value(subject).run()
        next(e for e in app.selectbox if e.label == "Model").set_value("quantum").run()
        next(e for e in app.button if e.label == "Analyze subject").click().run()
        assert not app.exception, "Recorded app inference raised an exception."
        assert not app.error, "Recorded app inference displayed an error."
        selection, result = app.session_state["analysis"]
        signature, selected_subject = selection
        assert selected_subject == subject
        assert [Path(item[0]).resolve() for item in signature] == [
            manifest, bundle_path, run_file(run, "folds.csv")
        ], "App selected another run/bundle path (possibly a same-named default run)."
        expected = oof[oof.subject_id.eq(subject) & oof.model.eq("quantum")].iloc[0]
        assert result["subject_id"] == subject and result["provenance"] == "held-out"
        for key in ("run_id", "dataset", "data_kind", "config_id", "fold", "model"):
            assert result[key] == bundle["metadata"][key], f"App export provenance differs: {key}."
        assert result["config_id"] == config.id
        assert result["fold"] == int(expected.fold)
        assert result["predicted_label"] == int(expected.predicted_label)
        assert np.isclose(result["decision_score"], expected.decision_score, rtol=1e-9, atol=1e-9)
        next(e for e in app.button if e.label == "Compute connection sensitivity").click().run()
        assert not app.exception and not app.error
        sensitivity = app.session_state["sensitivity"][1]
        assert len(sensitivity) == 6
        assert all(np.isfinite(row["score_change"]) for row in sensitivity)
        next(e for e in app.button if e.label == "Draw saved circuit").click().run()
        assert not app.exception and not app.error
        circuit = app.session_state["circuit_image"][1]
        assert circuit.startswith(b"\x89PNG\r\n\x1a\n"), "Circuit export is not a PNG."
        exported = {**result, "connection_sensitivity": sensitivity,
                    "sensitivity_reference": "Training-fold feature medians; no refitting"}
        payload = prediction_json(exported).encode("utf-8")
        download = next(e for e in app.get("download_button")
                        if e.proto.label == "Export analysis JSON")
        # AppTest tears down media storage after each run. Its content-addressed
        # URL still verifies the exact downloadable JSON bytes without a server.
        file_id = _calculate_file_id(payload, "application/json", "prediction.json")
        assert Path(download.proto.url).stem == file_id, "App JSON download differs from analysis."
        checks.append({"subject_id": subject, "matches_oof": True, "six_edges_finite": True,
                        "selected_run_and_bundle_match": True, "export_provenance_matches": True,
                        "actual_json_download_matches": True,
                        "actual_circuit_png_bytes": len(circuit),
                       "actual_circuit_png_sha256": hashlib.sha256(circuit).hexdigest(),
                       "strict_json_serializable": True})
    return {"startup_and_analysis_pass": True, "checks": checks}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repeat-run", type=Path)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.run, args.features, args.manifest = (
        path.resolve() for path in (args.run, args.features, args.manifest))
    report = {"status": "complete", "recorded_run": verify_run(args.run, args.features, args.manifest)}
    if args.repeat_run:
        for filename in ("folds.csv", "oof_predictions.csv", "fold_auc.csv"):
            a, b = (pd.read_csv(path / filename) for path in (args.run, args.repeat_run))
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)
        report["deterministic_repeat"] = {"matches": True, "files": ["folds", "oof", "fold_auc"],
                                          "run_id": args.repeat_run.name}
    report["application"] = verify_app(Path(__file__).resolve().parents[1], args.run,
                                       args.features, args.manifest, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"status": "complete", "report": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
