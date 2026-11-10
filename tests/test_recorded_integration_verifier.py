"""Verifier regressions with generated software fixtures, never research evidence.

The tiny fixture declares recorded metadata only to exercise recorded-workspace
guards. Every signal, archive, run, and decoy is created directly under tmp_path;
no recorded dataset is copied and no active workspace profile is written.
"""

import json
from dataclasses import replace
from pathlib import Path

import joblib
import pandas as pd
import pytest

from asd_eeg import workspace
from asd_eeg.evaluation.run_experiments import run_evaluation
from asd_eeg.features.cache import build_features, load_features, save_features
from asd_eeg.inference import export, pipeline
from asd_eeg.inference.compatibility import load_bundle
from asd_eeg.self_check import create_synthetic_manifest
from scripts import verify_recorded_integration as verifier

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def verifier_workspace(config, tmp_path, monkeypatch):
    settings = replace(
        config,
        preprocessing=replace(config.preprocessing, window_seconds=2, min_windows=1, max_windows=2),
        circuit=replace(config.circuit, passes=1),
        evaluation=replace(config.evaluation, folds=2),
    )
    manifest = create_synthetic_manifest(settings, tmp_path / "custom/cohort", subjects=4)
    table = pd.read_csv(manifest, dtype={"subject_id": str})
    table["data_kind"] = "recorded"  # Software-test selection fixture, not recorded evidence.
    table.to_csv(manifest, index=False)
    features = tmp_path / "custom/archives/selected.npz"
    dataset = build_features(manifest, settings, features)
    run = run_evaluation(dataset, tmp_path / "custom/experiments")
    # Match the extended artifact contract with explicit frozen session evidence.
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["feature_metadata"] = dataset.metadata
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    app_root = tmp_path / "app-root"
    (app_root / "app").mkdir(parents=True)
    # Execute the real application with an isolated root, without copying data/code.
    (app_root / "app/main.py").write_text(
        f"import runpy\nrunpy.run_path({str(ROOT / 'app/main.py')!r})\n", encoding="utf-8"
    )
    monkeypatch.setenv("NEUROQ_ROOT", str(app_root))
    # A different active/default results selection must never win over --run.
    monkeypatch.setattr(workspace, "workspace_paths", lambda root: {
        "manifest": str(manifest), "config": str(run / "config.json"),
        "features": str(features), "results": "results",
    })

    def reject_activation(*args, **kwargs):
        pytest.fail("The verifier must not activate or write a global workspace profile.")

    monkeypatch.setattr(workspace, "activate_workspace", reject_activation)
    folds = pd.read_csv(run / "folds.csv", dtype={"subject_id": str})
    cases = [folds.loc[folds.fold.eq(fold), "subject_id"].iloc[0] for fold in (0, 1)]
    return app_root, run, features, manifest, cases


def make_same_named_decoy(app_root, run, features):
    """A complete same-name run with identical OOF, so path checks are essential."""
    decoy = run_evaluation(load_features(features), app_root / "results")
    renamed = decoy.with_name(run.name)
    decoy.rename(renamed)
    summary_path = renamed / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_id"] = run.name
    summary["feature_metadata"] = load_features(features).metadata
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for path in renamed.glob("fold_*/*.joblib"):
        bundle = load_bundle(path)
        bundle["metadata"]["run_id"] = run.name
        joblib.dump(bundle, path)
    return renamed


def test_custom_run_app_verifies_actual_path_with_same_named_default_decoy(verifier_workspace):
    app_root, run, features, manifest, cases = verifier_workspace
    decoy = make_same_named_decoy(app_root, run, features)
    before = {path: path.read_bytes() for path in decoy.rglob("*") if path.is_file()}

    run_report = verifier.verify_run(run, features, manifest)
    assert run_report["frozen_workspace_matches"]
    assert len(run_report["all_bundle_checks"]) == 8
    app_report = verifier.verify_app(app_root, run, features, manifest, cases)
    assert app_report["startup_and_analysis_pass"]
    assert len(app_report["checks"]) == 2
    for check in app_report["checks"]:
        assert check["selected_run_and_bundle_match"]
        assert check["matches_oof"]
        assert check["export_provenance_matches"]
        assert check["actual_json_download_matches"]
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert not (app_root / workspace.PROFILE).exists()


@pytest.mark.parametrize("field,value", [
    ("config_id", "wrong-config"), ("fold", 99), ("model", "rbf"), ("run_id", "wrong-run"),
])
def test_app_rejects_wrong_export_provenance(verifier_workspace, monkeypatch, field, value):
    app_root, run, features, manifest, cases = verifier_workspace
    original = pipeline.analyze_subject

    def wrong_provenance(*args, **kwargs):
        return {**original(*args, **kwargs), field: value}

    monkeypatch.setattr(pipeline, "analyze_subject", wrong_provenance)
    with pytest.raises(AssertionError, match="export provenance differs"):
        verifier.verify_app(app_root, run, features, manifest, cases[:1])


def test_app_rejects_download_bytes_differing_from_analysis(verifier_workspace, monkeypatch):
    app_root, run, features, manifest, cases = verifier_workspace
    original = export.prediction_json

    def changed_download(prediction):
        if "run_id" in prediction:
            prediction = {**prediction, "run_id": "wrong-download-run"}
        return original(prediction)

    monkeypatch.setattr(export, "prediction_json", changed_download)
    with pytest.raises(AssertionError, match="JSON download differs"):
        verifier.verify_app(app_root, run, features, manifest, cases[:1])


@pytest.mark.parametrize("field,value,error", [
    ("feature_fingerprint", "wrong-fingerprint", "feature fingerprint"),
    ("dataset", "unrelated-software-fixture", "dataset or configuration"),
    ("data_kind", "synthetic", "recorded-data run"),
])
def test_run_rejects_frozen_provenance_changes(verifier_workspace, field, value, error):
    _, run, features, manifest, _ = verifier_workspace
    path = run / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary[field] = value
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        verifier.verify_run(run, features, manifest)


def test_run_rejects_changed_numeric_archive(verifier_workspace):
    _, run, features, manifest, _ = verifier_workspace
    dataset = load_features(features)
    dataset.X[0, 0] += .01
    save_features(dataset, features)
    with pytest.raises(ValueError, match="feature fingerprint"):
        verifier.verify_run(run, features, manifest)


def test_run_rejects_changed_frozen_configuration(verifier_workspace):
    _, run, features, manifest, _ = verifier_workspace
    path = run / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["circuit"]["passes"] = 2
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration differs"):
        verifier.verify_run(run, features, manifest)


def test_run_rejects_reordered_frozen_fold_rows(verifier_workspace):
    _, run, features, manifest, _ = verifier_workspace
    path = run / "folds.csv"
    folds = pd.read_csv(path, dtype={"subject_id": str})
    folds.iloc[::-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="feature-row order"):
        verifier.verify_run(run, features, manifest)


def test_run_rejects_reordered_bundle_training_references(verifier_workspace):
    _, run, features, manifest, _ = verifier_workspace
    path = run / "fold_0/quantum.joblib"
    bundle = load_bundle(path)
    bundle["metadata"]["train_subject_ids"].reverse()
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="training-reference order"):
        verifier.verify_run(run, features, manifest)


def test_main_passes_resolved_explicit_workspace_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    arguments = [Path("custom/run-selected"), Path("archives/selected.npz"), Path("cohort/data.csv")]
    resolved = [path.resolve() for path in arguments]
    calls = []
    monkeypatch.setattr(verifier, "verify_run", lambda *args: {"run_id": "run-selected"})

    def capture_app(root, run, features, manifest, cases):
        calls.append((root, run, features, manifest, cases))
        return {"startup_and_analysis_pass": True}

    monkeypatch.setattr(verifier, "verify_app", capture_app)
    assert verifier.main([
        "--run", str(arguments[0]), "--features", str(arguments[1]),
        "--manifest", str(arguments[2]), "--case", "software-subject",
        "--output", str(tmp_path / "report.json"),
    ]) == 0
    assert calls == [(ROOT, *resolved, ["software-subject"])]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
