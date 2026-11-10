"""Private browser configuration guards, using ONLY fictional temporary evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from asd_eeg.config import load_config

SPEC = importlib.util.spec_from_file_location(
    "recorded_demo_preparer", Path(__file__).resolve().parents[1] / "scripts/prepare_recorded_demo.py",
)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    run = root / "results/custom/run-fixture"
    run.mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scripts/record_tushar_demo.js").write_text("async (page, settings) => settings", encoding="utf-8")
    features = root / "artifacts/features.npz"
    features.parent.mkdir()
    features.write_bytes(b"FICTIONAL FIXTURE; NOT A REAL FEATURE ARCHIVE")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    ids = np.array(["fixture-held-out", "fixture-other"])
    X, y = np.zeros((2, 28)), np.array([0, 1])
    metadata = {"config_id": config.id, "dataset": "fixture-recorded-contract", "data_kind": "recorded"}
    dataset = SimpleNamespace(X=X, y=y, subject_ids=ids, metadata=metadata)
    fingerprint = hashlib.sha256(X.tobytes() + y.tobytes() + "\n".join(ids).encode()).hexdigest()
    summary = {**metadata, "subjects": 2, "class_counts": {"Control": 1, "ASD": 1},
               "feature_fingerprint": fingerprint}
    folds = pd.DataFrame({"subject_id": ids, "fold": [0, 1], "label": y})
    table = pd.DataFrame({"subject_id": ids, "recording_id": ["fixture-session", "fixture-other-session"]})
    case = {**metadata, "subject_id": ids[0], "run_id": run.name, "model": "quantum", "fold": 0,
            "provenance": "held-out", "predicted_label": 1, "predicted_class": "ASD",
            "decision_score": .1, "feature_vector": X[0].tolist(), "window_count": 3,
            "evaluation_truth_after_prediction": 0}
    frozen = {"config_id": config.id, "cases_selected_before_evaluation": [ids[0]],
              "feature_reproduction": {"archive_sha256": demo.digest(features)}}
    case_path, freeze_path = root / "artifacts/case.json", root / "artifacts/freeze.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    freeze_path.write_text(json.dumps(frozen), encoding="utf-8")
    pd.DataFrame({"subject_id": [ids[0]], "model": ["quantum"], "predicted_label": [1],
                  "true_label": [0], "decision_score": [.1]}).to_csv(run / "oof_predictions.csv", index=False)
    (run / "metrics.json").write_text('{"fixture": "nonclinical"}', encoding="utf-8")
    monkeypatch.setattr(demo, "load_config", lambda _: config)
    monkeypatch.setattr(demo, "read_manifest", lambda _: table)
    monkeypatch.setattr(demo, "load_features", lambda _: dataset)
    monkeypatch.setattr(demo, "check_feature_workspace", lambda *_: None)
    monkeypatch.setattr(demo, "check_run_workspace", lambda *_, **__: None)
    monkeypatch.setattr(demo, "read_run", lambda _: (summary, folds))
    monkeypatch.setattr(demo, "read_results", lambda _: ({name: {"accuracy": .5} for name in demo.MODELS}, None))
    args = dict(root=root, run=run, manifest=root / "data/local/manifest.csv",
                config_path=root / "configs/fixture.yaml", features=features,
                case_json=case_path, freeze=freeze_path, output=root / "artifacts/new-demo",
                url="http://localhost:8501/")
    return args, summary, case, frozen


def test_explicit_private_wrapper_binds_paths_hashes_case(evidence):
    args, _, case, _ = evidence
    wrapper = demo.prepare(**args)
    settings = json.loads((wrapper.parent / "demo_settings.json").read_text(encoding="utf-8"))
    assert settings["expected"] == case
    assert settings["workspace"]["Results directory"] == "results/custom"
    assert settings["oof_sha256"] == demo.digest(args["run"] / "oof_predictions.csv")
    assert settings["metrics_sha256"] == demo.digest(args["run"] / "metrics.json")
    assert settings["preview_recording"] == "fixture-session"
    assert settings["model_label"] == demo.MODELS["quantum"]
    assert "settings" in wrapper.read_text(encoding="utf-8")


@pytest.mark.parametrize("key,value", [
    ("run_id", "run-other"), ("config_id", "different"), ("dataset", "other"),
    ("data_kind", "synthetic"), ("model", "rbf"), ("fold", 1),
    ("provenance", "new-subject"), ("predicted_label", 0),
    ("evaluation_truth_after_prediction", 1), ("decision_score", .2),
    ("feature_vector", [1.] * 28),
])
def test_different_case_or_provenance_fails_before_writing(evidence, key, value):
    args, _, case, _ = evidence
    case[key] = value
    args["case_json"].write_text(json.dumps(case), encoding="utf-8")
    with pytest.raises(ValueError):
        demo.prepare(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("change", ["not-preselected", "archive-hash", "configuration", "run-fingerprint"])
def test_changed_freeze_fails_closed(evidence, change):
    args, summary, _, frozen = evidence
    if change == "not-preselected":
        frozen["cases_selected_before_evaluation"] = []
    elif change == "archive-hash":
        frozen["feature_reproduction"]["archive_sha256"] = "changed"
    elif change == "configuration":
        frozen["config_id"] = "changed"
    else:
        summary["feature_fingerprint"] = "changed"
    args["freeze"].write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(ValueError):
        demo.prepare(**args)
    assert not args["output"].exists()


def test_private_output_and_existing_files_are_protected(evidence):
    args, *_ = evidence
    args["output"] = args["root"] / "scripts/private-demo"
    with pytest.raises(ValueError, match="ignored artifacts"):
        demo.prepare(**args)
    args["output"] = args["root"] / "artifacts/existing"
    args["output"].mkdir()
    marker = args["output"] / "original.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        demo.prepare(**args)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_remote_endpoint_is_not_accepted(evidence):
    args, *_ = evidence
    args["url"] = "https://example.com/"
    with pytest.raises(ValueError, match="local Streamlit"):
        demo.prepare(**args)


def test_public_browser_script_requires_explicit_settings():
    source = (Path(__file__).resolve().parents[1] / "scripts/record_tushar_demo.js").read_text(encoding="utf-8")
    assert "async (page, settings)" in source
    assert "select('Experiment', settings.expected.run_id)" in source
    assert "select('Held-out subject', settings.expected.subject_id)" in source
    assert "select('Model', settings.model_label)" in source
    assert "settings.oof_sha256" in source
    assert "settings.metrics_sha256" in source
    assert "checkPrediction(JSON.parse" in source
