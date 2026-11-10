"""Laptop app flows with isolated generated fixtures, never research evidence.

Some fixture metadata is marked recorded solely to exercise recorded-workspace
selection guards. The generated signals/results remain in pytest temporary files.
"""

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from asd_eeg.data.manifest import read_manifest
from asd_eeg.features.cache import load_features, save_features
from asd_eeg.inference.compatibility import load_bundle
from asd_eeg.workspace import activate_workspace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def application_workspace(completed_check, tmp_path, monkeypatch):
    config = tmp_path / "configs/default.yaml"
    config.parent.mkdir()
    # Self-check explicitly uses three folds; restore its config rather than the
    # five-fold shipped defaults, just as a real recorded-workspace must do.
    config.write_text((Path(completed_check["run"]) / "config.json").read_text())
    table = read_manifest(completed_check["manifest"])
    table["data_kind"] = "recorded"  # software fixture metadata only, see module disclaimer
    manifest = tmp_path / "data/local/recordings.csv"
    manifest.parent.mkdir(parents=True)
    table.to_csv(manifest, index=False)
    source = Path(completed_check["run"])
    run = tmp_path / "results/software-fixtures" / source.name
    shutil.copytree(source, run)
    summary = json.loads((run / "summary.json").read_text())
    # Keep this a legacy run even when the generated source is a new evaluation.
    # Removing acceptance evidence must leave no alternate provenance source.
    summary.pop("feature_metadata", None)
    summary["data_kind"] = "recorded"
    (run / "summary.json").write_text(json.dumps(summary))
    for path in run.glob("fold_*/*.joblib"):
        bundle = load_bundle(path)
        bundle["metadata"]["data_kind"] = "recorded"
        joblib.dump(bundle, path)
    # Reproduce the existing acceptance artifact layout with generated fixtures.
    # Only this isolated copy is marked recorded; no research result is changed.
    evidence = tmp_path / "artifacts/software-fixture-acceptance"
    frozen = evidence / "features/subject_features.npz"
    dataset = load_features(Path(completed_check["directory"]) / "features.npz")
    dataset.metadata["data_kind"] = "recorded"
    save_features(dataset, frozen)
    (evidence / "acceptance.json").write_text(json.dumps({
        "status": "complete", "run": str(run), "config_id": summary["config_id"],
        "feature_reproduction": {
            "feature_fingerprint": summary["feature_fingerprint"],
            "archive_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        },
    }))
    activate_workspace(tmp_path, manifest, config, tmp_path / "artifacts/not-built.npz", run.parent)
    monkeypatch.setenv("NEUROQ_ROOT", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=30).run()
    assert not app.exception
    return app, run, manifest


def button(app, label):
    return next(element for element in app.button if element.label == label)


def selector(app, label):
    return next(element for element in app.selectbox if element.label == label)


def downloads(app):
    return [element.proto.label for element in app.get("download_button")]


def test_saved_results_are_reconciled_and_downloadable(application_workspace):
    app, _, _ = application_workspace
    app.sidebar.radio[0].set_value("Experiments").run()
    assert not app.exception
    assert not app.error
    assert "Download predictions" in downloads(app)
    assert "Download metrics" in downloads(app)
    assert "Download folds" in downloads(app)
    assert "Download run configuration" in downloads(app)
    assert not plt.get_fignums()


def test_analysis_sensitivity_and_actual_circuit_persist(application_workspace):
    app, _, _ = application_workspace
    app.sidebar.radio[0].set_value("Analysis").run()
    assert not app.exception
    button(app, "Analyze subject").click().run()
    assert not app.exception
    assert not app.error
    result = app.session_state["analysis"][1]
    assert result["provenance"] == "held-out"
    assert any("windows used from" in element.value for element in app.caption)
    assert "Export analysis JSON" in downloads(app)
    button(app, "Compute connection sensitivity").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.session_state["sensitivity"][1]) == 6
    assert "Download sensitivity table" in downloads(app)
    app.run()
    assert "Download sensitivity table" in downloads(app)
    button(app, "Draw saved circuit").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["circuit_image"][1].startswith(b"\x89PNG")
    assert "Download circuit figure" in downloads(app)
    app.run()
    assert "Download circuit figure" in downloads(app)
    assert not plt.get_fignums()


def test_changing_subject_never_displays_previous_analysis(application_workspace):
    app, _, _ = application_workspace
    app.sidebar.radio[0].set_value("Analysis").run()
    button(app, "Analyze subject").click().run()
    first = app.session_state["analysis"][1]["subject_id"]
    subjects = selector(app, "Held-out subject")
    second = next(sid for sid in subjects.options if sid != first)
    subjects.set_value(second).run()
    assert not app.exception
    assert "Export analysis JSON" not in downloads(app)
    assert not any("Prediction:" in element.value for element in app.subheader)


def test_damaged_bundle_is_an_actionable_app_error(application_workspace):
    app, run, _ = application_workspace
    app.sidebar.radio[0].set_value("Analysis").run()
    subject = selector(app, "Held-out subject").value
    folds = pd.read_csv(run / "folds.csv", dtype={"subject_id": str})
    fold = int(folds.loc[folds.subject_id == subject, "fold"].iloc[0])
    (run / f"fold_{fold}/quantum.joblib").write_bytes(b"broken software fixture")
    button(app, "Analyze subject").click().run()
    assert not app.exception
    assert any("restore" in element.value for element in app.error)
    assert "Export analysis JSON" not in downloads(app)


def test_failed_reanalysis_clears_previous_prediction(application_workspace, monkeypatch):
    from asd_eeg.inference import pipeline

    app, _, _ = application_workspace
    app.sidebar.radio[0].set_value("Analysis").run()
    button(app, "Analyze subject").click().run()
    assert "Export analysis JSON" in downloads(app)

    def reject(*args, **kwargs):
        raise ValueError("Deliberate software-test preprocessing failure")

    monkeypatch.setattr(pipeline, "analyze_subject", reject)
    button(app, "Analyze subject").click().run()
    assert not app.exception
    assert any("preprocessing failure" in element.value for element in app.error)
    assert "Export analysis JSON" not in downloads(app)


def test_incompatible_complete_runs_are_not_selectable(application_workspace):
    app, run, _ = application_workspace
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["dataset"] = "unrelated-software-dataset"
    summary_path.write_text(json.dumps(summary))
    app.sidebar.radio[0].set_value("Analysis").run()
    assert not app.exception
    assert not any(element.label == "Experiment" for element in app.selectbox)
    assert any("Unavailable saved runs" in element.label for element in app.expander)


@pytest.mark.parametrize("page", ["Analysis", "Experiments"])
@pytest.mark.parametrize("change", ["replaced", "reassigned", "missing-provenance"])
def test_changed_sessions_or_missing_provenance_make_run_unavailable(application_workspace, page, change):
    app, run, manifest = application_workspace
    if change == "missing-provenance":
        evidence = run.parents[2] / "artifacts/software-fixture-acceptance/acceptance.json"
        evidence.unlink()
    else:
        table = read_manifest(manifest)
        if change == "replaced":
            table.loc[len(table) - 1, "recording_id"] = "replacement-session"
        else:
            # Swap sessions between same-label subjects: preserve the entire cohort
            # and global recording-ID set while changing per-subject membership.
            first = table.index[-1]
            second = table.index[(table.subject_id != table.loc[first, "subject_id"])
                                 & (table.label == table.loc[first, "label"])][0]
            table.loc[[first, second], "subject_id"] = table.loc[[second, first], "subject_id"].to_numpy()
        table.to_csv(manifest, index=False)
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert not any(element.label in {"Experiment", "Experiment run"} for element in app.selectbox)
    assert any("Unavailable saved runs" in element.label for element in app.expander)
    assert "Export analysis JSON" not in downloads(app)


def test_corrupt_saved_metrics_do_not_crash_or_show_scores(application_workspace):
    app, run, _ = application_workspace
    (run / "metrics.json").write_text("[]")
    app.sidebar.radio[0].set_value("Experiments").run()
    assert not app.exception
    assert any("Could not display this saved run" in element.value for element in app.error)
    assert "Download metrics" not in downloads(app)


def test_blank_results_path_has_recovery_message(application_workspace):
    app, _, _ = application_workspace
    app.sidebar.text_input(key="results_path").set_value(" ").run()
    assert not app.exception
    assert any("must not be blank" in element.value for element in app.error)


def test_dataset_caption_uses_portable_workspace_relative_path(application_workspace):
    app, _, manifest = application_workspace
    app.sidebar.radio[0].set_value("Datasets").run()
    assert not app.exception
    captions = [element.value for element in app.caption]
    assert "Manifest: data/local/recordings.csv" in captions
    assert not any(str(manifest.parent.parent.parent) in value for value in captions)
