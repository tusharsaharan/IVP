"""Local-profile, explicit-QC, and BIDS handoff tests with labelled software fixtures."""

import json
import shutil
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from asd_eeg.cli import main, parser
from asd_eeg.config import Config
from asd_eeg.data.loaders import apply_channel_metadata
from asd_eeg.data.manifest import read_manifest
from asd_eeg.data.screening import screen_manifest
from asd_eeg.features.cache import load_features
from asd_eeg.workspace import (
    DEFAULT_PATHS,
    activate_workspace,
    check_feature_workspace,
    workspace_paths,
)

ROOT = Path(__file__).resolve().parents[1]


def raw_and_sidecar(tmp_path):
    names = ["F3", "F4", "P3", "P4", "EXG1", "Status"]
    raw = mne.io.RawArray(np.zeros((6, 512)), mne.create_info(names, 512,
                         ch_types=["eeg"] * 5 + ["stim"]), verbose="ERROR")
    path = tmp_path / "sub-test_task-Restingstate_eeg.bdf"
    sidecar = tmp_path / "sub-test_task-Restingstate_channels.tsv"
    table = pd.DataFrame({"name": names, "type": ["EEG"] * 4 + ["EMG", "TRIG"],
                          "status": ["good"] * 6})
    return raw, path, sidecar, table


def test_bids_external_channels_are_not_in_the_eeg_average(tmp_path):
    raw, path, sidecar, table = raw_and_sidecar(tmp_path)
    table.to_csv(sidecar, sep="\t", index=False)
    apply_channel_metadata(raw, path)
    raw.pick("eeg", exclude="bads")
    assert raw.ch_names == ["F3", "F4", "P3", "P4"]


def test_bids_bad_channels_are_excluded(tmp_path):
    raw, path, sidecar, table = raw_and_sidecar(tmp_path)
    table.loc[0, "status"] = "bad"
    table.to_csv(sidecar, sep="\t", index=False)
    apply_channel_metadata(raw, path)
    raw.pick("eeg", exclude="bads")
    assert raw.ch_names == ["F4", "P3", "P4"]


@pytest.mark.parametrize("problem", ["order", "duplicate", "type", "status", "missing_column"])
def test_inconsistent_bids_sidecars_fail_without_guessing(tmp_path, problem):
    raw, path, sidecar, table = raw_and_sidecar(tmp_path)
    if problem == "order":
        table = table.iloc[::-1]
    elif problem == "duplicate":
        table.loc[0, "name"] = table.loc[1, "name"]
    elif problem == "type":
        table.loc[0, "type"] = "UNKNOWN"
    elif problem == "status":
        table.loc[0, "status"] = "unknown"
    else:
        table = table.drop(columns="type")
    table.to_csv(sidecar, sep="\t", index=False)
    with pytest.raises(ValueError, match="BIDS"):
        apply_channel_metadata(raw, path)


def test_absent_sidecar_preserves_existing_readers(tmp_path):
    raw, path, _, _ = raw_and_sidecar(tmp_path)
    before = raw.get_channel_types()
    apply_channel_metadata(raw, path)
    assert raw.get_channel_types() == before


def test_profile_defaults_remain_backward_compatible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert workspace_paths(tmp_path) == DEFAULT_PATHS
    assert parser().parse_args(["features"]).config == "configs/default.yaml"


@pytest.mark.parametrize("values", [[], {}, {"manifest": "x"},
    {**DEFAULT_PATHS, "config": None}, {**DEFAULT_PATHS, "manifest": " "}])
def test_invalid_profile_has_actionable_cli_error(tmp_path, monkeypatch, capsys, values):
    (tmp_path / "data/local").mkdir(parents=True)
    (tmp_path / "data/local/workspace.json").write_text(json.dumps(values))
    monkeypatch.chdir(tmp_path)
    assert main(["validate"]) == 1
    assert "workspace" in capsys.readouterr().err.lower()


def test_activation_rejects_synthetic_manifest(completed_check, tmp_path):
    with pytest.raises(ValueError, match="not a synthetic"):
        activate_workspace(tmp_path, completed_check["manifest"], ROOT / "configs/default.yaml",
                           tmp_path / "features.npz", tmp_path / "results")
    assert not (tmp_path / "data/local/workspace.json").exists()


def test_cli_and_app_share_the_activated_local_profile(completed_check, tmp_path, monkeypatch):
    # Relabel only this software fixture's data-kind to exercise profile validation;
    # no numbers from the fixture are presented as recorded-data research.
    table = read_manifest(completed_check["manifest"])
    table["data_kind"] = "recorded"
    manifest = tmp_path / "data/local/test-recordings.csv"
    manifest.parent.mkdir(parents=True)
    table.to_csv(manifest, index=False)
    config = tmp_path / "configs/sfari_rest.yaml"
    config.parent.mkdir()
    shutil.copyfile(ROOT / "configs/sfari_rest.yaml", config)
    activate_workspace(tmp_path, manifest, config, tmp_path / "artifacts/test.npz",
                       tmp_path / "results/test")
    monkeypatch.chdir(tmp_path)
    args = parser().parse_args(["features"])
    assert args.manifest == "data/local/test-recordings.csv"
    assert args.config == "configs/sfari_rest.yaml"
    assert args.output == "artifacts/test.npz"
    monkeypatch.setenv("NEUROQ_ROOT", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=30).run()
    assert not app.exception
    assert app.session_state["manifest_path"] == args.manifest
    assert app.session_state["config_path"] == args.config
    assert app.session_state["features_path"] == args.output
    app.sidebar.radio[0].set_value("Datasets").run()
    assert not app.exception
    assert any("Manifest checked" in element.value for element in app.success)


def test_fixed_qc_preserves_inclusion_and_exclusion_audits(completed_check, config, tmp_path,
                                                         monkeypatch):
    from asd_eeg.data import screening

    original = screening.preprocess_recording
    table = read_manifest(completed_check["manifest"])
    excluded_id = table.recording_id.iloc[0]

    def reject_one(recording, settings):
        if recording.recording_id == excluded_id:
            raise ValueError("Deliberate fixed-QC software-test rejection")
        return original(recording, settings)

    monkeypatch.setattr(screening, "preprocess_recording", reject_one)
    result = screen_manifest(completed_check["manifest"], config, tmp_path)
    included = read_manifest(result["manifest"])
    assert excluded_id not in set(included.recording_id)
    audit = pd.read_csv(Path(result["manifest"]).parent / "recording_qc.csv")
    assert audit.status.eq("excluded").sum() == 1
    assert len(audit) == len(table)
    assert result["config_id"] == config.id
    assert result["included_recordings"] == len(table) - 1


def test_failed_qc_still_preserves_the_failed_summary(completed_check, config, tmp_path,
                                                    monkeypatch):
    from asd_eeg.data import screening

    def reject(*args):
        raise ValueError("Deliberate rejection of software-test fixtures")

    monkeypatch.setattr(screening, "preprocess_recording", reject)
    with pytest.raises(ValueError, match="fewer than two subjects"):
        screen_manifest(completed_check["manifest"], config, tmp_path)
    summary = json.loads(next(tmp_path.glob("screen-*/screen_summary.json")).read_text())
    assert summary["status"] == "failed"
    assert summary["included_recordings"] == 0


@pytest.mark.parametrize("problem", ["config", "dataset", "labels", "subjects", "recordings"])
def test_stale_feature_archive_is_blocked(completed_check, tmp_path, problem):
    dataset = load_features(Path(completed_check["directory"]) / "features.npz")
    config = Config.from_dict(dataset.metadata["config"])
    table = read_manifest(completed_check["manifest"])
    if problem == "config":
        dataset.metadata["config_id"] = "wrong"
    elif problem == "dataset":
        table["dataset"] = "different-dataset"
    elif problem == "labels":
        table.loc[0, "label"] = 1 - table.loc[0, "label"]
    elif problem == "subjects":
        table = table.iloc[1:]
    else:
        table.loc[0, "recording_id"] = "different-recording"
    manifest = tmp_path / "manifest.csv"
    table.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="Rebuild features"):
        check_feature_workspace(dataset, manifest, config)


def test_matching_feature_archive_is_accepted(completed_check):
    dataset = load_features(Path(completed_check["directory"]) / "features.npz")
    config = Config.from_dict(dataset.metadata["config"])
    check_feature_workspace(dataset, completed_check["manifest"], config)


def test_sfari_configuration_changes_only_the_verified_mains_frequency(config):
    from asd_eeg.config import load_config

    sfari = load_config(ROOT / "configs/sfari_rest.yaml").to_dict()
    sfari["preprocessing"]["line_hz"] = 50.0
    assert sfari == config.to_dict()
