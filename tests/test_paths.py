"""Cross-drive paths use software fixtures, not a recorded-data cohort."""

import json
import os
from pathlib import Path

import pytest

from asd_eeg import paths
from asd_eeg.config import load_config
from asd_eeg.data.manifest import read_manifest
from asd_eeg.data.screening import screen_manifest
from asd_eeg.workspace import activate_workspace, workspace_paths

ROOT = Path(__file__).resolve().parents[1]


def different_drives(*args, **kwargs):
    raise ValueError("path is on a different Windows drive or UNC share")


def test_same_drive_paths_stay_relative(tmp_path):
    recording = tmp_path / "recordings" / "sample.npz"
    assert paths.relative_or_absolute(recording, tmp_path) == "recordings/sample.npz"


def test_different_drive_paths_stay_absolute(tmp_path, monkeypatch):
    recording = tmp_path / "sample.npz"
    monkeypatch.setattr(paths.os.path, "relpath", different_drives)
    result = paths.relative_or_absolute(recording, tmp_path)
    assert Path(result).is_absolute()
    assert result == recording.resolve().as_posix()


@pytest.mark.skipif(os.name != "nt", reason="Windows drive semantics")
def test_actual_windows_cross_drive_path():
    assert paths.relative_or_absolute("Z:/recordings/sample.bdf", "C:/project") == (
        "Z:/recordings/sample.bdf"
    )


def test_activation_accepts_cross_drive_paths(completed_check, tmp_path, monkeypatch):
    # Mark the fixture as recorded only to exercise activation, never as research evidence.
    table = read_manifest(completed_check["manifest"])
    table["data_kind"] = "recorded"
    manifest = tmp_path / "fixture.csv"
    table.to_csv(manifest, index=False)
    monkeypatch.setattr(paths.os.path, "relpath", different_drives)
    result = activate_workspace(tmp_path, manifest, ROOT / "configs/default.yaml",
                                tmp_path / "features.npz", tmp_path / "results")
    assert all(Path(value).is_absolute() for value in result["paths"].values())
    assert workspace_paths(tmp_path) == result["paths"]
    assert json.loads(Path(result["profile"]).read_text()) == result["paths"]


def test_screening_accepts_cross_drive_paths(completed_check, tmp_path, monkeypatch):
    config = load_config(ROOT / "configs/default.yaml")
    monkeypatch.setattr(paths.os.path, "relpath", different_drives)
    result = screen_manifest(completed_check["manifest"], config, tmp_path)
    included = read_manifest(result["manifest"])
    assert included.eeg_path.map(lambda value: Path(value).is_absolute()).all()
    assert len(included) == len(read_manifest(completed_check["manifest"]))
