"""Workspace recovery uses isolated software fixtures, never new research data."""

import json
from pathlib import Path

import pytest

from asd_eeg.cli import main
from asd_eeg.data.manifest import read_manifest
from asd_eeg.workspace import DEFAULT_PATHS, PROFILE, activate_workspace, workspace_paths

ROOT = Path(__file__).resolve().parents[1]
BROKEN_PROFILES = ["{", "[]", "{}", json.dumps({**DEFAULT_PATHS, "config": None})]


@pytest.fixture
def broken_workspace(tmp_path, monkeypatch):
    profile = tmp_path / PROFILE
    profile.parent.mkdir(parents=True)
    profile.write_text("{", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return profile


@pytest.fixture
def activation_paths(completed_check, tmp_path):
    # Only fixture metadata is marked recorded to exercise the activation guard.
    table = read_manifest(completed_check["manifest"])
    table["data_kind"] = "recorded"
    manifest = tmp_path / "software-fixture.csv"
    table.to_csv(manifest, index=False)
    return {"manifest": manifest, "config": ROOT / "configs/default.yaml",
            "features": tmp_path / "features.npz", "results": tmp_path / "results"}


@pytest.mark.parametrize("arguments", [
    ["--help"], ["validate", "--help"], ["features", "--help"],
    ["evaluate", "--help"], ["predict", "--help"], ["screen", "--help"],
    ["prepare-sfari", "--help"], ["self-check", "--help"], ["use-workspace", "--help"],
])
def test_help_does_not_require_a_valid_workspace(broken_workspace, arguments, capsys):
    with pytest.raises(SystemExit) as caught:
        main(arguments)
    assert caught.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert broken_workspace.read_text() == "{"


@pytest.mark.parametrize("payload", BROKEN_PROFILES)
def test_activation_repairs_invalid_profile(broken_workspace, activation_paths, payload, capsys):
    broken_workspace.write_text(payload)
    arguments = ["use-workspace"]
    for name, value in activation_paths.items():
        arguments.extend([f"--{name}", str(value)])
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert workspace_paths(broken_workspace.parents[2]) == result["paths"]
    assert not list(broken_workspace.parent.glob("*.tmp"))


@pytest.mark.parametrize("arguments", [
    ["validate"], ["features"], ["evaluate"], ["screen"],
    ["predict", "--subject", "software-subject", "--bundle", "missing.joblib"],
])
def test_data_commands_still_fail_closed_for_invalid_profile(broken_workspace, arguments, capsys):
    assert main(arguments) == 1
    assert capsys.readouterr().err.startswith("Error:")
    assert broken_workspace.read_text() == "{"


def test_self_check_does_not_read_or_modify_invalid_profile(broken_workspace, monkeypatch, capsys):
    from asd_eeg import self_check

    calls = []

    def checked(config, output):
        calls.append((config, output))
        return {"status": "passed", "data_kind": "synthetic"}

    monkeypatch.setattr(self_check, "run_self_check", checked)
    assert main(["self-check", "--config", str(ROOT / "configs/default.yaml")]) == 0
    assert json.loads(capsys.readouterr().out)["data_kind"] == "synthetic"
    assert len(calls) == 1
    assert broken_workspace.read_text() == "{"


def test_sfari_planning_does_not_read_or_modify_invalid_profile(broken_workspace, monkeypatch, capsys):
    from asd_eeg.data import sfari

    calls = []

    def planned(source, output, download, workers):
        calls.append((source, output, download, workers))
        return {"status": "planned"}

    monkeypatch.setattr(sfari, "prepare_sfari", planned)
    assert main(["prepare-sfari", "--source", "software-metadata"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "planned"
    assert calls == [("software-metadata", "data/local/sfari", False, 4)]
    assert broken_workspace.read_text() == "{"


@pytest.mark.parametrize("field", list(DEFAULT_PATHS))
@pytest.mark.parametrize("empty", ["", " "])
def test_blank_activation_paths_leave_existing_profile_unchanged(
        broken_workspace, activation_paths, field, empty):
    activation_paths[field] = empty
    with pytest.raises(ValueError, match="nonempty"):
        activate_workspace(broken_workspace.parents[2], **activation_paths)
    assert broken_workspace.read_text() == "{"


@pytest.mark.parametrize("existing", [None, json.dumps(DEFAULT_PATHS)])
def test_failed_atomic_activation_preserves_previous_profile_and_cleans_temp(
        broken_workspace, activation_paths, monkeypatch, existing):
    from asd_eeg import workspace

    if existing is None:
        broken_workspace.unlink()
    else:
        broken_workspace.write_text(existing)

    def denied(source, destination):
        assert Path(source).parent == broken_workspace.parent
        assert Path(destination) == broken_workspace
        assert set(json.loads(Path(source).read_text())) == set(DEFAULT_PATHS)
        raise PermissionError("Deliberate software-test activation failure")

    monkeypatch.setattr(workspace.os, "replace", denied)
    with pytest.raises(PermissionError, match="activation failure"):
        activate_workspace(broken_workspace.parents[2], **activation_paths)
    if existing is None:
        assert not broken_workspace.exists()
    else:
        assert broken_workspace.read_text() == existing
    assert not list(broken_workspace.parent.glob("*.tmp"))


def test_invalid_activation_input_does_not_replace_profile(broken_workspace, activation_paths):
    activation_paths["config"] = ROOT / "missing-software-config.yaml"
    with pytest.raises(OSError):
        activate_workspace(broken_workspace.parents[2], **activation_paths)
    assert broken_workspace.read_text() == "{"
    assert not list(broken_workspace.parent.glob("*.tmp"))


@pytest.mark.parametrize("explicit", [False, True])
def test_valid_workspace_defaults_and_explicit_arguments_keep_their_precedence(
        broken_workspace, activation_paths, completed_check, explicit, capsys):
    activate_workspace(broken_workspace.parents[2], **activation_paths)
    before = broken_workspace.read_bytes()
    arguments = ["validate"]
    if explicit:
        arguments.extend(["--manifest", completed_check["manifest"]])
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["data_kind"] == ("synthetic" if explicit else "recorded")
    assert result["subjects"] == 12
    assert broken_workspace.read_bytes() == before
