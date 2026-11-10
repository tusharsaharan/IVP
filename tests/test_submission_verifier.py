"""Source-only archive guards; all fixtures are synthetic temporary files."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "submission_verifier", Path(__file__).resolve().parents[1] / "scripts/verify_submission.py",
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "repo"
    files = {
        "pyproject.toml": '[project]\nname = "synthetic-fixture"\n',
        "README.md": "Synthetic source fixture.\n",
        "configs/default.yaml": "nodes: []\n",
        "src/asd_eeg/__init__.py": '"""Synthetic fixture."""\n',
        "data/metadata/recordings.csv": ",".join(verifier.MANIFEST_COLUMNS) + "\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.mark.parametrize("name", [
    ".git/config", ".venv/pyvenv.cfg", "data/raw/subject.bdf", "data/local/manifest.csv",
    "results/run/summary.json", "artifacts/bundle.joblib", "neuroq-connect/README.md",
    "app/__pycache__/main.py", "app/.env", "docs/credentials.json", ".streamlit/secrets.toml",
    "docs/recordings/note.md", "src/pkg.egg-info/PKG-INFO", "docs/video.avi", "docs/model.joblib",
    "../README.md", "/README.md", "src\\outside.py", "docs/participant.csv",
    "src/double//file.py", "src/./file.py", "src/file.py:private.py",
    "scripts/record_demo.private.js", "docs/manifest.private.md",
])
def test_unsafe_paths_are_not_sources(name):
    assert not verifier.permitted_path(name)


@pytest.mark.parametrize("name", [
    "README.md", ".gitignore", ".streamlit/config.toml", "data/metadata/recordings.csv",
    "src/asd_eeg/cli.py", "scripts/verify_submission.py", "docs/REPRODUCTION.md",
    "tests/test_submission_verifier.py", ".github/workflows/ci.yml",
    "scripts/record_tushar_demo.js", "scripts/prepare_recorded_demo.py",
])
def test_expected_sources_are_permitted(name):
    assert verifier.permitted_path(name)


def test_archive_roundtrip_excludes_private_and_historical_files(source, tmp_path):
    for name in ["data/raw/payload.bdf", "results/run/summary.json", "app/__pycache__/bad.py",
                 "neuroq-connect/README.md", ".streamlit/secrets.toml", "docs/private.joblib"]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"DO NOT INCLUDE")
    archive = tmp_path / "source.zip"
    snapshot = verifier.create_archive(source, archive)
    assert not snapshot["final_freeze"]
    assert len(snapshot["files"]) == 5
    assert snapshot["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    extracted = tmp_path / "extracted"
    manifest = verifier.extract_verified(archive, extracted)
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((extracted / name).read_bytes()).hexdigest() == expected["sha256"]
    assert not (extracted / "data/raw").exists()
    assert not (extracted / "neuroq-connect").exists()


def test_populated_public_manifest_blocks_archive(source, tmp_path):
    with (source / "data/metadata/recordings.csv").open("a", encoding="utf-8") as handle:
        handle.write("synthetic,id,rec,path,1,fixture,synthetic\n")
    archive = tmp_path / "unsafe.zip"
    with pytest.raises(ValueError, match="header-only"):
        verifier.create_archive(source, archive)
    assert not archive.exists()


@pytest.mark.parametrize("content", [
    "user=" + "C:" + "\\Users\\fictional\\recording.bdf",
    "path=" + "/home" + "/fictional/project/recording.bdf",
    "-----BEGIN " + "PRIVATE KEY-----",
    "AKIA" + "A" * 16,
    "ghp_" + "a" * 36,
    "sub-" + "12345",
])
def test_sensitive_markers_fail_without_echoing_value(source, tmp_path, content):
    (source / "README.md").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="requires review") as error:
        verifier.create_archive(source, tmp_path / "unsafe.zip")
    assert content not in str(error.value)


def test_existing_archive_and_extract_are_preserved(source, tmp_path):
    archive = tmp_path / "source.zip"
    verifier.create_archive(source, archive)
    original = archive.read_bytes()
    with pytest.raises(FileExistsError):
        verifier.create_archive(source, archive)
    assert archive.read_bytes() == original
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    sentinel = extracted / "original.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        verifier.extract_verified(archive, extracted)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_tampering_and_traversal_are_blocked_before_extraction(source, tmp_path):
    original = tmp_path / "source.zip"
    verifier.create_archive(source, original)
    for changed_name, changed_bytes in [("README.md", b"tampered"), ("../escape.py", b"x")]:
        altered = tmp_path / ("altered-" + str(len(changed_bytes)) + ".zip")
        with zipfile.ZipFile(original) as archive, zipfile.ZipFile(altered, "w") as out:
            for name in archive.namelist():
                content = archive.read(name)
                if name == changed_name:
                    content = changed_bytes
                out.writestr(name, content)
            if changed_name.startswith(".."):
                out.writestr(changed_name, changed_bytes)
        destination = tmp_path / ("extracted-" + str(len(changed_bytes)))
        with pytest.raises(ValueError):
            verifier.extract_verified(altered, destination)
        assert not destination.exists()
    assert not (tmp_path / "escape.py").exists()


def test_duplicate_member_is_rejected(source, tmp_path):
    archive_path = tmp_path / "source.zip"
    verifier.create_archive(source, archive_path)
    with zipfile.ZipFile(archive_path, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("README.md", "duplicate")
    with pytest.raises(ValueError, match="Duplicate"):
        verifier.extract_verified(archive_path, tmp_path / "extracted")


def test_symlink_source_is_rejected(source, tmp_path):
    target = tmp_path / "private.md"
    target.write_text("private fixture", encoding="utf-8")
    link = source / "docs/link.md"
    link.parent.mkdir()
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation is unavailable for this account.")
    with pytest.raises(ValueError, match="link"):
        verifier.create_archive(source, tmp_path / "source.zip")


def test_manifest_contains_no_absolute_repository_path(source, tmp_path):
    snapshot = verifier.create_archive(source, tmp_path / "source.zip")
    assert str(source) not in json.dumps(snapshot)
