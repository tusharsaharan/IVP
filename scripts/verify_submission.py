"""Create a fail-closed source snapshot; optionally reproduce it in a fresh venv.

Only explicitly permitted text sources are read. Raw/local research trees are
never traversed. The archive is a timestamped snapshot, not a final code freeze.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import venv
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT_FILES = {
    ".gitignore", "AGENTS.md", "CONTRIBUTING.md", "DESIGN.md", "PRODUCT.md",
    "README.md", "README_SAHIL.md", "pyproject.toml", "requirements.lock.txt",
    ".streamlit/config.toml", "data/metadata/README.md", "data/metadata/recordings.csv",
}
SOURCE_DIRS = {"app", "configs", "docs", "scripts", "src", "tests", ".github"}
TEXT_SUFFIXES = {".py", ".js", ".md", ".yaml", ".yml", ".sh", ".toml", ".txt"}
BLOCKED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".playwright-cli", "build", "dist", "raw", "local", "results", "artifacts",
    "neuroq-connect", "recordings", "bundles", "node_modules",
}
BLOCKED_NAMES = {".env", "secrets.toml", "credentials", "credentials.json"}
MANIFEST_NAME = "SUBMISSION_MANIFEST.json"
MANIFEST_COLUMNS = [
    "dataset", "subject_id", "recording_id", "eeg_path", "label", "label_source", "data_kind",
]
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 16 * 1024 * 1024
SENSITIVE_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\s\"'<>]+", re.IGNORECASE),
    re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    re.compile(r"\bsub-\d{5}\b"),  # Recorded SFARI participant identities stay local.
)


def permitted_path(name: str) -> bool:
    """Allow source paths only, independently of the current Git index."""
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or ".." in path.parts or path.as_posix() != name):
        return False
    if any(part.casefold() in BLOCKED_PARTS for part in path.parts):
        return False
    if any(part.casefold() in BLOCKED_NAMES or part.startswith(".env") for part in path.parts):
        return False
    if any(".private." in part.casefold() for part in path.parts):
        return False
    if any(part.endswith(".egg-info") for part in path.parts):
        return False
    if name in ROOT_FILES:
        return True
    return path.parts[0] in SOURCE_DIRS and path.suffix.casefold() in TEXT_SUFFIXES


def is_link(path: Path) -> bool:
    # Windows junctions also bypass ordinary symlink checks. Python 3.11 has
    # no Path.is_junction, so use the reparse-point stat flag where available.
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(attributes & 0x400)


def source_paths(root: Path) -> list[Path]:
    """Do not descend into excluded directories or follow links."""
    candidates = []
    for name in sorted(ROOT_FILES):
        path = root / name
        parents = [root / Path(*Path(name).parts[:index])
                   for index in range(1, len(Path(name).parts) + 1)]
        if any(is_link(parent) for parent in parents):
            raise ValueError(f"Source link is not permitted: {name}")
        if path.is_file():
            candidates.append(path)
    for name in sorted(SOURCE_DIRS):
        base = root / name
        if is_link(base):
            raise ValueError(f"Source link is not permitted: {name}")
        if not base.is_dir():
            continue
        for folder, dirs, files in os.walk(base, followlinks=False):
            for dirname in list(dirs):
                path = Path(folder) / dirname
                if dirname.casefold() in BLOCKED_PARTS or dirname.endswith(".egg-info"):
                    dirs.remove(dirname)
                elif is_link(path):
                    raise ValueError("Source directory links are not permitted.")
            for filename in files:
                path = Path(folder) / filename
                relative = path.relative_to(root).as_posix()
                if permitted_path(relative):
                    if is_link(path):
                        raise ValueError(f"Source link is not permitted: {relative}")
                    candidates.append(path)
    return sorted(set(candidates), key=lambda path: path.relative_to(root).as_posix())


def checked_content(name: str, content: bytes) -> None:
    if len(content) > MAX_FILE_BYTES:
        raise ValueError(f"Oversized source file: {name}")
    text = content.decode("utf-8-sig")
    if "\x00" in text:
        raise ValueError(f"Binary content in source file: {name}")
    if any(pattern.search(text) for pattern in SENSITIVE_PATTERNS):
        # Never echo a potentially sensitive match into the public manifest.
        raise ValueError(f"Participant, personal-path or credential marker requires review: {name}")
    if name == "data/metadata/recordings.csv":
        rows = list(csv.reader(io.StringIO(text)))
        if rows != [MANIFEST_COLUMNS]:
            raise ValueError("The public recording manifest must be the header-only template.")


def git_snapshot(root: Path) -> dict:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, check=True,
        ).stdout != b""
        return {"head": head, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": "unknown", "working_tree_dirty": None}


def create_archive(root: Path, destination: Path) -> dict:
    root = root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("Use the maintained repository root containing pyproject.toml.")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing submission snapshot.")
    payloads = {}
    total = 0
    paths = source_paths(root)
    for path in paths:
        name = path.relative_to(root).as_posix()
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Oversized source file: {name}")
        content = path.read_bytes()
        checked_content(name, content)
        total += len(content)
        if total > MAX_SOURCE_BYTES:
            raise ValueError("Source snapshot exceeds the safety size limit.")
        payloads[name] = content
    required = {"README.md", "pyproject.toml", "configs/default.yaml", "src/asd_eeg/__init__.py"}
    if not required.issubset(payloads):
        raise ValueError("Submission source tree is missing required installation/smoke files.")
    manifest = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "source-only-concurrent-work-snapshot",
        "final_freeze": False,
        "git": git_snapshot(root),
        "source_bytes": total,
        "files": {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in payloads.items()
        },
        "exclusions": [
            "Git history; virtual environments; caches; historical nested project",
            "Raw EEG; populated/local manifests; models; results; local recordings",
            "Credentials; detected personal home paths; arbitrary binary documents/media",
        ],
        "limitations": [
            "Pattern checks are not a comprehensive secret/privacy audit.",
            "Concurrent changes after capture require a new snapshot and verification.",
            "No recorded-data experiment, clinical validity, report/PPT/AVI completion is claimed.",
        ],
    }
    # Detect additions/removals and edits while collecting source contents.
    if source_paths(root) != paths or any(
        (root / name).read_bytes() != content for name, content in payloads.items()
    ):
        raise RuntimeError("Source changed during capture; rerun after coordinating with other agents.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves any existing user snapshot even in a name race.
    with destination.open("xb") as handle:
        with zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in payloads.items():
                archive.writestr(name, content)
            archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
    return {
        **manifest,
        "archive_name": destination.name,
        "archive_bytes": destination.stat().st_size,
        "archive_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def extract_verified(archive_path: Path, destination: Path) -> dict:
    """Validate the entire ZIP before extracting to a new, empty directory."""
    if destination.exists():
        raise FileExistsError("Extraction requires a new directory; existing files are preserved.")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or names.count(MANIFEST_NAME) != 1:
            raise ValueError("Duplicate members or missing snapshot manifest.")
        if any(info.file_size > MAX_FILE_BYTES for info in archive.infolist()):
            raise ValueError("Oversized archive member.")
        if sum(info.file_size for info in archive.infolist()) > MAX_SOURCE_BYTES + MAX_FILE_BYTES:
            raise ValueError("Oversized extracted source snapshot.")
        manifest = json.loads(archive.read(MANIFEST_NAME))
        if manifest.get("schema") != 1 or set(names) != set(manifest["files"]) | {MANIFEST_NAME}:
            raise ValueError("Archive members do not match the snapshot manifest.")
        payloads = {}
        for name, expected in manifest["files"].items():
            if not permitted_path(name):
                raise ValueError(f"Unexpected archive member: {name}")
            content = archive.read(name)
            checked_content(name, content)
            if len(content) != expected["bytes"] or (
                hashlib.sha256(content).hexdigest() != expected["sha256"]
            ):
                raise ValueError(f"Snapshot hash mismatch: {name}")
            payloads[name] = content
    destination.mkdir(parents=True)
    for name, content in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (destination / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_stage(name: str, command: list[str], cwd: Path, logs: Path) -> dict:
    start = time.perf_counter()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    evidence = {"stage": name, "exit_code": result.returncode,
                "seconds": round(time.perf_counter() - start, 2)}
    (logs / f"{name}.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence), flush=True)
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect the local temporary log.")
    return evidence


def reproduce(archive: Path, temp_root: Path) -> dict:
    temp_root.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix="neuroq-submission-", dir=temp_root))
    extracted = working / "source"
    snapshot = extract_verified(archive, extracted)
    environment = working / "venv"
    start = time.perf_counter()
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
    evidence = [{"stage": "venv", "exit_code": 0,
                 "seconds": round(time.perf_counter() - start, 2)}]
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    cli = environment / ("Scripts/neuroq.exe" if os.name == "nt" else "bin/neuroq")
    stages = [
        ("install", [str(executable), "-m", "pip", "install", "--no-cache-dir",
                     str(extracted) + "[dev]"], working),
        ("pip-check", [str(executable), "-m", "pip", "check"], working),
        ("installed-import", [str(executable), "-I", "-c",
                              "import asd_eeg,sys; from pathlib import Path; "
                              "p=Path(asd_eeg.__file__).resolve(); print(sys.version); print(p); "
                              "assert p.is_relative_to(Path(sys.prefix).resolve())"], working),
        ("cli-help", [str(cli), "--help"], working),
        ("self-check", [str(cli), "self-check", "--config", str(extracted / "configs/default.yaml"),
                        "--output", str(working / "synthetic-software-check")], working),
        ("pytest", [str(executable), "-m", "pytest"], extracted),
        ("ruff", [str(executable), "-m", "ruff", "check", "."], extracted),
        ("versions", [str(executable), "-m", "pip", "freeze"], working),
    ]
    for name, command, cwd in stages:
        evidence.append(run_stage(name, command, cwd, working))
    disk_bytes = sum(path.stat().st_size for path in working.rglob("*") if path.is_file())
    report = {"status": "passed", "data_kind": "synthetic", "python": sys.version,
              "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "snapshot_utc": snapshot["created_utc"], "final_freeze": False,
              "stages": evidence, "temporary_bytes": disk_bytes,
              "note": "Software-only evidence; no recorded-data research metrics were reproduced."}
    (working / "reproduction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Local evidence retained at: {working}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="New ZIP path; never overwrites a snapshot")
    parser.add_argument("--smoke", action="store_true", help="Fresh extracted installation and tests")
    local_app_data = os.environ.get("LOCALAPPDATA")
    default_temp = (Path(local_app_data) / "Temp" / "opencode" if local_app_data
                    else Path(tempfile.gettempdir()) / "opencode")
    parser.add_argument("--temp-root", type=Path, default=default_temp)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or args.root / "artifacts/submission-20261110" / f"source-{stamp}.zip"
    try:
        snapshot = create_archive(args.root, output)
        sidecar = output.with_suffix(".manifest.json")
        with sidecar.open("x", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)
        print(json.dumps({"archive_name": output.name, "sha256": snapshot["archive_sha256"],
                          "files": len(snapshot["files"]), "final_freeze": False}), flush=True)
        if args.smoke:
            reproduce(output.resolve(), args.temp_root.resolve())
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
