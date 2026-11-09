"""Acquire only labelled SFARI resting-state recordings, without modifying source clones."""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

DATASET = "ds006780"
DOI = "doi:10.18112/openneuro.ds006780.v1.0.0"
BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds006780/"
LABEL_SOURCE = f"{DOI}; participants.tsv group: TD=0, ASD=1; participants.json and README"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RecordingDownload:
    relative_path: str
    subject_id: str
    label: int
    size: int
    sha256: str


def plan_download(source: str | Path) -> tuple[list[RecordingDownload], list[dict]]:
    """Freeze eligibility using explicit group labels, never EEG filenames or symptom scores."""
    source = Path(source).resolve()
    description = json.loads((source / "dataset_description.json").read_text(encoding="utf-8"))
    if description.get("DatasetDOI") != DOI:
        raise ValueError(f"Expected the SFARI snapshot {DOI}.")
    definitions = json.loads((source / "participants.json").read_text(encoding="utf-8"))
    levels = definitions.get("group", {}).get("Levels", {})
    if levels.get("ASD") != "autism spectrum disorder" or levels.get("TD") != "typically developing":
        raise ValueError("SFARI diagnostic group definitions do not match the documented mapping.")
    participants = pd.read_csv(source / "participants.tsv", sep="\t", dtype=str,
                               keep_default_na=False)
    if not {"participant_id", "group", "completed_Resting_state"}.issubset(participants):
        raise ValueError("SFARI metadata needs participant IDs, groups, and resting availability.")
    if participants.participant_id.duplicated().any():
        raise ValueError("SFARI participant IDs must be unique.")
    downloads, exclusions = [], []
    for row in participants.sort_values("participant_id").itertuples():
        if not re.fullmatch(r"sub-[A-Za-z0-9]+", row.participant_id):
            raise ValueError("Invalid BIDS participant ID in SFARI metadata.")
        reason = None
        if row.group not in {"ASD", "TD"}:
            reason = "Not an explicit ASD or TD group; siblings and unknown groups excluded"
        elif row.completed_Resting_state.lower() != "yes":
            reason = "Resting-state availability is not yes"
        if reason:
            exclusions.append({"subject_id": row.participant_id, "group": row.group,
                               "reason": reason})
            continue
        files = sorted((source / row.participant_id).rglob("*task-Restingstate*_eeg.bdf"))
        if not files:
            raise ValueError(f"Resting metadata exists but no BDF entries for {row.participant_id}.")
        for path in files:
            pointer = path.read_text(encoding="utf-8")
            match = re.search(r"SHA256E-s(\d+)--([a-f0-9]{64})\.bdf", pointer)
            if not match or ".git/annex/objects/" not in pointer:
                raise ValueError(f"Expected a SHA256E annex metadata pointer: {path.name}.")
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if (metadata.get("TaskName") != "Restingstate"
                    or metadata.get("SamplingFrequency") != 512
                    or metadata.get("PowerLineFrequency") != 60
                    or metadata.get("RecordingType") != "continuous"):
                raise ValueError(f"Recording metadata requires protocol review: {path.name}.")
            channels = pd.read_csv(path.with_name(path.name.replace("_eeg.bdf", "_channels.tsv")),
                                   sep="\t", keep_default_na=False)
            if not {"name", "type", "status"}.issubset(channels):
                raise ValueError(f"Incomplete channel metadata for {path.name}.")
            eeg = channels[channels.type.eq("EEG")]
            if (channels.name.duplicated().any() or len(eeg) != 64
                    or not {"F3", "F4", "P3", "P4"}.issubset(eeg.name)):
                raise ValueError(f"Review the SFARI montage for {path.name}.")
            downloads.append(RecordingDownload(path.relative_to(source).as_posix(),
                                               row.participant_id, int(row.group == "ASD"),
                                               int(match[1]), match[2]))
    if not downloads or {item.label for item in downloads} != {0, 1}:
        raise ValueError("SFARI selection needs both explicit ASD and TD participants.")
    return downloads, exclusions


def download_recording(item: RecordingDownload, output: Path) -> dict:
    """Resume our partial file, verify the annex size/hash, then atomically install it."""
    target = output / item.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == item.size and sha256_file(target) == item.sha256:
            return {**asdict(item), "status": "verified_existing"}
        raise ValueError(f"Existing destination is not the expected recording: {target.name}.")
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= item.size:
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(BASE_URL + item.relative_path, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an incorrect range for the partial download.")
                else:
                    offset = 0
                with partial.open("ab" if offset else "wb") as handle:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(block)
            if partial.stat().st_size != item.size or sha256_file(partial) != item.sha256:
                raise ValueError(f"Download size/SHA-256 does not match the snapshot: {target.name}.")
            partial.replace(target)
            return {**asdict(item), "status": "downloaded_verified"}
        except (OSError, urllib.error.URLError, http.client.HTTPException):
            if attempt == 2:
                raise
    raise RuntimeError("Recording download did not complete.")


def prepare_sfari(source: str | Path, output: str | Path, download: bool = False,
                  workers: int = 4) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("Use between one and eight download workers.")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use a separate destination, never modify the source metadata clone.")
    items, exclusions = plan_download(source)
    subjects = {item.subject_id: item.label for item in items}
    summary = {"dataset": DATASET, "snapshot": DOI, "subjects": len(subjects),
               "class_counts": {"ASD": sum(subjects.values()),
                                "Control": len(subjects) - sum(subjects.values())},
               "recordings": len(items), "download_bytes": sum(item.size for item in items),
               "excluded_subjects": len(exclusions), "status": "planned"}
    if not download:
        return summary
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < summary["download_bytes"] + 1024 ** 3:
        raise ValueError("Not enough free disk space for the selected EEG and a 1 GiB reserve.")
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_recording, item, output): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                records.append(future.result())
            except (OSError, ValueError, RuntimeError, http.client.HTTPException) as exc:
                failures.append({**asdict(item), "status": "failed", "error": str(exc)})
    report = {**summary, "status": "failed" if failures else "downloaded_verified",
              "files": sorted(records + failures, key=lambda item: item["relative_path"])}
    (output / "download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        raise ValueError(f"{len(failures)} downloads failed. See download_report.json; rerun to resume.")
    metadata_hashes = {}
    for filename in ("README", "dataset_description.json", "participants.json", "participants.tsv"):
        origin = source / filename
        shutil.copyfile(origin, output / filename)
        metadata_hashes[filename] = sha256_file(origin)
    rows = []
    for item in items:
        path = Path(item.relative_path)
        for suffix in ("_eeg.json", "_channels.tsv", "_events.tsv", "_events.json"):
            relative = path.with_name(path.name.replace("_eeg.bdf", suffix))
            if (source / relative).is_file():
                shutil.copyfile(source / relative, output / relative)
                metadata_hashes[relative.as_posix()] = sha256_file(source / relative)
        rows.append({"dataset": DATASET, "subject_id": item.subject_id,
                     "recording_id": path.stem, "eeg_path": path.as_posix(),
                     "label": item.label, "label_source": LABEL_SOURCE, "data_kind": "recorded"})
    pd.DataFrame(rows).to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(exclusions).to_csv(output / "eligibility_exclusions.csv", index=False)
    (output / "source_provenance.json").write_text(json.dumps(
        {"snapshot": DOI, "metadata_sha256": metadata_hashes, "selection":
         "Explicit ASD/TD, resting availability yes, all Restingstate BDF runs; no outcome selection"},
        indent=2), encoding="utf-8")
    return {**summary, "status": "downloaded_verified", "manifest": str(output / "candidates.csv")}
