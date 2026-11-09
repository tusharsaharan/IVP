"""Network-free acquisition checks; fixture payloads are software-test data, not EEG evidence."""

import hashlib
import io
import json

import pandas as pd
import pytest

from asd_eeg.data import sfari


@pytest.fixture
def metadata_clone(tmp_path):
    source = tmp_path / "metadata"
    source.mkdir()
    (source / "dataset_description.json").write_text(json.dumps({"DatasetDOI": sfari.DOI}))
    (source / "participants.json").write_text(json.dumps({"group": {"Levels": {
        "ASD": "autism spectrum disorder", "TD": "typically developing"}}}))
    (source / "README").write_text("Software-test metadata fixture. Not a real cohort.")
    pd.DataFrame({"participant_id": ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"],
                  "group": ["ASD", "TD", "ASD SIBLING", "NAN", "ASD"],
                  "completed_Resting_state": ["yes", "yes", "yes", "yes", "no"]}).to_csv(
        source / "participants.tsv", sep="\t", index=False)
    payload = b"software-test-payload" * 100
    digest = hashlib.sha256(payload).hexdigest()
    for subject in ("sub-01", "sub-02"):
        path = source / subject / "eeg" / f"{subject}_task-Restingstate_run-01_eeg.bdf"
        path.parent.mkdir(parents=True)
        path.write_text(f"../../.git/annex/objects/a/b/SHA256E-s{len(payload)}--{digest}.bdf")
        path.with_suffix(".json").write_text(json.dumps({"TaskName": "Restingstate",
            "SamplingFrequency": 512, "PowerLineFrequency": 60, "RecordingType": "continuous"}))
        names = ["F3", "F4", "P3", "P4"] + [f"test-{i}" for i in range(60)]
        pd.DataFrame({"name": names, "type": ["EEG"] * 64, "status": ["good"] * 64}).to_csv(
            path.with_name(path.name.replace("_eeg.bdf", "_channels.tsv")), sep="\t", index=False)
    return source, payload


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}


def test_plan_excludes_siblings_unknowns_and_unavailable_rest(metadata_clone):
    source, payload = metadata_clone
    items, exclusions = sfari.plan_download(source)
    assert [(item.subject_id, item.label) for item in items] == [("sub-01", 1), ("sub-02", 0)]
    assert {row["subject_id"] for row in exclusions} == {"sub-03", "sub-04", "sub-05"}
    assert all(item.size == len(payload) for item in items)


@pytest.mark.parametrize("problem", ["snapshot", "duplicate_subject", "group_definition", "montage",
                                   "sampling_rate", "line_frequency", "missing_recording"])
def test_metadata_requires_review_on_mismatch(metadata_clone, problem):
    source, _ = metadata_clone
    path = next((source / "sub-01").rglob("*.bdf"))
    if problem == "snapshot":
        (source / "dataset_description.json").write_text('{"DatasetDOI":"wrong"}')
    elif problem == "duplicate_subject":
        participants = pd.read_csv(source / "participants.tsv", sep="\t")
        pd.concat([participants, participants.iloc[:1]]).to_csv(source / "participants.tsv",
                                                               sep="\t", index=False)
    elif problem == "group_definition":
        (source / "participants.json").write_text('{}')
    elif problem == "montage":
        sidecar = path.with_name(path.name.replace("_eeg.bdf", "_channels.tsv"))
        table = pd.read_csv(sidecar, sep="\t")
        table.loc[0, "name"] = "missing-F3"
        table.to_csv(sidecar, sep="\t", index=False)
    elif problem == "missing_recording":
        path.unlink()
    else:
        metadata = json.loads(path.with_suffix(".json").read_text())
        metadata["SamplingFrequency" if problem == "sampling_rate" else "PowerLineFrequency"] = 999
        path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        sfari.plan_download(source)


def test_plan_is_read_only_and_never_downloads(metadata_clone, monkeypatch, tmp_path):
    source, _ = metadata_clone
    monkeypatch.setattr(sfari.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unexpected HTTP"))
    output = tmp_path / "destination"
    result = sfari.prepare_sfari(source, output)
    assert result["status"] == "planned"
    assert result["class_counts"] == {"ASD": 1, "Control": 1}
    assert not output.exists()


def test_downloads_are_verified_and_source_pointers_untouched(metadata_clone, monkeypatch, tmp_path):
    source, payload = metadata_clone
    monkeypatch.setattr(sfari.urllib.request, "urlopen", lambda *a, **k: Response(payload))
    output = tmp_path / "destination"
    result = sfari.prepare_sfari(source, output, download=True)
    assert result["status"] == "downloaded_verified"
    table = pd.read_csv(result["manifest"])
    assert table.label.tolist() == [1, 0]
    assert table.label_source.str.contains(sfari.DOI, regex=False).all()
    for item in sfari.plan_download(source)[0]:
        assert (output / item.relative_path).read_bytes() == payload
        assert ".git/annex" in (source / item.relative_path).read_text()
        assert (output / item.relative_path).with_suffix(".json").is_file()
    assert json.loads((output / "source_provenance.json").read_text())["metadata_sha256"]


@pytest.mark.parametrize("supports_range", [True, False])
def test_partial_download_resume_and_server_range_fallback(metadata_clone, monkeypatch, tmp_path,
                                                          supports_range):
    source, payload = metadata_clone
    item = sfari.plan_download(source)[0][0]
    target = tmp_path / "destination" / item.relative_path
    target.parent.mkdir(parents=True)
    partial = target.with_suffix(".bdf.part")
    partial.write_bytes(payload[:100])

    def response(request, **kwargs):
        assert request.get_header("Range") == "bytes=100-"
        return Response(payload[100:], 206, {"Content-Range": f"bytes 100-{len(payload)-1}/{len(payload)}"}) if supports_range else Response(payload)

    monkeypatch.setattr(sfari.urllib.request, "urlopen", response)
    sfari.download_recording(item, tmp_path / "destination")
    assert target.read_bytes() == payload
    assert not partial.exists()


def test_wrong_hash_never_publishes_a_payload(metadata_clone, monkeypatch, tmp_path):
    source, payload = metadata_clone
    item = sfari.plan_download(source)[0][0]
    monkeypatch.setattr(sfari.urllib.request, "urlopen", lambda *a, **k: Response(b"x" * len(payload)))
    with pytest.raises(ValueError, match="SHA-256"):
        sfari.download_recording(item, tmp_path)
    assert not (tmp_path / item.relative_path).exists()


def test_existing_verified_payload_needs_no_http(metadata_clone, monkeypatch, tmp_path):
    source, payload = metadata_clone
    item = sfari.plan_download(source)[0][0]
    path = tmp_path / item.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    monkeypatch.setattr(sfari.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unexpected HTTP"))
    assert sfari.download_recording(item, tmp_path)["status"] == "verified_existing"


def test_mismatching_existing_file_is_not_overwritten(metadata_clone, tmp_path):
    source, _ = metadata_clone
    item = sfari.plan_download(source)[0][0]
    path = tmp_path / item.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unrelated user file")
    with pytest.raises(ValueError, match="Existing destination"):
        sfari.download_recording(item, tmp_path)
    assert path.read_bytes() == b"unrelated user file"


def test_source_clone_cannot_be_a_destination(metadata_clone):
    source, _ = metadata_clone
    with pytest.raises(ValueError, match="separate destination"):
        sfari.prepare_sfari(source, source / "downloads", download=True)


def test_failed_download_preserves_audit_without_publishing_manifest(metadata_clone, monkeypatch,
                                                                  tmp_path):
    source, _ = metadata_clone

    def fail(*args, **kwargs):
        raise OSError("Deliberate HTTP failure")

    monkeypatch.setattr(sfari.urllib.request, "urlopen", fail)
    output = tmp_path / "destination"
    with pytest.raises(ValueError, match="downloads failed"):
        sfari.prepare_sfari(source, output, download=True)
    assert json.loads((output / "download_report.json").read_text())["status"] == "failed"
    assert not (output / "candidates.csv").exists()
