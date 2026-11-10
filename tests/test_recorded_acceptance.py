"""Acceptance orchestration must fail closed before any recorded-data evaluation."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asd_eeg.config import Config
from asd_eeg.contracts import FeatureDataset
from asd_eeg.features.cache import load_features, save_features
from asd_eeg.features.schema import feature_names

SPEC = importlib.util.spec_from_file_location(
    "recorded_acceptance", Path(__file__).resolve().parents[1] / "scripts/run_recorded_acceptance.py",
)
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


def table():
    return pd.DataFrame([
        {"dataset": "verified", "subject_id": "z", "recording_id": "r2", "label": 1,
         "data_kind": "recorded", "label_source": "explicit source", "eeg_path": "old/path"},
        {"dataset": "verified", "subject_id": "a", "recording_id": "r1", "label": 0,
         "data_kind": "recorded", "label_source": "explicit source", "eeg_path": "old/path"},
        {"dataset": "verified", "subject_id": "b", "recording_id": "r3", "label": 1,
         "data_kind": "recorded", "label_source": "explicit source", "eeg_path": "old/path"},
    ])


def test_qc_comparison_ignores_relocated_paths_and_row_order():
    left = table()
    right = left.iloc[::-1].copy()
    right["eeg_path"] = "relocated/path"
    assert acceptance.compare_cohorts(left, right)["identical_recording_metadata"]


@pytest.mark.parametrize("column,value", [
    ("label", 0), ("subject_id", "new"), ("recording_id", "new"),
    ("dataset", "other"), ("data_kind", "synthetic"), ("label_source", "guessed"),
])
def test_qc_comparison_rejects_changed_scientific_metadata(column, value):
    left = table()
    right = left.copy()
    right.loc[0, column] = value
    with pytest.raises(ValueError, match="differs"):
        acceptance.compare_cohorts(left, right)


def test_case_selection_is_fixed_without_prediction_information():
    assert acceptance.select_cases(table()) == ["a", "b"]
    assert acceptance.select_cases(table().iloc[::-1]) == ["a", "b"]


def test_case_selection_rejects_one_class():
    one = table()
    one["label"] = 1
    with pytest.raises(ValueError, match="both classes"):
        acceptance.select_cases(one)


@pytest.fixture
def source_fixture(tmp_path):
    participants = pd.DataFrame([
        {"participant_id": "a", "group": "TD", "completed_Resting_state": "yes"},
        {"participant_id": "b", "group": "ASD", "completed_Resting_state": "yes"},
    ])
    participants.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    (tmp_path / "participants.json").write_text(json.dumps({"group": {"Levels": {
        "ASD": "autism spectrum disorder", "TD": "typically developing",
    }}}), encoding="utf-8")
    files = [
        {"relative_path": "r1.bdf", "subject_id": "a", "label": 0, "sha256": "one"},
        {"relative_path": "r2.bdf", "subject_id": "b", "label": 1, "sha256": "two"},
    ]
    (tmp_path / "download_report.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    candidates = pd.DataFrame([
        {"subject_id": item["subject_id"], "recording_id": Path(item["relative_path"]).stem,
         "label": item["label"], "eeg_path": tmp_path / item["relative_path"]}
        for item in files
    ])
    return tmp_path, candidates


def test_candidate_provenance_uses_explicit_groups(source_fixture):
    source, candidates = source_fixture
    assert acceptance.verify_candidate_labels(candidates, source)["explicit_source_labels_match"]


def test_candidate_provenance_rejects_label_flip(source_fixture):
    source, candidates = source_fixture
    candidates.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="differ"):
        acceptance.verify_candidate_labels(candidates, source)


def test_candidate_provenance_rejects_missing_recording(source_fixture):
    source, candidates = source_fixture
    with pytest.raises(ValueError, match="every acquired"):
        acceptance.verify_candidate_labels(candidates.iloc[:1], source)


def test_candidate_provenance_rejects_duplicate_payload_across_subjects(source_fixture):
    source, candidates = source_fixture
    path = source / "download_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["files"][1]["sha256"] = report["files"][0]["sha256"]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="potential leakage"):
        acceptance.verify_candidate_labels(candidates, source)


def test_source_hash_verification_rejects_path_escape(tmp_path):
    (tmp_path / "download_report.json").write_text(json.dumps({
        "status": "downloaded_verified", "snapshot": "known", "files": [
            {"relative_path": "../outside.bdf", "size": 1, "sha256": "unknown"},
        ],
    }), encoding="utf-8")
    (tmp_path / "source_provenance.json").write_text(json.dumps({
        "snapshot": "known", "metadata_sha256": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        acceptance.verify_source(tmp_path)


@pytest.fixture
def archive_fixture(tmp_path, config):
    """Small synthetic archives with genuine session metadata; no recorded experiment."""
    cohort = table().sort_values("subject_id").reset_index(drop=True)
    cohort["data_kind"] = "synthetic"
    cohort["label_source"] = "explicit synthetic fixture"
    extra = cohort.iloc[[0]].copy()
    extra["recording_id"] = "r4"
    cohort = pd.concat([cohort, extra], ignore_index=True)
    for index, row in cohort.iterrows():
        payload = tmp_path / f"{row.recording_id}.npz"
        np.savez(payload, data=np.zeros((4, 128)), sfreq=128.0,
                 channels=np.array([node.channels[0] for node in config.nodes]))
        cohort.loc[index, "eeg_path"] = str(payload)
    manifest = tmp_path / "synthetic_manifest.csv"
    cohort.to_csv(manifest, index=False)
    subjects = cohort.groupby("subject_id", sort=True)
    dataset = FeatureDataset(
        X=np.zeros((len(subjects), config.n_features)),
        y=subjects.label.first().to_numpy(),
        subject_ids=subjects.label.first().index.to_numpy(dtype=str),
        metadata={
            "schema_version": 1, "config": config.to_dict(), "config_id": config.id,
            "dataset": "verified", "data_kind": "synthetic",
            "feature_names": list(feature_names(config)),
            "label_sources": ["explicit synthetic fixture"],
            "subjects": [{"subject_id": subject,
                          "quality": [{"recording_id": rid} for rid in rows.recording_id]}
                         for subject, rows in subjects],
        },
        window_counts=np.ones(len(subjects), dtype=int),
        window_connectivity=np.zeros((len(subjects), len(config.edges))),
        window_subject_index=np.arange(len(subjects)),
        window_recording_ids=subjects.recording_id.first().to_numpy(dtype=str),
    )
    archive = tmp_path / "previous.npz"
    save_features(dataset, archive)
    return manifest, archive, dataset


PROTOCOL_CHANGES = [
    ("evaluation", "svm_c", 2.0),
    ("circuit", "passes", 3),
    ("circuit", "bandwidth", 2.0),
    ("circuit", "edge_seed", 123),
    ("evaluation", "folds", 3),
    ("evaluation", "seed", 123),
]


def changed_protocol(config, section, setting, value):
    raw = config.to_dict()
    raw[section][setting] = value
    return Config.from_dict(raw)


def with_protocol(dataset, config):
    changed = deepcopy(dataset)
    changed.metadata["config"] = config.to_dict()
    changed.metadata["config_id"] = config.id
    return changed


def test_feature_reproduction_accepts_complete_matching_synthetic_archives(archive_fixture, config):
    manifest, archive, rebuilt = archive_fixture
    previous = load_features(archive)
    # Loaded JSON uses lists while freshly built configuration metadata uses tuples.
    acceptance.verify_feature_reproduction(previous, rebuilt, manifest, config)


@pytest.mark.parametrize("section,setting,value", PROTOCOL_CHANGES)
@pytest.mark.parametrize("archive_role", ["previous", "rebuilt"])
def test_feature_reproduction_rejects_model_circuit_and_fold_drift(
    archive_fixture, config, section, setting, value, archive_role,
):
    manifest, archive, original = archive_fixture
    requested = changed_protocol(config, section, setting, value)
    rebuilt = with_protocol(original, requested)
    previous = load_features(archive)
    if archive_role == "rebuilt":
        previous, rebuilt = rebuilt, previous
    assert np.array_equal(previous.X, rebuilt.X)
    assert np.array_equal(previous.y, rebuilt.y)
    assert np.array_equal(previous.subject_ids, rebuilt.subject_ids)
    with pytest.raises(ValueError, match="full configuration"):
        acceptance.verify_feature_reproduction(previous, rebuilt, manifest, requested)


@pytest.mark.parametrize("column,value", [("dataset", "other"), ("data_kind", "recorded")])
@pytest.mark.parametrize("archive_role", ["previous", "rebuilt"])
def test_feature_reproduction_rejects_dataset_and_data_kind_drift(
    archive_fixture, config, column, value, archive_role,
):
    manifest, archive, rebuilt = archive_fixture
    previous = load_features(archive)
    target = previous if archive_role == "previous" else rebuilt
    target.metadata[column] = value
    with pytest.raises(ValueError, match="dataset differs"):
        acceptance.verify_feature_reproduction(previous, rebuilt, manifest, config)


@pytest.mark.parametrize("problem", [
    "missing_subjects", "missing_subject", "duplicate_subject", "missing_quality",
    "missing_recording", "changed_recording", "duplicate_recording", "window_recording",
    "label_source", "config_id", "incomplete_config",
])
def test_feature_reproduction_requires_full_previous_provenance(archive_fixture, config, problem):
    manifest, archive, rebuilt = archive_fixture
    previous = load_features(archive)
    details = previous.metadata["subjects"]
    if problem == "missing_subjects":
        del previous.metadata["subjects"]
    elif problem == "missing_subject":
        details.pop()
    elif problem == "duplicate_subject":
        details.append(deepcopy(details[0]))
    elif problem == "missing_quality":
        del details[0]["quality"]
    elif problem == "missing_recording":
        details[0]["quality"].pop()
    elif problem == "changed_recording":
        details[0]["quality"][0]["recording_id"] = "foreign"
    elif problem == "duplicate_recording":
        details[0]["quality"].append(deepcopy(details[0]["quality"][0]))
    elif problem == "window_recording":
        previous.window_recording_ids[0] = previous.window_recording_ids[1]
    elif problem == "label_source":
        previous.metadata["label_sources"] = ["guessed"]
    elif problem == "config_id":
        previous.metadata["config_id"] = "wrong"
    else:
        # Omitting a default-valued section reconstructs the same Config/hash, but
        # does not supply the complete previously frozen protocol.
        del previous.metadata["config"]["circuit"]
    with pytest.raises(ValueError):
        acceptance.verify_feature_reproduction(previous, rebuilt, manifest, config)


@pytest.mark.parametrize("archive_role", ["previous", "rebuilt"])
def test_feature_reproduction_rejects_full_config_even_with_current_id(
    archive_fixture, config, archive_role,
):
    manifest, archive, rebuilt = archive_fixture
    previous = load_features(archive)
    target = previous if archive_role == "previous" else rebuilt
    target.metadata["config"]["evaluation"]["svm_c"] = 2.0
    with pytest.raises(ValueError, match="configuration"):
        acceptance.verify_feature_reproduction(previous, rebuilt, manifest, config)


@pytest.mark.parametrize("problem", ["values", "labels", "subject_order"])
def test_feature_reproduction_preserves_numeric_and_subject_comparison(
    archive_fixture, config, problem,
):
    manifest, archive, rebuilt = archive_fixture
    previous = load_features(archive)
    if problem == "values":
        previous.X[0, 0] = 0.5
    elif problem == "labels":
        previous.y[0] = 1 - previous.y[0]
    else:
        # Preserve the cohort and window references, but change feature-row order.
        previous.subject_ids = previous.subject_ids[::-1].copy()
        previous.y = previous.y[::-1].copy()
        previous.window_subject_index = len(previous.y) - 1 - previous.window_subject_index
    with pytest.raises(ValueError, match="differ"):
        acceptance.verify_feature_reproduction(previous, rebuilt, manifest, config)


@pytest.mark.parametrize("section,setting,value", PROTOCOL_CHANGES + [
    ("metadata", "dataset", "other"), ("metadata", "data_kind", "synthetic"),
])
def test_acceptance_fails_before_evaluation_on_previous_archive_drift(
    archive_fixture, config, tmp_path, monkeypatch, section, setting, value,
):
    """Stub recorded-data boundaries; only synthetic features and orchestration run."""
    manifest, archive, original = archive_fixture
    cohort = pd.read_csv(manifest)
    cohort["dataset"] = "ds006780"
    cohort["data_kind"] = "recorded"
    original.metadata.update(dataset="ds006780", data_kind="recorded")
    previous = deepcopy(original)
    requested = config
    if section == "metadata":
        previous.metadata[setting] = value
    else:
        requested = changed_protocol(config, section, setting, value)
    rebuilt = with_protocol(original, requested)
    monkeypatch.setattr(acceptance, "load_config", lambda _: requested)
    monkeypatch.setattr(acceptance, "read_manifest", lambda _: cohort.copy())
    monkeypatch.setattr("asd_eeg.workspace.read_manifest", lambda _: cohort.copy())
    monkeypatch.setattr(acceptance, "verify_source", lambda _: {})
    monkeypatch.setattr(acceptance, "verify_candidate_labels", lambda *_: {})
    monkeypatch.setattr(acceptance, "screen_manifest", lambda *_: {"manifest": str(manifest)})
    monkeypatch.setattr(acceptance, "build_features", lambda *_: rebuilt)
    monkeypatch.setattr(acceptance, "load_features", lambda _: previous)
    monkeypatch.setattr(acceptance, "git_commit", lambda: "synthetic-test")
    monkeypatch.setattr(acceptance, "software_versions", lambda: {})
    evaluations = []
    monkeypatch.setattr(acceptance, "run_evaluation", lambda *_: evaluations.append(True))
    output = tmp_path / "acceptance"
    results = tmp_path / "never-created-results"
    with pytest.raises(ValueError, match="differs"):
        acceptance.main([
            "--manifest", str(manifest), "--config", "unused.yaml", "--source", str(tmp_path),
            "--previous-manifest", str(manifest), "--previous-features", str(archive),
            "--output", str(output), "--results", str(results),
        ])
    assert evaluations == []
    assert not results.exists()
    assert not (output / "protocol_freeze.json").exists()
    evidence = json.loads((output / "acceptance.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "failed"
    assert "feature_reproduction" not in evidence
