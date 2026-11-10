"""Frozen session guards using generated fixtures only, never recorded evidence."""

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from asd_eeg.config import Config
from asd_eeg.data.manifest import read_manifest
from asd_eeg.features.cache import load_features, save_features
from asd_eeg.inference.artifacts import check_run_workspace, read_run


@pytest.fixture
def session_workspace(completed_check):
    summary, folds = read_run(completed_check["run"])
    table = read_manifest(completed_check["manifest"])
    dataset = load_features(Path(completed_check["directory"]) / "features.npz")
    config = Config.from_dict(dataset.metadata["config"])
    # Quality includes all supplied sessions, unlike capped window_recording_ids.
    # Add an unselected session for every subject to test that crucial distinction.
    extra = table.copy()
    extra["recording_id"] = extra.recording_id + "-not-in-window-table"
    table = pd.concat([table, extra], ignore_index=True)
    metadata = copy.deepcopy(dataset.metadata)
    for subject in metadata["subjects"]:
        subject["quality"].append({
            "recording_id": subject["quality"][0]["recording_id"] + "-not-in-window-table",
            "accepted_windows": 0,
        })
    summary["feature_metadata"] = metadata
    return summary, folds, table, config


def test_identical_whole_cohort_sessions_ignore_manifest_order(session_workspace):
    summary, folds, table, config = session_workspace
    check_run_workspace(summary, folds, table.iloc[::-1], config)


@pytest.mark.parametrize("field", ["dataset", "data_kind", "config_id", "config", "missing-config"])
def test_mismatched_embedded_metadata_fails_with_identical_membership(session_workspace, field):
    summary, folds, table, config = session_workspace
    metadata = summary["feature_metadata"]
    if field == "config":
        metadata["config"]["evaluation"]["seed"] += 1
    elif field == "missing-config":
        metadata.pop("config")
    else:
        metadata[field] = "different"
    with pytest.raises(ValueError, match="embedded feature provenance"):
        check_run_workspace(summary, folds, table, config)


@pytest.mark.parametrize("change", ["removed", "replaced", "reassigned", "added"])
def test_changed_sessions_with_identical_subjects_and_labels_are_rejected(session_workspace, change):
    summary, folds, table, config = session_workspace
    # Change a subject outside the first held-out fold, not merely a selected case.
    sid = folds.loc[folds.fold.ne(0), "subject_id"].iloc[-1]
    index = table.index[(table.subject_id == sid)
                        & table.recording_id.str.endswith("-not-in-window-table")][0]
    if change == "removed":
        table = table.drop(index)
    elif change == "replaced":
        table.loc[index, "recording_id"] = "replacement-session"
    elif change == "reassigned":
        label = table.loc[index, "label"]
        other = table.loc[(table.subject_id != sid) & (table.label == label), "subject_id"].iloc[0]
        table.loc[index, "subject_id"] = other
    else:
        table.loc[len(table)] = table.iloc[index]
        table.loc[len(table) - 1, "recording_id"] = "added-session"
    assert table.groupby("subject_id").label.first().to_dict() == folds.set_index("subject_id").label.to_dict()
    with pytest.raises(ValueError, match="recording sessions differ"):
        check_run_workspace(summary, folds, table, config)


@pytest.mark.parametrize("missing", ["metadata", "subjects", "one-subject", "quality", "recording-id"])
def test_missing_frozen_provenance_fails_closed(session_workspace, missing):
    summary, folds, table, config = session_workspace
    if missing == "metadata":
        summary.pop("feature_metadata")
    elif missing == "subjects":
        summary["feature_metadata"].pop("subjects")
    elif missing == "one-subject":
        summary["feature_metadata"]["subjects"].pop()
    elif missing == "quality":
        summary["feature_metadata"]["subjects"][-1].pop("quality")
    else:
        summary["feature_metadata"]["subjects"][-1]["quality"][-1].pop("recording_id")
    with pytest.raises(ValueError, match="provenance"):
        check_run_workspace(summary, folds, table, config)


@pytest.mark.parametrize("invalid", ["duplicate-subject", "duplicate-session", "foreign-subject", "blank-id"])
def test_invalid_frozen_session_provenance_is_rejected(session_workspace, invalid):
    summary, folds, table, config = session_workspace
    subjects = summary["feature_metadata"]["subjects"]
    if invalid == "duplicate-subject":
        subjects.append(copy.deepcopy(subjects[0]))
    elif invalid == "duplicate-session":
        subjects[-1]["quality"][-1]["recording_id"] = subjects[0]["quality"][0]["recording_id"]
    elif invalid == "foreign-subject":
        subjects[-1]["subject_id"] = "not-in-cohort"
    else:
        subjects[-1]["quality"][-1]["recording_id"] = " "
    with pytest.raises(ValueError, match="provenance"):
        check_run_workspace(summary, folds, table, config)


@pytest.fixture
def legacy_acceptance(completed_check, tmp_path):
    summary, folds = read_run(completed_check["run"])
    # New evaluations embed provenance; this fixture deliberately models an older
    # run that can only recover session membership from its original acceptance.
    summary.pop("feature_metadata", None)
    table = read_manifest(completed_check["manifest"])
    dataset = load_features(Path(completed_check["directory"]) / "features.npz")
    config = Config.from_dict(dataset.metadata["config"])
    # Use the existing recorded acceptance layout, with entirely synthetic data.
    evidence = tmp_path / "artifacts/generated-acceptance"
    frozen = evidence / "features/subject_features.npz"
    save_features(dataset, frozen)
    path = evidence / "acceptance.json"
    path.write_text(json.dumps({
        "status": "complete", "run": "Z:\\old-machine\\results\\" + summary["run_id"],
        "config_id": config.id,
        "feature_reproduction": {
            "feature_fingerprint": summary["feature_fingerprint"],
            "archive_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        },
    }))
    return summary, folds, table, config, tmp_path / "results" / summary["run_id"], frozen, path


def test_legacy_run_with_hash_bound_original_acceptance_remains_usable(legacy_acceptance):
    summary, folds, table, config, run, frozen, path = legacy_acceptance
    before = frozen.read_bytes(), path.read_bytes()
    check_run_workspace(summary, folds, table, config, run)
    assert (frozen.read_bytes(), path.read_bytes()) == before
    assert "feature_metadata" not in summary


def test_legacy_original_acceptance_checks_sessions(legacy_acceptance):
    summary, folds, table, config, run, _, _ = legacy_acceptance
    table.loc[len(table) - 1, "recording_id"] = "replacement-session"
    with pytest.raises(ValueError, match="recording sessions differ"):
        check_run_workspace(summary, folds, table, config, run)


@pytest.mark.parametrize("problem", ["missing-evidence", "archive-hash", "run-id", "fingerprint", "config", "cohort"])
def test_legacy_unbound_or_changed_archive_cannot_supply_provenance(legacy_acceptance, problem):
    summary, folds, table, config, run, frozen, path = legacy_acceptance
    evidence = json.loads(path.read_text())
    if problem == "missing-evidence":
        path.unlink()
    elif problem == "archive-hash":
        dataset = load_features(frozen)
        # Same numerical fingerprint, changed session provenance: SHA must block it.
        dataset.metadata["subjects"][-1]["quality"][0]["recording_id"] = "replacement"
        save_features(dataset, frozen)
    else:
        if problem == "run-id":
            evidence["run"] = "another-run"
        elif problem == "fingerprint":
            evidence["feature_reproduction"]["feature_fingerprint"] = "wrong"
        elif problem == "config":
            evidence["config_id"] = "wrong"
        else:
            dataset = load_features(frozen)
            dataset.y = 1 - dataset.y
            save_features(dataset, frozen)
            evidence["feature_reproduction"]["archive_sha256"] = hashlib.sha256(frozen.read_bytes()).hexdigest()
        path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="provenance|cohort"):
        check_run_workspace(summary, folds, table, config, run)


def test_explicit_provenance_root_supports_external_results(legacy_acceptance):
    summary, folds, table, config, _, _, path = legacy_acceptance
    check_run_workspace(summary, folds, table, config, Path("external/results") / summary["run_id"],
                        provenance_root=path.parent.parent)
