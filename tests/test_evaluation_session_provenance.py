"""New evaluation summaries freeze session provenance using synthetic fixtures only."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from asd_eeg.contracts import FeatureDataset
from asd_eeg.evaluation import run_experiments
from asd_eeg.features.schema import feature_names, feature_schema
from asd_eeg.inference.artifacts import check_run_workspace, read_run


class _SyntheticModel:
    """Fast deterministic test double; never represents a recorded-data model."""

    def predict(self, X):
        scores = X[:, 0] - 0.25
        return (scores > 0).astype(int), scores


@pytest.fixture
def synthetic_sessions(config, monkeypatch):
    config = replace(
        config,
        preprocessing=replace(config.preprocessing, min_windows=1, max_windows=1),
        evaluation=replace(config.evaluation, folds=2),
    )
    ids = np.array([f"synthetic-{i}" for i in range(4)])
    labels = np.array([0, 0, 1, 1])
    details, rows, used_recordings = [], [], []
    for sid, label in zip(ids.tolist(), labels.tolist(), strict=True):
        first, capped = f"{sid}-first", f"{sid}-beyond-cap"
        used_recordings.append(first)
        details.append({
            "subject_id": sid, "windows": 1, "accepted_windows": 2,
            "quality": [
                {"recording_id": first, "accepted_windows": 1},
                {"recording_id": capped, "accepted_windows": 1},
            ],
        })
        for rid in (first, capped):
            rows.append({"dataset": "synthetic-provenance", "data_kind": "synthetic",
                         "subject_id": sid, "recording_id": rid, "label": label})
    X = np.zeros((len(ids), config.n_features))
    X[:, :4 * len(config.nodes)] = 0.25
    X[:, 0] = [0.1, 0.2, 0.3, 0.4]
    metadata = {
        "schema_version": 1, "dataset": "synthetic-provenance", "data_kind": "synthetic",
        "config": config.to_dict(), "config_id": config.id,
        "feature_names": list(feature_names(config)), "feature_schema": feature_schema(config),
        "subjects": details, "label_sources": ["generated regression-test labels"],
    }
    dataset = FeatureDataset(
        X=X, y=labels, subject_ids=ids, metadata=metadata,
        window_counts=np.ones(len(ids), dtype=int),
        accepted_window_counts=np.full(len(ids), 2, dtype=int),
        window_connectivity=np.zeros((len(ids), len(config.edges))),
        window_subject_index=np.arange(len(ids)),
        window_recording_ids=np.array(used_recordings),
    )
    dataset.validate()
    monkeypatch.setattr(run_experiments, "fit_model", lambda *args: _SyntheticModel())
    monkeypatch.setattr(run_experiments, "software_versions", lambda: {"fixture": "synthetic"})
    monkeypatch.setattr(run_experiments, "git_commit", lambda: "unknown")
    return dataset, pd.DataFrame(rows), config


def test_new_summary_retains_all_quality_sessions_beyond_window_cap(synthetic_sessions, tmp_path):
    dataset, table, config = synthetic_sessions
    expected = json.loads(json.dumps(dataset.metadata))
    X, y, ids = dataset.X.copy(), dataset.y.copy(), dataset.subject_ids.copy()
    run = run_experiments.run_evaluation(dataset, tmp_path)
    summary, folds = read_run(run)

    assert summary["schema_version"] == 1
    assert summary["status"] == "complete"
    assert summary["feature_metadata"] == expected
    saved_ids = {record["recording_id"] for subject in summary["feature_metadata"]["subjects"]
                 for record in subject["quality"]}
    assert saved_ids == set(table.recording_id)
    assert set(dataset.window_recording_ids) < saved_ids
    assert all(subject["windows"] == config.preprocessing.max_windows
               and subject["accepted_windows"] > subject["windows"]
               for subject in summary["feature_metadata"]["subjects"])
    np.testing.assert_array_equal(dataset.X, X)
    np.testing.assert_array_equal(dataset.y, y)
    np.testing.assert_array_equal(dataset.subject_ids, ids)
    assert summary["feature_fingerprint"] == hashlib.sha256(
        X.tobytes() + y.tobytes() + "\n".join(ids).encode(),
    ).hexdigest()
    check_run_workspace(summary, folds, table.iloc[::-1], config, run)


def test_new_summary_does_not_alias_mutable_feature_metadata(synthetic_sessions, tmp_path,
                                                           monkeypatch):
    dataset, table, config = synthetic_sessions
    expected = json.loads(json.dumps(dataset.metadata))

    def mutate_source(*args):
        # The final summary must still contain the snapshot written before fitting.
        dataset.metadata["subjects"][-1]["quality"][-1]["recording_id"] = "mutated-after-start"
        dataset.metadata["label_sources"].append("mutated-after-start")
        return _SyntheticModel()

    monkeypatch.setattr(run_experiments, "fit_model", mutate_source)
    run = run_experiments.run_evaluation(dataset, tmp_path)
    summary, folds = read_run(run)
    assert summary["feature_metadata"] == expected
    assert json.loads(json.dumps(dataset.metadata)) != expected
    check_run_workspace(summary, folds, table, config, run)


@pytest.mark.parametrize("change", ["removed", "reassigned"])
def test_evaluator_summary_roundtrip_rejects_changed_capped_sessions(synthetic_sessions,
                                                                  tmp_path, change):
    dataset, table, config = synthetic_sessions
    run = run_experiments.run_evaluation(dataset, tmp_path)
    summary, folds = read_run(run)
    check_run_workspace(summary, folds, table, config, run)

    # Alter a subject outside the first held-out fold; subject/label membership stays fixed.
    sid = folds.loc[folds.fold.ne(0), "subject_id"].iloc[-1]
    index = table.index[(table.subject_id == sid)
                        & table.recording_id.str.endswith("-beyond-cap")][0]
    assert table.loc[index, "recording_id"] not in dataset.window_recording_ids
    if change == "removed":
        table = table.drop(index)
    else:
        label = table.loc[index, "label"]
        other = table.loc[(table.subject_id != sid) & (table.label == label), "subject_id"].iloc[0]
        table.loc[index, "subject_id"] = other
    assert table.groupby("subject_id").label.first().to_dict() == folds.set_index("subject_id").label.to_dict()
    with pytest.raises(ValueError, match="recording sessions differ"):
        check_run_workspace(summary, folds, table, config, run)


def test_legacy_quality_free_features_still_evaluate_without_invented_sessions(synthetic_sessions,
                                                                            tmp_path):
    dataset, table, config = synthetic_sessions
    for subject in dataset.metadata["subjects"]:
        subject.pop("quality")
    expected = json.loads(json.dumps(dataset.metadata))
    run = run_experiments.run_evaluation(dataset, tmp_path)
    summary, folds = read_run(run)
    assert summary["status"] == "complete"
    assert summary["feature_metadata"] == expected
    assert all("quality" not in subject for subject in summary["feature_metadata"]["subjects"])
    with pytest.raises(ValueError, match="recording provenance"):
        check_run_workspace(summary, folds, table, config, run)
