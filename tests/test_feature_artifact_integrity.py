"""Regression checks for feature handoffs, independent of clinical performance."""

import copy
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from asd_eeg.config import Config
from asd_eeg.features.cache import feature_schema_path, load_features, save_features
from asd_eeg.features.schema import feature_schema


@pytest.fixture
def feature_dataset(completed_check):
    return load_features(Path(completed_check["directory"]) / "features.npz")


def _write_unvalidated_archive(dataset, path):
    """Simulate an incompatible external handoff without using the validated writer."""
    arrays = {
        field.name: getattr(dataset, field.name)
        for field in fields(dataset)
        if field.name != "metadata" and getattr(dataset, field.name) is not None
    }
    arrays["metadata"] = np.array(json.dumps(dataset.metadata))
    np.savez_compressed(path, **arrays)


def _assert_rejected(dataset, path, message):
    with pytest.raises(ValueError, match=message):
        dataset.validate()
    with pytest.raises(ValueError, match=message):
        save_features(dataset, path)
    assert not path.exists(), "Validation must fail before creating an archive."
    _write_unvalidated_archive(dataset, path)
    with pytest.raises(ValueError, match=message):
        load_features(path)


def test_different_config_archives_do_not_overwrite_each_others_schema(feature_dataset, tmp_path):
    first = tmp_path / "subject_features.npz"
    second = tmp_path / "alternative.npz"
    save_features(feature_dataset, first)
    config = Config.from_dict(feature_dataset.metadata["config"])
    alternate = replace(config, evaluation=replace(config.evaluation, seed=123))
    other = copy.deepcopy(feature_dataset)
    other.metadata.update(config=alternate.to_dict(), config_id=alternate.id,
                          feature_schema=feature_schema(alternate))
    save_features(other, second)
    assert feature_schema_path(first).name == "feature_schema.json"
    assert feature_schema_path(second).name == "alternative.feature_schema.json"
    assert load_features(first).metadata["config_id"] == config.id
    assert load_features(second).metadata["config_id"] == alternate.id


def test_legacy_archive_ignores_an_unrelated_shared_sidecar(feature_dataset, tmp_path):
    legacy = replace(feature_dataset, metadata=copy.deepcopy(feature_dataset.metadata),
                     connectivity_median=None, connectivity_iqr=None, window_counts=None,
                     accepted_window_counts=None, window_connectivity=None,
                     window_subject_index=None, window_recording_ids=None)
    del legacy.metadata["feature_schema"]
    path = tmp_path / "features.npz"
    _write_unvalidated_archive(legacy, path)
    (tmp_path / "feature_schema.json").write_text('{"config_id": "unrelated"}', encoding="utf-8")
    loaded = load_features(path)
    assert loaded.connectivity_median is None
    assert loaded.window_connectivity is None
    np.testing.assert_array_equal(loaded.X, legacy.X)


def test_saving_legacy_metadata_replaces_a_stale_sidecar(feature_dataset, tmp_path):
    path = tmp_path / "subject_features.npz"
    feature_schema_path(path).write_text('{"config_id": "stale"}', encoding="utf-8")
    legacy_metadata = copy.deepcopy(feature_dataset.metadata)
    del legacy_metadata["feature_schema"]
    legacy = replace(feature_dataset, metadata=legacy_metadata)
    save_features(legacy, path)
    np.testing.assert_array_equal(load_features(path).X, legacy.X)


@pytest.mark.parametrize("part", ["schema_version", "feature_count", "config_id", "label_free",
                                 "node_order", "edge_order", "bands", "features"])
def test_malformed_embedded_schema_is_rejected(feature_dataset, tmp_path, part):
    schema = feature_dataset.metadata["feature_schema"]
    if isinstance(schema[part], list):
        schema[part].reverse()
    elif part == "label_free":
        schema[part] = False
    elif part == "config_id":
        schema[part] = "wrong"
    else:
        schema[part] = 999
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", "Embedded feature schema")


def test_schema_cannot_mislabel_a_feature_source(feature_dataset, tmp_path):
    feature_dataset.metadata["feature_schema"]["features"][0]["node_index"] = 1
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", "Embedded feature schema")


def test_reordered_feature_names_are_rejected(feature_dataset, tmp_path):
    feature_dataset.metadata["feature_names"].reverse()
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", "configured feature order")


def test_malformed_archive_specific_sidecar_is_rejected(feature_dataset, tmp_path):
    path = tmp_path / "features.npz"
    save_features(feature_dataset, path)
    sidecar = feature_schema(Config.from_dict(feature_dataset.metadata["config"]))
    sidecar["edge_order"].reverse()
    feature_schema_path(path).write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="Feature schema file"):
        load_features(path)


@pytest.mark.parametrize("invalid", [-0.5, 2.5])
def test_iqr_graph_range_is_enforced(feature_dataset, tmp_path, invalid):
    feature_dataset.connectivity_iqr[:, 0, 1] = invalid
    feature_dataset.connectivity_iqr[:, 1, 0] = invalid
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", "IQR values")


@pytest.mark.parametrize("invalid", [-0.5, 2.5])
def test_iqr_feature_range_is_enforced(feature_dataset, tmp_path, invalid):
    feature_dataset.X[:, 22] = invalid
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", "IQRs must lie")


@pytest.mark.parametrize("name", ["connectivity_median", "connectivity_iqr"])
def test_graphs_must_agree_with_feature_columns(feature_dataset, tmp_path, name):
    graph = getattr(feature_dataset, name)
    graph[0, 0, 1] = graph[0, 1, 0] = 0.5
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", "graphs do not match")


def test_graph_shape_is_checked_before_save_and_after_load(feature_dataset, tmp_path):
    feature_dataset.connectivity_median = np.tile(np.eye(3), (len(feature_dataset.X), 1, 1))
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", "configured nodes")


def test_empty_window_tables_are_rejected(feature_dataset, tmp_path):
    feature_dataset.window_connectivity = np.zeros((0, 6))
    feature_dataset.window_subject_index = np.array([], dtype=int)
    feature_dataset.window_recording_ids = np.array([], dtype=str)
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", "nonempty")


def test_used_window_counts_must_reconcile(feature_dataset, tmp_path):
    feature_dataset.window_counts += 1
    feature_dataset.accepted_window_counts += 2
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", "counts do not match")


def test_window_table_must_cover_every_subject(feature_dataset, tmp_path):
    feature_dataset.window_subject_index[:] = 0
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", "include every feature subject")


def test_window_summaries_must_agree_with_features(feature_dataset, tmp_path):
    feature_dataset.window_connectivity[:, 0] = 0.8
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", "summaries do not match")


@pytest.mark.parametrize("invalid_ids", ["numeric", "empty"])
def test_recording_ids_are_validated_without_coercion(feature_dataset, tmp_path, invalid_ids):
    if invalid_ids == "numeric":
        feature_dataset.window_recording_ids = np.arange(len(feature_dataset.window_connectivity))
        message = "string array"
    else:
        feature_dataset.window_recording_ids[0] = " "
        message = "needs a recording ID"
    _assert_rejected(feature_dataset, tmp_path / "invalid.npz", message)
