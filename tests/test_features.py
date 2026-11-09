import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asd_eeg.contracts import FeatureDataset
from asd_eeg.data.loaders import load_recording
from asd_eeg.data.manifest import read_manifest
from asd_eeg.features.cache import build_features, load_features, save_features
from asd_eeg.features.schema import feature_names, feature_schema
from asd_eeg.features.subject_features import extract_subject_features
from asd_eeg.preprocessing.pipeline import preprocess_recording

EDGE_ORDER = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _feature_archive(completed_check) -> tuple[FeatureDataset, Path]:
    path = Path(completed_check["directory"]) / "features.npz"
    return load_features(path), path


def test_feature_archive_has_fixed_order_graphs_and_schema(config, completed_check):
    dataset, path = _feature_archive(completed_check)
    assert dataset.X.shape[1] == 28
    assert dataset.connectivity_median.shape == (len(dataset.X), 4, 4)
    assert dataset.connectivity_iqr.shape == (len(dataset.X), 4, 4)
    assert dataset.window_connectivity.shape == (len(dataset.window_subject_index), 6)
    assert dataset.window_counts.shape == (len(dataset.X),)
    assert np.allclose(np.diagonal(dataset.connectivity_median, axis1=1, axis2=2), 1.0)
    assert np.allclose(np.diagonal(dataset.connectivity_iqr, axis1=1, axis2=2), 0.0)
    assert np.bincount(dataset.window_subject_index).tolist() == dataset.window_counts.tolist()
    for subject_index in range(len(dataset.X)):
        rows = dataset.window_connectivity[dataset.window_subject_index == subject_index]
        for edge_index, (left, right) in enumerate(EDGE_ORDER):
            assert dataset.connectivity_median[subject_index, left, right] == pytest.approx(
                np.median(rows[:, edge_index])
            )
            assert dataset.connectivity_iqr[subject_index, left, right] == pytest.approx(
                np.percentile(rows[:, edge_index], 75)
                - np.percentile(rows[:, edge_index], 25)
            )

    schema = json.loads((path.with_name("feature_schema.json")).read_text())
    assert schema["feature_count"] == 28
    assert schema["label_free"] is True
    assert [feature["name"] for feature in schema["features"]] == list(feature_names(config))
    assert [(edge["left_node_index"], edge["right_node_index"])
            for edge in schema["edge_order"]] == EDGE_ORDER
    assert schema["features"][0]["index"] == 0
    assert schema["features"][0]["band"] == "delta"
    assert schema["features"][15]["node_index"] == 3
    assert schema["features"][15]["band"] == "beta"
    assert schema["features"][16]["group"] == "connectivity_median"
    assert schema["features"][21]["group"] == "connectivity_median"
    assert schema["features"][22]["group"] == "connectivity_iqr"
    assert schema["features"][27]["group"] == "connectivity_iqr"
    assert dataset.metadata["feature_names"] == [f["name"] for f in schema["features"]]


def test_extraction_is_deterministic_and_label_free(config, completed_check, tmp_path):
    table = read_manifest(completed_check["manifest"])
    row = table.iloc[0]
    raw = load_recording(row)
    processed = preprocess_recording(raw, config)
    first = extract_subject_features([processed], config)
    second = extract_subject_features([processed], config)
    np.testing.assert_array_equal(first.vector, second.vector)
    np.testing.assert_array_equal(first.connectivity, second.connectivity)
    np.testing.assert_array_equal(first.connectivity_iqr, second.connectivity_iqr)
    np.testing.assert_array_equal(first.window_connectivity, second.window_connectivity)
    assert first.window_recording_ids.tolist() == second.window_recording_ids.tolist()
    assert first.accepted_window_count == second.accepted_window_count

    flipped = table.copy()
    flipped["label"] = 1 - flipped["label"]
    flipped_path = tmp_path / "flipped-labels.csv"
    flipped.to_csv(flipped_path, index=False)
    flipped_dataset = build_features(flipped_path, config, tmp_path / "features.npz")
    original, _ = _feature_archive(completed_check)
    assert flipped_dataset.subject_ids.tolist() == original.subject_ids.tolist()
    np.testing.assert_array_equal(flipped_dataset.X, original.X)
    np.testing.assert_array_equal(flipped_dataset.y, 1 - original.y)


def test_multiple_recordings_become_one_subject_row(config, completed_check, tmp_path):
    row = read_manifest(completed_check["manifest"]).iloc[0]
    raw = load_recording(row)
    split = raw.data.shape[1] // 2
    rows = []
    for index, chunk in enumerate((raw.data[:, :split], raw.data[:, split:])):
        recording_id = f"{row.recording_id}-part-{index}"
        path = tmp_path / f"{recording_id}.npz"
        np.savez_compressed(path, data=chunk, sfreq=np.array(raw.sfreq),
                            channels=np.array(raw.channels))
        rows.append({"dataset": row.dataset, "subject_id": row.subject_id,
                     "recording_id": recording_id, "eeg_path": str(path),
                     "label": int(row.label), "label_source": row.label_source,
                     "data_kind": row.data_kind})
    manifest = tmp_path / "recordings.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    two_recording_config = replace(
        config, preprocessing=replace(config.preprocessing, min_windows=1)
    )
    dataset = build_features(manifest, two_recording_config, tmp_path / "features.npz")
    assert dataset.X.shape == (1, 28)
    assert dataset.window_counts[0] == dataset.window_connectivity.shape[0]
    assert dataset.accepted_window_counts[0] >= dataset.window_counts[0]
    assert set(dataset.window_recording_ids) == {
        f"{row.recording_id}-part-0", f"{row.recording_id}-part-1"
    }
    assert dataset.window_subject_index.tolist() == [0] * dataset.window_connectivity.shape[0]


def test_flat_signal_fails_with_clear_quality_report(config, completed_check):
    row = read_manifest(completed_check["manifest"]).iloc[0]
    raw = load_recording(row)
    flat = replace(raw, data=np.zeros_like(raw.data))
    with pytest.raises(ValueError, match="flat"):
        preprocess_recording(flat, config)


def test_old_archive_without_graph_arrays_still_loads(completed_check, tmp_path):
    original, _ = _feature_archive(completed_check)
    minimal = FeatureDataset(original.X, original.y, original.subject_ids, original.metadata)
    path = tmp_path / "features.npz"
    save_features(minimal, path)
    loaded = load_features(path)
    assert loaded.connectivity_median is None
    assert loaded.window_connectivity is None
    np.testing.assert_array_equal(loaded.X, original.X)


def test_schema_matches_configuration(config):
    schema = feature_schema(config)
    assert schema["feature_count"] == config.n_features == 28
    assert len(schema["features"]) == 28
    assert schema["config_id"] == config.id
    assert [feature["index"] for feature in schema["features"]] == list(range(28))
    assert schema["edge_order"] == [
        {"index": index, "left_node_index": left, "right_node_index": right,
         "left_node": config.nodes[left].name, "right_node": config.nodes[right].name}
        for index, (left, right) in enumerate(config.edges)
    ]
    assert tuple(feature["name"] for feature in schema["features"]) == feature_names(config)


def test_archive_graph_shape_must_match_configuration(completed_check, tmp_path):
    original, _ = _feature_archive(completed_check)
    subjects = len(original.X)
    wrong = FeatureDataset(
        original.X, original.y, original.subject_ids, original.metadata,
        connectivity_median=np.tile(np.eye(3), (subjects, 1, 1)),
        connectivity_iqr=np.zeros((subjects, 3, 3)),
    )
    path = tmp_path / "wrong-graph.npz"
    save_features(wrong, path)
    with pytest.raises(ValueError, match="configured nodes"):
        load_features(path)
