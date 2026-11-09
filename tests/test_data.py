from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asd_eeg.data.loaders import load_recording
from asd_eeg.data.manifest import read_manifest
from asd_eeg.features.subject_features import extract_subject_features
from asd_eeg.preprocessing.pipeline import preprocess_recording


def test_manifest_rejects_conflicting_subject_labels(completed_check, tmp_path):
    table = read_manifest(completed_check["manifest"])
    table.loc[1, "subject_id"] = table.loc[0, "subject_id"]
    path = tmp_path / "conflict.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting"):
        read_manifest(path)


def test_manifest_rejects_reused_recording(completed_check, tmp_path):
    table = read_manifest(completed_check["manifest"])
    table.loc[1, "eeg_path"] = table.loc[0, "eeg_path"]
    path = tmp_path / "duplicate.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="same recording path"):
        read_manifest(path)


def test_feature_channel_order_invariance_and_missing_channel(config, completed_check):
    row = read_manifest(completed_check["manifest"]).iloc[0]
    raw = load_recording(row)
    first = extract_subject_features([preprocess_recording(raw, config)], config)
    reordered = replace(raw, data=raw.data[::-1].copy(), channels=raw.channels[::-1])
    second = extract_subject_features([preprocess_recording(reordered, config)], config)
    np.testing.assert_allclose(first.vector, second.vector, atol=1e-10)
    assert first.vector.shape == (28,)
    np.testing.assert_allclose(first.connectivity, first.connectivity.T)
    assert first.window_count >= config.preprocessing.min_windows
    missing = replace(raw, data=raw.data[:-1], channels=raw.channels[:-1])
    with pytest.raises(ValueError, match="Missing configured channels"):
        preprocess_recording(missing, config)


def test_bad_signal_units_fail_qc(config, completed_check):
    row = read_manifest(completed_check["manifest"]).iloc[0]
    raw = load_recording(row)
    with pytest.raises(ValueError, match="clean windows"):
        preprocess_recording(replace(raw, data=raw.data * 1e6), config)


def test_recorded_npz_can_be_loaded_without_labels(completed_check):
    table = pd.read_csv(completed_check["manifest"])
    row = table.iloc[0].drop(labels=["label", "label_source"])
    row["eeg_path"] = str(Path(completed_check["manifest"]).parent / row.eeg_path)
    recording = load_recording(row)
    assert recording.data.ndim == 2
