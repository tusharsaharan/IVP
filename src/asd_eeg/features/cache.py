"""Numeric archives with JSON metadata; no pickled feature arrays."""

import json
from pathlib import Path

import numpy as np

from asd_eeg.config import Config
from asd_eeg.contracts import FeatureDataset
from asd_eeg.data.loaders import load_recording
from asd_eeg.data.manifest import read_manifest
from asd_eeg.features.subject_features import extract_subject_features
from asd_eeg.preprocessing.pipeline import preprocess_recording


def build_features(manifest: str | Path, config: Config, output: str | Path) -> FeatureDataset:
    table = read_manifest(manifest)
    vectors, labels, subjects, details = [], [], [], []
    for subject, rows in table.groupby("subject_id", sort=True):
        recordings = [preprocess_recording(load_recording(row), config) for _, row in rows.iterrows()]
        features = extract_subject_features(recordings, config)
        vectors.append(features.vector)
        labels.append(int(rows.label.iloc[0]))
        subjects.append(subject)
        details.append({"subject_id": subject, "windows": features.window_count,
                        "quality": features.quality})
    metadata = {"schema_version": 1, "config": config.to_dict(), "config_id": config.id,
                "dataset": str(table.dataset.iloc[0]), "data_kind": str(table.data_kind.iloc[0]),
                "feature_names": list(features.feature_names), "subjects": details,
                "label_sources": sorted(set(table.label_source))}
    dataset = FeatureDataset(np.asarray(vectors), np.asarray(labels), np.asarray(subjects), metadata)
    save_features(dataset, output)
    return dataset


def save_features(dataset: FeatureDataset, path: str | Path) -> None:
    dataset.validate()
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Feature output must use the .npz extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=dataset.X, y=dataset.y, subject_ids=dataset.subject_ids.astype(str),
                        metadata=np.array(json.dumps(dataset.metadata)))


def load_features(path: str | Path) -> FeatureDataset:
    with np.load(path, allow_pickle=False) as archive:
        dataset = FeatureDataset(archive["X"], archive["y"], archive["subject_ids"].astype(str),
                                 json.loads(str(archive["metadata"].item())))
    dataset.validate()
    config = Config.from_dict(dataset.metadata["config"])
    if dataset.metadata.get("schema_version") != 1 or dataset.metadata["config_id"] != config.id:
        raise ValueError("Feature archive configuration or schema is inconsistent.")
    if dataset.X.shape[1] != config.n_features:
        raise ValueError("Feature archive width does not match its configuration.")
    if dataset.metadata.get("data_kind") not in {"recorded", "synthetic"}:
        raise ValueError("Feature archive must declare recorded or synthetic provenance.")
    return dataset
