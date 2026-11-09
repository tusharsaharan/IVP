"""Numeric archives with JSON metadata; no pickled feature arrays."""

import json
from pathlib import Path

import numpy as np

from asd_eeg.config import Config
from asd_eeg.contracts import FeatureDataset
from asd_eeg.data.loaders import load_recording
from asd_eeg.data.manifest import read_manifest
from asd_eeg.features.schema import feature_schema
from asd_eeg.features.subject_features import extract_subject_features
from asd_eeg.preprocessing.pipeline import preprocess_recording


def build_features(manifest: str | Path, config: Config, output: str | Path) -> FeatureDataset:
    table = read_manifest(manifest)
    vectors, labels, subjects, details = [], [], [], []
    medians, iqrs, window_counts, accepted_counts = [], [], [], []
    window_edges, window_subject_index, window_recording_ids = [], [], []
    for subject_index, (subject, rows) in enumerate(table.groupby("subject_id", sort=True)):
        recordings = [preprocess_recording(load_recording(row), config) for _, row in rows.iterrows()]
        features = extract_subject_features(recordings, config)
        vectors.append(features.vector)
        labels.append(int(rows.label.iloc[0]))
        subjects.append(subject)
        medians.append(features.connectivity)
        iqrs.append(features.connectivity_iqr)
        window_counts.append(features.window_count)
        accepted_counts.append(features.accepted_window_count)
        window_edges.append(features.window_connectivity)
        window_subject_index.extend([subject_index] * len(features.window_connectivity))
        window_recording_ids.extend(features.window_recording_ids.tolist())
        details.append({"subject_id": subject, "windows": features.window_count,
                        "accepted_windows": features.accepted_window_count,
                        "quality": features.quality})
    schema = feature_schema(config)
    metadata = {"schema_version": 1, "config": config.to_dict(), "config_id": config.id,
                "dataset": str(table.dataset.iloc[0]), "data_kind": str(table.data_kind.iloc[0]),
                "feature_names": list(features.feature_names), "feature_schema": schema,
                "subjects": details, "label_sources": sorted(set(table.label_source))}
    dataset = FeatureDataset(
        X=np.asarray(vectors),
        y=np.asarray(labels),
        subject_ids=np.asarray(subjects),
        metadata=metadata,
        connectivity_median=np.asarray(medians),
        connectivity_iqr=np.asarray(iqrs),
        window_counts=np.asarray(window_counts, dtype=int),
        accepted_window_counts=np.asarray(accepted_counts, dtype=int),
        window_connectivity=np.concatenate(window_edges, axis=0),
        window_subject_index=np.asarray(window_subject_index, dtype=int),
        window_recording_ids=np.asarray(window_recording_ids, dtype=str),
    )
    dataset.validate()
    save_features(dataset, output)
    return dataset


def _archive_arrays(dataset: FeatureDataset) -> dict[str, np.ndarray]:
    arrays = {
        "X": dataset.X,
        "y": dataset.y,
        "subject_ids": dataset.subject_ids.astype(str),
        "metadata": np.array(json.dumps(dataset.metadata)),
    }
    optional = {
        "connectivity_median": dataset.connectivity_median,
        "connectivity_iqr": dataset.connectivity_iqr,
        "window_counts": dataset.window_counts,
        "accepted_window_counts": dataset.accepted_window_counts,
        "window_connectivity": dataset.window_connectivity,
        "window_subject_index": dataset.window_subject_index,
        "window_recording_ids": dataset.window_recording_ids,
    }
    arrays.update({name: value for name, value in optional.items() if value is not None})
    return arrays


def save_features(dataset: FeatureDataset, path: str | Path) -> None:
    dataset.validate()
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Feature output must use the .npz extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **_archive_arrays(dataset))
    schema = dataset.metadata.get("feature_schema")
    if schema is not None:
        schema_path = path.with_name("feature_schema.json")
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def load_features(path: str | Path) -> FeatureDataset:
    path = Path(path)
    optional_names = ("connectivity_median", "connectivity_iqr", "window_counts",
                      "accepted_window_counts", "window_connectivity",
                      "window_subject_index", "window_recording_ids")
    optional: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        if "window_recording_ids" in archive:
            optional["window_recording_ids"] = archive["window_recording_ids"].astype(str)
        for name in optional_names:
            if name != "window_recording_ids" and name in archive:
                optional[name] = archive[name]
        dataset = FeatureDataset(archive["X"], archive["y"],
                                 archive["subject_ids"].astype(str),
                                 json.loads(str(archive["metadata"].item())), **optional)
    dataset.validate()
    config = Config.from_dict(dataset.metadata["config"])
    if dataset.metadata.get("schema_version") != 1 or dataset.metadata["config_id"] != config.id:
        raise ValueError("Feature archive configuration or schema is inconsistent.")
    if dataset.X.shape[1] != config.n_features:
        raise ValueError("Feature archive width does not match its configuration.")
    for name, matrix in (("median", dataset.connectivity_median),
                         ("IQR", dataset.connectivity_iqr)):
        if matrix is not None and matrix.shape[1:] != (len(config.nodes), len(config.nodes)):
            raise ValueError(f"Connectivity {name} matrices do not match the configured nodes.")
    if (dataset.window_connectivity is not None
            and dataset.window_connectivity.shape[1] != len(config.edges)):
        raise ValueError("Window connectivity does not match the configured edge order.")
    if dataset.metadata.get("data_kind") not in {"recorded", "synthetic"}:
        raise ValueError("Feature archive must declare recorded or synthetic provenance.")
    schema = dataset.metadata.get("feature_schema")
    if schema is not None and (schema.get("config_id") != config.id
                               or schema.get("feature_count") != config.n_features):
        raise ValueError("Embedded feature schema does not match the archive configuration.")
    schema_path = path.with_name("feature_schema.json")
    if schema_path.is_file():
        sidecar = json.loads(schema_path.read_text(encoding="utf-8"))
        if (sidecar.get("config_id") != config.id
                or sidecar.get("feature_count") != config.n_features):
            raise ValueError("Feature schema file does not match the archive configuration.")
    return dataset
