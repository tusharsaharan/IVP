import time
from pathlib import Path

from asd_eeg.data.loaders import load_recording
from asd_eeg.data.manifest import read_manifest
from asd_eeg.features.subject_features import extract_subject_features
from asd_eeg.inference.compatibility import check_subject, load_bundle
from asd_eeg.preprocessing.pipeline import preprocess_recording


def analyze_subject(manifest: str | Path, subject_id: str, bundle_path: str | Path,
                    require_held_out: bool = True) -> dict:
    started = time.perf_counter()
    bundle = load_bundle(bundle_path)
    table = read_manifest(manifest)
    rows = table[table.subject_id == subject_id]
    if rows.empty:
        raise ValueError(f"Subject '{subject_id}' is not present in this manifest.")
    provenance = check_subject(bundle, subject_id, str(rows.dataset.iloc[0]), require_held_out)
    if rows.data_kind.iloc[0] != bundle["metadata"]["data_kind"]:
        raise ValueError("Recording and model data kinds differ.")
    config = bundle["model"].config
    # Use the bundle's exact pipeline configuration, never the current app settings.
    recordings = [preprocess_recording(load_recording(row), config) for _, row in rows.iterrows()]
    features = extract_subject_features(recordings, config)
    labels, scores = bundle["model"].predict(features.vector[None, :])
    meta = bundle["metadata"]
    return {"subject_id": subject_id, "dataset": str(rows.dataset.iloc[0]),
            "data_kind": meta["data_kind"], "predicted_label": int(labels[0]),
            "predicted_class": meta["class_mapping"][int(labels[0])],
            "decision_score": float(scores[0]), "score_direction": "Positive favors ASD; negative favors Control",
            "provenance": provenance, "model": meta["model"], "fold": meta["fold"],
            "run_id": meta["run_id"], "config_id": config.id,
            "window_count": features.window_count, "node_names": [n.name for n in config.nodes],
            "feature_vector": features.vector.tolist(), "connectivity": features.connectivity.tolist(),
            "quality": features.quality, "processing_seconds": time.perf_counter() - started}
