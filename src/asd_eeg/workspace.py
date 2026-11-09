"""Shared local artifact paths for the command line and Streamlit application."""

import json
import os
from pathlib import Path

import numpy as np

from asd_eeg.config import load_config
from asd_eeg.data.manifest import read_manifest

DEFAULT_PATHS = {"manifest": "data/metadata/recordings.csv", "config": "configs/default.yaml",
                 "features": "artifacts/features/subject_features.npz", "results": "results"}
PROFILE = "data/local/workspace.json"


def check_feature_workspace(dataset, manifest: str | Path, config) -> None:
    """Prevent evaluation of a stale archive from another active cohort/configuration."""
    table = read_manifest(manifest)
    if (dataset.metadata["dataset"] != str(table.dataset.iloc[0])
            or dataset.metadata["data_kind"] != str(table.data_kind.iloc[0])):
        raise ValueError("Feature archive dataset differs from the active manifest. Rebuild features.")
    if dataset.metadata["config_id"] != config.id:
        raise ValueError("Feature archive configuration differs from the active pipeline. Rebuild features.")
    labels = table.groupby("subject_id", sort=True).label.first()
    order = np.argsort(dataset.subject_ids)
    if (labels.index.tolist() != dataset.subject_ids[order].tolist()
            or not np.array_equal(labels.to_numpy(), dataset.y[order])):
        raise ValueError("Feature archive subjects/labels differ from the active cohort. Rebuild features.")
    for subject in dataset.metadata.get("subjects", []):
        quality = subject.get("quality")
        if quality is not None:
            saved = {record["recording_id"] for record in quality}
            active = set(table[table.subject_id == subject["subject_id"]].recording_id)
            if saved != active:
                raise ValueError("Feature archive recordings differ from the active cohort. Rebuild features.")


def workspace_paths(root: str | Path) -> dict[str, str]:
    root = Path(root).resolve()
    profile = root / PROFILE
    if not profile.is_file():
        return DEFAULT_PATHS.copy()
    values = json.loads(profile.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) != set(DEFAULT_PATHS):
        raise ValueError("The local workspace profile needs manifest, config, features, and results.")
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise ValueError("Local workspace paths must be nonempty strings.")
    return values


def activate_workspace(root: str | Path, manifest: str | Path, config: str | Path,
                       features: str | Path, results: str | Path) -> dict:
    root = Path(root).resolve()
    paths = {"manifest": manifest, "config": config, "features": features, "results": results}
    resolved = {name: (root / value).resolve() for name, value in paths.items()}
    table = read_manifest(resolved["manifest"])
    if table.data_kind.iloc[0] != "recorded":
        raise ValueError("Activate a recorded-data manifest, not a synthetic software check.")
    load_config(resolved["config"])
    values = {name: Path(os.path.relpath(value, root)).as_posix()
              for name, value in resolved.items()}
    profile = root / PROFILE
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(json.dumps(values, indent=2), encoding="utf-8")
    return {"profile": str(profile), "paths": values}
