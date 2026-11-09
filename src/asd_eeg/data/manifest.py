"""Read explicit labels; never infer diagnoses from a filename."""

from pathlib import Path

import pandas as pd

REQUIRED = {"dataset", "subject_id", "recording_id", "eeg_path", "label", "label_source", "data_kind"}
SUPPORTED = {".npz", ".set", ".edf", ".bdf", ".fif"}


def read_manifest(path: str | Path, check_files: bool = True) -> pd.DataFrame:
    path = Path(path).resolve()
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED - set(table.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}.")
    if table.empty:
        raise ValueError("The manifest has no recordings. Add verified recording rows first.")
    for column in REQUIRED:
        table[column] = table[column].str.strip()
        if (table[column] == "").any():
            raise ValueError(f"Every recording needs a value in '{column}'.")
    if not table.label.isin(["0", "1"]).all():
        raise ValueError("Manifest labels must be 0 (Control) or 1 (ASD).")
    if not table.data_kind.isin(["recorded", "synthetic"]).all():
        raise ValueError("data_kind must be 'recorded' or 'synthetic'.")
    if table.dataset.nunique() != 1 or table.data_kind.nunique() != 1:
        raise ValueError("Use a separate manifest for each dataset and data kind.")
    if table.recording_id.duplicated().any():
        raise ValueError("Recording IDs must be unique.")
    if table.groupby("subject_id").label.nunique().max() > 1:
        raise ValueError("A subject has conflicting labels across recordings.")
    paths = [(path.parent / p).resolve() for p in table.eeg_path]
    if len(set(paths)) != len(paths):
        raise ValueError("The same recording path is listed more than once.")
    for resolved in paths:
        if resolved.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported format: {resolved.suffix}. Supported: {sorted(SUPPORTED)}")
        if check_files and not resolved.is_file():
            raise FileNotFoundError(f"Recording not found: {resolved}")
        if check_files and resolved.stat().st_size < 512:
            raise ValueError(f"{resolved.name} is too small to be EEG. Check download pointers.")
    table["eeg_path"] = [str(p) for p in paths]
    table["label"] = table.label.astype(int)
    return table
