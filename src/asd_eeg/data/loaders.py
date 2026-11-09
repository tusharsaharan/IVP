"""NPZ uses volts. MNE readers supply calibrated EEG values in volts."""

from pathlib import Path

import numpy as np
import pandas as pd

from asd_eeg.contracts import EEGRecording


def apply_channel_metadata(raw, path: Path) -> None:
    """Honor an accompanying BIDS channel table; never guess electrode equivalents."""
    if not path.stem.endswith("_eeg"):
        return
    sidecar = path.with_name(path.stem.removesuffix("_eeg") + "_channels.tsv")
    if not sidecar.is_file():
        return
    table = pd.read_csv(sidecar, sep="\t", dtype=str, keep_default_na=False)
    if not {"name", "type", "status"}.issubset(table):
        raise ValueError("BIDS channel metadata needs name, type, and status columns.")
    if table.name.duplicated().any() or table.name.tolist() != raw.ch_names:
        raise ValueError("BIDS channel names/order do not match the recording header.")
    kinds = {"EEG": "eeg", "EOG": "eog", "ECG": "ecg", "EMG": "emg",
             "TRIG": "stim", "MISC": "misc"}
    if not table.type.isin(kinds).all() or not table.status.isin(["good", "bad"]).all():
        raise ValueError("Unsupported BIDS channel type/status; review the recording metadata.")
    current = dict(zip(raw.ch_names, raw.get_channel_types(), strict=True))
    changes = {row.name: kinds[row.type] for row in table.itertuples()
               if current[row.name] != kinds[row.type]}
    if changes:
        raw.set_channel_types(changes, verbose="ERROR")
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(table[table.status.eq("bad")].name))


def load_recording(row) -> EEGRecording:
    path = Path(row["eeg_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if not {"data", "sfreq", "channels"}.issubset(archive.files):
                raise ValueError("EEG NPZ files require data, sfreq, and channels arrays.")
            data = np.asarray(archive["data"], dtype=float)
            sfreq = float(archive["sfreq"].item())
            channels = tuple(archive["channels"].astype(str).tolist())
    else:
        import mne

        readers = {
            ".set": mne.io.read_raw_eeglab,
            ".edf": mne.io.read_raw_edf,
            ".bdf": mne.io.read_raw_bdf,
            ".fif": mne.io.read_raw_fif,
        }
        reader = readers.get(path.suffix.lower())
        if reader is None:
            raise ValueError(f"Unsupported EEG format: {path.suffix}")
        try:
            raw = reader(str(path), preload=True, verbose="ERROR")
            apply_channel_metadata(raw, path)
            raw.pick("eeg", exclude="bads")
            data, sfreq, channels = raw.get_data(), float(raw.info["sfreq"]), tuple(raw.ch_names)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise ValueError(
                f"Could not load {path.name}. Check companion files, channel types, and that "
                f"the recording is continuous EEG. Reader detail: {exc}"
            ) from exc
    if data.ndim != 2 or data.shape[0] != len(channels) or data.shape[1] < 2:
        raise ValueError("EEG must have shape [channels, samples] with matching channel names.")
    if not np.isfinite(sfreq) or sfreq <= 0 or not np.isfinite(data).all():
        raise ValueError("Recording contains a nonfinite signal or invalid sampling rate.")
    if len(set(c.casefold() for c in channels)) != len(channels):
        raise ValueError("Channel names must be unique, ignoring case.")
    return EEGRecording(data, sfreq, channels, str(row["subject_id"]),
                        str(row["recording_id"]), str(row["dataset"]))
