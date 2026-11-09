"""NPZ uses volts. MNE readers supply calibrated EEG values in volts."""

from pathlib import Path

import numpy as np

from asd_eeg.contracts import EEGRecording


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
