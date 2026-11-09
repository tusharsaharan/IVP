from fractions import Fraction

import numpy as np
from scipy import signal

from asd_eeg.config import Config
from asd_eeg.contracts import EEGRecording, ProcessedRecording


def preprocess_recording(recording: EEGRecording, config: Config) -> ProcessedRecording:
    p = config.preprocessing
    names = {c.casefold() for c in recording.channels}
    required = {c.casefold() for n in config.nodes for c in n.channels}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Missing configured channels: {', '.join(missing)}. Review node mapping.")
    if p.high_hz >= recording.sfreq / 2:
        raise ValueError("The recording sampling rate is too low for the configured filter.")
    if recording.data.shape[1] / recording.sfreq < p.window_seconds * p.min_windows:
        raise ValueError(f"Need at least {p.window_seconds * p.min_windows:g} seconds of EEG.")
    data = recording.data.copy()
    if p.reference == "average":
        data -= data.mean(axis=0, keepdims=True)
    if 0 < p.line_hz < recording.sfreq / 2:
        b, a = signal.iirnotch(p.line_hz, 30, fs=recording.sfreq)
        data = signal.filtfilt(b, a, data, axis=-1)
    sos = signal.butter(4, [p.low_hz, p.high_hz], btype="bandpass", fs=recording.sfreq, output="sos")
    data = signal.sosfiltfilt(sos, data, axis=-1)
    ratio = Fraction(p.target_sfreq / recording.sfreq).limit_denominator(10000)
    if ratio != 1:
        data = signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)
    actual_sfreq = recording.sfreq * ratio.numerator / ratio.denominator
    width = round(actual_sfreq * p.window_seconds)
    starts = np.arange(0, data.shape[1] - width + 1, width)
    accepted = []
    rejected = {"nonfinite": 0, "amplitude": 0, "flat": 0}
    flat_channels: set[str] = set()
    for start in starts:
        window = data[:, start:start + width]
        if not np.isfinite(window).all():
            rejected["nonfinite"] += 1
        elif np.max(np.ptp(window, axis=1)) > p.max_peak_to_peak_uv * 1e-6:
            rejected["amplitude"] += 1
        elif np.min(np.std(window, axis=1)) <= 1e-12:
            rejected["flat"] += 1
            flat_channels.update(recording.channels[i] for i in
                                 np.flatnonzero(np.std(window, axis=1) <= 1e-12))
        else:
            accepted.append(int(start))
    if len(accepted) < p.min_windows:
        detail = ", ".join(f"{name}={count}" for name, count in rejected.items())
        if flat_channels:
            detail += f"; flat channels: {', '.join(sorted(flat_channels))}"
        raise ValueError(
            f"{recording.recording_id}: only {len(accepted)} clean windows; "
            f"{p.min_windows} required (rejected: {detail}). "
            "Check amplitude units, electrode contact, artifacts, and duration."
        )
    cleaned = EEGRecording(data, actual_sfreq, recording.channels, recording.subject_id,
                           recording.recording_id, recording.dataset)
    quality = {"recording_id": recording.recording_id, "total_windows": len(starts),
               "accepted_windows": len(accepted), "rejected_windows": len(starts) - len(accepted),
               "rejected_reasons": rejected, "flat_channels": sorted(flat_channels),
               "sampling_rate": actual_sfreq, "reference": p.reference}
    return ProcessedRecording(cleaned, np.asarray(accepted), width, quality)
