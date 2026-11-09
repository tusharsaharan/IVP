from __future__ import annotations

import numpy as np
from scipy import signal

from asd_eeg.config import Config
from asd_eeg.contracts import ProcessedRecording, SubjectFeatures


def feature_names(config: Config) -> tuple[str, ...]:
    names = [f"{node.name}: {band.name} log-relative power"
             for node in config.nodes for band in config.bands]
    for statistic in ("median", "IQR"):
        names.extend(f"{config.nodes[i].name} / {config.nodes[j].name}: connectivity {statistic}"
                     for i, j in config.edges)
    return tuple(names)


def extract_subject_features(recordings: list[ProcessedRecording], config: Config) -> SubjectFeatures:
    if not recordings:
        raise ValueError("At least one processed recording is required.")
    subjects = {r.recording.subject_id for r in recordings}
    datasets = {r.recording.dataset for r in recordings}
    if len(subjects) != 1 or len(datasets) != 1:
        raise ValueError("Feature extraction must receive recordings from exactly one subject/dataset.")
    powers, connections = [], []
    alpha = config.bands[2]
    for processed in sorted(recordings, key=lambda r: r.recording.recording_id):
        rec = processed.recording
        lookup = {name.casefold(): i for i, name in enumerate(rec.channels)}
        groups = [[lookup[c.casefold()] for c in node.channels] for node in config.nodes]
        sos = signal.butter(4, [alpha.low, alpha.high], btype="bandpass", fs=rec.sfreq, output="sos")
        filtered = signal.sosfiltfilt(sos, rec.data, axis=-1)
        envelopes = np.log(np.maximum(np.abs(signal.hilbert(filtered, axis=-1)) ** 2, 1e-24))
        regional_envelopes = np.stack([envelopes[index].mean(axis=0) for index in groups])
        for start in processed.window_starts:
            stop = start + processed.window_samples
            window = rec.data[:, start:stop]
            frequencies, psd = signal.welch(window, fs=rec.sfreq, nperseg=min(256, len(window[0])))
            full = (frequencies >= config.bands[0].low) & (frequencies < config.bands[-1].high)
            total = np.trapezoid(psd[:, full], frequencies[full], axis=1)
            relative = []
            for band in config.bands:
                mask = (frequencies >= band.low) & (frequencies < band.high)
                power = np.trapezoid(psd[:, mask], frequencies[mask], axis=1)
                relative.append(power / np.maximum(total, 1e-24))
            relative = np.stack(relative, axis=1)
            node_powers = np.stack([np.log10(np.maximum(relative[index].mean(axis=0), 1e-12))
                                    for index in groups])
            envelope_window = regional_envelopes[:, start:stop]
            if np.min(envelope_window.std(axis=1)) < 1e-12:
                continue
            correlation = np.corrcoef(envelope_window)
            edge_values = np.array([correlation[i, j] for i, j in config.edges])
            if np.isfinite(node_powers).all() and np.isfinite(edge_values).all():
                powers.append(node_powers.ravel())
                connections.append(edge_values)
    if len(powers) < config.preprocessing.min_windows:
        raise ValueError("Too few windows with finite power and connectivity features.")
    # Cap windows at subject level, deterministically spread across all accepted recordings.
    indices = np.linspace(0, len(powers) - 1, min(len(powers), config.preprocessing.max_windows), dtype=int)
    powers, connections = np.asarray(powers)[indices], np.asarray(connections)[indices]
    edge_median = np.median(connections, axis=0)
    edge_iqr = np.percentile(connections, 75, axis=0) - np.percentile(connections, 25, axis=0)
    vector = np.concatenate([np.median(powers, axis=0), edge_median, edge_iqr])
    graph = np.eye(len(config.nodes))
    for (i, j), value in zip(config.edges, edge_median, strict=True):
        graph[i, j] = graph[j, i] = value
    return SubjectFeatures(next(iter(subjects)), next(iter(datasets)), vector, graph,
                           len(indices), feature_names(config), config.id,
                           [p.quality for p in recordings])
