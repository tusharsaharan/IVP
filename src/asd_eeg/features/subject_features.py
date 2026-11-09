from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asd_eeg.config import Config
from asd_eeg.contracts import ProcessedRecording, SubjectFeatures
from asd_eeg.features.bandpower import node_band_power, relative_band_power
from asd_eeg.features.connectivity import (
    alpha_envelopes,
    correlation_matrix,
    edge_values,
    regional_envelopes,
)
from asd_eeg.features.schema import feature_names, feature_schema  # noqa: F401

__all__ = ["extract_subject_features", "feature_names", "feature_schema"]


@dataclass
class _WindowFeatures:
    power: np.ndarray
    connectivity: np.ndarray
    recording_id: str


def _even_indices(count: int, limit: int) -> np.ndarray:
    """Choose at most ``limit`` deterministic indices spread across the sequence."""
    if count < 1 or limit < 1:
        raise ValueError("Window selection requires at least one window and a positive limit.")
    if count <= limit:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, limit, dtype=int))


def _graph_matrix(edges: tuple[tuple[int, int], ...], values: np.ndarray,
                  node_count: int, diagonal: float) -> np.ndarray:
    matrix = np.zeros((node_count, node_count), dtype=float)
    np.fill_diagonal(matrix, diagonal)
    for (i, j), value in zip(edges, values, strict=True):
        matrix[i, j] = matrix[j, i] = value
    return matrix


def _validate_output(subject_id: str, vector: np.ndarray, connectivity: np.ndarray,
                     connectivity_iqr: np.ndarray, config: Config) -> None:
    if vector.shape != (config.n_features,):
        raise ValueError(
            f"{subject_id}: expected {config.n_features} features for {len(config.nodes)} nodes, "
            f"got {vector.shape}."
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{subject_id}: feature vector contains nonfinite values.")
    for name, matrix, diagonal in (("median", connectivity, 1.0),
                                   ("IQR", connectivity_iqr, 0.0)):
        if matrix.shape != (len(config.nodes), len(config.nodes)):
            raise ValueError(f"{subject_id}: connectivity {name} matrix has the wrong shape.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{subject_id}: connectivity {name} matrix is not finite.")
        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ValueError(f"{subject_id}: connectivity {name} matrix is not symmetric.")
        if not np.allclose(np.diag(matrix), diagonal, atol=1e-8):
            raise ValueError(f"{subject_id}: connectivity {name} matrix has an invalid diagonal.")
        lower, upper = (-1.0, 1.0) if name == "median" else (0.0, 2.0)
        if np.any(matrix < lower - 1e-8) or np.any(matrix > upper + 1e-8):
            raise ValueError(f"{subject_id}: connectivity {name} matrix is outside its valid range.")


def extract_subject_features(recordings: list[ProcessedRecording], config: Config) -> SubjectFeatures:
    """Build one deterministic feature row from a subject's processed recordings.

    Labels are not accepted by this function and no population-level fitting is
    performed. Recordings are ordered by recording_id before windows are pooled.
    """
    if not recordings:
        raise ValueError("At least one processed recording is required.")
    subjects = {r.recording.subject_id for r in recordings}
    datasets = {r.recording.dataset for r in recordings}
    if len(subjects) != 1 or len(datasets) != 1:
        raise ValueError("Feature extraction must receive recordings from exactly one subject/dataset.")
    subject_id = next(iter(subjects))
    alpha = next((band for band in config.bands if band.name == "alpha"), None)
    if alpha is None:
        raise ValueError("Connectivity extraction requires a configured alpha band.")

    windows: list[_WindowFeatures] = []
    flat_nodes: set[str] = set()
    rejected_nonfinite = 0
    for processed in sorted(recordings, key=lambda r: r.recording.recording_id):
        recording = processed.recording
        lookup = {name.casefold(): i for i, name in enumerate(recording.channels)}
        missing = sorted({channel.casefold() for node in config.nodes for channel in node.channels}
                         - set(lookup))
        if missing:
            raise ValueError(f"{subject_id}/{recording.recording_id}: missing configured channels: "
                             f"{', '.join(missing)}.")
        groups = [[lookup[c.casefold()] for c in node.channels] for node in config.nodes]
        envelopes = alpha_envelopes(recording.data, recording.sfreq, alpha)
        regional = regional_envelopes(envelopes, groups)
        for start in processed.window_starts:
            stop = int(start) + processed.window_samples
            if start < 0 or stop > recording.data.shape[1]:
                raise ValueError(f"{subject_id}/{recording.recording_id}: accepted window is out of range.")
            window = recording.data[:, start:stop]
            envelope_window = regional[:, start:stop]
            flat = [node.name for node, sd in zip(config.nodes, envelope_window.std(axis=1))
                    if sd <= 1e-12]
            if flat:
                flat_nodes.update(flat)
                continue
            try:
                relative = relative_band_power(window, recording.sfreq, config.bands)
                powers = node_band_power(relative, groups)
                matrix = correlation_matrix(envelope_window)
                connections = edge_values(matrix, config.edges)
            except ValueError:
                rejected_nonfinite += 1
                continue
            if not np.isfinite(powers).all():
                rejected_nonfinite += 1
                continue
            windows.append(_WindowFeatures(powers.ravel(), connections, recording.recording_id))

    if len(windows) < config.preprocessing.min_windows:
        details = [f"{len(windows)} feature-valid windows"]
        if flat_nodes:
            details.append(f"flat or near-constant alpha envelope at: {', '.join(sorted(flat_nodes))}")
        if rejected_nonfinite:
            details.append(f"{rejected_nonfinite} windows with nonfinite power or connectivity")
        raise ValueError(
            f"{subject_id}: too few usable windows; {config.preprocessing.min_windows} required "
            f"({'; '.join(details)}). Check electrode contact, amplitude units, artifacts, "
            "and the configured node mapping."
        )

    accepted_window_count = len(windows)
    used = _even_indices(accepted_window_count, config.preprocessing.max_windows)
    powers = np.stack([windows[index].power for index in used])
    connections = np.stack([windows[index].connectivity for index in used])
    window_recording_ids = np.asarray([windows[index].recording_id for index in used], dtype=str)

    node_power = np.median(powers, axis=0)
    edge_median = np.median(connections, axis=0)
    edge_iqr = (np.percentile(connections, 75, axis=0)
                - np.percentile(connections, 25, axis=0))
    vector = np.concatenate([node_power, edge_median, edge_iqr])
    connectivity = _graph_matrix(config.edges, edge_median, len(config.nodes), diagonal=1.0)
    connectivity_iqr = _graph_matrix(config.edges, edge_iqr, len(config.nodes), diagonal=0.0)
    _validate_output(subject_id, vector, connectivity, connectivity_iqr, config)
    return SubjectFeatures(
        subject_id=subject_id,
        dataset=next(iter(datasets)),
        vector=vector,
        connectivity=connectivity,
        window_count=len(used),
        feature_names=feature_names(config),
        config_id=config.id,
        quality=[processed.quality for processed in
                 sorted(recordings, key=lambda r: r.recording.recording_id)],
        connectivity_iqr=connectivity_iqr,
        window_connectivity=connections,
        window_recording_ids=window_recording_ids,
        accepted_window_count=accepted_window_count,
    )
