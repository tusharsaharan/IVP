"""Stable handoff types. Arrays use volts, seconds, and explicit axis ordering."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EEGRecording:
    data: np.ndarray  # [channels, samples], volts
    sfreq: float
    channels: tuple[str, ...]
    subject_id: str
    recording_id: str
    dataset: str


@dataclass
class ProcessedRecording:
    recording: EEGRecording
    window_starts: np.ndarray
    window_samples: int
    quality: dict


@dataclass
class SubjectFeatures:
    subject_id: str
    dataset: str
    vector: np.ndarray
    connectivity: np.ndarray
    window_count: int
    feature_names: tuple[str, ...]
    config_id: str
    quality: list[dict] = field(default_factory=list)
    connectivity_iqr: np.ndarray | None = None
    window_connectivity: np.ndarray | None = None
    window_recording_ids: np.ndarray | None = None
    accepted_window_count: int | None = None


@dataclass
class FeatureDataset:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    metadata: dict
    connectivity_median: np.ndarray | None = None
    connectivity_iqr: np.ndarray | None = None
    window_counts: np.ndarray | None = None
    accepted_window_counts: np.ndarray | None = None
    window_connectivity: np.ndarray | None = None
    window_subject_index: np.ndarray | None = None
    window_recording_ids: np.ndarray | None = None

    def validate(self) -> None:
        if self.X.ndim != 2 or not len(self.X) or not np.isfinite(self.X).all():
            raise ValueError("Features must be a nonempty, finite subjects-by-features matrix.")
        if self.y.shape != (len(self.X),) or self.subject_ids.shape != (len(self.X),):
            raise ValueError("Labels and subject IDs must align with the feature rows.")
        if len(set(self.subject_ids.tolist())) != len(self.X):
            raise ValueError("Feature rows must represent unique subjects.")
        if not np.isin(self.y, [0, 1]).all():
            raise ValueError("Labels must be 0 (Control) or 1 (ASD).")
        if self.connectivity_median is not None:
            self._validate_graph(self.connectivity_median, "median", diagonal=1.0, upper=1.0)
        if self.connectivity_iqr is not None:
            self._validate_graph(self.connectivity_iqr, "IQR", diagonal=0.0, upper=2.0)
        if (self.connectivity_median is not None and self.connectivity_iqr is not None
                and self.connectivity_median.shape != self.connectivity_iqr.shape):
            raise ValueError("Median and IQR connectivity matrices must have matching shapes.")
        for name, counts in (("window_counts", self.window_counts),
                             ("accepted_window_counts", self.accepted_window_counts)):
            if counts is not None:
                if counts.shape != (len(self.X),) or not np.isfinite(counts).all():
                    raise ValueError(f"{name} must contain one finite count per subject.")
                if np.any(counts < 1) or not np.allclose(counts, np.round(counts)):
                    raise ValueError(f"{name} must contain positive integer counts.")
        if (self.window_counts is not None and self.accepted_window_counts is not None
                and np.any(self.window_counts > self.accepted_window_counts)):
            raise ValueError("Used window counts cannot exceed accepted window counts.")
        if self.window_connectivity is not None:
            matrix = self.window_connectivity
            if matrix.ndim != 2 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
                raise ValueError("Window connectivity must be a nonempty, finite windows-by-edges matrix.")
            if np.any(matrix < -1.0 - 1e-8) or np.any(matrix > 1.0 + 1e-8):
                raise ValueError("Window connectivity values must lie within [-1, 1].")
            if self.window_subject_index is None or self.window_recording_ids is None:
                raise ValueError("Window connectivity needs subject indices and recording IDs.")
            if (self.window_subject_index.shape != (len(matrix),)
                    or self.window_recording_ids.shape != (len(matrix),)):
                raise ValueError("Window metadata must align with the window-connectivity rows.")
            if (not np.issubdtype(self.window_subject_index.dtype, np.integer)
                    or np.any(self.window_subject_index < 0)
                    or np.any(self.window_subject_index >= len(self.X))):
                raise ValueError("Window subject indices must reference a feature row.")
            if self.window_recording_ids.dtype.kind not in "US":
                raise ValueError("Window recording IDs must be a string array.")
        elif any(value is not None for value in
                 (self.window_subject_index, self.window_recording_ids)):
            raise ValueError("Window metadata cannot be supplied without window connectivity.")

    def _validate_graph(self, matrix: np.ndarray, name: str, diagonal: float,
                        upper: float) -> None:
        if matrix.ndim != 3 or matrix.shape[0] != len(self.X) or matrix.shape[1] != matrix.shape[2]:
            raise ValueError(
                f"Connectivity {name} must have shape [subjects, nodes, nodes]."
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"Connectivity {name} contains nonfinite values.")
        if not np.allclose(matrix, np.swapaxes(matrix, 1, 2), atol=1e-8):
            raise ValueError(f"Connectivity {name} matrices must be symmetric.")
        if not np.allclose(np.diagonal(matrix, axis1=1, axis2=2), diagonal, atol=1e-8):
            raise ValueError(f"Connectivity {name} matrices have an invalid diagonal.")
        if np.any(matrix < -1.0 - 1e-8) or np.any(matrix > upper + 1e-8):
            raise ValueError(f"Connectivity {name} values are outside their valid range.")
