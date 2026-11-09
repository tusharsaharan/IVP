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


@dataclass
class FeatureDataset:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    metadata: dict

    def validate(self) -> None:
        if self.X.ndim != 2 or not len(self.X) or not np.isfinite(self.X).all():
            raise ValueError("Features must be a nonempty, finite subjects-by-features matrix.")
        if self.y.shape != (len(self.X),) or self.subject_ids.shape != (len(self.X),):
            raise ValueError("Labels and subject IDs must align with the feature rows.")
        if len(set(self.subject_ids.tolist())) != len(self.X):
            raise ValueError("Feature rows must represent unique subjects.")
        if not np.isin(self.y, [0, 1]).all():
            raise ValueError("Labels must be 0 (Control) or 1 (ASD).")
