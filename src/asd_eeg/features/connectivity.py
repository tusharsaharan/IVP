"""Alpha-band envelope connectivity for the configured EEG graph."""

from __future__ import annotations

import numpy as np
from scipy import signal

from asd_eeg.config import Band


def alpha_envelopes(data: np.ndarray, sfreq: float, alpha: Band) -> np.ndarray:
    """Filter the alpha band and return log envelope power per electrode.

    The filter and Hilbert transform run on the continuous signal before
    windowing so window edges do not introduce a separate filter transient.
    """
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Connectivity data must be a nonempty channels-by-samples array.")
    if not np.isfinite(data).all():
        raise ValueError("Cannot compute connectivity from nonfinite samples.")
    if not 0 < alpha.low < alpha.high < sfreq / 2:
        raise ValueError("The alpha band must lie strictly inside the sampling Nyquist limit.")
    sos = signal.butter(4, [alpha.low, alpha.high], btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    envelope = np.log(np.maximum(np.abs(signal.hilbert(filtered, axis=-1)) ** 2, 1e-24))
    if not np.isfinite(envelope).all():
        raise ValueError("Alpha envelope is not finite.")
    return envelope


def regional_envelopes(envelopes: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Average electrode envelopes into the fixed node order."""
    return np.stack([envelopes[index].mean(axis=0) for index in groups])


def correlation_matrix(envelopes: np.ndarray) -> np.ndarray:
    """Return a symmetric node-by-node Pearson correlation matrix."""
    if envelopes.ndim != 2 or envelopes.shape[0] < 2 or envelopes.shape[1] < 2:
        raise ValueError("Connectivity requires at least two nodes and two samples.")
    if np.min(envelopes.std(axis=1)) <= 1e-12:
        raise ValueError("A regional alpha envelope is flat or near-constant.")
    matrix = np.corrcoef(envelopes)
    matrix = (matrix + matrix.T) / 2
    if not np.isfinite(matrix).all():
        raise ValueError("Connectivity correlation is not finite.")
    return matrix


def edge_values(matrix: np.ndarray, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Read edges in the configured authoritative order."""
    values = np.array([matrix[i, j] for i, j in edges], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Connectivity edge values are not finite.")
    if np.any(values < -1.0 - 1e-12) or np.any(values > 1.0 + 1e-12):
        raise ValueError("Connectivity edge values must lie within [-1, 1].")
    return values
