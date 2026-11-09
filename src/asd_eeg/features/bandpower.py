"""Window-level relative band power with explicit electrode-to-node aggregation."""

from __future__ import annotations

import numpy as np
from scipy import signal

from asd_eeg.config import Band


def relative_band_power(window: np.ndarray, sfreq: float,
                        bands: tuple[Band, ...]) -> np.ndarray:
    """Return per-electrode relative power with shape [channels, bands]."""
    if window.ndim != 2 or window.shape[1] < 2:
        raise ValueError("A power window must be a nonempty channels-by-samples array.")
    if not np.isfinite(window).all():
        raise ValueError("Cannot compute band power from nonfinite samples.")
    nperseg = min(256, window.shape[1])
    frequencies, psd = signal.welch(window, fs=sfreq, nperseg=nperseg)
    full = (frequencies >= bands[0].low) & (frequencies < bands[-1].high)
    if not full.any():
        raise ValueError("The configured bands do not cover any available frequency bin.")
    total = np.trapezoid(psd[:, full], frequencies[full], axis=1)
    if not np.isfinite(total).all():
        raise ValueError("Band-power normalization is not finite.")
    relative = []
    for band in bands:
        mask = (frequencies >= band.low) & (frequencies < band.high)
        if not mask.any():
            raise ValueError(f"Band '{band.name}' does not cover an available frequency bin.")
        power = np.trapezoid(psd[:, mask], frequencies[mask], axis=1)
        relative.append(power / np.maximum(total, 1e-24))
    values = np.stack(relative, axis=1)
    if not np.isfinite(values).all():
        raise ValueError("Relative band power is not finite.")
    return values


def node_band_power(relative: np.ndarray, groups: list[list[int]],
                    floor: float = 1e-12) -> np.ndarray:
    """Average electrodes within each node, then take log10 power.

    Returns shape [nodes, bands]. Electrode power is computed before regional
    aggregation so a node with several channels is not dominated by one electrode.
    """
    node_values = np.stack([relative[index].mean(axis=0) for index in groups])
    if not np.isfinite(node_values).all() or np.any(node_values < 0):
        raise ValueError("Node power values must be finite and nonnegative before logging.")
    return np.log10(np.maximum(node_values, floor))
