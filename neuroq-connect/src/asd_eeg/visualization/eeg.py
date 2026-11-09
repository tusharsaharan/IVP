"""
eeg.py — EEG signal visualization for NeuroQ-Connect.

All functions accept already-loaded numerical arrays.
No particular EEG file format is assumed.

Expected array shape: (n_channels, n_samples)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from asd_eeg.visualization.common import (
    apply_publication_style,
    ensure_output_dir,
    save_figure,
    validate_array,
    validate_labels,
    DEFAULT_FIGSIZE,
    DEFAULT_DPI,
    PUBLICATION_DPI,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_eeg_traces(
    eeg_data: np.ndarray,
    sfreq: float = 256.0,
    channel_names: Sequence[str] | None = None,
    title: str = "EEG Traces",
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "eeg_traces",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    channel_spacing: float = 1.5,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot multichannel EEG signals as vertically offset traces.

    Parameters
    ----------
    eeg_data:
        2-D array of shape ``(n_channels, n_samples)``.
    sfreq:
        Sampling frequency in Hz (used to build the time axis).
    channel_names:
        Optional list of channel label strings.  Defaults to
        ``["Ch 0", "Ch 1", …]``.
    title:
        Figure title.
    figsize:
        Width × height in inches.
    output_dir:
        Directory to save figures.  When ``None`` the figure is not saved.
    filename_stem:
        Base filename without extension.
    formats:
        File formats to save (e.g. ``["png", "pdf"]``).
    dpi:
        Resolution for saved files.
    channel_spacing:
        Vertical separation between traces (multiples of per-channel std).
    config:
        Optional config dict from :func:`~common.load_config`.
    close:
        Close the figure after saving.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    eeg_data = validate_array(eeg_data, "eeg_data", ndim=2, min_shape=(1, 1))
    n_channels, n_samples = eeg_data.shape

    if channel_names is None:
        channel_names = [f"Ch {i}" for i in range(n_channels)]
    else:
        channel_names = validate_labels(channel_names, n_channels, "channel_names")

    times = np.arange(n_samples) / float(sfreq)

    cfg_figsize = (
        figsize
        or (config or {}).get("figure", {}).get("default_figsize", DEFAULT_FIGSIZE)
    )
    fig_h = max(cfg_figsize[1], n_channels * 0.8)
    fig, ax = plt.subplots(figsize=(cfg_figsize[0], fig_h))

    # Compute per-channel scale so traces don't overlap
    stds = np.std(eeg_data, axis=1)
    stds[stds == 0] = 1.0  # avoid division by zero for flat channels
    median_std = float(np.median(stds))
    offset_step = median_std * channel_spacing

    for i, (trace, ch_name) in enumerate(zip(eeg_data, channel_names)):
        offset = (n_channels - 1 - i) * offset_step
        ax.plot(times, trace + offset, lw=0.9, color=f"C{i % 10}", label=ch_name)
        ax.text(
            times[-1] + 0.01 * times[-1],
            offset,
            ch_name,
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(title)
    ax.set_xlim(times[0], times[-1])
    ax.set_yticks([])  # Offsets make raw y-ticks meaningless
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_selected_channels(
    eeg_data: np.ndarray,
    channel_indices: Sequence[int],
    sfreq: float = 256.0,
    channel_names: Sequence[str] | None = None,
    title: str = "Selected EEG Channels",
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "eeg_selected_channels",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot a subset of EEG channels.

    Parameters
    ----------
    eeg_data:
        2-D array of shape ``(n_channels, n_samples)``.
    channel_indices:
        Indices of the channels to plot.
    sfreq:
        Sampling frequency in Hz.
    channel_names:
        Labels for *all* channels (indexing uses ``channel_indices``).
    title:
        Figure title.
    output_dir:
        Directory to save figures.
    filename_stem:
        Base filename without extension.

    Returns
    -------
    matplotlib.figure.Figure
    """
    eeg_data = validate_array(eeg_data, "eeg_data", ndim=2, min_shape=(1, 1))
    n_channels = eeg_data.shape[0]

    idx = list(channel_indices)
    for ci in idx:
        if not (0 <= ci < n_channels):
            raise ValueError(
                f"channel_indices contains {ci} which is out of range [0, {n_channels - 1}]."
            )

    if channel_names is None:
        channel_names = [f"Ch {i}" for i in range(n_channels)]
    else:
        channel_names = validate_labels(channel_names, n_channels, "channel_names")

    subset = eeg_data[idx, :]
    subset_names = [channel_names[i] for i in idx]

    return plot_eeg_traces(
        eeg_data=subset,
        sfreq=sfreq,
        channel_names=subset_names,
        title=title,
        figsize=figsize,
        output_dir=output_dir,
        filename_stem=filename_stem,
        formats=formats,
        dpi=dpi,
        config=config,
        close=close,
    )


def plot_eeg_time_window(
    eeg_data: np.ndarray,
    t_start: float,
    t_end: float,
    sfreq: float = 256.0,
    channel_names: Sequence[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "eeg_time_window",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot all EEG channels within a specified time window.

    Parameters
    ----------
    eeg_data:
        2-D array of shape ``(n_channels, n_samples)``.
    t_start:
        Start of the time window in seconds.
    t_end:
        End of the time window in seconds (exclusive).
    sfreq:
        Sampling frequency in Hz.

    Returns
    -------
    matplotlib.figure.Figure
    """
    eeg_data = validate_array(eeg_data, "eeg_data", ndim=2, min_shape=(1, 1))
    n_samples = eeg_data.shape[1]
    total_duration = n_samples / float(sfreq)

    if t_start < 0:
        raise ValueError(f"t_start must be >= 0; got {t_start}.")
    if t_end > total_duration:
        raise ValueError(
            f"t_end ({t_end:.3f} s) exceeds recording duration ({total_duration:.3f} s)."
        )
    if t_start >= t_end:
        raise ValueError(f"t_start ({t_start}) must be < t_end ({t_end}).")

    s_start = int(t_start * sfreq)
    s_end = int(t_end * sfreq)
    windowed = eeg_data[:, s_start:s_end]

    _title = title or f"EEG Traces — {t_start:.2f} s to {t_end:.2f} s"

    return plot_eeg_traces(
        eeg_data=windowed,
        sfreq=sfreq,
        channel_names=channel_names,
        title=_title,
        figsize=figsize,
        output_dir=output_dir,
        filename_stem=filename_stem,
        formats=formats,
        dpi=dpi,
        config=config,
        close=close,
    )


def plot_eeg_power_spectrum(
    eeg_data: np.ndarray,
    sfreq: float = 256.0,
    channel_names: Sequence[str] | None = None,
    fmin: float = 0.5,
    fmax: float = 50.0,
    title: str = "EEG Power Spectrum",
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "eeg_power_spectrum",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot the power spectral density (Welch method) for each channel.

    Parameters
    ----------
    eeg_data:
        2-D array ``(n_channels, n_samples)``.
    sfreq:
        Sampling frequency in Hz.
    fmin:
        Minimum frequency to display (Hz).
    fmax:
        Maximum frequency to display (Hz).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.signal import welch  # optional scipy dependency

    apply_publication_style(config)

    eeg_data = validate_array(eeg_data, "eeg_data", ndim=2, min_shape=(1, 1))
    n_channels = eeg_data.shape[0]

    if channel_names is None:
        channel_names = [f"Ch {i}" for i in range(n_channels)]
    else:
        channel_names = validate_labels(channel_names, n_channels, "channel_names")

    cfg_figsize = figsize or (config or {}).get("figure", {}).get("default_figsize", DEFAULT_FIGSIZE)
    fig, ax = plt.subplots(figsize=cfg_figsize)

    for i, (trace, ch_name) in enumerate(zip(eeg_data, channel_names)):
        freqs, psd = welch(trace, fs=sfreq, nperseg=min(256, trace.shape[0]))
        mask = (freqs >= fmin) & (freqs <= fmax)
        ax.semilogy(freqs[mask], psd[mask], lw=1.2, label=ch_name, color=f"C{i % 10}")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (µV²/Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, ncol=max(1, n_channels // 8))
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig
