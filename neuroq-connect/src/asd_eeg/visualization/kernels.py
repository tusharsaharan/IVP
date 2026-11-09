"""
kernels.py — Kernel matrix visualization for NeuroQ-Connect.

Supports:
- Classical RBF kernel matrices.
- Quantum kernel matrices.
- Training (square) kernel matrices.
- Test-vs-training (rectangular) kernel matrices.
- Side-by-side kernel comparison.
- Kernel value distribution plots.

Validation rules:
- Square training kernels: check symmetry and unit diagonal.
- Rectangular test-vs-training: no symmetry checks.
- All kernels: check finite values and correct dimensions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from asd_eeg.visualization.common import (
    CMAP_KERNEL,
    DEFAULT_FIGSIZE,
    PUBLICATION_DPI,
    apply_publication_style,
    save_figure,
    validate_array,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_kernel_matrix(
    matrix: Any,
    name: str = "kernel_matrix",
    check_square: bool = False,
    check_symmetric: bool = False,
    check_unit_diag: bool = False,
    sym_tol: float = 1e-6,
    diag_tol: float = 1e-6,
) -> np.ndarray:
    """Validate a kernel matrix and return as a float NumPy array."""
    mat = validate_array(matrix, name, ndim=2, min_shape=(1, 1))

    if not np.all(np.isfinite(mat)):
        raise ValueError(f"'{name}' contains non-finite values (NaN or Inf).")

    if check_square and mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"'{name}' must be square for a training kernel; "
            f"got shape {mat.shape}."
        )

    if check_symmetric and not np.allclose(mat, mat.T, atol=sym_tol):
        logger.warning(
            "'%s' is not symmetric (max asymmetry = %.2e). "
            "This may indicate a numerical issue.",
            name,
            float(np.max(np.abs(mat - mat.T))),
        )

    if check_unit_diag:
        diag_max_dev = float(np.max(np.abs(np.diag(mat) - 1.0)))
        if diag_max_dev > diag_tol:
            logger.warning(
                "'%s' diagonal deviates from 1 by up to %.2e.", name, diag_max_dev
            )

    return mat


# ---------------------------------------------------------------------------
# Internal axes helper (used by quantum.py)
# ---------------------------------------------------------------------------

def _kernel_heatmap_axes(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str = "Kernel Matrix",
    cmap: str = CMAP_KERNEL,
    annot: bool = False,
) -> None:
    """Draw a kernel heatmap on an existing Axes object."""
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=".2f" if annot else "",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Kernel Value"},
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_kernel_matrix(
    kernel_matrix: np.ndarray,
    title: str = "Kernel Matrix",
    is_training_kernel: bool = True,
    cmap: str = CMAP_KERNEL,
    annot: bool = False,
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "kernel_matrix",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot a single kernel matrix as a heatmap.

    Parameters
    ----------
    kernel_matrix:
        2-D float array.  Square for training kernels; rectangular
        ``(n_test, n_train)`` for test-vs-training matrices.
    is_training_kernel:
        When ``True``, symmetry and diagonal checks are applied.
    title:
        Figure title.
    annot:
        Annotate each cell with its value (useful for small matrices).
    output_dir:
        Directory to save figures.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    mat = _validate_kernel_matrix(
        kernel_matrix,
        name="kernel_matrix",
        check_square=is_training_kernel,
        check_symmetric=is_training_kernel,
        check_unit_diag=is_training_kernel,
    )

    cfg_figsize = figsize or (config or {}).get("figure", {}).get("default_figsize", (6, 5))
    fig, ax = plt.subplots(figsize=cfg_figsize)

    _kernel_heatmap_axes(ax, mat, title=title, cmap=cmap, annot=annot)
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_kernel_comparison(
    kernels: dict[str, np.ndarray],
    is_training_kernels: bool = True,
    cmap: str = CMAP_KERNEL,
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "kernel_comparison",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Side-by-side comparison of multiple kernel matrices.

    Parameters
    ----------
    kernels:
        Ordered dict mapping descriptive name → 2-D kernel matrix.
        Example: ``{"RBF Kernel": K_rbf, "Quantum Kernel": K_q}``.
    is_training_kernels:
        Apply symmetry/diagonal checks to all matrices.
    output_dir:
        Directory to save figures.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    if not kernels:
        raise ValueError("'kernels' dict must not be empty.")

    n = len(kernels)
    cfg_figsize = figsize or (n * 5, 4)
    fig, axes = plt.subplots(1, n, figsize=cfg_figsize)
    if n == 1:
        axes = [axes]

    for ax, (name, mat) in zip(axes, kernels.items()):
        mat = _validate_kernel_matrix(
            mat,
            name=name,
            check_square=is_training_kernels,
            check_symmetric=is_training_kernels,
            check_unit_diag=is_training_kernels,
        )
        _kernel_heatmap_axes(ax, mat, title=name, cmap=cmap)

    fig.suptitle("Kernel Matrix Comparison", fontsize=13, y=1.01)
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_kernel_distribution(
    kernels: dict[str, np.ndarray],
    title: str = "Kernel Value Distributions",
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "kernel_distribution",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Overlay KDE plots of kernel value distributions.

    Parameters
    ----------
    kernels:
        Dict mapping name → 2-D kernel matrix. Off-diagonal values are
        used for square matrices to avoid the trivial diagonal spike.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    if not kernels:
        raise ValueError("'kernels' dict must not be empty.")

    cfg_figsize = figsize or (config or {}).get("figure", {}).get("default_figsize", DEFAULT_FIGSIZE)
    fig, ax = plt.subplots(figsize=cfg_figsize)

    for i, (name, mat) in enumerate(kernels.items()):
        mat = validate_array(mat, name, ndim=2, min_shape=(1, 1))
        # Extract off-diagonal elements to avoid trivial diagonal spike
        n = mat.shape[0]
        if mat.shape[0] == mat.shape[1]:
            mask = ~np.eye(n, dtype=bool)
            values = mat[mask].ravel()
        else:
            values = mat.ravel()

        sns.kdeplot(values, ax=ax, label=name, fill=True, alpha=0.3, color=f"C{i % 10}")

    ax.set_xlabel("Kernel Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_train_test_kernel(
    K_train: np.ndarray,
    K_test: np.ndarray,
    title: str = "Training and Test-vs-Training Kernels",
    cmap: str = CMAP_KERNEL,
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "train_test_kernel",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot training and test-vs-training kernel matrices side by side.

    Symmetry and unit diagonal checks are applied only to ``K_train``.
    The test kernel is validated for dimensions: its number of columns
    must equal the number of training samples (rows of ``K_train``).

    Parameters
    ----------
    K_train:
        Square training kernel matrix ``(n_train, n_train)``.
    K_test:
        Rectangular test kernel matrix ``(n_test, n_train)``.
    title:
        Overall figure title.
    cmap:
        Colormap name.
    figsize:
        Width × height in inches.
    output_dir:
        Directory to save figures.
    filename_stem:
        Base filename without extension.
    formats:
        File formats to save.
    dpi:
        Resolution for saved files.
    config:
        Optional configuration dict.
    close:
        Close figure after saving.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    K_train = _validate_kernel_matrix(
        K_train, "K_train", check_square=True, check_symmetric=True
    )
    K_test = _validate_kernel_matrix(K_test, "K_test")

    if K_test.shape[1] != K_train.shape[0]:
        raise ValueError(
            f"K_test has {K_test.shape[1]} columns but K_train has "
            f"{K_train.shape[0]} rows — columns of test kernel must match training sample count."
        )

    cfg_figsize = figsize or (12, 5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg_figsize)
    _kernel_heatmap_axes(ax1, K_train, title="Training Kernel", cmap=cmap)
    _kernel_heatmap_axes(ax2, K_test, title="Test-vs-Training Kernel", cmap=cmap)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig
