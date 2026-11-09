"""
common.py — Shared utilities for all NeuroQ-Connect visualization modules.

Provides:
- Publication-quality matplotlib style settings.
- Consistent colour palettes.
- Output directory creation.
- Figure saving in multiple formats (PNG, PDF, SVG).
- Configuration loading from YAML.
- General input validation helpers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default style constants (overridden by load_config)
# ---------------------------------------------------------------------------

DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 150
PUBLICATION_DPI = 300

FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
ANNOTATION_SIZE = 9

ASD_COLOR = "#E15759"
TD_COLOR = "#4E79A7"
NEUTRAL_COLOR = "#76B7B2"
CMAP_CONNECTIVITY = "RdYlBu_r"
CMAP_KERNEL = "viridis"
CMAP_DIVERGING = "RdBu_r"

FORMATS = ("png", "pdf")


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load visualization settings from a YAML configuration file.

    If *config_path* is ``None`` the function searches for
    ``configs/visualization.yaml`` relative to the package root.

    Parameters
    ----------
    config_path:
        Path to a YAML configuration file, or ``None`` to use defaults.

    Returns
    -------
    dict
        Nested configuration dictionary.  Returns an empty dict if the file
        is not found (default values remain active).
    """
    if config_path is None:
        # Walk up from this file to find the project root (contains configs/)
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "configs" / "visualization.yaml"
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not Path(config_path).exists():
        logger.debug("No visualization.yaml found; using built-in defaults.")
        return {}

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    logger.debug("Loaded visualization config from %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply_publication_style(config: dict[str, Any] | None = None) -> None:
    """Apply a consistent, publication-quality matplotlib style.

    Call this once at the top of any script or test before creating figures.

    Parameters
    ----------
    config:
        Optional nested config dict (from :func:`load_config`).  When
        ``None``, built-in defaults are used.
    """
    cfg = config or {}
    fonts = cfg.get("fonts", {})
    title_size = fonts.get("title_size", TITLE_SIZE)
    label_size = fonts.get("label_size", LABEL_SIZE)
    tick_size = fonts.get("tick_size", TICK_SIZE)
    legend_size = fonts.get("legend_size", LEGEND_SIZE)
    font_family = fonts.get("family", FONT_FAMILY)

    mpl.rcParams.update(
        {
            # Font
            "font.family": font_family,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            # Figure
            "figure.dpi": cfg.get("figure", {}).get("dpi", DEFAULT_DPI),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            # Grid
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Lines
            "lines.linewidth": 1.5,
            # Saving
            "savefig.dpi": cfg.get("figure", {}).get("publication_dpi", PUBLICATION_DPI),
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create *output_dir* (and any missing parents) if it does not exist.

    Parameters
    ----------
    output_dir:
        Target directory path.

    Returns
    -------
    Path
        Resolved absolute path to the directory.
    """
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def save_figure(
    fig: mpl.figure.Figure,
    output_dir: str | Path,
    filename_stem: str,
    formats: Sequence[str] | None = None,
    dpi: int | None = None,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> list[Path]:
    """Save a matplotlib figure to one or more file formats.

    Parameters
    ----------
    fig:
        The matplotlib Figure to save.
    output_dir:
        Directory in which to save the file(s).
    filename_stem:
        Base filename without extension (e.g. ``"eeg_traces"``).
    formats:
        List of formats such as ``["png", "pdf"]``.  Defaults to config
        value or ``("png", "pdf")``.
    dpi:
        Resolution override.  Defaults to the ``publication_dpi`` config
        value or :data:`PUBLICATION_DPI`.
    config:
        Optional nested config dict.
    close:
        If ``True`` (default), close the figure after saving.

    Returns
    -------
    list[Path]
        Sorted list of paths to the saved files.
    """
    cfg = config or {}
    out_cfg = cfg.get("output", {})
    fig_cfg = cfg.get("figure", {})

    _formats = list(formats or out_cfg.get("formats", list(FORMATS)))
    _dpi = dpi or fig_cfg.get("publication_dpi", PUBLICATION_DPI)

    out_path = ensure_output_dir(output_dir)
    saved: list[Path] = []

    for fmt in _formats:
        dest = out_path / f"{filename_stem}.{fmt}"
        fig.savefig(dest, dpi=_dpi, bbox_inches="tight")
        logger.info("Saved figure → %s", dest)
        saved.append(dest)

    if close:
        plt.close(fig)

    return sorted(saved)


# ---------------------------------------------------------------------------
# Colour palette helpers
# ---------------------------------------------------------------------------

def get_color_palette(n: int, palette: str = "tab10") -> list[str]:
    """Return a list of *n* hex colour strings from a named palette.

    Parameters
    ----------
    n:
        Number of colours required.
    palette:
        Name of a seaborn / matplotlib colour palette.

    Returns
    -------
    list[str]
    """
    import seaborn as sns  # lazy import — only needed here

    return [mpl.colors.to_hex(c) for c in sns.color_palette(palette, n)]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_array(
    arr: Any,
    name: str = "array",
    ndim: int | None = None,
    min_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Validate that *arr* is a numeric NumPy array.

    Parameters
    ----------
    arr:
        Input to validate.
    name:
        Human-readable name used in error messages.
    ndim:
        Expected number of dimensions, or ``None`` to skip the check.
    min_shape:
        Tuple of minimum sizes per dimension, or ``None`` to skip.

    Returns
    -------
    np.ndarray
        The validated array.

    Raises
    ------
    TypeError
        If *arr* is not array-like or not numeric.
    ValueError
        If shape constraints are violated.
    """
    try:
        arr = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"'{name}' must be a numeric array-like; got {type(arr).__name__}.") from exc

    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"'{name}' must be {ndim}-D; got {arr.ndim}-D array.")

    if min_shape is not None:
        for axis, (actual, minimum) in enumerate(zip(arr.shape, min_shape)):
            if actual < minimum:
                raise ValueError(
                    f"'{name}' axis {axis} must have size >= {minimum}; got {actual}."
                )

    return arr


def validate_positive_int(value: Any, name: str = "value") -> int:
    """Validate that *value* is a positive integer."""
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"'{name}' must be an integer; got {type(value).__name__}.") from exc
    if v <= 0:
        raise ValueError(f"'{name}' must be positive; got {v}.")
    return v


def validate_labels(labels: Any, expected_length: int, name: str = "labels") -> list[str]:
    """Validate a list of string labels.

    Parameters
    ----------
    labels:
        List (or any sequence) of label strings.
    expected_length:
        Required number of labels.
    name:
        Human-readable parameter name for error messages.

    Returns
    -------
    list[str]
    """
    if not hasattr(labels, "__len__"):
        raise TypeError(f"'{name}' must be a sequence; got {type(labels).__name__}.")
    if len(labels) != expected_length:
        raise ValueError(
            f"'{name}' must have {expected_length} entries; got {len(labels)}."
        )
    return [str(lb) for lb in labels]
