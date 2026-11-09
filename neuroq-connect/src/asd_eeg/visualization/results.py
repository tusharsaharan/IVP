"""
results.py — Model evaluation and comparison visualizations.

For use with Dodiya Mahavirsinh's model training/evaluation outputs.

Models evaluated:
    1. Classical RBF-SVM
    2. Amplitude-fidelity quantum kernel SVM
    3. Proposed connectivity-guided quantum kernel SVM
    4. Reassigned-connection control model

All functions accept actual metrics and predictions.
No hardcoded scores, accuracy values, or confusion matrices.

IMPORTANT: SVM decision scores are NOT probabilities.
           They are never labelled as "ASD probability" in any figure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc

from asd_eeg.visualization.common import (
    apply_publication_style,
    save_figure,
    validate_array,
    get_color_palette,
    ASD_COLOR,
    TD_COLOR,
    PUBLICATION_DPI,
    DEFAULT_FIGSIZE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_MODEL_NAMES = [
    "RBF-SVM",
    "Amplitude-Fidelity QK-SVM",
    "Connectivity-Guided QK-SVM",
    "Reassigned-Connection Control",
]

STANDARD_METRICS = ["accuracy", "precision", "recall", "f1_score"]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_metrics_dict(metrics: dict[str, float], name: str = "metrics") -> None:
    for key, val in metrics.items():
        try:
            float(val)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"metrics['{key}'] must be numeric; got {type(val).__name__}."
            ) from exc


def _validate_labels_and_scores(
    y_true: np.ndarray,
    y_pred_or_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = validate_array(y_true, "y_true", ndim=1, min_shape=(1,))
    y_pred_or_score = validate_array(y_pred_or_score, "y_pred_or_score", ndim=1, min_shape=(1,))
    if y_true.shape != y_pred_or_score.shape:
        raise ValueError(
            f"y_true and y_pred_or_score must have the same length; "
            f"got {y_true.shape} vs {y_pred_or_score.shape}."
        )
    return y_true, y_pred_or_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_model_comparison(
    model_metrics: dict[str, dict[str, float]],
    metric_keys: Sequence[str] | None = None,
    title: str = "Model Comparison",
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "model_comparison",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Grouped bar chart comparing multiple models on multiple metrics.

    Parameters
    ----------
    model_metrics:
        Dict mapping model name → dict of metric_name → float.
        Example::
            {
                "RBF-SVM": {"accuracy": 0.72, "f1_score": 0.70},
                "QK-SVM":  {"accuracy": 0.78, "f1_score": 0.76},
            }
    metric_keys:
        Which metrics to plot.  Defaults to keys present in the first model.
    title:
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    if not model_metrics:
        raise ValueError("'model_metrics' must not be empty.")

    model_names = list(model_metrics.keys())
    _metric_keys = list(metric_keys or list(next(iter(model_metrics.values())).keys()))

    for name in model_names:
        _validate_metrics_dict(model_metrics[name], name)

    n_models = len(model_names)
    n_metrics = len(_metric_keys)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    colors = get_color_palette(n_models)
    cfg_figsize = figsize or (config or {}).get("figure", {}).get("default_figsize", DEFAULT_FIGSIZE)
    fig, ax = plt.subplots(figsize=cfg_figsize)

    for i, (model_name, metrics) in enumerate(model_metrics.items()):
        values = [float(metrics.get(k, float("nan"))) for k in _metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width=width * 0.9,
                      label=model_name, color=colors[i], edgecolor="grey", linewidth=0.5)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=45)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in _metric_keys])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] | None = None,
    model_name: str = "Model",
    normalize: str | None = "true",
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "confusion_matrix",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot a confusion matrix using scikit-learn's ConfusionMatrixDisplay.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels (from the trained model).
    class_names:
        Display names for classes.  Defaults to ``["TD", "ASD"]``.
    model_name:
        Model name included in the title.
    normalize:
        Passed to ConfusionMatrixDisplay: ``"true"``, ``"pred"``,
        ``"all"``, or ``None`` for raw counts.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    y_true, y_pred = _validate_labels_and_scores(y_true, y_pred)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    _class_names = list(class_names) if class_names is not None else ["TD", "ASD"]

    cfg_figsize = figsize or (5, 4)
    fig, ax = plt.subplots(figsize=cfg_figsize)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=_class_names,
        normalize=normalize,
        colorbar=False,
        ax=ax,
        cmap="Blues",
    )
    norm_label = f" (normalized by {normalize})" if normalize else ""
    ax.set_title(f"Confusion Matrix — {model_name}{norm_label}")
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
    model_name: str = "Model",
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "roc_curve",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot an ROC curve from SVM decision scores.

    **Important:** ``decision_scores`` are raw SVM decision function outputs,
    NOT probabilities.  The figure title and labels reflect this.
    SVM decision scores can be used to rank predictions for ROC analysis
    but must NOT be interpreted as class probabilities.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0 / 1).
    decision_scores:
        SVM decision function values (can be negative).

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    y_true, decision_scores = _validate_labels_and_scores(y_true, decision_scores)

    unique_labels = np.unique(y_true.astype(int))
    if len(unique_labels) < 2:
        raise ValueError(
            "y_true must contain at least two distinct classes for ROC curve."
        )

    cfg_figsize = figsize or (5, 5)
    fig, ax = plt.subplots(figsize=cfg_figsize)

    RocCurveDisplay.from_predictions(
        y_true, decision_scores, name=model_name, ax=ax
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_title(f"ROC Curve — {model_name}\n(SVM decision scores, NOT probabilities)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_roc_comparison(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray]],
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "roc_comparison",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Overlay ROC curves for multiple models.

    Parameters
    ----------
    roc_data:
        Dict mapping model name → ``(y_true, decision_scores)`` tuple.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    if not roc_data:
        raise ValueError("'roc_data' must not be empty.")

    colors = get_color_palette(len(roc_data))
    cfg_figsize = figsize or (6, 5)
    fig, ax = plt.subplots(figsize=cfg_figsize)

    for (model_name, (y_true, scores)), color in zip(roc_data.items(), colors):
        y_true, scores = _validate_labels_and_scores(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true.astype(int), scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.5, color=color, label=f"{model_name} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_title("ROC Curve Comparison\n(SVM decision scores, NOT probabilities)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_fold_performance(
    fold_metrics: list[dict[str, float]],
    metric_key: str = "accuracy",
    model_name: str = "Model",
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "fold_performance",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot fold-wise performance for a single model.

    Parameters
    ----------
    fold_metrics:
        List of dicts, one per CV fold, each containing at minimum
        ``{metric_key: float}``.
    metric_key:
        Which metric to plot.
    model_name:
        Model name for the title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    if not fold_metrics:
        raise ValueError("'fold_metrics' must not be empty.")

    values = [float(d[metric_key]) for d in fold_metrics]
    folds = list(range(1, len(values) + 1))

    cfg_figsize = figsize or (config or {}).get("figure", {}).get("default_figsize", DEFAULT_FIGSIZE)
    fig, ax = plt.subplots(figsize=cfg_figsize)

    ax.plot(folds, values, "o-", color="#4E79A7", lw=2, ms=7, label=metric_key.replace("_", " ").title())
    mean_val = float(np.mean(values))
    ax.axhline(mean_val, color="orange", lw=1.5, ls="--",
               label=f"Mean = {mean_val:.3f}")
    ax.fill_between(folds, values, mean_val, alpha=0.15, color="#4E79A7")

    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(f"Fold-wise {metric_key.replace('_', ' ').title()} — {model_name}")
    ax.set_xticks(folds)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_metric_heatmap(
    model_metrics: dict[str, dict[str, float]],
    metric_keys: Sequence[str] | None = None,
    title: str = "Metric Heatmap",
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "metric_heatmap",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Heatmap of all model metrics for the final research report.

    Parameters
    ----------
    model_metrics:
        Dict mapping model name → metric dict.
    metric_keys:
        Which metrics to include.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    if not model_metrics:
        raise ValueError("'model_metrics' must not be empty.")

    model_names = list(model_metrics.keys())
    _metric_keys = list(metric_keys or list(next(iter(model_metrics.values())).keys()))

    matrix = np.array(
        [[float(model_metrics[m].get(k, float("nan"))) for k in _metric_keys]
         for m in model_names]
    )

    cfg_figsize = figsize or (len(_metric_keys) * 1.6 + 2, len(model_names) * 0.8 + 1.5)
    fig, ax = plt.subplots(figsize=cfg_figsize)

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlGnBu",
        annot=True,
        fmt=".3f",
        xticklabels=[k.replace("_", " ").title() for k in _metric_keys],
        yticklabels=model_names,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Score"},
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig
