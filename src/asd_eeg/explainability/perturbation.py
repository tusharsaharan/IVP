"""
perturbation.py — Connection-perturbation sensitivity analysis.

Feature vector layout (28 dimensions, 0-indexed):
    Indices 0–15  : Band-power features (4 bands × 4 nodes = 16 values)
    Indices 16–21 : Median alpha connectivity for 6 connections (EDGES order)
    Indices 22–27 : IQR/variability for the same 6 connections

EDGES (canonical order, same as connectivity.py):
    [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

Analysis procedure for each held-out subject:
    1. Compute original decision score using the trained model.
    2. For each of the 6 connections:
       a. Copy the subject feature vector.
       b. Replace median connectivity (index 16+i) with the training reference.
       c. Replace IQR variability (index 22+i) with the training reference.
       d. Compute perturbed decision score.
       e. Record score change = perturbed - original.
    3. Return structured results (can be saved as CSV / JSON).

Constraints:
    - Model is NEVER retrained.
    - Reference values come ONLY from training data (injected by the caller).
    - Held-out test data is NEVER used to compute reference values.
    - Feature indices are enforced by module-level constants.
    - Analysis measures sensitivity only — NOT causality or biomarkers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from asd_eeg.visualization.common import (
    DEFAULT_FIGSIZE,
    PUBLICATION_DPI,
    apply_publication_style,
    save_figure,
    validate_array,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature layout constants — keep in sync with Aryan Pandey's pipeline
# ---------------------------------------------------------------------------

N_FEATURES = 28
N_BAND_POWER_FEATURES = 16   # Indices 0–15
N_CONNECTIONS = 6
MEDIAN_CONN_START = 16       # Indices 16–21
IQR_CONN_START = 22          # Indices 22–27
MEDIAN_CONN_END = MEDIAN_CONN_START + N_CONNECTIONS   # 22 (exclusive)
IQR_CONN_END = IQR_CONN_START + N_CONNECTIONS         # 28 (exclusive)

EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
]

REFERENCE_STRATEGIES = ("training_mean", "training_median")


# ---------------------------------------------------------------------------
# Data classes (plain dicts — no dataclasses to keep student-friendly)
# ---------------------------------------------------------------------------

PerturbationResult = dict  # keys defined in _make_result below


def _make_result(
    subject_id: str,
    edge_index: int,
    edge_label: str,
    original_score: float,
    perturbed_score: float,
    reference_median: float,
    reference_iqr: float,
) -> PerturbationResult:
    delta = perturbed_score - original_score
    return {
        "subject_id": subject_id,
        "edge_index": edge_index,
        "edge_label": edge_label,
        "original_score": float(original_score),
        "perturbed_score": float(perturbed_score),
        "score_change": float(delta),
        "reference_median_conn": float(reference_median),
        "reference_iqr_conn": float(reference_iqr),
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_feature_vector(features: Any, name: str = "features") -> np.ndarray:
    vec = validate_array(features, name, ndim=1, min_shape=(N_FEATURES,))
    if vec.shape[0] != N_FEATURES:
        raise ValueError(
            f"Feature vector '{name}' must have exactly {N_FEATURES} elements; "
            f"got {vec.shape[0]}."
        )
    return vec


def _validate_training_features(training_features: Any) -> np.ndarray:
    mat = validate_array(training_features, "training_features", ndim=2,
                         min_shape=(1, N_FEATURES))
    if mat.shape[1] != N_FEATURES:
        raise ValueError(
            f"training_features must have {N_FEATURES} columns; "
            f"got {mat.shape[1]}."
        )
    return mat


def _validate_edge_index(edge_index: int) -> None:
    if not (0 <= edge_index < N_CONNECTIONS):
        raise ValueError(
            f"edge_index must be in [0, {N_CONNECTIONS - 1}]; got {edge_index}."
        )


# ---------------------------------------------------------------------------
# Reference value computation
# ---------------------------------------------------------------------------

def compute_training_references(
    training_features: np.ndarray,
    strategy: str = "training_mean",
) -> dict[str, np.ndarray]:
    """Compute per-connection reference values from training data only.

    Parameters
    ----------
    training_features:
        2-D array of shape ``(n_train_subjects, 28)`` containing feature
        vectors from the **training fold only**.  Must NOT include held-out
        test subjects.
    strategy:
        How to compute the reference value:
        - ``"training_mean"``  : arithmetic mean over training subjects.
        - ``"training_median"``: median over training subjects.

    Returns
    -------
    dict with keys:
        "median_conn"  : ndarray of shape (6,) — reference median connectivity
        "iqr_conn"     : ndarray of shape (6,) — reference IQR connectivity

    Notes
    -----
    These reference values must be computed exclusively from the training fold.
    Using test/validation data to compute references would constitute data
    leakage and invalidate the analysis.
    """
    if strategy not in REFERENCE_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {REFERENCE_STRATEGIES}; got '{strategy}'."
        )

    mat = _validate_training_features(training_features)

    agg = np.mean if strategy == "training_mean" else np.median

    median_conn_refs = agg(mat[:, MEDIAN_CONN_START:MEDIAN_CONN_END], axis=0)
    iqr_conn_refs = agg(mat[:, IQR_CONN_START:IQR_CONN_END], axis=0)

    return {
        "median_conn": median_conn_refs,
        "iqr_conn": iqr_conn_refs,
    }


# ---------------------------------------------------------------------------
# Core perturbation analysis
# ---------------------------------------------------------------------------

def run_perturbation_analysis(
    subject_features: np.ndarray,
    scoring_fn: Callable[[np.ndarray], float],
    training_references: dict[str, np.ndarray],
    subject_id: str = "subject",
    node_names: Sequence[str] | None = None,
) -> list[PerturbationResult]:
    """Run connection-perturbation sensitivity analysis for one subject.

    For each of the six connections:
    - Replace its median connectivity feature with the training reference.
    - Replace its IQR connectivity feature with the training reference.
    - Compute the perturbed decision score.
    - Record the change.

    The model is NEVER retrained.

    Parameters
    ----------
    subject_features:
        1-D array of exactly 28 features for the held-out subject.
    scoring_fn:
        A callable that accepts a 1-D array of 28 features and returns
        a scalar decision score.  Typically ``model.decision_function``.
        Example::
            scoring_fn = lambda x: svm_model.decision_function(x.reshape(1, -1))[0]
    training_references:
        Dict returned by :func:`compute_training_references`.  Must contain
        ``"median_conn"`` and ``"iqr_conn"`` arrays of length 6.
        These MUST be derived from the training fold only.
    subject_id:
        Identifier for the subject (used in result dicts).
    node_names:
        Optional labels for the 4 nodes (used to build edge labels).

    Returns
    -------
    list[PerturbationResult]
        One result dict per connection (6 total), in EDGES order.

    Notes
    -----
    This analysis measures model sensitivity to controlled feature
    substitutions.  It does NOT imply causality, biological mechanism,
    or validated autism biomarkers.
    """
    vec = _validate_feature_vector(subject_features)

    median_refs = validate_array(
        training_references["median_conn"], "training_references['median_conn']",
        ndim=1, min_shape=(N_CONNECTIONS,)
    )
    iqr_refs = validate_array(
        training_references["iqr_conn"], "training_references['iqr_conn']",
        ndim=1, min_shape=(N_CONNECTIONS,)
    )
    if median_refs.shape[0] != N_CONNECTIONS or iqr_refs.shape[0] != N_CONNECTIONS:
        raise ValueError(
            f"training_references arrays must have length {N_CONNECTIONS}."
        )

    _node_names = list(node_names) if node_names is not None else [f"N{i}" for i in range(4)]

    edge_labels = [f"{_node_names[u]}-{_node_names[v]}" for u, v in EDGES]

    # Original score
    try:
        original_score = float(scoring_fn(vec))
    except Exception as exc:
        raise RuntimeError(
            f"scoring_fn raised an error on original features: {exc}"
        ) from exc

    results: list[PerturbationResult] = []

    for i, (edge, label) in enumerate(zip(EDGES, edge_labels)):
        perturbed = vec.copy()

        # Replace median connectivity feature
        perturbed[MEDIAN_CONN_START + i] = median_refs[i]
        # Replace IQR connectivity feature
        perturbed[IQR_CONN_START + i] = iqr_refs[i]

        try:
            perturbed_score = float(scoring_fn(perturbed))
        except Exception as exc:
            raise RuntimeError(
                f"scoring_fn raised an error on perturbed features (edge {i}): {exc}"
            ) from exc

        results.append(
            _make_result(
                subject_id=subject_id,
                edge_index=i,
                edge_label=label,
                original_score=original_score,
                perturbed_score=perturbed_score,
                reference_median=float(median_refs[i]),
                reference_iqr=float(iqr_refs[i]),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Multi-subject batch analysis
# ---------------------------------------------------------------------------

def run_batch_perturbation(
    subjects_features: np.ndarray,
    scoring_fn: Callable[[np.ndarray], float],
    training_references: dict[str, np.ndarray],
    subject_ids: Sequence[str] | None = None,
    node_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run perturbation analysis for multiple subjects.

    Parameters
    ----------
    subjects_features:
        2-D array of shape ``(n_subjects, 28)``.
    scoring_fn:
        As in :func:`run_perturbation_analysis`.
    training_references:
        As in :func:`run_perturbation_analysis`.
    subject_ids:
        Optional list of subject identifiers.  Defaults to ``["s0", "s1", …]``.

    Returns
    -------
    pandas.DataFrame with columns matching :data:`PerturbationResult` keys.
    """
    mat = _validate_training_features(subjects_features)
    n = mat.shape[0]
    _ids = list(subject_ids) if subject_ids is not None else [f"s{i}" for i in range(n)]

    all_results: list[PerturbationResult] = []
    for i, (vec, sid) in enumerate(zip(mat, _ids)):
        subject_results = run_perturbation_analysis(
            subject_features=vec,
            scoring_fn=scoring_fn,
            training_references=training_references,
            subject_id=sid,
            node_names=node_names,
        )
        all_results.extend(subject_results)

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_perturbation_results(
    results: list[PerturbationResult] | pd.DataFrame,
    output_dir: str | Path,
    filename_stem: str = "perturbation_results",
    formats: Sequence[str] = ("csv", "json"),
) -> list[Path]:
    """Save structured perturbation results to CSV and/or JSON.

    Parameters
    ----------
    results:
        Either a list of result dicts or a DataFrame.
    output_dir:
        Directory to save files.
    filename_stem:
        Base filename.
    formats:
        Formats to save (``"csv"`` and/or ``"json"``).

    Returns
    -------
    list[Path]
        Paths to the saved files.
    """
    from asd_eeg.visualization.common import ensure_output_dir
    out_path = ensure_output_dir(output_dir)

    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = results

    saved: list[Path] = []

    if "csv" in formats:
        dest = out_path / f"{filename_stem}.csv"
        df.to_csv(dest, index=False)
        logger.info("Saved perturbation CSV → %s", dest)
        saved.append(dest)

    if "json" in formats:
        dest = out_path / f"{filename_stem}.json"
        dest.write_text(json.dumps(df.to_dict(orient="records"), indent=2))
        logger.info("Saved perturbation JSON → %s", dest)
        saved.append(dest)

    return saved


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_perturbation_sensitivity(
    results: list[PerturbationResult] | pd.DataFrame,
    subject_id: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "perturbation_sensitivity",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Bar chart of decision-score changes from connection perturbation.

    Parameters
    ----------
    results:
        List of result dicts (one per connection) or a DataFrame.
    subject_id:
        If *results* is a multi-subject DataFrame, filter to this subject.
    title:
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    This chart shows model sensitivity to feature substitutions.
    It does NOT prove causality or biological significance.
    """
    apply_publication_style(config)

    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = results.copy()

    if subject_id is not None:
        df = df[df["subject_id"] == subject_id]
        if df.empty:
            raise ValueError(f"No results found for subject_id='{subject_id}'.")

    if df.empty:
        raise ValueError("'results' is empty; nothing to plot.")

    # If multiple subjects remain, aggregate by edge
    if df["subject_id"].nunique() > 1:
        logger.info(
            "Multiple subjects detected; plotting mean score change per connection."
        )
        df = df.groupby("edge_label", sort=False)["score_change"].mean().reset_index()
        df.columns = ["edge_label", "score_change"]
        agg_note = " (mean across subjects)"
    else:
        agg_note = ""

    cfg_conn = (config or {}).get("perturbation", {})
    bar_color = cfg_conn.get("bar_color", "#5778a4")
    highlight_color = cfg_conn.get("highlight_color", "#e49444")

    edge_labels = df["edge_label"].tolist()
    changes = df["score_change"].tolist()

    colors = [highlight_color if abs(c) == max(abs(v) for v in changes) else bar_color
              for c in changes]

    cfg_figsize = figsize or (config or {}).get("figure", {}).get("default_figsize", DEFAULT_FIGSIZE)
    fig, ax = plt.subplots(figsize=cfg_figsize)

    bars = ax.barh(edge_labels, changes, color=colors, edgecolor="grey", linewidth=0.6)
    ax.axvline(0, color="black", lw=1.0, ls="-")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)

    _sid = subject_id or (df["subject_id"].iloc[0] if "subject_id" in df.columns else "")
    _title = title or f"Connection-Perturbation Sensitivity{agg_note}"
    if _sid:
        _title += f"\nSubject: {_sid}"

    ax.set_xlabel("Decision Score Change (perturbed − original)")
    ax.set_ylabel("Connection")
    ax.set_title(_title)

    # Disclaimer note
    fig.text(
        0.5, -0.04,
        "Note: Score changes reflect model sensitivity only — not causality or clinical significance.",
        ha="center", va="center", fontsize=8, color="grey", style="italic",
    )

    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_case_study(
    subject_features: np.ndarray,
    predicted_label: int,
    decision_score: float,
    perturbation_results: list[PerturbationResult] | pd.DataFrame | None = None,
    true_label: int | None = None,
    subject_id: str = "Subject",
    class_names: Sequence[str] | None = None,
    node_names: Sequence[str] | None = None,
    is_synthetic: bool = False,
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "case_study",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Generate a case-study summary figure for a single held-out subject.

    Panels:
        Left  : Connectivity feature bar chart (median + IQR features).
        Right : Perturbation sensitivity (if provided), or feature summary.

    Parameters
    ----------
    subject_features:
        1-D feature vector of length 28.
    predicted_label:
        Model prediction (integer class index).
    decision_score:
        SVM decision function value (NOT a probability).
    perturbation_results:
        Optional list of perturbation result dicts for this subject.
    true_label:
        Ground-truth label (if available from the held-out set).
    subject_id:
        Identifier (use anonymized IDs; no real patient identities).
    class_names:
        Display names for classes.  Defaults to ``["TD", "ASD"]``.
    is_synthetic:
        If ``True``, adds a prominent "SYNTHETIC DATA — ILLUSTRATIVE" banner.
    node_names:
        Labels for the 4 nodes.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Do NOT include real patient names, dates of birth, or identifiable
    clinical information.  Use anonymized identifiers only.
    """
    apply_publication_style(config)

    vec = _validate_feature_vector(subject_features)
    _class_names = list(class_names) if class_names is not None else ["TD", "ASD"]
    _node_names = list(node_names) if node_names is not None else [f"N{i}" for i in range(4)]
    edge_labels = [f"{_node_names[u]}-{_node_names[v]}" for u, v in EDGES]

    n_panels = 2 if perturbation_results is not None else 1
    cfg_figsize = figsize or (n_panels * 6, 5)
    fig, axes = plt.subplots(1, n_panels, figsize=cfg_figsize)
    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: Connectivity features ---
    ax0 = axes[0]
    median_vals = vec[MEDIAN_CONN_START:MEDIAN_CONN_END]
    iqr_vals = vec[IQR_CONN_START:IQR_CONN_END]

    x = np.arange(N_CONNECTIONS)
    width = 0.35
    ax0.bar(x - width / 2, median_vals, width, label="Median Connectivity", color="#4E79A7", edgecolor="grey")
    ax0.bar(x + width / 2, iqr_vals, width, label="IQR Variability", color="#F28E2B", edgecolor="grey")
    ax0.set_xticks(x)
    ax0.set_xticklabels(edge_labels, rotation=30, ha="right")
    ax0.set_xlabel("Connection")
    ax0.set_ylabel("Feature Value")
    ax0.set_title("Connectivity Features")
    ax0.legend(fontsize=9)

    # --- Panel 2: Perturbation sensitivity or placeholder ---
    if perturbation_results is not None:
        ax1 = axes[1]
        df = pd.DataFrame(perturbation_results) if isinstance(perturbation_results, list) else perturbation_results
        changes = df["score_change"].tolist()
        elabels = df["edge_label"].tolist()
        bar_colors = ["#E15759" if c < 0 else "#4E79A7" for c in changes]
        ax1.barh(elabels, changes, color=bar_colors, edgecolor="grey")
        ax1.axvline(0, color="black", lw=1.0)
        ax1.set_xlabel("Decision Score Change")
        ax1.set_title("Perturbation Sensitivity")
        ax1.text(
            0.5, -0.12,
            "Sensitivity only — not causality",
            transform=ax1.transAxes, ha="center", fontsize=8, color="grey", style="italic"
        )

    # --- Title with prediction info ---
    pred_name = _class_names[predicted_label] if 0 <= predicted_label < len(_class_names) else str(predicted_label)
    true_info = f" | True: {_class_names[true_label]}" if true_label is not None and 0 <= true_label < len(_class_names) else ""
    suptitle = (
        f"Case Study — {subject_id}\n"
        f"Prediction: {pred_name}{true_info}  |  Decision Score: {decision_score:.4f}"
    )
    fig.suptitle(suptitle, fontsize=12, y=1.02)

    if is_synthetic:
        fig.text(
            0.5, 1.07,
            "⚠  SYNTHETIC DATA — ILLUSTRATIVE EXAMPLE ONLY",
            ha="center", va="bottom", fontsize=12,
            color="red", weight="bold",
            transform=fig.transFigure,
        )

    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def connection_sensitivity(bundle: dict, feature_vector: np.ndarray) -> list[dict]:
    """Compatibility interface for app and inference pipeline."""
    model = bundle["model"]
    config = model.config
    original = float(model.predict(feature_vector[None, :])[1][0])
    rows = []
    for index, (i, j) in enumerate(config.edges):
        changed = feature_vector.copy()
        positions = [4 * len(config.nodes) + index, 4 * len(config.nodes) + len(config.edges) + index]
        changed[positions] = model.feature_reference[positions]
        score = float(model.predict(changed[None, :])[1][0])
        rows.append({
            "connection": f"{config.nodes[i].name} / {config.nodes[j].name}",
            "original_score": original,
            "perturbed_score": score,
            "score_change": score - original,
        })
    return rows
