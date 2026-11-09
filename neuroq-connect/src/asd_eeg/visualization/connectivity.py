"""
connectivity.py — Four-node EEG connectivity visualization.

The six inter-node connections ALWAYS follow this canonical order:
    EDGES = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

All functions accept connectivity values from the actual feature-extraction
pipeline and do NOT assume which brain regions the nodes represent unless
explicit node labels are supplied.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from asd_eeg.visualization.common import (
    CMAP_CONNECTIVITY,
    PUBLICATION_DPI,
    apply_publication_style,
    save_figure,
    validate_array,
    validate_labels,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project constants — must stay consistent with feature extraction pipeline
# ---------------------------------------------------------------------------

EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
]
N_NODES = 4
N_EDGES = 6

DEFAULT_NODE_POSITIONS: dict[int, tuple[float, float]] = {
    0: (0.0, 1.0),
    1: (1.0, 1.0),
    2: (0.0, 0.0),
    3: (1.0, 0.0),
}

DEFAULT_NODE_NAMES = ["Node 0", "Node 1", "Node 2", "Node 3"]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_edge_weights(weights: Any) -> np.ndarray:
    w = validate_array(weights, "edge_weights", ndim=1, min_shape=(N_EDGES,))
    if w.shape[0] != N_EDGES:
        raise ValueError(
            f"edge_weights must have exactly {N_EDGES} values "
            f"(one per canonical edge); got {w.shape[0]}."
        )
    return w


def _build_graph(weights: np.ndarray, node_names: list[str]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(N_NODES))
    for (u, v), w in zip(EDGES, weights):
        G.add_edge(u, v, weight=float(w))
    return G


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_connectivity_graph(
    edge_weights: np.ndarray,
    node_names: Sequence[str] | None = None,
    node_positions: dict[int, tuple[float, float]] | None = None,
    title: str = "EEG Connectivity Graph",
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "connectivity_graph",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Draw a four-node EEG connectivity graph with weighted edges.

    Parameters
    ----------
    edge_weights:
        1-D array of exactly six connection strengths in canonical EDGES order.
    node_names:
        Labels for the four nodes.  Defaults to ``["Node 0", …, "Node 3"]``.
    node_positions:
        Dict mapping node index → (x, y) position.  Defaults to a square layout.
    title:
        Figure title.
    output_dir:
        Directory to save figures.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    weights = _validate_edge_weights(edge_weights)

    _node_names = list(node_names) if node_names is not None else DEFAULT_NODE_NAMES[:]
    validate_labels(_node_names, N_NODES, "node_names")

    pos = node_positions or DEFAULT_NODE_POSITIONS

    cfg_conn = (config or {}).get("connectivity", {})
    node_size = cfg_conn.get("node_size", 1800)
    edge_width_scale = cfg_conn.get("edge_width_scale", 5.0)
    node_font_size = cfg_conn.get("node_font_size", 11)

    G = _build_graph(weights, _node_names)
    labels = {i: _node_names[i] for i in range(N_NODES)}
    edge_list = list(G.edges())
    edge_weights_list = [G[u][v]["weight"] for u, v in edge_list]

    # Normalize edge widths and colours
    w_arr = np.array(edge_weights_list)
    w_range = w_arr.max() - w_arr.min()
    w_norm = (w_arr - w_arr.min()) / (w_range if w_range > 0 else 1.0)
    edge_widths = 1.0 + w_norm * edge_width_scale
    cmap = plt.get_cmap(CMAP_CONNECTIVITY)
    edge_colors = [cmap(v) for v in w_norm]

    cfg_figsize = figsize or (config or {}).get("figure", {}).get("default_figsize", (7, 6))
    fig, ax = plt.subplots(figsize=cfg_figsize)

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="#4E79A7", alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=node_font_size,
                            font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list,
        width=edge_widths, edge_color=edge_colors, ax=ax, alpha=0.85
    )

    # Edge weight annotations
    edge_label_dict = {(u, v): f"{w:.3f}" for (u, v), w in zip(EDGES, weights)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_label_dict, font_size=8, ax=ax
    )

    # Colour bar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=float(weights.min()), vmax=float(weights.max())),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Connection Strength", fontsize=10)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_connectivity_matrix(
    connectivity_matrix: np.ndarray,
    node_names: Sequence[str] | None = None,
    title: str = "Connectivity Matrix",
    cmap: str = CMAP_CONNECTIVITY,
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "connectivity_matrix",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot a 4×4 connectivity matrix as a heatmap.

    Parameters
    ----------
    connectivity_matrix:
        2-D array of shape ``(4, 4)`` with pairwise connectivity values.
    node_names:
        Labels for rows and columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    mat = validate_array(connectivity_matrix, "connectivity_matrix", ndim=2)
    if mat.shape != (N_NODES, N_NODES):
        raise ValueError(
            f"connectivity_matrix must be ({N_NODES}, {N_NODES}); got {mat.shape}."
        )

    _node_names = list(node_names) if node_names is not None else DEFAULT_NODE_NAMES[:]
    validate_labels(_node_names, N_NODES, "node_names")

    cfg_figsize = figsize or (config or {}).get("figure", {}).get("default_figsize", (6, 5))
    fig, ax = plt.subplots(figsize=cfg_figsize)

    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".3f",
        xticklabels=_node_names,
        yticklabels=_node_names,
        linewidths=0.5,
        linecolor="grey",
        cbar_kws={"label": "Connectivity"},
    )
    ax.set_title(title)
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def plot_edge_weights_bar(
    edge_weights: np.ndarray,
    node_names: Sequence[str] | None = None,
    title: str = "Edge Connection Strengths",
    figsize: tuple[float, float] | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "edge_weights_bar",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Bar chart of the six edge weights in canonical EDGES order.

    Parameters
    ----------
    edge_weights:
        1-D array of exactly six connection strengths.
    node_names:
        Labels for the four nodes (used to build edge labels).

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style(config)

    weights = _validate_edge_weights(edge_weights)
    _node_names = list(node_names) if node_names is not None else DEFAULT_NODE_NAMES[:]
    validate_labels(_node_names, N_NODES, "node_names")

    edge_labels = [f"{_node_names[u]}–{_node_names[v]}" for u, v in EDGES]

    cfg_figsize = figsize or (config or {}).get("figure", {}).get("default_figsize", (9, 4))
    fig, ax = plt.subplots(figsize=cfg_figsize)

    colors = plt.get_cmap(CMAP_CONNECTIVITY)(
        (weights - weights.min()) / max(weights.max() - weights.min(), 1e-9)
    )
    bars = ax.bar(edge_labels, weights, color=colors, edgecolor="grey", linewidth=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel("Connection")
    ax.set_ylabel("Connection Strength")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig
