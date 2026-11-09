"""
quantum.py — Quantum circuit diagram visualization for NeuroQ-Connect.

Adapter-based design:
    When Vivek G's PennyLane circuit is available, pass the QNode (or the
    circuit function decorated with ``@qml.qnode``) directly to
    :func:`visualize_circuit`.

    If the circuit is not yet available, an explicitly labelled
    **illustrative placeholder** circuit is generated for testing only.
    This placeholder is never presented as the research model.

How to integrate Vivek's circuit
---------------------------------
1. Import your QNode (or circuit function):
       from your_module import my_qnode

2. Call::
       from asd_eeg.visualization.quantum import visualize_circuit
       visualize_circuit(
           circuit_fn=my_qnode,
           n_wires=<number_of_qubits>,
           sample_input=<representative_input_array>,
           output_dir="results/figures",
       )

3. The figure is saved to ``results/figures/quantum_circuit.<fmt>``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from asd_eeg.visualization.common import (
    PUBLICATION_DPI,
    apply_publication_style,
    save_figure,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_pennylane_draw(
    circuit_fn: Callable,
    sample_input: np.ndarray,
    n_wires: int,
) -> plt.Figure | None:
    """Attempt to draw a PennyLane circuit using its built-in drawer.

    Returns a matplotlib Figure on success, or ``None`` if PennyLane is
    unavailable or the draw fails.
    """
    try:
        import pennylane as qml
    except ImportError:
        logger.info("PennyLane not installed; cannot draw actual circuit.")
        return None

    try:
        fig, _ = qml.draw_mpl(circuit_fn)(sample_input)
        return fig
    except Exception as exc:  # noqa: BLE001
        logger.warning("qml.draw_mpl failed: %s", exc)
        return None


def _build_illustrative_circuit_figure(n_wires: int = 4) -> plt.Figure:
    """Return an ILLUSTRATIVE circuit figure for testing only.

    This circuit is a generic example. It does NOT represent the
    actual NeuroQ-Connect quantum kernel circuit.
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(10, max(3, n_wires * 0.9)))

    wire_y = list(range(n_wires - 1, -1, -1))

    # Wire lines
    for y in wire_y:
        ax.axhline(y, color="black", lw=1.2, zorder=1)
        ax.text(-0.4, y, f"q[{n_wires - 1 - y}]", va="center", ha="right", fontsize=10)

    # Generic gate blocks (illustrative)
    gates: list[tuple[float, int, str]] = []
    for w in range(n_wires):
        gates.append((0.5, w, "H"))
    for w in range(n_wires):
        gates.append((1.5, w, "RZ"))
    for w in range(n_wires - 1):
        gates.append((2.5, w, "CNOT"))
    for w in range(n_wires):
        gates.append((3.5, w, "RY"))

    for x, w, name in gates:
        y = wire_y[w]
        rect = plt.Rectangle(
            (x - 0.2, y - 0.25), 0.4, 0.5,
            facecolor="#AEC6CF", edgecolor="black", lw=0.8, zorder=2
        )
        ax.add_patch(rect)
        ax.text(x, y, name, va="center", ha="center", fontsize=7, zorder=3)

    ax.set_xlim(-0.6, 4.3)
    ax.set_ylim(-0.8, n_wires - 0.2)
    ax.axis("off")
    ax.set_title(
        "ILLUSTRATIVE CIRCUIT — FOR TESTING ONLY\n"
        "This is NOT the NeuroQ-Connect quantum kernel circuit.\n"
        "Replace with the actual circuit from Vivek G.",
        fontsize=10,
        color="red",
        pad=10,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_circuit(
    circuit_fn: Callable | None = None,
    n_wires: int = 4,
    sample_input: np.ndarray | None = None,
    title: str = "Quantum Circuit Diagram",
    output_dir: str | Path | None = None,
    filename_stem: str = "quantum_circuit",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Visualize the quantum circuit.

    If *circuit_fn* is a PennyLane QNode and *sample_input* is provided,
    the actual circuit is drawn using ``qml.draw_mpl``.

    Otherwise, an explicitly labelled **illustrative placeholder** is
    generated for testing.  It is marked clearly in the figure title.

    Parameters
    ----------
    circuit_fn:
        A PennyLane QNode or decorated circuit function (from Vivek G).
        When ``None``, the illustrative placeholder is used.
    n_wires:
        Number of qubits.  Used for the placeholder and as a fallback.
    sample_input:
        Representative input array passed to *circuit_fn* to trigger
        tracing.  Required when *circuit_fn* is provided.
    title:
        Title for the actual circuit figure (ignored for the placeholder).
    output_dir:
        Directory to save figures.
    filename_stem:
        Base filename without extension.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    **Integration instructions for Vivek G:**

    1. Ensure PennyLane is installed: ``pip install pennylane``
    2. Decorate the circuit with ``@qml.qnode(dev)`` as usual.
    3. Call::
           visualize_circuit(
               circuit_fn=your_qnode,
               n_wires=<n_qubits>,
               sample_input=<array_of_shape_(n_features,)>,
               output_dir="results/figures",
           )
    4. The saved file will be ``results/figures/quantum_circuit.png``.
    """
    apply_publication_style(config)

    fig: plt.Figure | None = None

    if circuit_fn is not None:
        if sample_input is None:
            warnings.warn(
                "circuit_fn was provided but sample_input is None; "
                "falling back to illustrative circuit.",
                UserWarning,
                stacklevel=2,
            )
        else:
            fig = _try_pennylane_draw(circuit_fn, sample_input, n_wires)
            if fig is not None:
                fig.suptitle(title, fontsize=12)

    if fig is None:
        fig = _build_illustrative_circuit_figure(n_wires=n_wires)

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def visualize_kernel_circuit_pair(
    kernel_matrix: np.ndarray,
    circuit_fn: Callable | None = None,
    n_wires: int = 4,
    sample_input: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    filename_stem: str = "circuit_kernel_pair",
    formats: Sequence[str] | None = None,
    dpi: int = PUBLICATION_DPI,
    config: dict[str, Any] | None = None,
    close: bool = True,
) -> plt.Figure:
    """Side-by-side figure: circuit diagram (left) and kernel heatmap (right).

    This helper assembles the two panels so the relationship between
    the circuit and its kernel matrix is visually clear.

    Parameters
    ----------
    kernel_matrix:
        Square or rectangular 2-D kernel matrix.
    circuit_fn:
        Optional PennyLane QNode (from Vivek G).
    """
    from asd_eeg.visualization.kernels import _kernel_heatmap_axes  # local import

    apply_publication_style(config)

    fig, (ax_circ, ax_kern) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Circuit panel ---
    if circuit_fn is not None and sample_input is not None:
        inner = _try_pennylane_draw(circuit_fn, sample_input, n_wires)
        if inner is not None:
            # Transfer the inner figure's axes content into ax_circ
            inner.canvas.draw()
            import io

            import PIL.Image
            buf = io.BytesIO()
            inner.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            img = PIL.Image.open(buf)
            ax_circ.imshow(np.array(img))
            ax_circ.axis("off")
            ax_circ.set_title("Quantum Circuit", fontsize=11)
            plt.close(inner)
        else:
            _placeholder_axes(ax_circ, n_wires)
    else:
        _placeholder_axes(ax_circ, n_wires)

    # --- Kernel panel ---
    _kernel_heatmap_axes(ax_kern, kernel_matrix, title="Quantum Kernel Matrix")

    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, filename_stem, formats=formats, dpi=dpi, config=config, close=close)
    elif close:
        plt.close(fig)

    return fig


def _placeholder_axes(ax: plt.Axes, n_wires: int) -> None:
    """Draw a minimal illustrative circuit on an existing Axes."""
    wire_y = list(range(n_wires - 1, -1, -1))
    for y in wire_y:
        ax.axhline(y, color="black", lw=1.0, zorder=1)
        ax.text(-0.3, y, f"q[{n_wires - 1 - y}]", va="center", ha="right", fontsize=8)
    for w in range(n_wires):
        y = wire_y[w]
        rect = plt.Rectangle((0.3, y - 0.2), 0.3, 0.4,
                              facecolor="#AEC6CF", edgecolor="black", lw=0.6, zorder=2)
        ax.add_patch(rect)
        ax.text(0.45, y, "H", va="center", ha="center", fontsize=7, zorder=3)
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-0.8, n_wires - 0.2)
    ax.axis("off")
    ax.set_title("Circuit (Illustrative — Not Actual)", fontsize=9, color="red")
