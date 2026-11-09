"""
Visualization sub-package for NeuroQ-Connect.

Modules:
    common       — Shared utilities, styles, and I/O helpers.
    eeg          — Multichannel EEG signal plots.
    connectivity — Four-node connectivity graphs and heatmaps.
    quantum      — Quantum circuit diagrams (adapter-based).
    kernels      — Kernel matrix visualizations.
    results      — Model evaluation and comparison plots.
"""

from asd_eeg.visualization import common, connectivity, eeg, kernels, quantum, results

__all__ = ["common", "connectivity", "eeg", "kernels", "quantum", "results"]
