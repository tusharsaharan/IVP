"""
generate_test_figures.py — Illustrative example figure generation.

IMPORTANT: ALL DATA IN THIS SCRIPT IS SYNTHETIC.
This script demonstrates how to call every visualization and
explainability function.  It does NOT contain real EEG recordings,
real subject data, or actual experimental results.

Generated figures are saved to results/figures/ and results/case_studies/
with the prefix "example_" or "synthetic_" to distinguish them from
actual research outputs.

Usage:
    python examples/generate_test_figures.py

Requirements:
    pip install -e ".[dev]"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root without installing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Headless — no display required

from asd_eeg.explainability.perturbation import (
    compute_training_references,
    plot_case_study,
    plot_perturbation_sensitivity,
    run_batch_perturbation,
    run_perturbation_analysis,
    save_perturbation_results,
)
from asd_eeg.visualization import connectivity as conn_mod
from asd_eeg.visualization import eeg as eeg_mod
from asd_eeg.visualization import kernels as kern_mod
from asd_eeg.visualization import quantum as quantum_mod
from asd_eeg.visualization import results as res_mod
from asd_eeg.visualization.common import apply_publication_style, load_config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
CASE_STUDIES_DIR = PROJECT_ROOT / "results" / "case_studies"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CASE_STUDIES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config and style
# ---------------------------------------------------------------------------

config = load_config()
apply_publication_style(config)

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2024)
FORMATS = ["png", "pdf"]

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_synthetic_eeg(n_channels: int = 4, duration_s: float = 5.0, sfreq: float = 256.0) -> np.ndarray:
    """Return a synthetic EEG array (n_channels, n_samples) with realistic-looking oscillations."""
    n_samples = int(duration_s * sfreq)
    t = np.linspace(0, duration_s, n_samples)
    data = np.zeros((n_channels, n_samples))
    freqs = [1.0, 10.0, 25.0, 40.0]  # Delta, Alpha, Beta, Gamma
    for ch in range(n_channels):
        f = freqs[ch % len(freqs)]
        data[ch] = (
            RNG.standard_normal(n_samples) * 5.0
            + 20.0 * np.sin(2 * np.pi * f * t + RNG.uniform(0, 2 * np.pi))
        )
    return data


def make_symmetric_kernel(n: int) -> np.ndarray:
    """Return a positive-definite symmetric kernel matrix with approximate unit diagonal."""
    A = RNG.uniform(0.1, 1.0, (n, n))
    K = A @ A.T
    d = np.sqrt(np.diag(K))
    K = K / np.outer(d, d)
    np.fill_diagonal(K, 1.0)
    return K


def make_training_features(n: int = 40) -> np.ndarray:
    """Return synthetic 28-dim feature matrix for n subjects."""
    return RNG.standard_normal((n, 28))


def make_linear_scorer(ref_vec: np.ndarray) -> callable:
    """Return a linear scoring function (simulates a fitted SVM decision function)."""
    weights = RNG.standard_normal(28)
    def _score(x: np.ndarray) -> float:
        return float(weights @ x)
    return _score


# ---------------------------------------------------------------------------
# 1. EEG visualizations
# ---------------------------------------------------------------------------

def demo_eeg():
    print("\n[1/8] EEG visualizations...")
    eeg_data = make_synthetic_eeg(n_channels=4, duration_s=5.0)
    node_names = ["F3", "F4", "P3", "P4"]

    eeg_mod.plot_eeg_traces(
        eeg_data,
        sfreq=256.0,
        channel_names=node_names,
        title="SYNTHETIC EEG Traces (Illustrative)",
        output_dir=FIGURES_DIR,
        filename_stem="example_eeg_traces",
        formats=FORMATS,
    )
    print("  Saved: example_eeg_traces.[png,pdf]")

    eeg_mod.plot_selected_channels(
        eeg_data,
        channel_indices=[0, 2],
        channel_names=node_names,
        title="SYNTHETIC EEG — Selected Channels (Illustrative)",
        output_dir=FIGURES_DIR,
        filename_stem="example_eeg_selected",
        formats=FORMATS,
    )
    print("  Saved: example_eeg_selected.[png,pdf]")

    eeg_mod.plot_eeg_time_window(
        eeg_data,
        t_start=1.0,
        t_end=3.0,
        sfreq=256.0,
        channel_names=node_names,
        title="SYNTHETIC EEG — Time Window 1–3 s (Illustrative)",
        output_dir=FIGURES_DIR,
        filename_stem="example_eeg_window",
        formats=FORMATS,
    )
    print("  Saved: example_eeg_window.[png,pdf]")


# ---------------------------------------------------------------------------
# 2. Connectivity visualizations
# ---------------------------------------------------------------------------

def demo_connectivity():
    print("\n[2/8] Connectivity visualizations...")
    node_names = ["F3", "F4", "P3", "P4"]
    edge_weights = RNG.uniform(0.1, 0.9, 6)

    conn_mod.plot_connectivity_graph(
        edge_weights,
        node_names=node_names,
        title="SYNTHETIC Connectivity Graph (Illustrative)",
        output_dir=FIGURES_DIR,
        filename_stem="example_connectivity_graph",
        formats=FORMATS,
    )
    print("  Saved: example_connectivity_graph.[png,pdf]")

    # Build 4x4 matrix from edge weights
    import networkx as nx
    G = nx.Graph()
    for (u, v), w in zip(conn_mod.EDGES, edge_weights):
        G.add_edge(u, v, weight=float(w))
    mat = np.zeros((4, 4))
    for u, v, d in G.edges(data=True):
        mat[u, v] = d["weight"]
        mat[v, u] = d["weight"]

    conn_mod.plot_connectivity_matrix(
        mat,
        node_names=node_names,
        title="SYNTHETIC Connectivity Matrix (Illustrative)",
        output_dir=FIGURES_DIR,
        filename_stem="example_connectivity_matrix",
        formats=FORMATS,
    )
    print("  Saved: example_connectivity_matrix.[png,pdf]")

    conn_mod.plot_edge_weights_bar(
        edge_weights,
        node_names=node_names,
        title="SYNTHETIC Edge Weights (Illustrative)",
        output_dir=FIGURES_DIR,
        filename_stem="example_edge_weights_bar",
        formats=FORMATS,
    )
    print("  Saved: example_edge_weights_bar.[png,pdf]")


# ---------------------------------------------------------------------------
# 3. Quantum circuit visualization
# ---------------------------------------------------------------------------

def demo_quantum():
    print("\n[3/8] Quantum circuit visualization...")
    quantum_mod.visualize_circuit(
        circuit_fn=None,  # No actual circuit yet — illustrative placeholder
        n_wires=4,
        output_dir=FIGURES_DIR,
        filename_stem="example_quantum_circuit_placeholder",
        formats=FORMATS,
    )
    print("  Saved: example_quantum_circuit_placeholder.[png,pdf]")
    print("  NOTE: This is a PLACEHOLDER. Replace circuit_fn with Vivek's QNode.")


# ---------------------------------------------------------------------------
# 4. Kernel visualizations
# ---------------------------------------------------------------------------

def demo_kernels():
    print("\n[4/8] Kernel matrix visualizations...")
    n_train = 24
    n_test = 8

    K_rbf = make_symmetric_kernel(n_train)
    K_qk  = make_symmetric_kernel(n_train)
    K_test = RNG.uniform(0.0, 1.0, (n_test, n_train))

    kern_mod.plot_kernel_matrix(
        K_rbf,
        title="SYNTHETIC RBF Kernel Matrix (Illustrative)",
        is_training_kernel=True,
        output_dir=FIGURES_DIR,
        filename_stem="example_kernel_rbf",
        formats=FORMATS,
    )
    print("  Saved: example_kernel_rbf.[png,pdf]")

    kern_mod.plot_kernel_matrix(
        K_qk,
        title="SYNTHETIC Quantum Kernel Matrix (Illustrative)",
        is_training_kernel=True,
        output_dir=FIGURES_DIR,
        filename_stem="example_kernel_quantum",
        formats=FORMATS,
    )
    print("  Saved: example_kernel_quantum.[png,pdf]")

    kern_mod.plot_kernel_comparison(
        {"RBF (Synthetic)": K_rbf, "Quantum (Synthetic)": K_qk},
        is_training_kernels=True,
        output_dir=FIGURES_DIR,
        filename_stem="example_kernel_comparison",
        formats=FORMATS,
    )
    print("  Saved: example_kernel_comparison.[png,pdf]")

    kern_mod.plot_kernel_distribution(
        {"RBF (Synthetic)": K_rbf, "Quantum (Synthetic)": K_qk},
        title="SYNTHETIC Kernel Value Distributions (Illustrative)",
        output_dir=FIGURES_DIR,
        filename_stem="example_kernel_distribution",
        formats=FORMATS,
    )
    print("  Saved: example_kernel_distribution.[png,pdf]")

    kern_mod.plot_train_test_kernel(
        K_rbf, K_test,
        output_dir=FIGURES_DIR,
        filename_stem="example_kernel_train_test",
        formats=FORMATS,
    )
    print("  Saved: example_kernel_train_test.[png,pdf]")


# ---------------------------------------------------------------------------
# 5. Model evaluation visualizations
# ---------------------------------------------------------------------------

def demo_results():
    print("\n[5/8] Model evaluation visualizations...")

    # Synthetic metrics — clearly NOT real experimental results
    model_metrics = {
        "RBF-SVM (Synthetic)": {
            "accuracy": float(RNG.uniform(0.60, 0.80)),
            "precision": float(RNG.uniform(0.58, 0.78)),
            "recall": float(RNG.uniform(0.60, 0.80)),
            "f1_score": float(RNG.uniform(0.59, 0.79)),
        },
        "Amplitude-Fidelity QK (Synthetic)": {
            "accuracy": float(RNG.uniform(0.62, 0.82)),
            "precision": float(RNG.uniform(0.60, 0.80)),
            "recall": float(RNG.uniform(0.62, 0.82)),
            "f1_score": float(RNG.uniform(0.61, 0.81)),
        },
        "Connectivity QK (Synthetic)": {
            "accuracy": float(RNG.uniform(0.65, 0.85)),
            "precision": float(RNG.uniform(0.63, 0.83)),
            "recall": float(RNG.uniform(0.65, 0.85)),
            "f1_score": float(RNG.uniform(0.64, 0.84)),
        },
        "Reassigned Control (Synthetic)": {
            "accuracy": float(RNG.uniform(0.58, 0.78)),
            "precision": float(RNG.uniform(0.56, 0.76)),
            "recall": float(RNG.uniform(0.58, 0.78)),
            "f1_score": float(RNG.uniform(0.57, 0.77)),
        },
    }

    res_mod.plot_model_comparison(
        model_metrics,
        title="SYNTHETIC Model Comparison (Illustrative — NOT actual results)",
        output_dir=FIGURES_DIR,
        filename_stem="example_model_comparison",
        formats=FORMATS,
    )
    print("  Saved: example_model_comparison.[png,pdf]")

    res_mod.plot_metric_heatmap(
        model_metrics,
        title="SYNTHETIC Metric Heatmap (Illustrative)",
        output_dir=FIGURES_DIR,
        filename_stem="example_metric_heatmap",
        formats=FORMATS,
    )
    print("  Saved: example_metric_heatmap.[png,pdf]")

    # Synthetic labels and decision scores
    n = 30
    y_true = RNG.integers(0, 2, size=n)
    y_pred = (y_true + RNG.integers(0, 2, size=n)) % 2
    scores = RNG.standard_normal(n)

    res_mod.plot_confusion_matrix(
        y_true, y_pred,
        model_name="Synthetic Model",
        output_dir=FIGURES_DIR,
        filename_stem="example_confusion_matrix",
        formats=FORMATS,
    )
    print("  Saved: example_confusion_matrix.[png,pdf]")

    res_mod.plot_roc_curve(
        y_true, scores,
        model_name="Synthetic Model (SVM decision scores)",
        output_dir=FIGURES_DIR,
        filename_stem="example_roc_curve",
        formats=FORMATS,
    )
    print("  Saved: example_roc_curve.[png,pdf]")

    fold_metrics = [
        {"accuracy": float(RNG.uniform(0.60, 0.85))} for _ in range(5)
    ]
    res_mod.plot_fold_performance(
        fold_metrics, metric_key="accuracy",
        model_name="Synthetic Model",
        output_dir=FIGURES_DIR,
        filename_stem="example_fold_performance",
        formats=FORMATS,
    )
    print("  Saved: example_fold_performance.[png,pdf]")

    roc_data = {
        "RBF-SVM (Syn)": (y_true, RNG.standard_normal(n)),
        "QK-SVM (Syn)":  (y_true, RNG.standard_normal(n)),
    }
    res_mod.plot_roc_comparison(
        roc_data,
        output_dir=FIGURES_DIR,
        filename_stem="example_roc_comparison",
        formats=FORMATS,
    )
    print("  Saved: example_roc_comparison.[png,pdf]")


# ---------------------------------------------------------------------------
# 6. Perturbation sensitivity analysis
# ---------------------------------------------------------------------------

def demo_perturbation():
    print("\n[6/8] Perturbation sensitivity analysis...")

    training_data = make_training_features(n=40)
    subject_vec = RNG.standard_normal(28)
    training_refs = compute_training_references(training_data, strategy="training_mean")

    # Synthetic scorer
    weights = RNG.standard_normal(28)
    scorer = lambda x: float(weights @ x)

    results = run_perturbation_analysis(
        subject_features=subject_vec,
        scoring_fn=scorer,
        training_references=training_refs,
        subject_id="synthetic_subject_01",
        node_names=["F3", "F4", "P3", "P4"],
    )

    plot_perturbation_sensitivity(
        results,
        subject_id="synthetic_subject_01",
        title="SYNTHETIC Perturbation Sensitivity (Illustrative)",
        output_dir=FIGURES_DIR,
        filename_stem="example_perturbation_sensitivity",
        formats=FORMATS,
    )
    print("  Saved: example_perturbation_sensitivity.[png,pdf]")

    # Save structured results
    paths = save_perturbation_results(
        results,
        output_dir=FIGURES_DIR,
        filename_stem="example_perturbation_results",
        formats=("csv", "json"),
    )
    for p in paths:
        print(f"  Saved: {p.name}")

    # Batch
    batch_df = run_batch_perturbation(
        training_data[:5], scorer, training_refs,
        subject_ids=[f"syn_{i}" for i in range(5)],
        node_names=["F3", "F4", "P3", "P4"],
    )
    plot_perturbation_sensitivity(
        batch_df,
        title="SYNTHETIC Batch Perturbation (Illustrative — 5 subjects)",
        output_dir=FIGURES_DIR,
        filename_stem="example_perturbation_batch",
        formats=FORMATS,
    )
    print("  Saved: example_perturbation_batch.[png,pdf]")


# ---------------------------------------------------------------------------
# 7 & 8. Held-out subject case studies
# ---------------------------------------------------------------------------

def demo_case_studies():
    print("\n[7/8] Case-study figures (synthetic)...")

    training_data = make_training_features(n=40)
    training_refs = compute_training_references(training_data, strategy="training_mean")
    weights = RNG.standard_normal(28)
    scorer = lambda x: float(weights @ x)

    for i, (pred, true_lbl, decision) in enumerate(
        [(1, 1, 0.52), (0, 1, -0.18)]
    ):
        subject_vec = RNG.standard_normal(28)
        pert_results = run_perturbation_analysis(
            subject_vec, scorer, training_refs,
            subject_id=f"SYNTHETIC_CS_{i+1:02d}",
            node_names=["F3", "F4", "P3", "P4"],
        )

        plot_case_study(
            subject_features=subject_vec,
            predicted_label=pred,
            decision_score=decision,
            perturbation_results=pert_results,
            true_label=true_lbl,
            subject_id=f"SYNTHETIC_CS_{i+1:02d}",
            class_names=["TD", "ASD"],
            node_names=["F3", "F4", "P3", "P4"],
            is_synthetic=True,
            output_dir=CASE_STUDIES_DIR,
            filename_stem=f"synthetic_case_study_{i+1:02d}",
            formats=FORMATS,
        )
        print(f"  Saved: synthetic_case_study_{i+1:02d}.[png,pdf]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("NeuroQ-Connect — Illustrative Figure Generation")
    print("ALL DATA IS SYNTHETIC — FOR DEMONSTRATION ONLY")
    print("=" * 60)

    demo_eeg()
    demo_connectivity()
    demo_quantum()
    demo_kernels()
    demo_results()
    demo_perturbation()
    demo_case_studies()

    print("\n" + "=" * 60)
    print("DONE. Figures saved to:")
    print(f"  {FIGURES_DIR}")
    print(f"  {CASE_STUDIES_DIR}")
    print("=" * 60)
    print("\nReminder: All generated figures are illustrative examples.")
    print("Do not present them as actual research results.")


if __name__ == "__main__":
    main()
