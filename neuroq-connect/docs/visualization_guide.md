# NeuroQ-Connect Visualization Guide

This document explains how to use every visualization and explainability
function in the `asd_eeg` package.

> [!IMPORTANT]
> All data passed to visualization functions must be **real, already-loaded numerical arrays**.
> No function generates synthetic data internally when real data is missing.
> Synthetic data is only used in `examples/generate_test_figures.py` for demonstration.

---

## 1. Setup

```bash
pip install -e ".[dev]"
```

## 2. Load configuration

```python
from asd_eeg.visualization.common import load_config, apply_publication_style

config = load_config()          # reads configs/visualization.yaml
apply_publication_style(config) # apply globally before creating figures
```

---

## 3. EEG visualizations (`asd_eeg.visualization.eeg`)

```python
from asd_eeg.visualization import eeg as eeg_mod

# Full traces (all channels)
eeg_mod.plot_eeg_traces(
    eeg_data,                        # shape: (n_channels, n_samples)
    sfreq=256.0,
    channel_names=["F3", "F4", "P3", "P4"],
    title="Subject EEG",
    output_dir="results/figures",
    filename_stem="subject_01_eeg",
    formats=["png", "pdf"],
)

# Selected channels
eeg_mod.plot_selected_channels(
    eeg_data,
    channel_indices=[0, 2],
    sfreq=256.0,
    channel_names=channel_names,
    output_dir="results/figures",
    filename_stem="selected_channels",
)

# Time window
eeg_mod.plot_eeg_time_window(
    eeg_data,
    t_start=5.0,  # seconds
    t_end=10.0,
    sfreq=256.0,
    output_dir="results/figures",
    filename_stem="eeg_window",
)

# Power spectrum (requires scipy)
eeg_mod.plot_eeg_power_spectrum(
    eeg_data,
    sfreq=256.0,
    fmin=0.5,
    fmax=50.0,
    output_dir="results/figures",
    filename_stem="power_spectrum",
)
```

---

## 4. Connectivity visualizations (`asd_eeg.visualization.connectivity`)

```python
from asd_eeg.visualization import connectivity as conn_mod

# Edge weights: 1-D array of exactly 6 values in canonical EDGES order
# EDGES = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

edge_weights = connectivity_features[16:22]   # median connectivity features

conn_mod.plot_connectivity_graph(
    edge_weights,
    node_names=["F3", "F4", "P3", "P4"],
    output_dir="results/figures",
    filename_stem="connectivity_graph",
)

conn_mod.plot_connectivity_matrix(
    matrix_4x4,
    node_names=["F3", "F4", "P3", "P4"],
    output_dir="results/figures",
    filename_stem="connectivity_matrix",
)

conn_mod.plot_edge_weights_bar(
    edge_weights,
    node_names=["F3", "F4", "P3", "P4"],
    output_dir="results/figures",
    filename_stem="edge_weights",
)
```

---

## 5. Quantum circuit (`asd_eeg.visualization.quantum`)

```python
from asd_eeg.visualization import quantum as q_mod

# When Vivek's circuit is available:
from vivek_module import my_qnode   # PennyLane QNode

q_mod.visualize_circuit(
    circuit_fn=my_qnode,
    n_wires=4,
    sample_input=representative_feature_vector,
    output_dir="results/figures",
    filename_stem="quantum_circuit",
)

# Without the actual circuit (illustrative placeholder):
q_mod.visualize_circuit(
    circuit_fn=None,
    n_wires=4,
    output_dir="results/figures",
    filename_stem="quantum_circuit_placeholder",
)
```

> [!WARNING]
> The illustrative placeholder circuit is NOT the NeuroQ-Connect model.
> Never include it in the final research report without replacing it.

---

## 6. Kernel visualizations (`asd_eeg.visualization.kernels`)

```python
from asd_eeg.visualization import kernels as kern_mod

kern_mod.plot_kernel_matrix(K_train, is_training_kernel=True, output_dir="results/figures")
kern_mod.plot_kernel_matrix(K_test, is_training_kernel=False, output_dir="results/figures")

kern_mod.plot_kernel_comparison(
    {"RBF": K_rbf, "Quantum": K_qk},
    is_training_kernels=True,
    output_dir="results/figures",
)

kern_mod.plot_kernel_distribution(
    {"RBF": K_rbf, "Quantum": K_qk},
    output_dir="results/figures",
)

kern_mod.plot_train_test_kernel(K_train, K_test, output_dir="results/figures")
```

---

## 7. Model evaluation (`asd_eeg.visualization.results`)

```python
from asd_eeg.visualization import results as res_mod

# Model metrics dict (from Dodiya's evaluation output)
model_metrics = {
    "RBF-SVM":          {"accuracy": ..., "precision": ..., "recall": ..., "f1_score": ...},
    "Connectivity QK":  {"accuracy": ..., ...},
}

res_mod.plot_model_comparison(model_metrics, output_dir="results/figures")
res_mod.plot_metric_heatmap(model_metrics, output_dir="results/figures")

# Confusion matrix (integer labels)
res_mod.plot_confusion_matrix(y_true, y_pred, model_name="QK-SVM", output_dir="results/figures")

# ROC curve (SVM decision scores, NOT probabilities)
res_mod.plot_roc_curve(y_true, decision_scores, model_name="QK-SVM", output_dir="results/figures")
res_mod.plot_roc_comparison({"RBF": (y_true, scores_rbf), "QK": (y_true, scores_qk)})

# Fold-wise performance
fold_metrics = [{"accuracy": 0.75}, {"accuracy": 0.80}, ...]   # one dict per fold
res_mod.plot_fold_performance(fold_metrics, metric_key="accuracy", model_name="QK-SVM")
```

> [!CAUTION]
> SVM decision scores are NOT probabilities. Never label them as "ASD probability".
> The figure titles and axis labels in `results.py` enforce this distinction.

---

## 8. Perturbation analysis (`asd_eeg.explainability.perturbation`)

```python
from asd_eeg.explainability.perturbation import (
    compute_training_references,
    run_perturbation_analysis,
    save_perturbation_results,
    plot_perturbation_sensitivity,
    plot_case_study,
)

# Step 1: Compute references from TRAINING data only
training_refs = compute_training_references(X_train, strategy="training_mean")

# Step 2: Scoring function (never retrain — use saved model)
scorer = lambda x: svm_model.decision_function(x.reshape(1, -1))[0]

# Step 3: Run analysis for one held-out subject
results = run_perturbation_analysis(
    subject_features=x_held_out,     # 28-dim vector
    scoring_fn=scorer,
    training_references=training_refs,
    subject_id="anonymized_S01",
    node_names=["F3", "F4", "P3", "P4"],
)

# Step 4: Visualize
plot_perturbation_sensitivity(results, output_dir="results/figures")

# Step 5: Save structured results
save_perturbation_results(results, output_dir="results/figures", formats=("csv", "json"))
```

> [!NOTE]
> This analysis shows **model sensitivity** to feature substitutions only.
> It does NOT prove causality or biological significance.

---

## 9. Case-study figures

```python
from asd_eeg.explainability.perturbation import plot_case_study

plot_case_study(
    subject_features=x_held_out,
    predicted_label=pred,
    decision_score=score,
    perturbation_results=pert_results,
    true_label=y_true_held_out,
    subject_id="anonymized_S01",
    class_names=["TD", "ASD"],
    node_names=["F3", "F4", "P3", "P4"],
    is_synthetic=False,              # True only for illustrative examples
    output_dir="results/case_studies",
    filename_stem="case_study_S01",
)
```
