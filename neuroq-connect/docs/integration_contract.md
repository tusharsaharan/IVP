# NeuroQ-Connect — Integration Contract

**Module:** Visualization and Explainability (this module)  
**Author:** Visualization & Explainability Lead  
**Last updated:** 2026-11

This document defines the exact input formats, data structures, and function interfaces required between all team members and the `asd_eeg` visualization and explainability module.

---

## Sahil Gupta — Dataset and Preprocessing

### Expected inputs to `asd_eeg.visualization.eeg`

| Parameter | Type | Shape / Format | Notes |
|---|---|---|---|
| `eeg_data` | `np.ndarray` | `(n_channels, n_samples)` | Float array in µV |
| `sfreq` | `float` | Scalar | Sampling frequency in Hz (e.g., 256.0) |
| `channel_names` | `Sequence[str]` | Length `n_channels` | e.g. `["F3", "F4", "P3", "P4"]` |
| `subject_id` | `str` | Anonymized identifier | e.g. `sub-001` (no clinical PII) |
| `y_labels` | `np.ndarray` | `(n_subjects,)` int | 0 = TD (Typically Developing), 1 = ASD |

### Module Functions:
```python
from asd_eeg.visualization.eeg import (
    plot_eeg_traces,
    plot_selected_channels,
    plot_eeg_time_window,
    plot_eeg_power_spectrum,
)
```

### Usage Example:
```python
plot_eeg_traces(
    eeg_data=X_eeg,              # (4, n_samples) float array
    sfreq=256.0,
    channel_names=["F3", "F4", "P3", "P4"],
    output_dir="results/figures",
    filename_stem=f"subject_{subject_id}_traces",
)
```

---

## Aryan Pandey — Connectivity and Feature Extraction

### Feature vector layout (28 dimensions, 0-indexed)

The canonical 28-dimensional subject feature vector must follow this exact order:

| Index range | Content | Dimension |
|---|---|---|
| `0–15` | Band-power features (4 frequency bands × 4 nodes) | 16 |
| `16–21` | Median alpha connectivity for 6 connections (EDGES order) | 6 |
| `22–27` | IQR / variability for the same 6 connections (EDGES order) | 6 |

### Canonical EDGES order (strictly enforced across project)

```python
EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3)
]
```

### Connectivity features expected by `asd_eeg.visualization.connectivity`

```python
# 1. 1-D edge weights: values at indices 16:22 of the feature vector
edge_weights = feature_vector[16:22]   # shape: (6,)

# 2. Or a 4×4 pairwise connectivity matrix
connectivity_matrix: np.ndarray        # shape: (4, 4)
```

### Usage Example:
```python
from asd_eeg.visualization.connectivity import (
    plot_connectivity_graph,
    plot_connectivity_matrix,
    plot_edge_weights_bar,
)

plot_connectivity_graph(
    edge_weights=feature_vector[16:22],
    node_names=["Node 0", "Node 1", "Node 2", "Node 3"],
    output_dir="results/figures",
    filename_stem="connectivity_graph",
)
```

---

## Vivek G — Quantum Implementation

### Actual Circuit Adapter Interface (`asd_eeg.visualization.quantum`)

The visualization module uses an adapter pattern that directly visualizes Vivek's PennyLane QNode or circuit function using `qml.draw_mpl`:

```python
import pennylane as qml
from asd_eeg.visualization.quantum import visualize_circuit

# Vivek's QNode definition:
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def my_circuit(x):
    # e.g., AngleEmbedding, Entanglement layers, etc.
    qml.AngleEmbedding(x[:4], wires=range(4))
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    return qml.expval(qml.PauliZ(0))

# Pass Vivek's circuit directly to the visualization module:
visualize_circuit(
    circuit_fn=my_circuit,
    n_wires=4,
    sample_input=sample_feature_vector,  # 1-D array passed to trace the circuit
    output_dir="results/figures",
    filename_stem="quantum_circuit",
    formats=["png", "pdf"],
)
```

> [!NOTE]
> If `circuit_fn` is `None` or PennyLane is not installed, an explicitly labelled
> **illustrative placeholder circuit** is generated for testing purposes only.

### Kernel Matrices expected by `asd_eeg.visualization.kernels`

- Training kernel: `K_train` of shape `(n_train, n_train)` (symmetric, positive semi-definite).
- Test kernel: `K_test` of shape `(n_test, n_train)` (rectangular).

---

## Dodiya Mahavirsinh — Model Training and Evaluation

### Model metrics format

Dodiya's evaluation script should export or provide a dictionary matching this structure.

> [!NOTE]
> **Illustrative Placeholders:** The numerical values shown below (accuracy, precision, recall, F1-score) are strictly illustrative placeholder examples to define dictionary schema and data types. They do **not** represent actual experimental findings, benchmark outcomes, or validated classification performance.

```python
# Illustrative schema placeholder — replace with actual evaluation metrics:
model_metrics: dict[str, dict[str, float]] = {
    "RBF-SVM": {
        "accuracy": 0.725,   # Illustrative placeholder (not actual result)
        "precision": 0.710,  # Illustrative placeholder
        "recall": 0.740,     # Illustrative placeholder
        "f1_score": 0.725,   # Illustrative placeholder
    },
    "Amplitude-Fidelity QK-SVM": {
        "accuracy": 0.760,   # Illustrative placeholder
        "precision": 0.750,  # Illustrative placeholder
        "recall": 0.780,     # Illustrative placeholder
        "f1_score": 0.765,   # Illustrative placeholder
    },
    "Connectivity-Guided QK-SVM": {
        "accuracy": 0.820,   # Illustrative placeholder
        "precision": 0.810,  # Illustrative placeholder
        "recall": 0.835,     # Illustrative placeholder
        "f1_score": 0.822,   # Illustrative placeholder
    },
    "Reassigned-Connection Control": {
        "accuracy": 0.690,   # Illustrative placeholder
        "precision": 0.680,  # Illustrative placeholder
        "recall": 0.700,     # Illustrative placeholder
        "f1_score": 0.690,   # Illustrative placeholder
    },
}
```

### Predictions and Decision Scores

```python
y_true:          np.ndarray   # shape (n_subjects,), int {0, 1}
y_pred:          np.ndarray   # shape (n_subjects,), int {0, 1}
decision_scores: np.ndarray   # shape (n_subjects,), float (from model.decision_function)
```

> [!CAUTION]
> SVM decision scores are **continuous unbounded margins, NOT probabilities**.
> Never provide `predict_proba` values or label SVM decision scores as "ASD probabilities".

### Fold-specific training reference values for Explainability

For perturbation sensitivity analysis, reference values must be calculated **exclusively from each fold's training split**:

```python
from asd_eeg.explainability.perturbation import compute_training_references

# Compute references strictly on training split (no test leakage!)
training_refs = compute_training_references(X_train_fold, strategy="training_mean")
```

### Model Scoring Callable for Perturbation Analysis

```python
# A callable accepting a single (28,) feature vector and returning a scalar float
scorer = lambda x: float(saved_svm_model.decision_function(x.reshape(1, -1))[0])
```

---

## Tushar Saharan — Integration and Application

### Calling Visualization and Explainability in the Application Pipeline

```python
from asd_eeg.visualization.common import load_config, apply_publication_style
from asd_eeg.visualization import eeg, connectivity, kernels, results
from asd_eeg.explainability.perturbation import (
    compute_training_references,
    run_perturbation_analysis,
    save_perturbation_results,
    plot_perturbation_sensitivity,
    plot_case_study,
)

# 1. Initialize configuration and styling
config = load_config("configs/visualization.yaml")
apply_publication_style(config)

# 2. Run sensitivity analysis on a held-out subject
training_refs = compute_training_references(X_train, strategy="training_mean")
pert_results = run_perturbation_analysis(
    subject_features=subject_x,
    scoring_fn=lambda x: model.decision_function(x.reshape(1, -1))[0],
    training_references=training_refs,
    subject_id="sub-102",
    node_names=["F3", "F4", "P3", "P4"],
)

# 3. Export structured results (CSV & JSON) and figures
save_perturbation_results(pert_results, output_dir="results/figures", filename_stem="sub-102_pert")
plot_perturbation_sensitivity(pert_results, output_dir="results/figures", filename_stem="sub-102_pert_plot")
plot_case_study(
    subject_features=subject_x,
    predicted_label=int(model.predict(subject_x.reshape(1, -1))[0]),
    decision_score=float(model.decision_function(subject_x.reshape(1, -1))[0]),
    perturbation_results=pert_results,
    true_label=int(y_test_subject),
    subject_id="sub-102",
    output_dir="results/case_studies",
    filename_stem="sub-102_case_study",
)
```

---

## Pranav Karhale — Report and Presentation

### Generated Asset Directories

| Type of Asset | Path | Description |
|---|---|---|
| Publication Figures | `results/figures/` | Clean 300 DPI figures for report / slides |
| Case Studies | `results/case_studies/` | Held-out subject case-study panels |

### File Formats
- **PNG** (300 DPI): Ideal for slides, Word documents, and quick previews.
- **PDF**: Vector graphics ideal for LaTeX reports.
- **SVG**: Scalable vector graphics for editing in Illustrator/Inkscape.

### Distinguishing Real Findings from Synthetic Demos
- Any file prefixed with `example_` or `synthetic_` is generated from random data for demonstration and unit testing.
- Only files generated from real experimental evaluation pipelines (without the `synthetic_` prefix) should be cited in final scientific manuscripts.
