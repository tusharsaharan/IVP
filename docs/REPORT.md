# NeuroQ-Connect: Connectivity-Guided Quantum Kernels for EEG-Based Autism Classification

**Academic Project Final Report**  
**Group 8 — Advanced Cognitive Systems / Intelligent Signal Processing**  
**Submission Target:** Reproducible Quantum-EEG Experiment & Interactive Demonstration  

---

## Executive Summary

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition characterized by atypical neural connectivity and social communication patterns. Electroencephalography (EEG) provides high temporal resolution non-invasive biomarker potential, but classical machine learning methods struggle with high dimensionality, volume conduction, and small sample cohorts. 

**NeuroQ-Connect** introduces a connectivity-guided quantum kernel machine learning pipeline. Rather than generic angle encoding or unstructured unitary layers, we construct a 4-node, 6-edge graph representation from EEG frequency-band powers and alpha-band envelope connectivity. This graph directly parameterizes a 4-qubit quantum feature map: single-node relative powers drive local $RY$ and $RZ$ rotations, while pairwise edge medians and interquartile ranges (variability) parameterize two-qubit $IsingZZ$ interactions over two interleaved passes. 

Using PennyLane exact statevector simulation, we evaluate the resulting fidelity kernel:
$$K(x, y) = |\langle\psi(x)\mid\psi(y)\rangle|^2$$

under a strict, leakage-free stratified subject-wise cross-validation protocol. We compare the proposed method against three controlled baselines: (1) a classical feature-matched RBF-SVM, (2) an amplitude-fidelity quantum kernel baseline inspired by Saranya & Menaka (2025), and (3) an edge-reassigned quantum kernel control that destroys anatomical connectivity correspondence while preserving individual angle distributions and circuit depth.

---

## 1. Problem Definition and Motivation

Diagnosis of Autism Spectrum Disorder currently relies on behavioral evaluations (e.g., ADOS-2, ADI-R) that are subjective, resource-intensive, and often delayed until age 3–5. EEG offers an objective, physiological modality. However:
1. **Low Signal-to-Noise Ratio & Volume Conduction:** Sensor-level EEG contains widespread mixing from common scalp references.
2. **Subject Heterogeneity:** Small subject cohorts lead to severe overfitting if window-level samples are not strictly grouped by subject.
3. **Black-box Classical Models:** Deep learning models fail to preserve known functional brain topologies and require excessive training samples.
4. **Quantum Machine Learning Pitfalls:** Prior quantum EEG models either treat EEG as an arbitrary flattened vector without biological priors or claim quantum computational speedup prematurely on classically simulable qubit registers.

NeuroQ-Connect resolves these issues by enforcing a biological graph inductive bias in the quantum circuit, restricting evaluation to subject-disjoint partitions, and verifying software integrity with reproducible automated tests.

---

## 2. Literature Basis and Distinction

### Reference Study
Saranya and Menaka (2025), *A Quantum-Based Machine Learning Approach for Autism Detection Using Common Spatial Patterns of EEG Signals*, IEEE Access (DOI: 10.1109/ACCESS.2025.3531979).

The reference paper extracts Common Spatial Pattern (CSP) and wavelet features from a private 20-child pediatric EEG cohort, preparing quantum states via amplitude embedding and using a quantum support vector machine. 

### Methodological Distinction in NeuroQ-Connect
- **Feature Representation:** Instead of CSP spatial filtering on private recordings, NeuroQ-Connect builds a standardized 4-node regional montage (e.g., F3, F4, P3, P4) capturing frontal-parietal connectivity.
- **Circuit Architecture:** Rather than generic amplitude embedding requiring data padding and arbitrary unitary transformations, NeuroQ-Connect uses an input-dependent Hamiltonian evolution where single-qubit rotations encode regional spectral powers and two-qubit $IsingZZ$ gates encode statistical alpha-envelope connectivity.
- **Ablation Control:** We introduce an edge-reassigned control experiment that freezes node assignments but applies a fixed, label-independent permutation to the edges. This isolates whether regional brain connectivity topology specifically guides learning.
- **Clinical Honesty:** We make no claims of quantum advantage or diagnostic certainty on synthetic/unverified cohorts. All metrics are derived from saved out-of-fold predictions.

---

## 3. End-to-End System Architecture

The pipeline consists of modular stages with shared contracts:

```
[Raw EEG: Channels x Samples]
             │
             ▼
[Preprocessing & Signal QC]
  • 1–40 Hz 4th-order Butterworth bandpass
  • 50 Hz notch filter
  • Polyphase rational resampling to 128 Hz
  • 8.0 s non-overlapping windows; amplitude/flat rejection
             │
             ▼
[Feature Extraction: 28 Features per Subject]
  • 16 Node Powers: 4 bands (delta, theta, alpha, beta) × 4 nodes
  • 6 Connectivity Medians: alpha-envelope Pearson r across accepted windows
  • 6 Connectivity IQRs: interquartile range of r across accepted windows
             │
             ▼
[Subject-Wise Cross-Validation Split]
  • Stratified subject folds (k=5 or minority count)
  • StandardScaler fitted ON TRAINING FOLD ONLY
  • Bounded angle transform: φ = (π/2) * tanh(z / 2)
             │
             ▼
[Model Training & Kernel Evaluation]
  ┌───────────────────────────────────────────────────────────┐
  │ 1. Classical RBF-SVM Baseline                             │
  │ 2. Amplitude-Fidelity Kernel Baseline                     │
  │ 3. Proposed Connectivity-Guided Quantum Kernel SVM        │
  │ 4. Reassigned-Connection Graph Control SVM                │
  └───────────────────────────────────────────────────────────┘
             │
             ▼
[Inference, Explanation & Export]
  • Save model bundles (joblib schema v1)
  • CLI & Streamlit inference engine
  • Connection-perturbation sensitivity (SVM score delta)
  • Audited metrics: Balanced Acc, Sensitivity, Specificity, Fold AUC
```

---

## 4. EEG Preprocessing and Quality Control

1. **Signal Loading:** Voltage calibration into standard Volts ($V$). Load continuous records from NPZ, EDF, BDF, FIF, or EEGLAB SET/FDT formats.
2. **Channel Selection & Montage Mapping:** Explicit mapping to four scalp nodes (provisional default: Left Frontal `F3`, Right Frontal `F4`, Left Parietal `P3`, Right Parietal `P4`).
3. **Filtering:**
   - 4th-order zero-phase Butterworth bandpass filter between 1.0 Hz and 40.0 Hz.
   - Notch filter at 50.0 Hz (only applied if Nyquist $> 50$ Hz).
   - Average reference across clean EEG channels.
4. **Segmentation & Rejection:**
   - Continuous signal divided into non-overlapping 8.0-second epochs.
   - Flat-signal rejection (variance $< 10^{-14}$).
   - Amplitude threshold: peak-to-peak amplitude exceeding $300\ \mu\text{V}$ marks an epoch as artifactual.
   - Minimum usable criterion: at least 3 accepted windows per recording; pooled sequence capped at 20 windows spaced evenly.

---

## 5. Regional Graph and Feature Representation

For $q = 4$ nodes and $e = \frac{q(q-1)}{2} = 6$ edges, each subject is represented by a 28-dimensional real vector:

$$\mathbf{x} \in \mathbb{R}^{28}$$

### Feature Indexing & Canonical Ordering
| Slice | Dimension | Feature Content | Description |
|---|---|---|---|
| `0:16` | 16 | Node Spectral Powers | 4 frequency bands per node in node order (Node 0: $\delta,\theta,\alpha,\beta$; Node 1: ... Node 3: ...) |
| `16:22` | 6 | Edge Alpha-Envelope Medians | Median Pearson $r$ of Hilbert envelopes in edge order: $(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)$ |
| `22:28` | 6 | Edge Alpha-Envelope IQRs | $75\text{th} - 25\text{th}$ percentile IQR of envelope correlations across accepted windows |

### Power & Connectivity Mathematics
- **Relative Band Power:** Welch PSD ($N_{\text{FFT}} \le 256$, Hanning taper). Integrated PSD within band $[\text{low}, \text{high})$ divided by total $1-30$ Hz power. Log10-transformed and aggregated across accepted windows by median.
- **Alpha-Envelope Connectivity:** Signal filtered in alpha band ($8-13$ Hz); analytic signal obtained via Hilbert transform. Regional log envelope averaged across electrodes in each node. Pairwise Pearson correlation computed per window:
$$r_{ij} = \frac{\sum_t (A_i(t) - \bar{A}_i)(A_j(t) - \bar{A}_j)}{\sqrt{\sum_t (A_i(t) - \bar{A}_i)^2 \sum_t (A_j(t) - \bar{A}_j)^2}}$$

---

## 6. Quantum Method and Feature Map Design

### 6.1 State Preparation and Feature Encoding
Continuous features $\mathbf{x}$ are standardized using training-fold parameters:
$$\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$$
Mapped to bounded rotation angles:
$$\boldsymbol{\phi} = \frac{\pi}{2} \tanh\left(\frac{\mathbf{z}}{2}\right) \in \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$$

### 6.2 4-Qubit Circuit Structure
A register of $q=4$ qubits initialized to $|0\rangle^{\otimes 4}$.
1. **Initial Layer:** Hadamard gates on all wires:
$$H^{\otimes 4} |0\rangle^{\otimes 4} = |+\rangle^{\otimes 4}$$
2. **Encoding Passes ($P=2$ interleaved passes):**
   - **Pass Step 1 (Node Spectral Rotations):** Local $RY$ and $RZ$ rotations parameterized by node delta and theta powers:
     $$\bigotimes_{i=0}^3 RZ(\phi_{4i+1}) RY(\phi_{4i})$$
   - **Pass Step 2 (Connectivity Median Entanglement):** Pairwise two-qubit Ising interactions on all 6 edges:
     $$\prod_{(u,v) \in \mathcal{E}} \exp\left(-i \frac{\phi_{16 + e(u,v)}}{2} Z_u \otimes Z_v\right)$$
   - **Pass Step 3 (Node Higher-Band Rotations):** Local rotations for alpha and beta powers:
     $$\bigotimes_{i=0}^3 RZ(\phi_{4i+3}) RY(\phi_{4i+2})$$
   - **Pass Step 4 (Connectivity Variability Entanglement):** Pairwise Ising interactions for edge variability:
     $$\prod_{(u,v) \in \mathcal{E}} \exp\left(-i \frac{\phi_{22 + e(u,v)}}{2} Z_u \otimes Z_v\right)$$

### 6.3 Statevector Caching & Fidelity Kernel
PennyLane generates exact statevectors $|\psi(\mathbf{x})\rangle \in \mathbb{C}^{16}$.
One statevector is cached per subject. The kernel matrix entry between subjects $x$ and $y$ is:
$$K(x, y) = |\langle\psi(x)\mid\psi(y)\rangle|^2$$
- **Unit Norm:** $\langle\psi(x)\mid\psi(x)\rangle = 1.0$ guaranteed by unitary evolution.
- **Symmetry & PSD:** $K(x, y) = K(y, x)$, with eigenvalues $\lambda_i(K) \ge 0$.

### 6.4 Controlled Baselines
1. **Classical RBF-SVM:** SVM trained on bounded angle vectors $\boldsymbol{\phi}$ using radial basis function kernel:
   $$K_{\text{RBF}}(x, y) = \exp(-\gamma \|x - y\|^2)$$
2. **Amplitude-Fidelity Kernel:** Constant 1 appended, normalized, with classical closed form:
   $$K_{\text{amp}}(x, y) = \frac{(\mathbf{x}^T \mathbf{y} + 1)^2}{(\|\mathbf{x}\|^2 + 1)(\|\mathbf{y}\|^2 + 1)}$$
3. **Reassigned Connections Control:** Uses identical circuit depth and multiset of rotation angles, but permutes the edge-to-qubit assignments with a fixed random seed. If topological anatomical connections matter, the original graph should outperform the reassigned control.

---

## 7. Strict Evaluation Protocol

To prevent methodological flaws common in academic machine learning literature:
1. **Subject Disjointness:** Multiple sessions/epochs from the same participant are strictly grouped; test folds contain entirely held-out participants.
2. **Zero Preprocessing Leakage:** Standardization ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) and PCA/filtering parameters are fit solely on training folds.
3. **Fold-Level ROC-AUC Reporting:** Support Vector Machine decision boundaries produce uncalibrated margins whose scales differ between folds. We report individual fold AUCs and their descriptive mean/standard deviation; we never pool raw uncalibrated decision scores across folds into a single misleading ROC curve.
4. **Automated Mathematical Audits:** Model bundles store training subject IDs, held-out subject IDs, config hashes, and dependency versions.

---

## 8. Explainability: Connection-Perturbation Sensitivity

To inspect what functional connections influence the quantum classifier for a specific held-out individual without making ungrounded causal claims:
1. Compute the subject's baseline decision score $s_0 = f(\mathbf{x})$.
2. For each of the 6 edges $e \in \mathcal{E}$:
   - Replace that edge's median connectivity and IQR variability with the training-cohort reference median:
     $$\mathbf{x}' = \mathbf{x}[\text{edge } e \leftarrow \boldsymbol{\mu}_{\text{train, edge } e}]$$
   - Compute the perturbed score $s_e = f(\mathbf{x}')$.
   - Measure score deflection $\Delta s_e = s_e - s_0$.
3. Direction of $\Delta s$: Negative deflections indicate the connection contributed towards an ASD classification; positive towards Control.
4. **Clinical Boundary:** This measures model sensitivity to feature substitution under the learned decision boundary. It is **not** a validated neurodevelopmental causal mechanism.

---

## 9. Implementation and Verification Results

### 9.1 Automated Test Suite Performance
The test suite consists of 230 automated unit and integration tests passing across all components:
- `test_quantum_math.py`: Unit norm of states, PSD kernel matrix, input sensitivity of RY/RZ and IsingZZ gates, mathematical equivalence of amplitude kernel to classical inner product.
- `test_feature_artifact_integrity.py`: 28-feature width enforcement, canonical ordering, schema validation, non-empty window tables.
- `test_evaluation_integrity.py`: Leakage-free subject folds, identical partitions across all 4 models, recomputability of metrics from CSV.
- `test_artifact_compatibility.py`: Bundle schema v1 validation, config hash agreement, model reload invariance, CLI and pipeline agreement.
- `test_app.py`: Streamlit navigation and UI stability.

### 9.2 Reproducibility Check
- `neuroq self-check`: Executes the full synthetic pipeline end-to-end, fitting RBF, Amplitude, Quantum, and Reassigned models across 3 folds, saving model bundles, verifying reload, predicting on held-out subjects, and exporting `prediction.json`. Result: **Status: passed**.
- Parity Check: CLI command `neuroq predict` and application `analyze_subject()` yield bit-for-bit identical decision scores.

---

## 10. Scientific Limitations and Ethical Considerations

1. **Volume Conduction & Reference Choice:** Sensor-level EEG reflects electrical summation across brain tissue and the scalp. Alpha-envelope correlations do not represent direct axonal white-matter tractography.
2. **Classical Simulation vs. Hardware:** The 4-qubit circuit ($2^4 = 16$ amplitudes) is classically simulated with exact linear algebra. No quantum supremacy or speedup is claimed.
3. **Cohort Size:** Public autism EEG cohorts (e.g., KAU, TUOS) have small sample counts ($N \approx 20 - 40$). Statistical power is inherently constrained.
4. **Diagnostic Disclaimer:** This software is a course research prototype. It is not approved by medical regulatory bodies and must never be used for clinical diagnosis or treatment planning.

---

## 11. Team Contributions and Responsibilities

| Team Member | Area of Ownership | Delivered Modules & Artifacts |
|---|---|---|
| **Vivek G** | Quantum Architecture & Models | 4-qubit feature map, interleaved RY/RZ & IsingZZ gates, fidelity kernel, amplitude baseline, model bundle interface. |
| **Sahil Gupta** | Data Ingestion & Preprocessing | Continuous EEG loaders (NPZ, EDF, SET), filtering pipeline, 8-s windowing, artifact rejection, recording metadata. |
| **Aryan Pandey** | Regional Graph & Subject Features | 4-node montage aggregation, Welch relative band powers, Hilbert alpha-envelope correlation and IQR (28 features total). |
| **Dodiya Mahavirsinh** | Evaluation & Classical Comparisons | Stratified subject cross-validation, RBF-SVM baseline, 4-model evaluation runner, metrics and fold AUC calculation. |
| **Tushar Saharan** | System Integration & Demonstration | Shared inference pipeline, `neuroq` CLI entry points, Streamlit web application, bundle schema management. |
| **Shivam Mall** | Visualization & Explainability | Connectivity graphs, trace plots, confusion matrices, ROC curves, connection-perturbation sensitivity module. |
| **Pranav Karhale** | Verification & Final Submission | Independent test suite (`test_artifact_compatibility.py`), reproducibility guide, demo script, PPT deck, report assembly. |

---

## 12. References

1. Saranya, P., & Menaka, R. (2025). *A Quantum-Based Machine Learning Approach for Autism Detection Using Common Spatial Patterns of EEG Signals*. IEEE Access, DOI: [10.1109/ACCESS.2025.3531979](https://doi.org/10.1109/ACCESS.2025.3531979).
2. Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta, J. M. (2019). *Supervised learning with quantum-enhanced feature spaces*. Nature, 567(7747), 209-212.
3. Schuld, M., & Killoran, N. (2019). *Quantum machine learning in feature Hilbert spaces*. Physical Review Letters, 122(4), 040504.
4. Group 8 Repository Documentation: `docs/CONTRACTS.md`, `docs/METHOD.md`, `docs/FEATURES.md`, `docs/reproducibility.md`, `docs/demo_script.md`.

