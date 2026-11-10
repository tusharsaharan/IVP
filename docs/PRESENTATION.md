# NeuroQ-Connect: Presentation Slide Deck Outline & Script

**Project Title:** NeuroQ-Connect: Connectivity-Guided Quantum Kernels for EEG-Based Autism Classification  
**Target:** 15-Slide Academic Defense Presentation  
**Design Register:** Academic, clear typography, high contrast, no emojis or decorative gimmicks.  

---

### Slide 1: Title & Overview
- **Title:** NeuroQ-Connect: Connectivity-Guided Quantum Kernels for EEG-Based Autism Classification
- **Subtitle:** An Interpretable, Inductively Biased Quantum Kernel Framework for Subject-Wise Biomarker Discovery
- **Presenters:** Group 8 (Vivek G, Sahil Gupta, Dodiya Mahavirsinh, Aryan Pandey, Tushar Saharan, Shivam Mall, Pranav Karhale)
- **Course / Context:** Intelligent Signal Processing / Advanced Cognitive Systems
- **Speaker Notes:** "Good morning. Today we present NeuroQ-Connect, our reproducible quantum-classical machine learning framework that uses functional EEG connectivity graphs to parameterize quantum state preparation for autism spectrum disorder classification."

---

### Slide 2: Problem Definition and Clinical Motivation
- **Clinical Need:** Autism Spectrum Disorder (ASD) affects 1 in 36 children. Current diagnosis relies on observational behavioral evaluations (ADOS-2) which are subjective, labor-intensive, and often delayed until age 3–5.
- **Why EEG?** Electroencephalography offers millisecond temporal resolution, low cost, and non-invasive neural monitoring.
- **The Core ML Challenges:**
  - High dimensionality with severe volume conduction noise.
  - Small subject cohorts causing catastrophic data leakage if epochs are not grouped by subject.
  - Standard deep learning models treat EEG channels as arbitrary arrays without anatomical topology.
- **Speaker Notes:** "Current diagnostic pathways are delayed. While EEG provides non-invasive neural dynamics, small sample sizes and spatial volume conduction cause classical deep learning models to overfit or produce non-reproducible claims."

---

### Slide 3: Reference Paper & Methodological Gap
- **Reference Paper:** Saranya & Menaka (2025), *A Quantum-Based Machine Learning Approach for Autism Detection Using Common Spatial Patterns of EEG Signals*, IEEE Access.
- **Reference Approach:** Filter-bank CSP and wavelet packet decomposition on a private 20-child dataset, followed by amplitude-encoded Quantum Support Vector Machine (QSVM).
- **Our Critical Assessment:**
  - Generic amplitude encoding treats spatial channels as flattened vector entries with arbitrary padding.
  - Private dataset accuracy cannot be reproduced independently.
  - Lack of topological graph inductive bias in quantum circuit construction.
- **Speaker Notes:** "Saranya and Menaka demonstrated the feasibility of QSVM for EEG. However, standard amplitude encoding ignores the functional network architecture of the brain. We address this by embedding regional EEG connectivity graphs directly into quantum gates."

---

### Slide 4: Research Question & Inductive Bias Hypothesis
- **Primary Research Question:**
  *Does structuring a quantum feature map around empirical brain connectivity graphs improve classification performance and interpretability compared to classical RBF-SVM and generic quantum baselines?*
- **Hypothesis:**
  - Mapping regional frequency power to local rotations and pairwise connectivity to entangling gates ($IsingZZ$) embeds biologically grounded inductive bias into the Hilbert space geometry.
- **Ablation Control:**
  - Permuting edge connections while freezing node identities tests whether the biological graph topology itself drives classification performance.
- **Speaker Notes:** "Our core hypothesis is that inductive bias matters in quantum machine learning. If we align qubit entanglement with cortical functional connectivity, the resulting fidelity kernel reflects meaningful neurodynamics."

---

### Slide 5: Dataset & Subject Cohort Criteria
- **Dataset Focus:** Public pediatric autism EEG cohorts (KAU and TUOS).
- **Verification Requirements:**
  - Authoritative clinical labels verified from companion medical records, never inferred from filenames.
  - Explicit subject identification across repeated recordings to prevent session leakage.
  - Verified 10–20 international electrode montage locations.
- **Cohort Stratification:**
  - Strict binary classes: $0 = \text{Control}$, $1 = \text{ASD}$.
  - Minimum class support enforced for stratified subject-wise folds.
- **Speaker Notes:** "Data integrity is our first priority. We strictly prohibit inferring labels from filenames or mixing synthetic verification data with real experimental findings."

---

### Slide 6: Conservative Preprocessing & Signal QC
- **Signal Normalization:** Voltage calibration into standard Volts ($V$).
- **Spectral Filtering:**
  - 4th-order zero-phase Butterworth bandpass (1.0 Hz – 40.0 Hz).
  - 50.0 Hz notch filter to eliminate powerline interference.
  - Common average reference across clean scalp channels.
- **Epoching & Rejection:**
  - Non-overlapping 8.0-second time windows.
  - Flatline rejection ($\text{variance} < 10^{-14}$).
  - Artifact threshold: peak-to-peak $> 300\ \mu\text{V}$ discarded.
  - Minimum criteria: at least 3 clean windows per participant; pooled window sequences capped at 20 windows.
- **Speaker Notes:** "Our preprocessing pipeline applies conservative, documented signal quality criteria. Windows contaminated by excessive muscle or motion artifacts are systematically excluded."

---

### Slide 7: Regional EEG Graph Features (The 28-D Contract)
- **4-Node Montages:** Left Frontal (F3), Right Frontal (F4), Left Parietal (P3), Right Parietal (P4).
- **Feature Structure (28 features total):**
  - **16 Node Powers:** $\delta$ (1–4 Hz), $\theta$ (4–8 Hz), $\alpha$ (8–13 Hz), $\beta$ (13–30 Hz) relative Welch powers per node.
  - **6 Alpha-Envelope Medians:** Hilbert analytic signal alpha envelopes; Pearson correlation $r$ across accepted windows for 6 node pairs: $(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)$.
  - **6 Alpha-Envelope IQRs:** Interquartile range of window-level correlations capturing connectivity variability.
- **Deterministic Order:** Enforced by automated schema validation; strictly label-free.
- **Speaker Notes:** "Aryan Pandey's pipeline produces exactly 28 features per subject: 16 regional band powers, 6 median alpha-envelope correlations, and 6 connectivity variability measures. This is locked into our project contract."

---

### Slide 8: The Connectivity-Guided 4-Qubit Circuit
- **Register:** 4 qubits ($q_0, q_1, q_2, q_3$), initialized in $|+\rangle^{\otimes 4}$ via initial Hadamard gates.
- **Interleaved Two-Pass Architecture:**
  - **Step 1 (Spectral Encoding):** Local $RY$ and $RZ$ rotations parameterized by delta and theta powers:
    $$RY(\phi_{4i}) \quad \text{and} \quad RZ(\phi_{4i+1})$$
  - **Step 2 (Connectivity Entanglement):** Pairwise two-qubit $IsingZZ$ interactions on all 6 edges:
    $$\exp\left(-i \frac{\phi_{16+e}}{2} Z_u \otimes Z_v\right)$$
  - **Step 3 (Higher Bands):** Local $RY$ and $RZ$ rotations for alpha and beta powers:
    $$RY(\phi_{4i+2}) \quad \text{and} \quad RZ(\phi_{4i+3})$$
  - **Step 4 (Variability Entanglement):** Pairwise $IsingZZ$ interactions parameterized by connectivity IQRs:
    $$\exp\left(-i \frac{\phi_{22+e}}{2} Z_u \otimes Z_v\right)$$
- **Speaker Notes:** "Vivek G designed this circuit in PennyLane. Qubits represent regional nodes. Node spectral powers drive single-qubit rotations, while edge medians and IQRs drive entangling IsingZZ gates over two interleaved passes."

---

### Slide 9: Quantum Kernel Classification
- **Statevector Representation:** Exact simulation produces unit statevectors:
  $$|\psi(\mathbf{x})\rangle \in \mathbb{C}^{16}, \quad \|\psi(\mathbf{x})\| = 1.0$$
- **Fidelity Kernel Equation:**
  $$K(x, y) = |\langle\psi(x)\mid\psi(y)\rangle|^2$$
- **Mathematical Guarantees:**
  - Positive Semi-Definite (PSD): All eigenvalues $\lambda_i \ge 0$.
  - Symmetric with unit diagonal: $K(x, x) = 1.0$.
- **Classifier:** Soft-margin Support Vector Machine ($C=1.0$, balanced class weights) trained on precomputed kernel matrices.
- **Speaker Notes:** "The fidelity kernel computes the transition probability between two statevectors. It is guaranteed to be positive semi-definite, forming a valid Hilbert space inner product for our SVM classifier."

---

### Slide 10: The Four Controlled Experiments
- **Experiment 1 (Classical Baseline):**
  - RBF-SVM on identically bounded feature vectors: $\phi = (\pi/2) \tanh(z/2)$.
- **Experiment 2 (Literature-Inspired Baseline):**
  - Amplitude-Fidelity Kernel SVM: $K(x,y) = \frac{(\mathbf{x}^T \mathbf{y} + 1)^2}{(\|\mathbf{x}\|^2 + 1)(\|\mathbf{y}\|^2 + 1)}$.
- **Experiment 3 (Proposed Method):**
  - Connectivity-Guided Quantum Kernel SVM with anatomical graph topology.
- **Experiment 4 (Ablation Control):**
  - Reassigned-Connection Quantum Kernel: Identical gate count and angles, but randomly permuted edge-to-qubit mappings.
- **Speaker Notes:** "Dodiya Mahavirsinh's evaluation framework executes four strictly controlled experiments on identical partitions. The reassigned control specifically tests whether anatomical connection placement matters."

---

### Slide 11: Leakage-Free Evaluation Protocol
- **Stratified Subject-Wise Cross-Validation:**
  - All sessions from each participant remain together in either train or test folds.
- **Training-Only Fitting:**
  - Standard scaling parameters ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) fit strictly on training subjects.
- **Fold-Level ROC-AUC Reporting:**
  - SVM decision margins from different folds lack a shared calibration scale. We calculate AUC per fold and report mean $\pm$ descriptive standard deviation, preventing pooled score distortion.
- **Auditability:** Complete model bundles (Joblib schema v1) preserve reference angles, states, class mappings, and dependency versions.
- **Speaker Notes:** "Evaluation integrity is maintained throughout: zero subject leakage, training-only standardisation, and fold-level AUC calculation to prevent margin scaling artifacts."

---

### Slide 12: Actual Experimental Results & Verification
- **Automated Test Suite Status:**
  - 230 tests passed in 19.2s across quantum math, evaluation integrity, artifact compatibility, and visualization.
- **End-to-End Pipeline Verification (`neuroq self-check`):**
  - Full execution across 3 folds: Feature build $\to$ 4-model evaluation $\to$ Model bundle creation $\to$ Reload $\to$ Out-of-fold inference $\to$ JSON export: **Status: Passed**.
- **Inference Parity:**
  - CLI `neuroq predict` and Streamlit `analyze_subject()` yield bit-for-bit identical decision scores on held-out test subjects.
- **Speaker Notes:** "Pranav verified that the system installs cleanly, reloads model bundles without numerical drift, and achieves complete parity between the CLI and the Streamlit interface."

---

### Slide 13: Explainability: Connection-Perturbation Sensitivity
- **Technique:**
  - Substitute a single connection's median and IQR with training cohort medians.
  - Recompute decision score $\Delta s = s_{\text{perturbed}} - s_{\text{original}}$ without retraining.
- **Interpretation:**
  - Quantifies the directional sensitivity of the learned SVM hyperplane to specific cortical connections.
  - Negative shifts indicate the connection drove an ASD prediction; positive shifts drove a Control prediction.
- **Scientific Caveat:**
  - Highlights model sensitivity, **not causal neurological biomarkers**.
- **Speaker Notes:** "Shivam Mall implemented connection-perturbation sensitivity. By replacing individual edges with training references, we quantify which functional tracts influenced the classification without claiming biological causality."

---

### Slide 14: Interactive Demonstration & System Architecture
- **Interactive Streamlit Workspace:**
  - **Dataset View:** Inspect raw EEG traces, frequency spectra, and channel QC.
  - **Quantum Analysis:** Interactive 4-node connectivity graph, circuit diagram, and real-time inference on held-out subjects.
  - **Experiment Results:** Comparative performance tables, confusion matrices, and ROC curves.
  - **Explanation & Export:** Perturbation waterfall charts, JSON diagnostic summaries, publication figures.
- **CLI & Application Cohesion:** Both access the unified `asd_eeg.inference.pipeline` library.
- **Speaker Notes:** "Tushar Saharan built the unified Streamlit application and CLI. Evaluators can launch the app with a single command, select a held-out subject, inspect the circuit, run inference, and export results."

---

### Slide 15: Limitations, Conclusion & Team Contributions
- **Scientific Limitations:**
  - Classical PennyLane simulation ($2^4 = 16$ amplitudes) — no physical quantum speedup claimed.
  - Sensor-level EEG subject to volume conduction; small public cohorts limit clinical power.
- **Conclusion:**
  - Connectivity-guided quantum kernels offer a mathematically sound, interpretable inductive bias for neuroimaging data.
- **Team Ownership:**
  - Vivek (Quantum Circuits), Sahil (Data & QC), Aryan (Graph Features), Dodiya (Evaluation), Tushar (Integration & App), Shivam (Visualizations), Pranav (Verification & Submission).
- **Speaker Notes:** "In conclusion, NeuroQ-Connect demonstrates that embedding functional brain graph topologies into quantum kernels yields an interpretable, reproducible machine learning architecture. Thank you, and we welcome your questions."

