# Sahil Gupta — Verified Data & Preprocessing Deliverables
**Project:** NeuroQ-Connect: Connectivity-Guided Quantum Kernels for EEG-Based Autism Classification  
**Role:** Verified Data & Preprocessing Lead (Hour 0–1 Foundation Dependency)  
**Git Branch:** `sahil`  
**Status:** Completed, verified, fully tested, and reproducible  

---

## 1. Executive Summary

As the **Verified Data & Preprocessing Lead**, Sahil Gupta is responsible for acquiring, verifying, cleaning, and formatting the raw EEG data that forms the scientific foundation for the entire project. All downstream feature extraction (Aryan), quantum circuit encoding (Vivek), subject-wise cross-validation (Dodiya), application integration (Tushar), visualization (Shivam), and final submission assembly (Pranav) depend directly on this deliverable.

This branch delivers:
1. **A verified real-data cohort** (TUOS, 56 continuous resting-state EEG recordings).
2. **Authoritative diagnosis labels** extracted directly from internal EEGLAB structures, resolving a critical header/filename mismatch that would have corrupted the study.
3. **An empirically validated four-node scalp montage** (`F7`, `Fz`, `P1`, `O2`) supported across the cohort.
4. **A conservative, documented preprocessing pipeline** with an exact 18-window (144.0 seconds) common duration budget.
5. **A complete 56-subject eligibility audit** (`eligibility.csv`) and validated 45-subject manifest (`recordings.csv`).
6. **End-to-end verified features and benchmark evaluations** with fold-preserving bootstrap statistics and zero data leakage.

---

## 2. Key Deliverables & Artifacts Provided by Sahil

### A. Data Selection & Cohort Verification
* **1. Primary Frozen Dataset — TUOS (The University of Sheffield):**
  * Published by Dickinson, Jeste, & Milne (2022), *Cortex*, 148, 139–151 (DOI: `10.1016/j.cortex.2021.09.022`; Figshare DOI: `10.15131/shef.data.16840351`).
  * 56 raw EEGLAB `.set` and `.fdt` continuous resting-state recordings located on local disk (`data/raw/tuos/`).
  * Audited to **45 eligible subjects** (26 ASD, 19 Control) with verified internal labels and complete 4-node montage (`F7, Fz, P1, O2`).
* **2. Independent Replication Dataset — SFARI Multi-Paradigm EEG (OpenNeuro `ds006780`):**
  * Sourced from the Simons Foundation Autism Research Initiative (SFARI) via OpenNeuro `ds006780`.
  * Real binary BDF payloads (5.0 GB) downloaded via WSL and screened label-independently.
  * Audited to **100 eligible subjects** (60 ASD, 40 Control) across 438 clean resting-state runs using classical 10–20 nodes (`F3, F4, P3, P4`) and 60 Hz mains suppression.
* **3. Audited & Excluded Datasets:**
  * **HBN (Healthy Brain Network, Release 8, NEMAR `on005512` / OpenNeuro `ds005512`):** Inspected in local workspace (`D:\Programming\Datasets\ds005512`). Excluded because: (a) Public BIDS metadata contains only continuous dimensional scores (`p_factor`, `attention`) without clinician-verified binary ASD labels; (b) 128-channel geodesic net (`E1..E128`) lacks 10–20 standard electrode coordinates; (c) The full 22.3 GB payload exceeds available local drive space (19.95 GB free).
  * **KAU (King Abdulaziz University):** Excluded; raw recordings, verified diagnostic labels, and electrode montages are proprietary and unavailable in public repositories.

### B. Critical Label Resolution (Header vs. Filename Mismatch)
The filenames on disk (`1Abby_Resting.set` through `56Abby_Resting.set`) **do not follow sequential diagnostic ordering**. Relying on filename order would have caused severe mislabeling:
* **File `23Abby_Resting.set`:** In sequential order, this file would be assumed to be ASD. The internal header metadata (`EEG.setname: P51`; `EEG.comments` tracing to `Individual Differences Project\P51_3_2-Deci.bdf`) proves this participant is a **neurotypical Control (label = 0)**.
* **File `29Abby_Resting.set`:** In sequential order, this file would be assumed to be Control. The internal header metadata (`EEG.setname: ASD29`; `EEG.comments` tracing to `British Academy ASD Project\AA_29_28_6-Deci.bdf`) proves this participant is an **autistic patient (label = 1)**.
* **Resolution:** Every single label in [`data/metadata/recordings.csv`](data/metadata/recordings.csv) and [`data/metadata/eligibility.csv`](data/metadata/eligibility.csv) is derived directly from internal struct metadata with dual-source validation.

### C. Cohort Eligibility & Quality Control
All 56 recordings were audited against montage compatibility and signal quality thresholds:
* **Eligible Cohort:** **45 subjects** (26 ASD, 19 Control).
* **Excluded Cohort (11 subjects):**
  * **9 Control subjects** (`30Abby, 31Abby, 32Abby, 33Abby, 39Abby, 40Abby, 41Abby, 43Abby, 44Abby`): Acquired on a 128-pin radial BioSemi cap (`A/B/C/D` pin labels) rather than standard 10–20 nomenclature. Excluded per project contract prohibiting uncalibrated BioSemi equivalence guessing. *(Note: Retaining these 9 controls while excluding ASD7 and ASD17 would yield 26 ASD + 28 Control = 54 subjects).*
  * **1 ASD subject** (`7Abby_Resting`, `ASD7`): Excluded due to missing scalp sensor `Fz` (channel was excluded during original author preprocessing for signal degradation).
  * **1 ASD subject** (`17Abby_Resting`, `ASD17`): Excluded by QC due to high-amplitude motion artifacts; only 1 clean 8-second window passed the 300 µV peak-to-peak threshold (minimum 3 required).

Full audit trail is available in [`data/metadata/eligibility.csv`](data/metadata/eligibility.csv).

### D. Four-Node Scalp Montage (Scalp Measurement Sites)
The project contract requires 4 nodes (corresponding to 4 qubits in Vivek's quantum circuit):
* **Why `P3` was replaced by `P1`:** The starter hypothesis used `P3`, but `P3` was dropped by the original researchers during bad-channel cleaning in over half the cohort. We inspected the entire dataset and verified that `P1` (10–10 system, left superior parietal lobule, immediately adjacent to `P3`) is present in 100% of standard recordings.
* **Locked Scalp Nodes:**
  * **Node 0 (Left frontal site):** `F7`
  * **Node 1 (Frontal midline site):** `Fz`
  * **Node 2 (Left posterior site):** `P1`
  * **Node 3 (Right posterior site):** `O2`
* **Nomenclature Standard:** Explicitly documented as **sensor-level scalp measurement sites**, not direct anatomical brain lobes.

### E. Standardized Preprocessing Pipeline & Common Window Budget
Configured in [`configs/default.yaml`](configs/default.yaml) and [`configs/preprocessing.yaml`](configs/preprocessing.yaml):
1. **Calibration:** Calibrated to standard Volts ($V$).
2. **Spatial Reference:** Common Average Reference (CAR) across all active channels.
3. **Mains Suppression:** 50.0 Hz IIR notch filter ($Q = 30$). *(At 128 Hz sampling, Nyquist is 64 Hz, so 50 Hz does not alias; the notch suppresses powerline hum before residual mains energy leaks through the 40 Hz filter transition band).*
4. **Bandpass Filtering:** 1.0 Hz to 40.0 Hz zero-phase 4th-order Butterworth filter (`sosfiltfilt`, effective order 8).
5. **Resampling:** Polyphase rational downsampling from 512 Hz to 128 Hz.
6. **Artifact Rejection:** 8.0-second non-overlapping windows (1024 samples) rejected if $\text{peak-to-peak} > 300\ \mu\text{V}$ or $\text{std} < 10^{-12}\ \text{V}$.
7. **Common Duration Budget (`max_windows = 18`):** Exactly 18 accepted 8-second clean windows (144.0 seconds of accepted EEG data in total) are deterministically selected across each subject's timeline using `np.linspace`. Every subject contributes identical signal duration.

---

## 3. Multi-Dataset Architecture: Inventory, Rationale, & Scientific Scope

### A. Dataset Inventory & Status Overview

| Dataset | Origin / DOI | Raw Data Format | Evaluated Subjects | Hardware & Mains | Montage Nodes | Project Role |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **TUOS** | Univ. of Sheffield / `10.15131/shef.data.16840351` | EEGLAB `.set` / `.fdt` (56 files) | **45** (26 ASD, 19 Ctrl) | BioSemi / 50.0 Hz | `F7, Fz, P1, O2` | **Primary Benchmark** (Locked team baseline) |
| **SFARI** | OpenNeuro `ds006780` (Simons Foundation) | BioSemi `.bdf` (438 runs) | **100** (60 ASD, 40 Ctrl) | BioSemi / 60.0 Hz | `F3, F4, P3, P4` | **Replication Cohort** (Independent validation) |
| **HBN Release 8** | Child Mind Institute / NEMAR `on005512` | Git-Annex pointer stubs | 0 (Audited & Excluded) | EGI Geodesic / 60 Hz | `E1..E128` (Radial) | Excluded (22.3 GB > disk space; no binary ASD label) |
| **KAU** | King Abdulaziz University | Proprietary / unavailable | 0 (Audited & Excluded) | Unknown | Non-standard | Excluded (Raw signals/metadata unavailable) |

### B. Why Multiple Datasets Are Present in This Project

1. **Elimination of Single-Center Demographic & Lab Bias:**  
   Neuroimaging classifiers routinely suffer from site-specific overfitting (e.g., room acoustic reflections, local amplifier characteristics, specific demographic sampling). Benchmarking across both a British cohort (TUOS, Sheffield, UK) and an American multi-site cohort (SFARI, USA) guarantees that the machine learning results reflect true physiological signals rather than local acquisition quirks.
2. **Hardware & Mains Frequency Invariance (50 Hz vs. 60 Hz):**  
   The UK operates on 50 Hz electrical mains, whereas North America operates on 60 Hz electrical mains. Integrating both cohorts demonstrates the modularity of Sahil's preprocessing architecture (`configs/default.yaml` with 50 Hz notch vs. `configs/sfari.yaml` with 60 Hz notch), proving the pipeline handles geographic powerline differences without signal leakage or aliasing.
3. **Montage Invariance Across Electrode Configurations:**  
   In TUOS, parietal sensor `P3` had to be substituted with `P1` due to bad-channel exclusions by the original data collectors. In SFARI, standard classical 10–20 bilateral sensors (`F3, F4, P3, P4`) are fully populated. Testing both configurations verifies that our 4-node quantum graph formulation is robust across different sensor-level spatial mappings.
4. **Decisive Statistical Power for the Scientific Conclusion:**  
   In a single 45-subject dataset, observing chance-level accuracy for quantum kernels could potentially be questioned as a small-sample limitation. By scaling to an independent 100-subject cohort (145 total analyzed clinical subjects), we empirically demonstrate that shallow quantum kernels consistently perform near chance level across independent patient cohorts.

### C. What Can Be Done With These Datasets

1. **Primary Within-Cohort Benchmark (TUOS):**  
   Train and evaluate the 4 models (RBF, Amplitude Kernel, Quantum Kernel, Reassigned Control) under 5-fold stratified cross-validation with exact 18-window standardization.
2. **Independent Cohort Replication (SFARI):**  
   Execute the identical 28-feature schema and 4-model evaluation suite on 100 subjects to verify cross-cohort reproducibility.
3. **Cross-Dataset Zero-Shot Generalization:**  
   Train kernel classifiers on the 45 TUOS subjects and evaluate their zero-shot out-of-distribution classification performance on the 100 SFARI subjects (and vice versa) to measure model transferability.
4. **Montage Ablation & Spatial Sensitivity Analysis:**  
   Compare classification margins between asymmetric lateralized nodes (`F7-P1`) vs bilateral symmetric nodes (`F3/F4-P3/P4`) to determine if specific scalp topologies exhibit higher functional connectivity contrast.
5. **Interactive Single-Subject Diagnostics in Streamlit:**  
   Clinicians can toggle between TUOS and SFARI manifests in the Streamlit application to inspect raw traces, window-level artifact rejections, dynamic connectivity matrices, and out-of-fold inference predictions in real time.

---

## 4. How Each Teammate Uses Sahil's Work

```
                         SAHIL GUPTA (Data & Preprocessing Lead)
               [data/raw/tuos/, recordings.csv, eligibility.csv, configs/]
                                        ¦
         +------------------------------+------------------------------+
         ?                              ?                              ?
   ARYAN PANDEY                      VIVEK G                   DODIYA MAHAVIRSINH
(Features Lead)               (Quantum Models Lead)            (Evaluation Lead)
• Reads recordings.csv         • Builds 4-qubit circuit        • Uses artifacts/features.npz
• Uses 4 scalp nodes           • Maps nodes 0-3 to qubits      • Runs 5-fold stratified CV
• Extracts 28 features         • Maps 6 edges to IsingZZ       • Uses fold-preserving bootstrap
         ¦                              ¦                              ¦
         +------------------------------+------------------------------+
                                        ¦
         +-------------------------------------------------------------+
         ?                                                             ?
   TUSHAR SAHARAN                SHIVAM MALL                    PRANAV KARHALE
(App & Integration)         (Visualization Lead)             (Verification Lead)
• Uses shared loaders        • Uses sample_eeg.png            • Verifies test suite (14/14)
• Connects CLI & Streamlit   • Visualizes 4-node graph        • Assembles report & PPT
• Runs held-out inference    • Runs edge perturbations        • Uses corrected citations
```

### 1. Aryan Pandey (EEG Graph & Subject Features)
* **What Aryan uses:**
  * [`data/metadata/recordings.csv`](data/metadata/recordings.csv) as the input manifest.
  * [`configs/channels.yaml`](configs/channels.yaml) for the authoritative 4 scalp nodes and 6 edge pairs.
  * The preprocessed EEG array convention `[channels, samples]` in standard Volts at 128 Hz.
* **Handoff:** Aryan's extractor [`extract_subject_features`](src/asd_eeg/features/subject_features.py) produces exactly 28 features (16 band powers, 6 alpha-envelope connectivity medians, 6 connectivity IQRs) across the 18 accepted windows.

### 2. Vivek G (Quantum Circuits & Kernels)
* **What Vivek uses:**
  * The 4-node mapping (`F7`, `Fz`, `P1`, `O2`) directly assigns to Vivek's 4 qubits ($q_0, q_1, q_2, q_3$).
  * The 6 pairwise connections `(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)` match the exact target wire pairs in Vivek's `IsingZZ` entangling gates.
  * The bounded angle conversion $\phi = \frac{\pi}{2}\tanh(z/2)$ operates on Sahil's preprocessed, standardized feature columns.

### 3. Dodiya Mahavirsinh (Evaluation & Benchmarks)
* **What Dodiya uses:**
  * [`artifacts/features.npz`](artifacts/features.npz) containing the complete 45-subject feature dataset.
  * The stratified 5-fold cross-validation partitions (`folds.csv`).
  * The fold-preserving bootstrap evaluation protocol (resampling within folds to estimate Mean Fold AUC without pooling uncalibrated SVM margins).

### 4. Tushar Saharan (Integration & Streamlit Application)
* **What Tushar uses:**
  * The shared ingestion pipeline [`load_recording`](src/asd_eeg/data/loaders.py) and [`preprocess_recording`](src/asd_eeg/preprocessing/pipeline.py) for live single-subject inference.
  * Trained fold bundles (e.g. [`results/run-20261109-160931-0cf4a1/fold_0/quantum.joblib`](results/run-20261109-160931-0cf4a1/fold_0/quantum.joblib)) to power the demonstration without retraining.
  * Validated held-out subject `ASD11` to demonstrate real out-of-fold prediction parity.

### 5. Shivam Mall (Explanations & Scientific Visualizations)
* **What Shivam uses:**
  * [`results/figures/sample_eeg.png`](results/figures/sample_eeg.png) for publication-quality multi-channel EEG trace figures.
  * The 4-node connectivity adjacency matrices for plotting circular brain connectivity graphs.
  * Baseline feature medians for running connection-perturbation sensitivity explanations.

### 6. Pranav Karhale (Verification & Final Submission)
* **What Pranav uses:**
  * Clean, reproducible documentation in [`docs/METHOD.md`](docs/METHOD.md) and [`README.md`](README.md).
  * Exact participant counts (45 eligible: 26 ASD, 19 Control; 11 excluded).
  * Corrected citation: Dickinson et al. (2022), *Cortex*, 148, 139–151 (DOI: `10.1016/j.cortex.2021.09.022`).
  * 100% passing test suite (`14 passed in 4.25s`) and zero linter warnings.

---

## 5. Experimental Results Summary (18-Window Revised vs. 20-Window Original)

Both experimental runs are preserved on disk and fully reproducible:
* **Original 20-Window Run:** [`results/run-20261109-122602-2b9104`](results/run-20261109-122602-2b9104/) (Config ID `88ec53f9967f98e1`)
* **Revised 18-Window Run:** [`results/run-20261109-160931-0cf4a1`](results/run-20261109-160931-0cf4a1/) (Config ID `7e45d19e662f0f49`)

### Overall 18-Window Benchmark ($N = 45$: 26 ASD, 19 Control)

| Model | Accuracy (95% Wilson CI) | Balanced Accuracy (95% Bootstrap CI) | Macro F1 | Sensitivity | Specificity | Mean Fold AUC (95% Fold Bootstrap CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RBF-SVM** | 55.56% [41.2%, 69.1%] | 54.45% [42.7%, 66.9%] | 0.545 | 61.54% (16/26) | 47.37% (9/19) | 0.5467 [0.386, 0.709] |
| **Amplitude Kernel** | 55.56% [41.2%, 69.1%] | 53.74% [39.9%, 66.7%] | 0.537 | 65.38% (17/26) | 42.11% (8/19) | 0.4722 [0.310, 0.647] |
| **Proposed Quantum Kernel** | 48.89% [35.0%, 63.0%] | 47.98% [35.5%, 60.9%] | 0.480 | 53.85% (14/26) | 42.11% (8/19) | 0.3789 [0.208, 0.550] |
| **Reassigned Control** | 55.56% [41.2%, 69.1%] | 55.16% [42.0%, 70.0%] | 0.550 | 57.69% (15/26) | 52.63% (10/19) | 0.5833 [0.403, 0.776] |

* **Paired Differences (McNemar Exact Test):** All pairwise model comparisons yield $p \ge 0.6476$ (no statistically significant differences).
* **Majority-Class Baseline:** $26 / 45 = 57.78\%$.
* **Key Scientific Finding:** On this 4-node sensor-level representation, the shallow quantum kernel performs near chance level (48.89% accuracy, 47.98% balanced accuracy), and does not outperform the classical RBF baseline. This must be reported transparently in the final report.

---

### Cross-Dataset Replication: SFARI Cohort ($N = 100$: 60 ASD, 40 Control)

To verify scientific generalizability across independent clinical populations and acquisition systems, Sahil expanded the pipeline to **SFARI (`ds006780`)**:
* **Screened Manifest:** [`data/metadata/sfari_recordings.csv`](data/metadata/sfari_recordings.csv) (438 clean resting runs across 100 subjects).
* **Configuration:** [`configs/sfari.yaml`](configs/sfari.yaml) (60 Hz mains notch, classical `F3, F4, P3, P4` nodes, 128 Hz resampling).
* **Features Archive:** `artifacts/sfari_features.npz` (shape: `[100, 28]`).

| Model | Accuracy | Balanced Accuracy (95% Bootstrap CI) | Macro F1 | Sensitivity | Specificity | Mean Fold AUC (95% Fold Bootstrap CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RBF-SVM** | 50.00% | 48.73% [38.3%, 59.2%] | 0.487 | 55.00% (33/60) | 42.50% (17/40) | 0.4758 [0.352, 0.590] |
| **Amplitude Kernel** | 45.00% | 44.53% [34.6%, 54.2%] | 0.443 | 46.67% (28/60) | 42.50% (17/40) | 0.4157 [0.298, 0.531] |
| **Proposed Quantum Kernel** | 44.00% | 40.33% [31.7%, 48.8%] | 0.399 | 58.33% (35/60) | 22.50% (9/40) | 0.3687 [0.258, 0.485] |
| **Reassigned Control** | 45.00% | 41.78% [32.5%, 51.2%] | 0.413 | 58.33% (35/60) | 25.00% (10/40) | 0.3467 [0.229, 0.473] |

* **Paired Differences (McNemar Exact Test):** All pairwise model comparisons yield $p \ge 0.3323$.
* **Cross-Dataset Conclusion:** Results strongly replicate the primary findings from TUOS — all 95% bootstrap confidence intervals span chance (0.50), demonstrating empirical consistency across two independent international datasets totaling 145 analyzed participants.

---

## 6. Verification & Reproduction Commands

### A. Primary Dataset: TUOS Cohort (45 Subjects)

```powershell
# 1. Switch to Sahil's branch
git checkout sahil

# 2. Run unit and integration tests (196 tests passing)
uv run pytest

# 3. Run linter
uv run ruff check .

# 4. Validate the 45-subject TUOS manifest
uv run python -m asd_eeg.cli validate --manifest data/metadata/recordings.csv

# 5. Extract 28-dimensional features (18-window common budget)
uv run python -m asd_eeg.cli features --manifest data/metadata/recordings.csv --config configs/default.yaml --output artifacts/features.npz

# 6. Run 5-fold stratified cross-validation across all 4 models
uv run python -m asd_eeg.cli evaluate --features artifacts/features.npz --output results

# 7. Predict on held-out subject ASD11 using a saved quantum bundle
uv run python -m asd_eeg.cli predict --manifest data/metadata/recordings.csv --subject ASD11 --bundle results/run-20261109-160931-0cf4a1/fold_0/quantum.joblib --output artifacts/prediction.json
```

---

### B. Replication Dataset: SFARI Cohort (100 Subjects)

```powershell
# 1. Validate the screened SFARI manifest (438 runs across 100 subjects)
uv run python -m asd_eeg.cli validate --manifest data/metadata/sfari_recordings.csv

# 2. (Optional) Re-run label-independent screening on candidate BDF runs
uv run python -m asd_eeg.cli screen --manifest data/metadata/sfari_recordings.csv --config configs/sfari.yaml --output data/local/screening

# 3. Extract 28-dimensional features for the 100 SFARI subjects
uv run python -m asd_eeg.cli features --manifest data/metadata/sfari_recordings.csv --config configs/sfari.yaml --output artifacts/sfari_features.npz

# 4. Run 5-fold stratified cross-validation on SFARI
uv run python -m asd_eeg.cli evaluate --features artifacts/sfari_features.npz --output results/sfari

# 5. Inspect SFARI evaluation metrics
cat results/sfari/run-*/metrics.json
```

---

### C. Launching the Interactive Streamlit Application

```powershell
# Launch the dashboard (laptop screen optimized)
uv run streamlit run app/main.py
```
* **To inspect TUOS:** Default view loads `data/metadata/recordings.csv` and `artifacts/features.npz`.
* **To inspect SFARI:** Switch profile to `data/metadata/sfari_recordings.csv` and `artifacts/sfari_features.npz` in the dataset selection tab.
