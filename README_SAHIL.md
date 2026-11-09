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
* **Selected & Frozen Dataset:** **TUOS** (*The University of Sheffield*), published by Dickinson, Jeste, & Milne (2022), *Cortex*, 148, 139–151 (DOI: `10.1016/j.cortex.2021.09.022`; Figshare DOI: `10.15131/shef.data.16840351`).
* **Why Candidate Datasets Were Rejected:**
  * **KAU:** Unavailable; raw recordings, verified labels, and channel montage files were not provided.
  * **OpenNeuro (`ds006780`, `ds005512`):** Inspected in the raw archive and proven to be Git-Annex pointer stubs (~200 bytes) containing no binary EEG data payloads.
* **Raw Files on Local Disk:** 56 EEGLAB `.set` header and companion `.fdt` binary data files located in `data/raw/tuos/` (git-ignored for data privacy and repository cleanliness).

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

## 3. How Each Teammate Uses Sahil's Work

```
                         SAHIL GUPTA (Data & Preprocessing Lead)
               [data/raw/tuos/, recordings.csv, eligibility.csv, configs/]
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
   ARYAN PANDEY                      VIVEK G                   DODIYA MAHAVIRSINH
(Features Lead)               (Quantum Models Lead)            (Evaluation Lead)
• Reads recordings.csv         • Builds 4-qubit circuit        • Uses artifacts/features.npz
• Uses 4 scalp nodes           • Maps nodes 0-3 to qubits      • Runs 5-fold stratified CV
• Extracts 28 features         • Maps 6 edges to IsingZZ       • Uses fold-preserving bootstrap
         │                              │                              │
         └──────────────────────────────┼──────────────────────────────┘
                                        │
         ┌──────────────────────────────┴──────────────────────────────┐
         ▼                                                             ▼
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

## 4. Experimental Results Summary (18-Window Revised vs. 20-Window Original)

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

## 5. Verification & Reproduction Commands

To reproduce Sahil's entire pipeline from a terminal:

```powershell
# 1. Switch to Sahil's branch
git checkout sahil

# 2. Run unit and integration tests (14 tests)
python -m pytest

# 3. Run linter
python -m ruff check .

# 4. Validate the 45-subject manifest
neuroq validate --manifest data/metadata/recordings.csv

# 5. Extract 28-dimensional features (18-window common budget)
neuroq features --manifest data/metadata/recordings.csv --config configs/default.yaml --output artifacts/features.npz

# 6. Run 5-fold stratified evaluation across all 4 models
neuroq evaluate --features artifacts/features.npz --output results

# 7. Predict on held-out subject ASD11 using a saved quantum bundle
neuroq predict --manifest data/metadata/recordings.csv --subject ASD11 --bundle results/run-20261109-160931-0cf4a1/fold_0/quantum.joblib --output artifacts/prediction.json
```
