# Method, Limitations, and Empirical Findings

## Baseline cohort (TUOS)

- **Dataset:** Continuous resting-state EEG from Dickinson, Jeste, & Milne (2022), *Cortex*, 148, 139–151 (DOI: `10.1016/j.cortex.2021.09.022`; Figshare DOI: `10.15131/shef.data.16840351`).
- **Eligible participants:** 45 subjects (26 ASD, 19 Control) with standard 10–20 electrode coverage and valid resting recordings.
- **Excluded participants:** 11 subjects (9 controls on non-standard radial cap without 10–20 calibration; 1 ASD missing Fz; 1 ASD with insufficient clean signal).
- **Label derivation:** Extracted directly from internal EEGLAB struct comments and setnames (`AA_*` / `ASD*` -> ASD=1; `P*` / `ID*` -> Control=0), not from unverified disk file order.
- **Scalp nodes:** 4 sensor-level sites: `F7` (Node 0), `Fz` (Node 1), `P1` (Node 2, adjacent to original P3), and `O2` (Node 3).
- **Edges:** 6 pairwise connections: `(F7, Fz)`, `(F7, P1)`, `(F7, O2)`, `(Fz, P1)`, `(Fz, O2)`, `(P1, O2)`.
- **Signal conditioning:** Resampled to 128 Hz; common average reference (CAR); 50.0 Hz notch filter ($Q = 30$); 1.0–40.0 Hz zero-phase 4th-order Butterworth bandpass filter.
- **Duration budget:** 18 clean 8.0-second non-overlapping windows (144.0 seconds total accepted signal per participant), deterministically sampled across each recording.

## 28-dimensional feature representation

Extracted identically across all subjects and models:
1. **Regional band powers (16 features):** Relative power in Delta (1–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), and Beta (13–30 Hz) across the 4 nodes (ordered node-major: $0_\delta, 0_\theta, 0_\alpha, 0_\beta, 1_\delta, \dots, 3_\beta$).
2. **Connectivity medians (6 features):** Median Pearson correlation of Hilbert envelope across accepted windows for each of the 6 edges (ordered: $e_{01}, e_{02}, e_{03}, e_{12}, e_{13}, e_{23}$).
3. **Connectivity IQRs (6 features):** Interquartile range of Hilbert envelope correlation across accepted windows for each of the 6 edges (ordered identically to medians).

## Models compared under stratified 5-fold cross-validation

1. **RBF-SVM:** Classical Support Vector Classifier with radial basis function kernel on standardized 28 features.
2. **Amplitude Kernel:** Classical RBF kernel using only the 16 regional band powers.
3. **Proposed Quantum Kernel:** 4-qubit parameterized quantum circuit:
   - Initial state: $|+\rangle^{\otimes 4}$ via Hadamard gates.
   - Node encoding: Single-qubit $R_Z(\theta_i)$ rotations with $\theta_i = \pi \cdot \tilde{x}_i^{\text{band}}$ from normalized band power.
   - Edge encoding: Two-qubit $ZZ(\gamma_{ij})$ entangling gates with $\gamma_{ij} = \pi \cdot \tilde{w}_{ij}$ from envelope connectivity.
   - Kernel value: State overlap $|\langle\psi(x)|\psi(x')\rangle|^2$.
4. **Reassigned Control:** Quantum kernel with edge weights shuffled across non-adjacent node pairs to test topological specificity.

## Empirical findings (TUOS 18-window benchmark, N = 45)

All evaluation metrics are computed out-of-fold under stratified 5-fold cross-validation:

| Model | Accuracy (95% Wilson CI) | Balanced Accuracy (95% Bootstrap CI) | Macro F1 | Sensitivity | Specificity | Mean Fold AUC (95% Fold Bootstrap CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RBF-SVM** | 55.56% [41.2%, 69.1%] | 54.45% [42.7%, 66.9%] | 0.545 | 61.54% (16/26) | 47.37% (9/19) | 0.5467 [0.386, 0.709] |
| **Amplitude Kernel** | 55.56% [41.2%, 69.1%] | 53.74% [39.9%, 66.7%] | 0.537 | 65.38% (17/26) | 42.11% (8/19) | 0.4722 [0.310, 0.647] |
| **Proposed Quantum Kernel** | 48.89% [35.0%, 63.0%] | 47.98% [35.5%, 60.9%] | 0.480 | 53.85% (14/26) | 42.11% (8/19) | 0.3789 [0.208, 0.550] |
| **Reassigned Control** | 55.56% [41.2%, 69.1%] | 55.16% [42.0%, 70.0%] | 0.550 | 57.69% (15/26) | 52.63% (10/19) | 0.5833 [0.403, 0.776] |

### Key scientific observations

1. **No quantum advantage:** The proposed quantum kernel achieves 48.89% accuracy and 47.98% balanced accuracy, performing near chance level ($50\%$) and below the classical RBF-SVM (55.56% accuracy, 54.45% balanced accuracy).
2. **Confidence intervals span chance:** All 95% bootstrap confidence intervals for balanced accuracy and fold-level AUC cross the 0.50 chance threshold:
   - **Quantum Kernel:** Mean Fold AUC = 0.3789 (95% CI: [0.208, 0.550]); Balanced Acc = 0.4798 (95% CI: [0.355, 0.609])
   - **RBF-SVM:** Mean Fold AUC = 0.5467 (95% CI: [0.386, 0.709]); Balanced Acc = 0.5445 (95% CI: [0.427, 0.669])
   - **Amplitude Kernel:** Mean Fold AUC = 0.4722 (95% CI: [0.310, 0.647]); Balanced Acc = 0.5374 (95% CI: [0.399, 0.667])
   - **Reassigned Control:** Mean Fold AUC = 0.5833 (95% CI: [0.403, 0.776]); Balanced Acc = 0.5516 (95% CI: [0.420, 0.700])
3. **Paired classification differences:** McNemar's exact test on out-of-fold contingency tables shows no statistically significant classification differences between any pair of models ($p \ge 0.6476$).
4. **Sensor-level limitation:** Sensor-level alpha envelope correlation reflects functional statistical coupling between scalp measurement sites, but is subject to volume conduction and common reference effects. It must not be interpreted as anatomical or causal connectivity.
5. **Class balance:** With 26 ASD and 19 Control subjects, the majority-class baseline is $26/45 = 57.78\%$.

## Optional SFARI recorded-data workflow

TUOS remains primary. Tushar's SFARI acquisition, fixed-QC screening, and local workspace activation remain available as an independent workflow; they do not replace the TUOS manifest or defaults. See [SFARI.md](SFARI.md). KAU is out of scope.

Use snapshot DOI `10.18112/openneuro.ds006780.v1.0.0`. Eligibility comes from `participants.tsv` diagnostic group and resting-state availability, with definitions and provenance in `participants.json` and README. Map only explicit ASD=1 and TD=0; siblings and unknown groups are excluded, never treated as TD. Do not infer diagnoses from filenames or questionnaire scores. The inspected metadata initially lists 103 candidate participants (63 ASD, 40 TD) and 528 resting-state BDF runs; these are pre-QC counts, not final sample sizes.

Acquire all eligible resting-state BDF runs into a separate ignored directory and verify annex SHA-256 hashes and byte counts. Retain source metadata and provenance hashes. Metadata describes continuous BioSemi 64-channel EEG at 512 Hz, 60 Hz mains, and directly named F3/F4/P3/P4. `configs/sfari_rest.yaml` preserves this node order and a 20-window cap, separately from TUOS's F7/Fz/P1/O2 and 18-window cap. Both retain the 28-column node/edge ordering convention, training-only transforms, model settings, fold generator, and bundle schema; their feature archives and trained bundles are not interchangeable.

MNE calibrates BDF to volts. Apply ordered BIDS channel types/status before selecting EEG so external EXG channels (EMG) and the Status trigger do not enter average reference or QC. Honor bad-channel metadata and fail on missing required nodes rather than substituting electrodes.

Screen each recording with the shared preprocessor before evaluation, retaining runs with at least three clean eight-second windows. Save all recording exclusions and participants with no surviving recordings. Pool sessions only within their original participant IDs and select up to 20 evenly spaced accepted windows per SFARI participant. Do not relax QC or select tasks, subsets, or settings using held-out performance. Basic fixed-rule QC is not ICA or a clinical artifact-removal protocol. Unequal duration, age/sex imbalance, reference effects, prior filtering, and differential attrition remain limitations. SFARI describes eyes-open rest, unlike the primary TUOS eyes-closed recordings; no cross-dataset validity is established.

Activate a dataset-specific ignored local profile with separate feature/results paths. Active evaluation checks dataset/configuration/cohort/labels/session compatibility. Restart Streamlit after changing profiles. Raw EEG, local bundles, generated results, credentials, and personal paths remain outside Git. Real-data SFARI acquisition, final QC counts, and held-out verification must be completed and reviewed by Sahil and Vivek before making recorded-data claims.

## Integration decision

The integration of PRs #4 and #5 retains Sahil's TUOS defaults and manifest as primary and Tushar's SFARI configuration/workspace tools as optional, as explicitly approved for this merge. The workspace regression test now checks both distinct node maps, mains frequencies, window caps, and configuration IDs while requiring identical remaining preprocessing/model settings and the shared 28-feature/edge-order convention. No feature algorithm, label mapping, fold generation, dependency, or model-reference ordering is changed by the conflict resolution. Rebuild features and bundles whenever the selected dataset or configuration changes; never combine synthetic software checks with recorded-data results.

## Cross-dataset replication: SFARI (ds006780, v1.0.0)

To test generalizability across independent clinical cohorts and recording hardware, the pipeline was replicated on **SFARI** (OpenNeuro `ds006780`, snapshot 1.0.0, DOI: [`10.18112/openneuro.ds006780.v1.0.0`](https://doi.org/10.18112/openneuro.ds006780.v1.0.0)).
- **Cohort Audit:** 136 total subjects audited. 33 subjects excluded (28 siblings per protocol, 4 incomplete resting state, 1 missing group).
- **Candidate Pool:** 103 subjects (63 ASD, 40 TD) across 517 resting-state BDF runs.
- **Fixed-QC Screening:** 3 subjects excluded for pervasive motion/amplitude artifacts across all runs (`sub-11025`, `sub-11562`, `sub-11936`).
- **Screened Eligible Cohort:** Exactly **100 participants (60 ASD, 40 Control)** across 438 clean resting-state runs.
- **Preprocessing Adaptation:** 60.0 Hz notch filter (US powerline interference), 1.0–40.0 Hz bandpass, CAR reference, 128 Hz resampling, classical four-node montage (`F3, F4, P3, P4`, verified in BioSemi 64-channel array).
- **5-Fold Cross-Validation Results (100 Subjects):**
  - **RBF-SVM:** Accuracy = 50.0%, Balanced Acc = 48.73% (95% CI: [0.383, 0.592]), Mean Fold AUC = 0.4758 (95% CI: [0.352, 0.590])
  - **Amplitude Kernel:** Accuracy = 45.0%, Balanced Acc = 44.53% (95% CI: [0.346, 0.542]), Mean Fold AUC = 0.4157 (95% CI: [0.298, 0.531])
  - **Proposed Quantum Kernel:** Accuracy = 44.0%, Balanced Acc = 40.33% (95% CI: [0.317, 0.488]), Mean Fold AUC = 0.3687 (95% CI: [0.258, 0.485])
  - **Reassigned Control:** Accuracy = 45.0%, Balanced Acc = 41.78% (95% CI: [0.325, 0.512]), Mean Fold AUC = 0.3467 (95% CI: [0.229, 0.473])
- **Statistical Significance:** McNemar exact tests show no significant classification differences between any pair of models ($p \ge 0.3323$), replicating our TUOS finding that connectivity-guided quantum kernels do not provide an empirical advantage over classical RBF baselines on resting-state scalp EEG.

## Explanation

Connection sensitivity replaces the median and IQR of one edge with raw-feature medians from that model's training subjects and recomputes its decision score. This is an input perturbation check; correlated-feature changes can be off-distribution. It is not a causal biomarker.

## Reference and distinction

Saranya and Menaka (2025), IEEE Access, DOI: 10.1109/ACCESS.2025.3531979. Their paper uses wavelet/filter-bank CSP and engineered features from a private 20-child cohort. The current regional-envelope graph representation and circuit are an adaptation. Their reported accuracy must never be attributed to this code or another dataset. The proposed contribution is a controlled investigation of graph-conditioned encoding, with novelty and benefit to be assessed empirically.
