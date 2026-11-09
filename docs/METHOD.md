# Verified Method, Dataset Specification, and Scientific Notes

## Dataset selection and provenance: TUOS (The University of Sheffield)

The research evaluation freezes **TUOS** as the primary real-data cohort.
- **Reference Publication:** Dickinson, A., Jeste, S., & Milne, E. (2022), *Electrophysiological signatures of brain aging in autism spectrum disorder*, **Cortex**, 148, 139–151. DOI: [`10.1016/j.cortex.2021.09.022`](https://doi.org/10.1016/j.cortex.2021.09.022).
- **Archive Package:** Figshare DOI: `10.15131/shef.data.16840351`.
- **Candidate Assessment:**
  - **KAU:** Unavailable; missing raw continuous recordings, verified labels, and channel montage.
  - **OpenNeuro (ds006780, ds005512):** The inspected metadata clones initially contained only Git-Annex pointer stubs (~200 bytes), not binary EEG payloads. This does not establish that recordings are unavailable. The optional SFARI acquisition workflow below retrieves and verifies `ds006780` payloads separately.
  - **TUOS:** Extracted from Figshare package `16840351`. Contains 56 continuous eyes-closed resting-state EEG recordings (28 autistic adults and 28 neurotypical controls, aged 18–68) recorded with BioSemi ActiveTwo hardware at 512 Hz for ~150–160 seconds.

### Authoritative label provenance and header/filename mismatch resolution

Labels were **never** inferred from filename ordering. The original dataset exhibits a known mismatch between sequential file numbering and participant groups:
1. File `23Abby_Resting.set` contains header `setname: P51` and comments tracing to `Individual Differences Project\EEG Data\P51_3_2-Deci.bdf`. This is a **Control** subject (label = 0), not ASD.
2. File `29Abby_Resting.set` contains header `setname: ASD29` and comments tracing to `British Academy ASD Project\EEG Data\Raw Data\AA_29_28_6-Deci.bdf`. This is an **ASD** subject (label = 1).
3. Authoritative labels are extracted directly from the EEGLAB `.set` struct metadata (`setname` and `comments` recording path).

### Cohort eligibility and exclusion criteria

All 56 recordings were evaluated against rigorous inclusion criteria:
- **Eligible subjects:** 45 participants (**26 ASD, 19 Control**)
- **Exclusion criteria (11 recordings excluded):**
  - **Montage mismatch (9 recordings: P1, P5, P6, P9, P20, P24, P25, P29, P31):** Recorded using a 128-pin BioSemi cap labeled with radial pin coordinates (`A1..A32, B1..B32, C1..C32, D1..D32`) rather than standard 10-20 international nomenclature. Excluded per project contract prohibiting uncalibrated BioSemi equivalence guessing. *(Note: Retaining these 9 controls while excluding ASD7 and ASD17 would yield 26 ASD + 28 Control = 54 subjects).*
  - **Montage channel absence (1 recording: ASD7):** Missing channel `Fz` (channel was excluded during original author preprocessing for signal degradation).
  - **Quality control amplitude rejection (1 recording: ASD17):** Excessive motion/amplitude artifacts; only 1 clean 8-second window passed the 300 µV peak-to-peak threshold (minimum 3 required).

Full eligibility status and exclusion rationales are recorded in `data/metadata/eligibility.csv`. The verified manifest is saved at `data/metadata/recordings.csv`.

## Verified EEG preprocessing pipeline

1. **Continuous signal calibration:** Scaled to standard volts ($V$).
2. **Spatial referencing:** Common average reference (CAR) across all active channels.
3. **Mains notch filter:** 50.0 Hz IIR notch filter with quality factor $Q = 30$. *(Note on sampling: At target sampling rate 128 Hz, Nyquist is 64 Hz, so 50 Hz does not alias. The notch provides active suppression of European 50 Hz laboratory powerline interference, preventing high-amplitude mains energy from leaking through the 40 Hz filter transition band).*
4. **Bandpass filtering:** Fourth-order zero-phase Butterworth filter with cutoffs at 1.0 Hz and 40.0 Hz (effective order 8 bidirectional `sosfiltfilt`).
5. **Resampling:** Rational polyphase resampling from 512.0 Hz to 128.0 Hz.
6. **Windowing & QC:** Non-overlapping 8.0-second epochs (1024 samples at 128 Hz). Windows are accepted only when:
   - All channels are finite and non-NaN.
   - Channel standard deviation exceeds $10^{-12}$ V (flat-line check).
   - Maximum peak-to-peak amplitude is $\le 300\ \mu\text{V}$.
7. **Window budget:** `configs/default.yaml` caps each TUOS subject at 18 accepted 8-second windows, selected deterministically using linear index spacing (`np.linspace`) when the pool exceeds the cap. Eighteen windows represent 144.0 seconds of accepted EEG, not necessarily consecutive time. The shared pipeline requires at least three accepted windows per recording; it does not require every subject to reach 18. Check the saved accepted/used-window counts before claiming equal duration across the cohort.

## Four-node scalp montage (scalp measurement sites)

The four nodes represent sensor-level **scalp measurement sites** (not direct or source-localized brain regions):
- **Node 0 (Left frontal site):** `F7` (anterior lateral sensor)
- **Node 1 (Frontal midline site):** `Fz` (anterior midline sensor)
- **Node 2 (Left posterior site):** `P1` (centroparietal sensor, 10-10 system)
- **Node 3 (Right posterior site):** `O2` (occipital sensor)

Pairwise sensor-level connectivity is evaluated across six edges:
- `(0, 1)`: Left frontal (F7) — Frontal midline (Fz)
- `(0, 2)`: Left frontal (F7) — Left posterior (P1)
- `(0, 3)`: Left frontal (F7) — Right posterior (O2)
- `(1, 2)`: Frontal midline (Fz) — Left posterior (P1)
- `(1, 3)`: Frontal midline (Fz) — Right posterior (O2)
- `(2, 3)`: Left posterior (P1) — Right posterior (O2)

## Feature extraction

The extractor is label-free and accepts only processed recordings, a configuration, and the fixed node/edge orders. Labels are not an input. The authoritative order is node-major power, then edge medians, then edge IQRs. For four nodes the width is exactly 28.

Features follow the strict 28-dimensional contract:
1. **Scalp site band powers (features 1–16):** Welch PSD normalized by 1–30 Hz total power (at most 256 samples per segment). Relative power for each electrode is the integrated PSD in a half-open band divided by the integrated 1-30 Hz PSD. Bands are delta [1,4), theta [4,8), alpha [8,13), and beta [13,30). Per-channel relative powers are averaged within a node, $\log_{10}$ transformed, and then median aggregated across the 18 used windows.
2. **Connectivity medians (features 17–22):** Continuous filtered signals receive an alpha bandpass followed by the Hilbert transform. Log envelope power is averaged across each node's electrodes. Pearson correlation of those regional log envelopes is computed separately inside each accepted window, never across the whole recording. The fixed edge order is `(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)`. Edge medians across the 18 used windows become features 17–22.
3. **Connectivity variability (features 23–28):** Interquartile range (IQR, 75th percentile minus 25th percentile) of alpha-envelope correlations across the 18 used windows in the same edge order.

Accepted windows are pooled across a subject's recordings in `recording_id` order. If the pool exceeds `max_windows`, windows are selected at deterministic evenly spaced indices (`np.linspace`). The archive records accepted and used window counts, the subject median and IQR connectivity matrices, and the used-window edge correlations together with their subject and recording identifiers. `feature_schema.json` describes every feature's name, source node or edge, units, and aggregation. The configured feature order must not be changed without rebuilding features and saved model bundles.

Integration validation does not change the scientific feature definitions or settings. Schemas are archive-specific (`feature_schema.json` for the default `subject_features.npz`, otherwise `<archive-stem>.feature_schema.json`). Ordered definitions are checked against the configuration; graph values and window summaries must agree with the feature columns. Optional legacy arrays may be absent, but inconsistent supplied handoff data is rejected before saving or evaluation. These are software-integrity checks, not evidence of clinical validity.

## Quantum and baseline models

Features are standardized on training folds only and bounded by $\phi = \frac{\pi}{2} \tanh(z / 2)$.
1. **RBF-SVM:** Classical radial basis function kernel on bounded 28-dimensional features.
2. **Amplitude-fidelity kernel SVM:** Normalized amplitude overlap kernel $(x^T y + 1)^2 / ((\|x\|^2+1)(\|y\|^2+1))$.
3. **Connectivity-guided quantum-kernel SVM:** Four-qubit PennyLane statevector circuit with node rotations (RY/RZ) and edge-conditioned IsingZZ entangling gates over 2 passes. Fidelity kernel $K(x,y) = |\langle\psi(x)\mid\psi(y)\rangle|^2$.
4. **Reassigned quantum kernel:** Control experiment with fixed, label-independent permutation of edge assignments.

The map is fixed, not a variationally trained QNN; the learned classifier is an SVM. The four-qubit statevector model is classically tractable and demonstrates neither quantum computational advantage nor clinical validity. Shared references and volume conduction remain limitations of the non-orthogonalized sensor-level connectivity.

## Statistical methodology & evaluation notes

Fixed-setting stratified subject-wise cross-validation is used. Fold count is resolved per run; default is 5. Class balance is preserved in every fold. Preprocessing scalers are fit strictly on training folds and applied to test folds.

Classification metrics are calculated from all out-of-fold labels. AUC is calculated within each fold and averaged. The evaluation module freezes subject-fold assignments once per run and recomputes all classification and AUC summaries through the shared `summarize_oof` function. AUC summaries include the mean, population standard deviation (`ddof=0`), and number of defined folds; standard deviation is descriptive fold-to-fold spread, not a confidence interval. Single-class fold AUC is undefined and omitted from its summary. OOF validation rejects duplicates, differing model cohorts/partitions, conflicting labels, nonfinite scores, and invalid fold IDs. Split creation never truncates fractional labels into diagnosis classes. Source-commit provenance is tied to the installed source checkout, not the caller's current directory.

The statistics below are Sahil's reported TUOS results, with run/configuration identifiers in [README_SAHIL.md](../README_SAHIL.md). This merge verifies software integration, not those recorded-data results or the separate bootstrap calculations; their underlying local artifacts must be checked before submission. No hyperparameter search, external validation, or clinical validation is supplied by this integration. Changing settings after inspecting held-out scores requires a separately identified exploratory run.

1. **Confidence intervals:** Wilson score intervals apply specifically to the unweighted overall accuracy proportion ($k / N$). They do not apply to balanced accuracy or ROC-AUC.
2. **Fold-preserving stratified bootstrap (2,000 resamples):**
   - Uncalibrated SVM decision margins from distinct folds cannot be pooled into a single global ranking without risking scale distortions.
   - To estimate sampling uncertainty without pooling, the bootstrap resamples ASD and Control participants independently within each fold.
   - For every iteration, fold AUC is computed within each of the 5 test folds and averaged to yield the resampled Mean Fold AUC.
   - Results on the 18-window configuration:
     - **RBF-SVM:** Mean Fold AUC = 0.5467 (95% CI: [0.386, 0.709]); Balanced Acc = 0.5445 (95% CI: [0.427, 0.669])
     - **Amplitude Kernel:** Mean Fold AUC = 0.4722 (95% CI: [0.310, 0.647]); Balanced Acc = 0.5374 (95% CI: [0.399, 0.667])
     - **Proposed Quantum Kernel:** Mean Fold AUC = 0.3789 (95% CI: [0.208, 0.550]); Balanced Acc = 0.4798 (95% CI: [0.355, 0.609])
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

## Explanation

Connection sensitivity replaces the median and IQR of one edge with raw-feature medians from that model's training subjects and recomputes its decision score. This is an input perturbation check; correlated-feature changes can be off-distribution. It is not a causal biomarker.

## Reference and distinction

Saranya and Menaka (2025), IEEE Access, DOI: 10.1109/ACCESS.2025.3531979. Their paper uses wavelet/filter-bank CSP and engineered features from a private 20-child cohort. The current regional-envelope graph representation and circuit are an adaptation. Their reported accuracy must never be attributed to this code or another dataset. The proposed contribution is a controlled investigation of graph-conditioned encoding, with novelty and benefit to be assessed empirically.
