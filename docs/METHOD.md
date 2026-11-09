# Verified Method, Dataset Specification, and Scientific Notes

## Dataset selection and provenance: TUOS (The University of Sheffield)

The research evaluation freezes **TUOS** as the primary real-data cohort.
- **Reference Publication:** Dickinson, Jeste, & Milne (2021), *Electrophysiological signatures of brain aging in autism spectrum disorder*, Cortex, 142, 340-353. DOI: `10.15131/shef.data.16840351`.
- **Candidate Assessment:**
  - **KAU:** Unavailable; missing raw continuous recordings, verified labels, and channel montage.
  - **OpenNeuro (ds006780, ds005512):** Present only as Git-Annex symlink pointer stubs (~200 bytes) without binary EEG data payloads.
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
  - **Montage mismatch (9 recordings: P1, P5, P6, P9, P20, P24, P25, P29, P31):** Recorded using a 128-pin BioSemi cap labeled with radial pin coordinates (`A1..A32, B1..B32, C1..C32, D1..D32`) rather than standard 10-20 international nomenclature. Excluded per project contract prohibiting uncalibrated BioSemi equivalence guessing.
  - **Montage channel absence (1 recording: ASD7):** Missing channel `Fz` (channel was excluded during original author preprocessing for signal degradation).
  - **Quality control amplitude rejection (1 recording: ASD17):** Excessive motion/amplitude artifacts; only 1 clean 8-second window passed the 300 µV peak-to-peak threshold (minimum 3 required).

Full eligibility status and exclusion rationales are recorded in `data/metadata/eligibility.csv`. The verified manifest is saved at `data/metadata/recordings.csv`.

## Verified EEG preprocessing pipeline

1. **Continuous signal calibration:** Scaled to standard volts.
2. **Bandpass filtering:** Fourth-order zero-phase Butterworth filter with cutoffs at 1.0 Hz and 40.0 Hz.
3. **Mains notch filter:** 50.0 Hz IIR notch filter with quality factor $Q = 30$.
4. **Resampling:** Rational polyphase resampling to 128.0 Hz.
5. **Spatial referencing:** Common average reference (CAR) across all active channels.
6. **Windowing & QC:** Non-overlapping 8.0-second epochs (1024 samples at 128 Hz). Windows are accepted only when:
   - All channels are finite and non-NaN.
   - Channel standard deviation exceeds $10^{-12}$ V (flat-line check).
   - Maximum peak-to-peak amplitude is $\le 300\ \mu\text{V}$.
7. **Subject pooling:** Up to 20 accepted clean windows (160 seconds maximum) pooled per subject.

## Four-node montage selection

The verified four-node configuration spans anterior, posterior, midline, and bilateral cortical regions:
- **Node 0 (Left frontal):** `F7`
- **Node 1 (Frontal midline):** `Fz`
- **Node 2 (Left posterior):** `P1`
- **Node 3 (Right posterior):** `O2`

Pairwise connectivity is evaluated across six edges:
- `(0, 1)`: Left frontal — Frontal midline
- `(0, 2)`: Left frontal — Left posterior (intra-hemispheric left)
- `(0, 3)`: Left frontal — Right posterior (cross-hemispheric fronto-posterior)
- `(1, 2)`: Frontal midline — Left posterior
- `(1, 3)`: Frontal midline — Right posterior
- `(2, 3)`: Left posterior — Right posterior (inter-hemispheric posterior)

## Feature extraction

Features follow the strict 28-dimensional contract:
1. **Node band powers (features 1–16):** Welch PSD normalized by 1–30 Hz total power. Relative powers for delta [1,4), theta [4,8), alpha [8,13), and beta [13,30) bands, $\log_{10}$ transformed and median aggregated across windows.
2. **Connectivity medians (features 17–22):** Median alpha-envelope Pearson correlation across accepted windows in fixed edge order `(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)`.
3. **Connectivity variability (features 23–28):** Interquartile range (IQR, 75th percentile minus 25th percentile) of alpha-envelope correlations across accepted windows in the same edge order.

## Quantum and baseline models

Features are standardized on training folds only and bounded by $\phi = \frac{\pi}{2} \tanh(z / 2)$.
1. **RBF-SVM:** Classical radial basis function kernel on bounded 28-dimensional features.
2. **Amplitude-fidelity kernel SVM:** Normalized amplitude overlap kernel $(x^T y + 1)^2 / ((\|x\|^2+1)(\|y\|^2+1))$.
3. **Connectivity-guided quantum-kernel SVM:** Four-qubit PennyLane statevector circuit with node rotations (RY/RZ) and edge-conditioned IsingZZ entangling gates over 2 passes. Fidelity kernel $K(x,y) = |\langle\psi(x)\mid\psi(y)\rangle|^2$.
4. **Reassigned quantum kernel:** Control experiment with fixed, label-independent permutation of edge assignments.

## Scientific limitations

1. Sensor-level alpha envelope correlation reflects functional connectivity but is subject to volume conduction and common reference effects.
2. Exact statevector simulation on 4 qubits is classically tractable and serves as an algorithmic investigation, not a demonstration of physical quantum speedup.
3. The sample size ($N = 45$: 26 ASD, 19 Control) is appropriate for preliminary feasibility studies and out-of-fold validation, but requires multi-cohort expansion before clinical interpretation.
