# Starter method and decisions requiring review

## Status

This is a working starting implementation for the team to validate and extend. The default electrode mapping and preprocessing settings are configurable hypotheses, not verified KAU or TUOS specifications. No real-cohort result is distributed with the repository.

## EEG pipeline

Continuous EEG is loaded in volts. Bad channels marked in MNE metadata are excluded. Required configured channels must exist. Optional average referencing uses all loaded EEG channels; this choice must be reviewed against the acquisition reference and compared consistently within a cohort. A notch is applied only when below the recording Nyquist frequency. A fourth-order 1-40 Hz Butterworth bandpass and rational polyphase resampling to 128 Hz are the defaults.

Non-overlapping eight-second windows are accepted when all channels are finite, non-flat, and below the configured peak-to-peak limit. Each recording needs at least three accepted windows by default. This is basic signal QC, not a replacement for a dataset-specific artifact-cleaning protocol. The extractor pools accepted windows across a subject's recordings and takes up to 20 windows evenly across the pooled sequence. Channel ordering, duration budgets, reference, and session-selection policy need to be frozen before the main study.

## Features

The extractor is label-free and accepts only processed recordings, a configuration, and the fixed node/edge orders. Labels are not an input. The authoritative order is node-major power, then edge medians, then edge IQRs. For four nodes the width is exactly 28.

Welch PSD uses at most 256 samples per segment. Relative power for each electrode is the integrated PSD in a half-open band divided by the integrated 1-30 Hz PSD. Bands are delta [1,4), theta [4,8), alpha [8,13), and beta [13,30). Electrode relative powers are averaged within a node, log10 transformed, and then median aggregated across used windows.

For connectivity, continuous filtered signals receive an alpha bandpass followed by the Hilbert transform. Log envelope power is averaged across each node's electrodes. Pearson correlation of those regional log envelopes is computed separately inside each accepted window, never across the whole recording. The fixed edge order is `(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)`. Edge medians and the 75th-minus-25th percentile IQRs become the final 12 features. This is sensor-level statistical connectivity, not source-localized or causal connectivity. It is not orthogonalized; shared references and volume conduction remain limitations. Window summaries preserve variability but not full temporal order.

Accepted windows are pooled across a subject's recordings in `recording_id` order. If the pool exceeds `max_windows`, windows are selected at deterministic evenly spaced indices. Powers use the median across used windows. The archive records accepted and used window counts, the subject median and IQR connectivity matrices, and the used-window edge correlations together with their subject and recording identifiers. `feature_schema.json` describes every feature's name, source node or edge, units, and aggregation. The configured feature order must not be changed without rebuilding features and saved model bundles.

Integration validation does not change the scientific feature definitions or settings. Schemas are archive-specific (`feature_schema.json` for the default `subject_features.npz`, otherwise `<archive-stem>.feature_schema.json`). Ordered definitions are checked against the configuration; graph values and window summaries must agree with the feature columns. Optional legacy arrays may be absent, but inconsistent supplied handoff data is rejected before saving or evaluation. These are software-integrity checks, not evidence of clinical validity.


## Quantum map

Training-only standardized features are bounded with `(pi/2)*tanh(z/2)`. Each qubit corresponds to one configured node. Start with Hadamards. For each of two passes:

1. Apply node delta RY and theta RZ rotations.
2. Apply pairwise IsingZZ rotations using connectivity medians.
3. Apply node alpha RY and beta RZ rotations.
4. Apply pairwise IsingZZ rotations using connectivity IQRs.

The map is fixed, not a variationally trained QNN. The learned classifier is an SVM. Input-dependent interactions and interleaved noncommuting rotations avoid the simple cancellation caused by appending the same terminal unitary to every feature state.

The fidelity kernel is `abs(<psi(x)|psi(y)>)**2`, a positive-semidefinite density-matrix inner product in exact simulation. Four qubits have 16 amplitudes. This small model is classically tractable and does not demonstrate quantum computational advantage.

## Comparisons

- RBF-SVM on the same bounded inputs.
- Amplitude-fidelity kernel: append a constant 1, zero-pad to a power of two, normalize, and square the overlap. Its direct classical expression is `(x.y + 1)^2 / ((||x||^2+1)(||y||^2+1))`. Four-node inputs require 5 amplitude-encoding qubits if explicitly prepared; the graph circuit uses 4. This baseline is paper-inspired, not a reproduction of the paper's original feature pipeline or constant-padding choice.
- Proposed graph-conditioned quantum kernel.
- The same circuit with a fixed seeded permutation of edge inputs across qubit pairs. The permutation is shared by all subjects and both edge statistics. Node assignments remain fixed. Gate counts and each subject's angle multiset are preserved. For two nodes there is only one edge, so this control is identical and cannot test correspondence.

## Evaluation

Fixed-setting stratified subject-wise cross-validation is used. Fold count is reduced to the minority-class subject count if necessary; at least two subjects per class are required. All sessions are grouped through one row per subject. The same partitions, preprocessing features, SVM C, and class weighting apply across methods. Standardization is refitted within each fold. No hyperparameter search is implemented in the starter. Any future tuning requires an inner training-only validation loop.

Classification metrics are calculated from all out-of-fold labels. AUC is calculated within each fold and averaged. The initial protocol supplies no confidence intervals, label-permutation significance tests, noise/hardware experiments, or cross-dataset validation; the report must not claim those experiments. Repeated interactive evaluation does not turn test folds into a validation set for choosing a winning configuration.

## Explanation

Connection sensitivity replaces the median and IQR of one edge with raw-feature medians from that model's training subjects and recomputes its decision score. This is an input perturbation check; correlated-feature changes can be off-distribution. It is not a causal biomarker.

## Reference and distinction

Saranya and Menaka (2025), IEEE Access, DOI: 10.1109/ACCESS.2025.3531979. Their paper uses wavelet/filter-bank CSP and engineered features from a private 20-child cohort. The current regional-envelope graph representation and circuit are an adaptation. Their reported accuracy must never be attributed to this code or another dataset. The proposed contribution is a controlled investigation of graph-conditioned encoding, with novelty and benefit to be assessed empirically.
