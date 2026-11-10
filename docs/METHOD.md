# Implemented method, limitations, and reported handoffs

The algorithm descriptions below are checked against the integrated root package.
Numerical tables inherited from Sahil are **unverified historical claims**, not
completed results available in this checkout. Obtain their complete run artifacts
and regenerate metrics before submission. `STATUS.md` separates implementation
verification from scientific evidence; no clinical validity is established.

The November 10 completion sprint also reproduced **Tushar's independent SFARI
20-window experiment**, described below and in `TUSHAR_HANDOFF.md`. Its recorded
outputs are separate from every inherited TUOS/SFARI numerical claim.

## Baseline cohort (TUOS)

- **Dataset:** Continuous resting-state EEG from Dickinson, Jeste, & Milne (2022), *Cortex*, 148, 139–151 (DOI: `10.1016/j.cortex.2021.09.022`; Figshare DOI: `10.15131/shef.data.16840351`).
- **Reported eligible participants:** Sahil's ledger lists 45 subjects (26 ASD, 19 Control); TUOS payloads and reproduced feature/QC artifacts are not available in this checkout.
- **Reported exclusions:** 11 subjects (9 controls without a verified standard-electrode map; 1 ASD missing Fz; 1 ASD with insufficient clean signal). Obtain and reproduce the source-based exclusion audit.
- **Label provenance:** Sahil reports interpreting original EEGLAB comments/setnames against the dataset's group definitions, including corrections to disk numbering. That interpretation needs source verification. Neither the loader nor the feature extractor derives diagnoses from filenames/header prefixes: every manifest row must carry an explicit authoritative label source.
- **Scalp nodes:** 4 sensor-level sites: `F7` (Node 0), `Fz` (Node 1), `P1` (Node 2, adjacent to original P3), and `O2` (Node 3).
- **Edges:** 6 pairwise connections: `(F7, Fz)`, `(F7, P1)`, `(F7, O2)`, `(Fz, P1)`, `(Fz, O2)`, `(P1, O2)`.
- **Signal conditioning:** Load calibrated volts, select good EEG channels, apply common average reference, a 50.0 Hz notch ($Q = 30$), 1.0–40.0 Hz zero-phase fourth-order Butterworth filtering, then polyphase resampling to 128 Hz. QC rejects nonfinite, near-flat, and >300 microvolt peak-to-peak windows; each recording needs at least three accepted eight-second windows. This is fixed-rule QC, not ICA/ASR or a clinical cleaning protocol. Honor prior cleaning and bad-channel metadata.
- **Duration cap:** **At most 18** feature-valid eight-second non-overlapping windows per subject (at most 144 seconds), pooled across recordings in `recording_id` order and deterministically sampled at evenly spaced indices. Subjects with 3-17 usable windows remain eligible. Do not claim equal accepted duration without checking saved counts; the selected windows need not be consecutive.

## 28-dimensional feature representation

Extracted identically across all subjects and models:
1. **Regional band powers (16 features):** Welch PSD with at most 256 samples per segment. Integrate each half-open delta [1,4), theta [4,8), alpha [8,13), and beta [13,30) band relative to the integrated 1-30 Hz PSD. Average electrode powers within each node, log10-transform, then take the median across used windows. Order is node-major: $0_\delta, 0_\theta, 0_\alpha, 0_\beta, 1_\delta, \dots, 3_\beta$.
2. **Connectivity medians (6 features):** Alpha-band-filter continuous EEG, take Hilbert log envelope power, average across each node's electrodes, and compute Pearson correlations separately inside each used window. Aggregate medians in fixed edge order: $e_{01}, e_{02}, e_{03}, e_{12}, e_{13}, e_{23}$.
3. **Connectivity IQRs (6 features):** The 75th-minus-25th percentile of those same used-window edge correlations, in the same edge order.

Feature extraction is label-free and performs no population fitting. Archives retain
configuration/hash, feature names/schema, subject grouping, quality counts, median/IQR
graphs, and used-window edge values/recording IDs. Supplied graph fields must match
the feature columns and reproduce their aggregations. Schema sidecars are archive-
specific. Preserve feature and node/edge order at training and inference.

## Models compared under stratified 5-fold cross-validation

Every model receives **all 28 matching features**. A `StandardScaler` is fitted
on training subjects only, then angles are `(pi/2) * tanh(z/2)`. No held-out
subject enters scaling, fitting, or training-derived perturbation references.

1. **RBF-SVM:** Classical radial-basis SVM on the same bounded 28-feature angles.
2. **Amplitude-fidelity SVM:** Append a constant 1, zero-pad to a power of two,
   normalize, and square the state overlap. Its exact classical expression is
   `(x.y + 1)^2 / ((||x||^2+1)(||y||^2+1))`. This is not an RBF or a power-only
   baseline. An explicit 28-feature amplitude state needs five qubits.
3. **Graph-conditioned quantum kernel:** Four Hadamard-initialized qubits, one per
   configured scalp node. Each of two fixed passes applies node delta RY/theta RZ,
   edge-median IsingZZ, node alpha RY/beta RZ, then edge-IQR IsingZZ gates. Circuit
   bandwidth scales the bounded angles. The kernel is
   `abs(<psi(x)|psi(x')>)**2`, a PSD density-matrix inner product in exact simulation.
   This is a fixed data-dependent re-uploading map, **not a trained variational QNN**.
4. **Reassigned-edge control:** The same circuit with one seeded permutation of
   edge inputs across all pairs, shared by every subject and both edge statistics.
   Node identity, gate count, and each subject's edge-angle multiset are preserved;
   no "non-adjacent" constraint is imposed. With only one edge the control is identical.
   The shipped seed (1729) gives permutation `[0, 1, 2, 4, 5, 3]`: the three edges
   touching node 0 keep their own values and only edges (1,2), (1,3), (2,3) rotate,
   so the control perturbs half of the graph. It is pinned by a test and unchanged
   because frozen bundles depend on it; a full derangement would be a new experiment.
   See `MODEL.md` for the model-owner review.

Four qubits have 16 amplitudes and are classically tractable. Execution uses a
deterministic PennyLane statevector simulator, not physical quantum hardware;
no computational speedup is established. Kernel-SVM inference preserves the
**full original training-reference column order**, not only support vectors.

Subject-wise stratified folds are generated once and shared by all four models.
Requested folds are five (seed 42), reduced to minority-class count if necessary;
at least two subjects per class are required. All sessions of a subject contribute
one subject row. SVM C and class weighting are fixed across methods. There is no
hyperparameter search; future tuning requires inner training-only validation.

Global out-of-fold labels regenerate classification metrics. AUC is computed within
each fold and averaged because independently fitted SVM margins need not share a
scale. Single-class fold AUC is undefined and excluded. Population SD (`ddof=0`)
describes fold spread, not uncertainty. A separate OOF-only postprocessor now provides
stratified subject bootstrap, Wilson intervals and exact paired McNemar calculations
with Holm correction (`python -m asd_eeg.evaluation.statistics`). It does not refit
models, bootstrap pooled cross-fold scores, or guarantee nominal inference when
overlapping CV training sets induce dependent errors; see `STATISTICS.md`. The shared
runner still does not perform label permutations, hardware/noise experiments, or
cross-dataset transfer. Record source commit, versions, configuration,
folds, predictions, and bundles; only `status: complete` runs are valid results.

## Reported historical TUOS findings (unverified)

The following table is inherited from Sahil's reported 18-window run, not regenerated
here. Its complete `summary.json`, `oof_predictions.csv`, bundles, and uncertainty
scripts must be obtained before accepting these numbers. Existing tests validate
software fixtures, not this table or clinical performance.

| Model | Accuracy (95% Wilson CI) | Balanced Accuracy (95% Bootstrap CI) | Macro F1 | Sensitivity | Specificity | Mean Fold AUC (95% Fold Bootstrap CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RBF-SVM** | 55.56% [41.2%, 69.1%] | 54.45% [42.7%, 66.9%] | 0.545 | 61.54% (16/26) | 47.37% (9/19) | 0.5467 [0.386, 0.709] |
| **Amplitude Kernel** | 55.56% [41.2%, 69.1%] | 53.74% [39.9%, 66.7%] | 0.537 | 65.38% (17/26) | 42.11% (8/19) | 0.4722 [0.310, 0.647] |
| **Proposed Quantum Kernel** | 48.89% [35.0%, 63.0%] | 47.98% [35.5%, 60.9%] | 0.480 | 53.85% (14/26) | 42.11% (8/19) | 0.3789 [0.208, 0.550] |
| **Reassigned Control** | 55.56% [41.2%, 69.1%] | 55.16% [42.0%, 70.0%] | 0.550 | 57.69% (15/26) | 52.63% (10/19) | 0.5833 [0.403, 0.776] |

### Interpretation limits for the reported table

1. **Reported comparison, not a universal conclusion:** The table reports 48.89% quantum accuracy and 47.98% balanced accuracy versus 55.56% and 54.45% for RBF. If reproduced, it shows no observed benefit in this fixed experiment; it does not prove that quantum methods universally lack an advantage.
2. **Confidence intervals span chance:** All 95% bootstrap confidence intervals for balanced accuracy and fold-level AUC cross the 0.50 chance threshold:
   - **Quantum Kernel:** Mean Fold AUC = 0.3789 (95% CI: [0.208, 0.550]); Balanced Acc = 0.4798 (95% CI: [0.355, 0.609])
   - **RBF-SVM:** Mean Fold AUC = 0.5467 (95% CI: [0.386, 0.709]); Balanced Acc = 0.5445 (95% CI: [0.427, 0.669])
   - **Amplitude Kernel:** Mean Fold AUC = 0.4722 (95% CI: [0.310, 0.647]); Balanced Acc = 0.5374 (95% CI: [0.399, 0.667])
   - **Reassigned Control:** Mean Fold AUC = 0.5833 (95% CI: [0.403, 0.776]); Balanced Acc = 0.5516 (95% CI: [0.420, 0.700])
3. **Unverified uncertainty/significance:** The reported intervals and McNemar claim ($p \ge 0.6476$) require source OOF predictions and separately reproducible statistical scripts. They are not outputs of the integrated runner. Overlap with chance or a nonsignificant test does not prove model equivalence.
4. **Sensor-level limitation:** Sensor-level alpha envelope correlation reflects functional statistical coupling between scalp measurement sites, but is subject to volume conduction and common reference effects. It must not be interpreted as anatomical or causal connectivity.
5. **Class balance:** With 26 ASD and 19 Control subjects, the majority-class baseline is $26/45 = 57.78\%$.

## Optional SFARI recorded-data workflow

TUOS remains primary. Tushar's SFARI acquisition, fixed-QC screening, and local workspace activation remain available as an independent workflow; they do not replace the TUOS manifest or defaults. See [SFARI.md](SFARI.md). KAU is out of scope.

Use snapshot DOI `10.18112/openneuro.ds006780.v1.0.0`. Eligibility comes from `participants.tsv` diagnostic group and resting-state availability, with definitions and provenance in `participants.json` and README. Map only explicit ASD=1 and TD=0; siblings and unknown groups are excluded, never treated as TD. Do not infer diagnoses from filenames or questionnaire scores. The inspected metadata initially lists 103 candidate participants (63 ASD, 40 TD) and 528 resting-state BDF runs; these are pre-QC counts, not final sample sizes.

Acquire all eligible resting-state BDF runs into a separate ignored directory and verify annex SHA-256 hashes and byte counts. Retain source metadata and provenance hashes. Metadata describes continuous BioSemi 64-channel EEG at 512 Hz, 60 Hz mains, and directly named F3/F4/P3/P4. `configs/sfari_rest.yaml` retains Tushar's 20-window protocol; `configs/sfari.yaml` retains Sahil's 18-window protocol. TUOS uses F7/Fz/P1/O2 and 18 windows. All retain the feature-order convention, training-only transforms, model settings, fold generator, and bundle schema, but different configurations/dataset IDs/cohorts do not have interchangeable archives or bundles.

MNE calibrates BDF to volts. Apply ordered BIDS channel types/status before selecting EEG so external EXG channels (EMG) and the Status trigger do not enter average reference or QC. Honor bad-channel metadata and fail on missing required nodes rather than substituting electrodes.

Screen each recording with the shared preprocessor before evaluation, retaining runs with at least three clean eight-second windows. Save all exclusions and participants with no surviving recordings. Pool sessions only within their original participant IDs and select at most the chosen configuration's 18 or 20 feature-valid windows. Do not relax QC or select tasks, subsets, or settings using held-out performance. Fixed-rule QC is not ICA or clinical artifact removal. Unequal duration, age/sex imbalance, reference effects, prior filtering, and differential attrition remain limitations. SFARI describes eyes-open rest, unlike TUOS eyes-closed recordings; separate within-cohort CV does not demonstrate cross-dataset transfer.

Activate a dataset-specific ignored local profile with separate feature/results paths. Active evaluation checks dataset/configuration/cohort/labels/session compatibility. Restart Streamlit after changing profiles. Raw EEG, local bundles, generated results, credentials, and personal paths remain outside Git. Real-data SFARI acquisition, final QC counts, and held-out verification must be completed and reviewed by Sahil and Vivek before making recorded-data claims.

## Integration decision

The November 10 integration retains Sahil's TUOS defaults and both distinct SFARI
protocols without tuning settings against scores. Regression tests check node maps,
mains frequencies, window caps, distinct configuration IDs, and identical remaining
model settings. Cross-drive paths now work in both screening and activation; uniquely
named QC output directories still refuse collisions rather than overwriting audits.
Populated incoming manifests, ledgers, and original notes are preserved under ignored
`data/local/branch-handoff-20261110/` locally; the public manifest is header-only.
Existing local profiles/raw EEG/features are not changed. Scientific algorithms,
labels, folds, dependencies, bundle schema, and training-reference order are unchanged.
Rebuild artifacts whenever configuration/cohort changes and never mix synthetic
software checks with recorded-data results. Historical personal paths remain in
existing Git history; no force-push/history rewrite is part of this integration.

## Reproduced Tushar SFARI experiment (November 10, 2026)

The separately named recorded run is `run-20261110-045223-cfdc86`, configuration
`7bcd185ff8336878` (`configs/sfari_rest.yaml`). All 528 acquired BDF payloads and
preserved source metadata were rehashed; explicit ASD/TD participant labels and
candidate-to-payload grouping reconciled. No identical payload hash crossed subject
IDs. Repeated fixed QC reproduced the same 448 included recordings and 100 subjects
(60 ASD, 40 Control). Rebuilding all raw-input features reproduced the existing
100-by-28 matrix within 1e-10. Used windows range from 3 to 20; accepted duration is
not equalized. No source, QC, model or fold setting was tuned after inspecting scores.

| Model | Accuracy | Balanced accuracy | Macro-F1 | Sensitivity | Specificity | Mean fold AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RBF-SVM | 0.4800 | 0.4542 | 0.4536 | 0.5833 | 0.3250 | 0.4313 |
| Amplitude fidelity | 0.4700 | 0.4500 | 0.4502 | 0.5500 | 0.3500 | 0.4271 |
| Graph quantum | 0.4600 | 0.4292 | 0.4270 | 0.5833 | 0.2750 | 0.4813 |
| Reassigned edges | 0.5000 | 0.4708 | 0.4694 | 0.6167 | 0.3250 | 0.4917 |

Five fixed stratified subject folds use seed 42; every subject has one OOF row per
model. Classification metrics regenerate from saved predictions, and AUC is averaged
within folds. A deterministic second run reproduced folds, predictions, scores and
fold AUC. All 20 saved bundles passed training-only scaler/reference and saved-OOF
checks. Two raw-input cases, chosen lexicographically within classes before evaluation,
matched saved OOF predictions/features across all four models. The quantum predictions
on those two cases were both incorrect; the demonstration does not hide those errors.

The 60% always-ASD accuracy baseline exceeds every model. All observed balanced
accuracies are below 0.50. This fixed small-cohort experiment establishes neither
useful discrimination nor a quantum benefit. A descriptive 10,000-replicate
stratified subject bootstrap (PCG64 seed 42, 95% pointwise percentile intervals) gives
balanced-accuracy intervals of [0.3583, 0.5500] for RBF, [0.3542, 0.5458] for amplitude,
[0.3375, 0.5250] for quantum and [0.3792, 0.5625] for reassigned edges. All six exact
paired McNemar comparisons have Holm-adjusted p=1.0. These intervals/tests are
conditional on frozen OOF predictions, neglect retraining and CV training dependence,
and do **not** demonstrate equivalence or clinical validity.

The preserved 438-recording handoff is a subset of the reproduced 448-recording
manifest, with identical subjects and labels. Ten additional recordings across three
subjects explain the retained-count difference; the older 517-candidate acquisition
audit is unavailable, so the cause of its omissions remains unresolved. The 18/20
subject window cap is not consulted by recording-level QC and cannot explain the
recording selection difference. Do not reuse these new metrics for the older
18-window protocol, harmonize dataset IDs silently, or claim TUOS reproduction.

## Reported historical SFARI benchmark (unverified)

Sahil reports a separate within-cohort SFARI evaluation using `configs/sfari.yaml`
(18-window cap). No supporting completed run is available here. The figures below
remain attributed historical claims, not verified transfer/clinical evidence.
The current local 20-window handoff instead has 528 candidate runs and 448
retained runs, versus Sahil's reported 517 and 438. Reconcile source acquisition,
QC exclusions, dataset identifiers, and configuration/run provenance before reporting
one final sample. Some inherited balanced accuracies differ from the arithmetic
mean of the stated sensitivity and specificity; regenerate all metrics from OOF.
- **Cohort Audit:** 136 total subjects audited. 33 subjects excluded (28 siblings per protocol, 4 incomplete resting state, 1 missing group).
- **Candidate Pool:** 103 subjects (63 ASD, 40 TD) across 517 resting-state BDF runs.
- **Fixed-QC Screening:** 3 subjects reported excluded for pervasive motion/amplitude artifacts across all runs. Participant identities and recording-level exclusions remain in ignored local evidence, not public documentation.
- **Screened Eligible Cohort:** Exactly **100 participants (60 ASD, 40 Control)** across 438 clean resting-state runs.
- **Preprocessing Adaptation:** 60.0 Hz notch filter (US powerline interference), 1.0–40.0 Hz bandpass, CAR reference, 128 Hz resampling, classical four-node montage (`F3, F4, P3, P4`, verified in BioSemi 64-channel array).
- **5-Fold Cross-Validation Results (100 Subjects):**
  - **RBF-SVM:** Accuracy = 50.0%, Balanced Acc = 48.73% (95% CI: [0.383, 0.592]), Mean Fold AUC = 0.4758 (95% CI: [0.352, 0.590])
  - **Amplitude Kernel:** Accuracy = 45.0%, Balanced Acc = 44.53% (95% CI: [0.346, 0.542]), Mean Fold AUC = 0.4157 (95% CI: [0.298, 0.531])
  - **Proposed Quantum Kernel:** Accuracy = 44.0%, Balanced Acc = 40.33% (95% CI: [0.317, 0.488]), Mean Fold AUC = 0.3687 (95% CI: [0.258, 0.485])
  - **Reassigned Control:** Accuracy = 45.0%, Balanced Acc = 41.78% (95% CI: [0.325, 0.512]), Mean Fold AUC = 0.3467 (95% CI: [0.229, 0.473])
- **Reported significance claim:** McNemar $p \ge 0.3323$ is unverified until source predictions and statistical scripts are supplied. Do not interpret the table as proof of montage invariance, cross-dataset generalization, model equivalence, or a clinical biomarker.

## Explanation

### Reproduction safeguards added after independent review

The acceptance workflow now compares the previous and rebuilt archives' **complete
configuration**, dataset/data kind, subject labels, label sources and recording
membership before evaluation. Unchanged numeric features alone cannot establish an
unchanged protocol: SVM, circuit and fold/seed changes may leave feature extraction
unchanged. These comparisons do not alter the frozen SFARI settings or saved run.

Saved-run application reproduction requires original whole-cohort session provenance,
including recordings beyond the window cap. Future summaries retain original feature
metadata; legacy runs need original acceptance/hash-bound archive evidence. Removing,
replacing or reassigning sessions is not silently presented as reproduction of OOF.
The integration verifier explicitly sets its supplied manifest/archive/results paths
and checks selected bundle paths, rather than assuming `results/RUN_ID`.

Public demo code contains no participant selection, truth or margin. Ignored local
settings bind its run/configuration/model/case to pre-evaluation evidence. Actual OOF
and metrics downloads are hash-checked and prediction exports checked before
case-specific captions. Participant-level details and original personal paths remain
local. None of these fixes is a new research experiment or score-driven configuration
change; source/run provenance remains attributed to its original creation.

Connection sensitivity replaces the median and IQR of one edge with raw-feature medians from that model's training subjects and recomputes its decision score. This is an input perturbation check; correlated-feature changes can be off-distribution. It is not a causal biomarker.

## Project-only workspace recovery follow-up (November 10, 2026)

CLI help and workspace-independent commands now parse before reading the local
profile, so `use-workspace` can repair a malformed profile. Data commands remain
fail-closed, and valid-profile defaults retain explicit argument precedence.
Activation rejects blank paths, validates recorded-data inputs/configuration, and
atomically saves the same four-field local profile. Failed validation or replacement
preserves previous settings. These are integration/recovery safeguards only:
no node map, filtering, feature definition/order, quantum circuit, classifier,
fold, seed, bundle schema, diagnosis label, or frozen recorded result changed.
The expanded CI matrix and self-check exercise software using generated fixtures;
they do not reproduce private recordings or resolve scientific owner approval.

## Reference and distinction

Saranya and Menaka (2025), IEEE Access, DOI: 10.1109/ACCESS.2025.3531979. Their paper uses wavelet/filter-bank CSP and engineered features from a private 20-child cohort. The current regional-envelope graph representation and circuit are an adaptation. Their reported accuracy must never be attributed to this code or another dataset. The proposed contribution is a controlled investigation of graph-conditioned encoding, with novelty and benefit to be assessed empirically.
