# Conditional uncertainty from frozen OOF predictions

Owner handoff: Dodiya Mahavirsinh (evaluation) and Pranav Karhale (verification).
This separate analysis does not change the evaluation runner, bundles, shared
contracts, labels, folds, feature definitions, or dependencies. Its test fixtures
are synthetic software checks, not recorded-data evidence. No numerical research
results are supplied here.

## Run locally after evaluation finishes

Use the **individual complete run directory**, not the parent results directory:

```powershell
python -m asd_eeg.evaluation.statistics --run results/sfari/RUN_ID --output results/sfari/RUN_ID/statistics-seed42-r10000.json --seed 42 --replicates 10000 --confidence 0.95
```

Replace the example run path with the actual frozen run; this does not select or
change a dataset-specific configuration. TUOS and the two distinct SFARI window
caps remain separate experiments. The command needs no raw EEG, network access,
credentials, or retraining. Keep generated output local/ignored like the run.

`--run` and `--output` are required. Output is one **new JSON file**, not a
directory. Existing files or directories are refused, including on a write race;
there is no overwrite switch. Select a different filename for another analysis.
The CLI exits with status 1 and an error on invalid/incomplete inputs. Defaults
are seed 42, 10,000 replicates, and confidence 0.95; specify them explicitly in
reported commands. The API requires explicit `seed` and `replicates`, with at
least two replicates allowed for tests (not a recommended research budget).

## Required inputs and validation

The command reads snapshots of `summary.json`, `config.json`, `folds.csv`,
`oof_predictions.csv`, `metrics.json`, `fold_auc.csv`, and
`model_comparison.csv`. It requires:

- `summary.json` with schema 1 and `status: complete`, a consistent configuration
  hash, seed, requested/actual fold counts, subject count and class counts, plus
  dataset, data kind, run ID, feature fingerprint and source commit provenance.
- Every currently configured comparison model, with one prediction per subject
  per model and identical cohorts, true labels, and held-out fold assignments.
  The authoritative `summarize_oof` validation is reused, not replaced. Missing,
  duplicate, conflicting, nonbinary, nonfinite, blank-ID and mismatched rows fail.
- `folds.csv` must agree with the predictions and reproduce
  `make_subject_folds` using the original CSV subject order and saved settings.
  There must be at least two subjects per class. Keep the original fold table;
  reordering its rows can change the replay of the seeded partition.
- Saved classification and descriptive within-fold AUC metrics must regenerate
  from OOF. Fold AUC/model-comparison rows cannot be missing or duplicated. JSON
  numbers must be finite, numeric table values must be finite, and duplicate
  JSON keys/CSV headers are refused. The only allowed blank numeric entries are
  independently regenerated undefined single-class-fold AUC values/summaries.
- Every expected per-model/per-fold bundle file must exist and be nonempty.
  **Bundles are not deserialized, hashed or inspected.** Existence does not prove
  valid training references, training-only fitting or absence of leakage; those
  remain the runner's invariants and separate integrity tests. This tool checks
  consistency of the saved artifacts, not the truth of their provenance.

The JSON records SHA-256 hashes of the exact source tables/configuration,
analysis/validator/split source hashes, package/Python versions, run metadata,
bootstrap settings, and limitations. It does not copy subject IDs, raw scores,
EEG or local source paths into the result. Artifact hashes are reproducibility
identifiers, not authentication. Repeating the command to different filenames
with identical snapshots, source, versions and settings produces identical JSON
bytes; changing CSV row order changes its file hash even when estimates agree.
Inputs should be frozen during analysis, not modified concurrently.

## Subject-stratified bootstrap

The unit is a subject's **whole matched vector of OOF predicted labels across
models**. All sessions/windows have already contributed to that one subject's
features; they are not independent observations. IDs are sorted lexicographically
for a row-order-independent in-memory calculation. Models are sorted by name.

For each replicate, using `numpy.random.Generator(PCG64(seed))`:

1. Draw `n_control` subject indices with replacement within true label 0.
2. Draw `n_ASD` indices with replacement within true label 1.
3. Use the exact same two draws for every model and every paired difference.
4. Recompute accuracy, sensitivity (`TP/n_ASD`), specificity (`TN/n_control`),
   and balanced accuracy (`(sensitivity + specificity)/2`).

No model is refitted; predicted labels and held-out assignments remain fixed.
Class counts are preserved, not sampled. Accuracy therefore uses the **observed
cohort class proportions**, not an estimate of real-world prevalence. Each
reported estimate is the original full-cohort value, not the bootstrap mean.
Two-sided percentile limits are the `(1-confidence)/2` and
`1-(1-confidence)/2` sample quantiles, using NumPy's explicit `method="linear"`.
Limits are **pointwise**, not simultaneous or multiplicity-adjusted. The paired
limits use differences within each shared replicate (`model_a - model_b`), not
subtraction of independently computed interval endpoints.

The sampling assumption is exchangeable subjects within each true-label class,
conditional on this frozen cohort and predictions. Site/family/other clustering,
selection/QC uncertainty, demographics, diagnostic uncertainty and prevalence
variation are not represented. Small samples can yield discrete, unstable or
degenerate limits; more replicates reduce Monte Carlo error, not sample-size or
design limitations. A confidence interval crossing chance, or two intervals
overlapping, is not a formal demonstration of equivalence or lack of a difference.

## Wilson score intervals

For a binomial proportion `p=k/n`, let `z` be the standard-normal upper quantile
at tail probability `(1-confidence)/2`. The Wilson interval (no continuity
correction) has center and radius:

```text
center = (p + z^2/(2*n)) / (1 + z^2/n)
radius = z * sqrt(p*(1-p)/n + z^2/(4*n^2)) / (1 + z^2/n)
limits = [center - radius, center + radius], bounded to [0, 1]
```

Denominators are explicit in JSON:

| Outcome | Successes | Trials |
| --- | --- | --- |
| Accuracy | TP + TN | All subjects |
| Sensitivity (ASD = 1) | TP | TP + FN = ASD subjects |
| Specificity (Control = 0) | TN | TN + FP = Control subjects |

Balanced accuracy is not a single binomial proportion and receives **no Wilson
interval**. Wilson's binomial independence assumptions are not made true simply
by calling predictions OOF; see the CV dependence caveat below.

## Exact paired McNemar and multiplicity

Every unordered model pair is tested once, on the same matched subjects. Let
`b` count subjects where A is correct and B wrong; let `c` count A wrong and B
correct. Conditional on the number of discordant subjects `d=b+c`, the null of
equal marginal correctness uses `Binomial(d, 0.5)`. The reported raw p-value is
SciPy `binomtest(b, d, p=0.5, alternative="two-sided")`, not an asymptotic
chi-squared approximation or continuity-corrected test. If `d=0`, p is 1.
Both-correct, both-wrong and both discordant counts are saved.

McNemar concerns **accuracy/correctness only**; it does not test balanced accuracy,
class-specific rates, decision scores or AUC. The exact calculation is exact
under its independent matched-subject binomial null, **not a guarantee of valid
type-I error for dependent cross-validation outputs**.

For `m` pairs, Holm step-down controls the family-wise error under valid marginal
p-values: sort `p_(1) <= ... <= p_(m)`, calculate
`min(1, max_{j<=i} ((m-j+1)*p_(j)))`, then restore pair order. The default four
models yield six tests. Raw and Holm-adjusted p-values, family size and outcome
are retained; no pair is selected after inspecting results. Correction covers
only this analysis's pair family, not other outcomes, earlier runs or protocol
exploration. It cannot repair invalid marginal p-values due to CV dependence.
Nonsignificance is **not equivalence**; no equivalence margin or test is supplied.

## Scientific interpretation limits

Each subject was held out from its own model fit, but CV models have **overlapping
training subjects**. As a result, errors across subjects/folds may be dependent.
These bootstrap/Wilson limits and exact McNemar calculations summarize the
**fixed OOF predictions conditionally**, not the full sampling uncertainty of a
learning algorithm trained on a new cohort. Nominal coverage and type-I error are
not guaranteed for this CV design. Treat p-values as qualified exploratory/
descriptive comparisons, not stand-alone confirmatory evidence. There is no
repeated training/bootstrap, nested tuning, label permutation, independent test
set or external validation here.

No pooled cross-fold SVM-score or AUC bootstrap is performed. Scores from separate
fold fits are not calibrated probabilities or necessarily on a shared scale.
The runner's mean within-fold AUC and population fold SD remain descriptive
outputs; fold SD is not a confidence interval and folds are not independent
replicate experiments. This module deliberately supplies no AUC uncertainty.

Freeze QC, eligibility, montage, window cap, feature definitions, model settings
and folds **before** reading held-out results. Decisions changed after reading
performance require a new transparently labelled exploratory run. Within-cohort
CV, even on recorded SFARI data, does not establish TUOS-to-SFARI transfer,
clinical utility, causal connectivity or quantum advantage. Results from
synthetic checks must never be mixed with recorded-data tables.

This document records the assumptions of the new optional module. The statement
in `METHOD.md` that the **shared runner** itself does not implement these tests
remains accurate; do not retrospectively attach this tool's output to historical
unverified tables or claim their inherited intervals have been reproduced.

## Verification

```powershell
python -m pytest tests/test_statistics.py tests/test_evaluation_handoffs.py tests/test_evaluation_integrity.py
python -m ruff check src/asd_eeg/evaluation/statistics.py tests/test_statistics.py
```

Tests check known Wilson limits and class denominators, deterministic shared
stratified resampling against an independent calculation, paired differences,
exact binomial p-values, Holm ordering/ties, identical predictions, strict OOF
and complete-run rejection, score-offset invariance, provenance hashes, exclusive
output writing, and the module CLI. Full-suite checks remain the integration
owner's responsibility. Independent/external recorded-data validation remains
future scientific work, not a passing software test.
