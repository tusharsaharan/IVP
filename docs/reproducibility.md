# Reproducibility guide

## Environment

Python 3.11, 3.12, or 3.13 is required. The locked environment pinned to Python 3.13 on Windows is in `requirements.lock.txt`. All other Python versions use the compatible dependency ranges declared in `pyproject.toml`.

Create and activate a virtual environment before installing:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux activate with `source .venv/bin/activate`. If PowerShell activation is restricted, prefix commands with `.venv\Scripts\python.exe -m` and omit the activation step; activation is optional.

For the exact tested Python 3.13 environment:

```powershell
python -m pip install -r requirements.lock.txt -e .
```

## Software verification (synthetic only)

After installation, verify the software pipeline end-to-end using clearly labelled synthetic data:

```powershell
neuroq self-check
```

Outputs go to `artifacts/software-check/`. The check generates synthetic EEG, extracts features, runs all four models, verifies held-out inference, and exports a prediction. No real EEG data or research result is involved.

Run the full test suite:

```powershell
python -m pytest
```

Run the linter:

```powershell
python -m ruff check .
```

Both commands must pass before submission.

## Connecting real data

Sahil supplies the verified dataset. The workflow below applies once `data/metadata/recordings.csv` contains authoritative rows:

```powershell
neuroq validate --manifest data/metadata/recordings.csv
neuroq features --manifest data/metadata/recordings.csv --config configs/default.yaml --output artifacts/features/subject_features.npz
neuroq evaluate --features artifacts/features/subject_features.npz --output results
```

Each evaluation creates a new time-stamped run folder. Reruns do not overwrite earlier results.

## Reproducing a saved prediction

```powershell
neuroq predict --manifest data/metadata/recordings.csv --subject SUBJECT_ID --bundle results/RUN_ID/fold_0/quantum.joblib --output artifacts/prediction.json
```

The subject must be held out in the selected fold. Use `--allow-new-subject` only for genuinely new subjects not present in any training fold. Training subjects are always blocked.

The equivalent Python call:

```python
from asd_eeg.inference.pipeline import analyze_subject
result = analyze_subject("data/metadata/recordings.csv", subject_id, bundle_path)
```

The CLI and the pipeline function use the same code path and must produce identical `predicted_label` and `decision_score`.

## Recomputing metrics

All classification metrics in `results/RUN_ID/metrics.json` can be regenerated from `results/RUN_ID/oof_predictions.csv`:

```python
import pandas as pd
from asd_eeg.evaluation.metrics import summarize_oof

oof = pd.read_csv("results/RUN_ID/oof_predictions.csv")
metrics = summarize_oof(oof)
```

`summarize_oof` rejects duplicates, conflicting labels, nonfinite scores, and differing model cohorts before computing. The result must equal `metrics.json` within floating-point tolerance. Wall-clock runtime is not recomputable from the CSV.

> [!IMPORTANT]
> AUC is computed per fold and then averaged. Uncalibrated SVM margins from separate folds do not share a scale; pooling them into a single ROC curve produces an incorrect result.

## Verifying a fresh installation

Pranav performs this check before submission:

1. Create a new directory outside the repository.
2. Extract the submission archive into it.
3. Create a new virtual environment and install:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -e ".[dev]"
   ```
4. Run `python -m pytest` and `python -m ruff check .`.
5. Run `neuroq self-check` and confirm it prints `"status": "passed"`.
6. If real data is available, run the full pipeline from `validate` through `evaluate`.
7. Launch `python -m streamlit run app/main.py` and verify the application opens.
8. Confirm that CLI and app inference return the same decision score for a held-out subject.

Report any failure with the exact command, full error message, and Python version.

## Configuration reproducibility

The configuration is serialised into every run folder as `config.json` and embedded in every model bundle. Its SHA-256 hash (`config_id`) is recorded in `summary.json`, `folds.csv`, and each bundle's metadata. Features built from a different configuration produce a different hash and are incompatible with bundles from a prior configuration.

Changing any configuration setting — node list, band definitions, preprocessing parameters, circuit passes, SVM C, number of folds, or random seed — requires regenerating features, re-running evaluation, and saving new bundles. Existing bundles must not be used with a modified configuration.

## Git provenance

Each run records the Git commit of the installed source checkout in `summary.json` under `git_commit`. This is determined from the package source tree, not the caller's working directory. Non-Git installations record `"unknown"`. Use the commit to identify exactly which version of the code produced a result.

## What is not recomputable

- Wall-clock runtime recorded per model in `metrics.json`.
- Figures generated from real EEG: re-run `neuroq features` and re-plot with Shivam's visualization functions.
- The AVI demonstration recording.

## Scientific limitations

- Stratified subject-wise cross-validation with fixed settings. No hyperparameter search is implemented.
- AUC is averaged across folds; fold-to-fold standard deviation is descriptive, not a confidence interval.
- No significance testing, confidence intervals, or cross-dataset validation is performed.
- The quantum circuit is simulated classically with PennyLane statevectors. No physical quantum hardware is used. No quantum speedup is claimed.
- Perturbation sensitivity measures model sensitivity to controlled feature substitutions. It does not establish causal biomarkers.
- Results depend on the dataset, cohort, and preprocessing choices frozen before evaluation. Changing protocol after reviewing held-out results requires a new exploratory run and transparent reporting.

