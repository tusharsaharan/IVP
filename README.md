# NeuroQ-Connect

Shared foundation for Group 8's EEG autism-classification project. A light Streamlit workspace connects verified recordings, regional connectivity features, quantum kernels, subject-wise evaluation, and saved-model inference.

**Status:** integrated research software with a reproduced local SFARI four-model experiment and real held-out CLI/application agreement. Recorded accuracy is 46–50%; no clinical utility or quantum advantage is established. TUOS remains the primary protocol retained from Sahil's handoff, using F7/Fz/P1/O2, but its payloads/results are not reproduced here. SFARI resting-state EEG (`ds006780`) is a separate workflow; KAU is out of scope. Populated manifests, raw recordings, local profiles, model bundles, and generated results are excluded from Git. The synthetic check verifies software only. See [verified progress and remaining work](docs/STATUS.md), [Tushar's completed handoff](docs/TUSHAR_HANDOFF.md), and [independent installation verification](docs/REPRODUCTION.md).

## Install and launch

Python 3.11, 3.12, or 3.13 is required. Run from the repository root.

```powershell
git clone https://github.com/tusharsaharan/IVP.git
cd IVP
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m streamlit run app/main.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. If PowerShell activation is restricted, use `.venv\Scripts\python.exe` instead of `python`; activation is optional.

The application opens at http://localhost:8501. All recordings and generated artifacts stay on the local machine and are excluded from Git.

For the exact tested Python 3.13 environment, install with `python -m pip install -r requirements.lock.txt -e .`. Other supported Python versions use the compatible dependency ranges in `pyproject.toml`.

## First working software check

This generates clearly labelled synthetic EEG, extracts real numeric features, runs all four models, and verifies held-out inference. It does not produce research evidence.

```powershell
neuroq self-check
```

Outputs go to `artifacts/software-check/`. In the application, open **Software check** from the workspace. Real experiments use `results/` instead.

## Connect the team's real data

The primary TUOS handoff is documented in [README_SAHIL.md](README_SAHIL.md) and [the method notes](docs/METHOD.md). Sahil reports 45 eligible subjects (26 ASD, 19 Control); those counts and results require the original label/QC evidence and recording payloads before reproduction. Obtain permitted SET/FDT pairs under ignored `data/raw/tuos/` and create an authoritative manifest under `data/local/tuos/recordings.csv`. Resolve paths relative to that manifest. The tracked `data/metadata/recordings.csv` is deliberately header-only, not a ready-to-run cohort.

`configs/default.yaml` uses F7/Fz/P1/O2, 50 Hz mains, and an 18-window cap. Check saved window counts before claiming identical accepted duration. For another verified dataset, use a separate authoritative manifest and dataset-specific configuration agreed with Sahil and Aryan, not guessed labels or electrode equivalents.

SFARI is separate: [the acquisition and integration guide](docs/SFARI.md) downloads explicit ASD/TD resting-state recordings, verifies hashes, and saves QC exclusions. Both SFARI profiles use F3/F4/P3/P4 and 60 Hz: `configs/sfari_rest.yaml` retains Tushar's 20-window cap; `configs/sfari.yaml` retains Sahil's 18-window cap. They have different configuration IDs and must not share archives/bundles. HBN scores and unlabelled recordings are not substitutes for verified diagnoses.

Once TUOS recordings and their audited manifest are available, activate it first so
an existing SFARI profile cannot supply conflicting defaults:

```powershell
neuroq use-workspace --manifest data/local/tuos/recordings.csv --config configs/default.yaml --features artifacts/tuos/subject_features.npz --results results/tuos
neuroq validate
neuroq features
neuroq evaluate
```

The equivalent evaluation module entry point uses the same pipeline and default feature path:

```powershell
python -m asd_eeg.evaluation.run_experiments --features artifacts/tuos/subject_features.npz --output results/tuos
```

Each evaluation creates a new run folder with configuration, subject folds, predictions, metrics, and per-fold model bundles. Reruns do not overwrite earlier results.

```powershell
neuroq predict --subject SUBJECT_ID --bundle results/tuos/RUN_ID/fold_FOLD_INDEX/quantum.joblib --output artifacts/tuos/prediction.json
```

Use the run's `folds.csv` to select `FOLD_INDEX` for `SUBJECT_ID`. Prediction requires a subject held out in that fold. Use `--allow-new-subject` only for genuinely unseen subjects from the compatible dataset; training subjects remain blocked. Scores are SVM decision scores, not calibrated probabilities.

`neuroq use-workspace` writes the ignored `data/local/workspace.json`. Subsequent CLI commands and a fresh Streamlit session use its manifest, pipeline configuration, feature archive, and results folder. Explicit CLI path arguments override defaults. The evaluation module entry point remains a standalone archive runner; supply its feature/output arguments when using a custom workspace. The app and active CLI evaluation reject stale feature archives with a different dataset, configuration, cohort, labels, or recorded-session IDs.

If a local profile already selects SFARI, explicitly activate TUOS before launching the app. Keep dataset outputs separate:

```powershell
neuroq use-workspace --manifest data/local/tuos/recordings.csv --config configs/default.yaml --features artifacts/tuos/subject_features.npz --results results/tuos
```

Restart Streamlit after changing profiles. Activating a profile does not download recordings or train models.

## Implemented baseline protocol

- Continuous NPZ, EEGLAB SET, EDF, BDF, and FIF readers. SET companion files must remain alongside their recording.
- Fixed filtering/reference, non-overlapping windows, amplitude/flat-signal checks, and explicit channel matching.
- Four powers per node, plus median and IQR of alpha-envelope connectivity per edge. Four nodes produce 28 features. Every archive writes a validated schema alongside subject median/IQR graphs and used-window edge correlations. The default archive uses `feature_schema.json`; other filenames use `<archive-stem>.feature_schema.json` to avoid collisions. See [the feature handoff](docs/FEATURES.md) for shapes and ordering.
- Training-fold-only standardization and bounded angle encoding.
- Feature-matched RBF-SVM, amplitude-fidelity SVM, graph-conditioned quantum kernel, and fixed edge-reassignment control.
- Deterministic, shallow PennyLane statevector simulation. No physical quantum hardware or quantum speedup is claimed.
- Fixed-setting stratified subject-wise cross-validation, fold AUC, out-of-fold classification metrics, and complete inference bundles.
- Separate reproducible OOF-only Wilson/stratified-bootstrap summaries and exact paired McNemar comparisons, with explicit CV-dependence limitations; see [statistics](docs/STATISTICS.md).
- Held-out connection-perturbation sensitivity and JSON export.

This starter deliberately exposes the scientific decisions in configuration and [the method notes](docs/METHOD.md). It is not an exact reproduction of the reference paper. Changing the protocol after reviewing held-out results requires a new exploratory run and transparent reporting.

## Team ownership

| Member | Primary directories | Handoff |
| --- | --- | --- |
| Vivek G | `models/`, `training/` | Circuit, kernels, reproducible model bundles |
| Sahil Gupta | `data/`, `preprocessing/` | Verified manifest, montage, QC and usable recordings |
| Dodiya Mahavirsinh | `evaluation/`, `baselines/` | Frozen folds, comparisons, saved predictions |
| Aryan Pandey | `features/` | Feature matrix, feature schema, graph summaries |
| Tushar Saharan | `inference/`, `app/`, CLI | Shared pipeline, application and exports |
| Shivam Mall | `explainability/`, `visualization/` | Sensitivity checks, scientific plots and captions |
| Pranav Karhale | `tests/`, `docs/` | Reproducibility, report/PPT/AVI and submission checks |

Paths above are relative to `src/asd_eeg/` where applicable. See [team handoffs](docs/TEAM.md), [contracts](docs/CONTRACTS.md), and [contribution instructions](CONTRIBUTING.md).

## Verify a change

```powershell
python -m pytest
python -m ruff check .
```

GitHub Actions runs both checks on pushes and pull requests. Tests cover quantum-kernel mathematics, manifest integrity, feature extraction, subject isolation, artifact compatibility, CLI, and application startup.

## Reference

Saranya and Menaka (2025), *A Quantum-Based Machine Learning Approach for Autism Detection Using Common Spatial Patterns of EEG Signals*, IEEE Access. https://doi.org/10.1109/ACCESS.2025.3531979

The source paper uses its own pediatric dataset. Its reported accuracy is not a result of this repository.
