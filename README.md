# NeuroQ-Connect

Shared foundation for Group 8's EEG autism-classification project. A light Streamlit workspace connects verified recordings, regional connectivity features, quantum kernels, subject-wise evaluation, and saved-model inference.

**Status:** runnable starter implementation. No verified KAU/TUOS cohort or clinical performance is included. The optional synthetic check verifies software only and is labelled separately throughout the interface.

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

1. Sahil fills `data/metadata/recordings.csv` using authoritative labels. Keep recordings outside Git.
2. Sahil and Aryan update `configs/default.yaml` for the verified montage, reference, and recording protocol. Default F3/F4/P3/P4 nodes are a starting configuration, not a confirmed dataset specification.
3. Validate, extract, and evaluate:

```powershell
neuroq validate --manifest data/metadata/recordings.csv
neuroq features --manifest data/metadata/recordings.csv --config configs/default.yaml --output artifacts/features/subject_features.npz
neuroq evaluate --features artifacts/features/subject_features.npz --output results
```

Each evaluation creates a new run folder with configuration, subject folds, predictions, metrics, and per-fold model bundles. Reruns do not overwrite earlier results.

```powershell
neuroq predict --manifest data/metadata/recordings.csv --subject SUBJECT_ID --bundle results/RUN_ID/fold_0/quantum.joblib --output artifacts/prediction.json
```

By default prediction requires a subject held out in the selected fold. Use `--allow-new-subject` only for genuinely unseen subjects; training subjects remain blocked. Scores are SVM decision scores, not calibrated probabilities.

## Implemented baseline protocol

- Continuous NPZ, EEGLAB SET, EDF, BDF, and FIF readers. SET companion files must remain alongside their recording.
- Fixed filtering/reference, non-overlapping windows, amplitude/flat-signal checks, and explicit channel matching.
- Four powers per node, plus median and IQR of alpha-envelope connectivity per edge. Four nodes produce 28 features. Every archive writes a validated `feature_schema.json` alongside subject median/IQR graphs and used-window edge correlations. See [the feature handoff](docs/FEATURES.md) for shapes and ordering.
- Training-fold-only standardization and bounded angle encoding.
- Feature-matched RBF-SVM, amplitude-fidelity SVM, graph-conditioned quantum kernel, and fixed edge-reassignment control.
- Deterministic, shallow PennyLane statevector simulation. No physical quantum hardware or quantum speedup is claimed.
- Fixed-setting stratified subject-wise cross-validation, fold AUC, out-of-fold classification metrics, and complete inference bundles.
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
