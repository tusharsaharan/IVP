# SFARI resting-state acquisition and Tushar's integration handoff

## Scope

TUOS is the primary experiment. This optional independent workflow uses SFARI `ds006780`, snapshot `1.0.0`, DOI <https://doi.org/10.18112/openneuro.ds006780.v1.0.0>. KAU is not required. The dataset describes ASD, typically developing (TD), and sibling groups with diagnostic provenance. Only explicit ASD/TD participants with resting-state availability enter this comparison. Siblings and unknown groups are not controls. Real recording acquisition, final QC counts, and held-out verification remain prerequisites for SFARI research claims.

The inspected snapshot lists 103 resting-state candidates (63 ASD, 40 TD), 528 BDF runs, and approximately 4.15 GB of EEG. Final counts come from the saved QC summary, not these pre-QC totals. No metrics in the software tests are recorded-data research results.

The November 10 local 20-window audit retains 100 subjects (60 ASD, 40 Control)
and **448** runs, with a compatible `[100, 28]` feature archive but no completed
recorded-data evaluation. Sahil separately reports **438** runs under an 18-window
handoff. Those counts are not interchangeable; see `STATUS.md` and `README_SAHIL.md`.

## Acquire recordings

Run commands from the repository root after installation. Use an existing **unmodified metadata clone** with annex pointers, or create one:

```powershell
git clone --branch 1.0.0 https://github.com/OpenNeuroDatasets/ds006780.git data/local/sfari-metadata
```

On Windows, the annex entries are small text pointers until their actual payloads are fetched. This workflow reads the pointer SHA-256/size without requiring git-annex. It never overwrites the metadata clone.

Preview the selection and byte count, then download:

```powershell
neuroq prepare-sfari --source data/local/sfari-metadata --output data/local/sfari
neuroq prepare-sfari --source data/local/sfari-metadata --output data/local/sfari --download --workers 4
```

`--source` can instead point to the team's existing `ds006780` metadata folder. Only resting-state BDF payloads for eligible participants are fetched from public OpenNeuro storage. Files are published only after their byte counts and SHA-256 match the snapshot's annex keys. Rerunning skips verified files and resumes partial transfers. A mismatching existing completed file is never overwritten. Download failures retain an audit and do not publish a candidate manifest.

Local outputs include:

- `candidates.csv`: one row per acquired recording, explicit source-based diagnosis, original participant grouping, relative paths.
- `eligibility_exclusions.csv`: excluded siblings, unknown groups, and unavailable rest.
- `download_report.json`: each payload's size, SHA-256, and transfer status.
- `source_provenance.json`: snapshot identity, selection policy, and metadata hashes.
- BDF files and their BIDS channel/event metadata, with original names.

These outputs are under ignored `data/local/`. Do not commit raw data, private paths, participant lists, bundles, or generated results.

## Screen before evaluation

```powershell
neuroq screen --manifest data/local/sfari/candidates.csv --config configs/sfari_rest.yaml --output data/local/sfari/qc
```

The command returns the new screened manifest path. Each invocation creates a unique `screen-...` directory with `recordings.csv`, full `recording_qc.json`, an inclusion/error CSV, and `screen_summary.json`. Selection uses the shared fixed recording QC, not labels or model performance. Subjects with no surviving recording are listed explicitly. If fewer than two subjects remain in either class, evaluation must stop; the failed audit remains available.

The commands here use `configs/sfari_rest.yaml`: F3/F4/P3/P4, 60 Hz mains,
and at most 20 feature-valid windows per subject. Sahil's `configs/sfari.yaml`
retains an 18-window alternative; it differs only in that cap and has a different
configuration ID. To reproduce it, use that config for screening/activation and
separate `artifacts/sfari-18/` and `results/sfari-18/` paths. Neither cap enforces
an exact accepted duration. TUOS uses different nodes and 50 Hz. Do not reuse
archives or bundles across those configurations or incompatible dataset IDs.
The loader honors ordered BIDS channel names/types and bad-channel flags; external
EMG and triggers do not enter EEG reference/QC/features. No guessed map is used.

## Activate, train, and demonstrate

Replace `SCREEN_ID` below with the directory returned by screening:

```powershell
neuroq use-workspace --manifest data/local/sfari/qc/SCREEN_ID/recordings.csv --config configs/sfari_rest.yaml --features artifacts/sfari/subject_features.npz --results results/sfari
neuroq validate
neuroq features
neuroq evaluate
python -m streamlit run app/main.py
```

Activation writes ignored `data/local/workspace.json` and overrides the TUOS CLI/application defaults until another profile is activated. A fresh application session and CLI defaults now agree on the manifest, config, features, and separate SFARI results directory. Existing Streamlit sessions should be restarted after changing profiles. Explicit command arguments still work; no profile is required for another verified dataset. Archive compatibility checks block stale configuration/cohort/label/session selections. They do not replace raw-file hash verification. To return to TUOS, use the activation command in the main README.

In the application:

1. **Datasets:** inspect the checked manifest, load a real recording, and inspect channels/signal.
2. **Experiments:** open the complete saved four-model result; review cohort counts and fold AUC definitions.
3. **Analysis:** choose a participant. The app automatically chooses that subject's held-out fold bundle.
4. Inspect connectivity, actual circuit gate angles, decision score, and training-reference perturbation sensitivity; export JSON and figures.

For the equivalent terminal prediction, replace `RUN_ID`, `SUBJECT_ID`, and `FOLD_INDEX` with entries from the saved `folds.csv`:

```powershell
neuroq predict --subject SUBJECT_ID --bundle results/sfari/RUN_ID/fold_FOLD_INDEX/quantum.joblib --output artifacts/sfari/prediction.json
```

Training subjects remain blocked. Decision scores are not calibrated probabilities or medical diagnoses. The recorded experiment is distinct from `neuroq self-check` and its synthetic artifacts.

## Ownership and freeze

- **Tushar:** acquisition orchestration, local profile, CLI/application integration, archive compatibility, held-out demo and exports.
- **Sahil:** review eligibility/label provenance, hardware/reference metadata, raw signal/QC, and exact exclusions.
- **Aryan:** review real feature/window/graph artifacts using the unchanged 28-column order.
- **Vivek:** review node/circuit assumptions and restored model reference ordering.
- **Mahavir:** verify saved shared folds and recompute all metrics from OOF predictions.
- **Shivam:** review/export real plots and perturbation captions.
- **Pranav:** reconcile final cohort counts, configuration/run IDs, limitations, and reproduction instructions across submissions.

Freeze dataset/QC/model settings before reviewing scores. Basic fixed QC, possible age/sex and duration imbalance, reference/volume-conduction effects, and lack of external/clinical validation must remain explicit. No new dependency or bundle-schema change is required.

## Historical download helpers

The Bash scripts in `scripts/` are retained convenience tools for WSL/Linux.
The direct S3 sync does not pin the snapshot, verify annex SHA-256 values, restrict
ASD/TD eligibility, or generate the project's candidate/QC manifest. HBN downloads
do not establish binary ASD labels. They are **not substitutes** for the verified
`prepare-sfari`/`screen` workflow above. Do not run installers or download both
datasets merely to reproduce this project; HBN remains out of scope. Review disk
space, versions, and source integrity before using optional scripts.
