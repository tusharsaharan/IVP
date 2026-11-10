# Tushar Saharan: completed integration and application handoff

**November 10, 2026.** Tushar's software/integration acceptance workflow is
demonstrated on the available local SFARI data. This is not approval on behalf of
other owners, reproduction of unavailable TUOS payloads, or a clinical validation.

## Delivered and verified

- Hash-verified acquisition of explicit ASD/TD resting-state recordings from
  `ds006780` v1.0.0; all 528 payloads and preserved metadata rehashed.
- Explicit participant/source label mapping checked, candidate payloads reconciled,
  and no identical payload hash crossing subjects found.
- Repeated label-independent recording QC: 448 recordings, 100 subjects, 60 ASD and
  40 Control, identical to the original Tushar audit.
- Rebuilt all 100-by-28 features from raw recordings; compatible source/config/schema,
  graphs and used windows (3–20) retained. Numerical features match the original
  archive within 1e-10; no pipeline/default change was made to improve scores.
- Fixed four-model evaluation on the same five subject folds; training-only scaler
  and feature references; 20 reloadable bundles preserving original reference order.
- Deterministic repeat: folds, OOF predictions/scores and fold AUC match.
- Two cases selected **before** evaluation, one per class, match raw inference to OOF
  across all four models. AppTest verifies actual quantum inference, six finite edge
  perturbations, actual circuit PNG and strict JSON. CLI independently matches OOF.
- Safe active-cohort/config/**complete session membership** run selection; damaged results are rejected instead of
  displayed; inference does not infer labels from filenames or use training subjects.
- Persistent application sensitivity/circuit outputs; analysis JSON, prediction,
  metrics, folds, configuration, sensitivity CSV and scientific PNG downloads.
- Independent fresh README and extracted-package installation, including one real
  installed-CLI held-out prediction using original permitted local data in place.

## Recorded result and its meaning

Authoritative local run: `results/run-20261110-045223-cfdc86/`.
Config: `7bcd185ff8336878`, `configs/sfari_rest.yaml`; five folds, seed 42.

| Model | Accuracy | Balanced accuracy | Mean fold AUC |
| --- | ---: | ---: | ---: |
| RBF | 48% | 45.42% | 0.4313 |
| Amplitude | 47% | 45.00% | 0.4271 |
| Quantum | 46% | 42.92% | 0.4813 |
| Reassigned | 50% | 47.08% | 0.4917 |

None beats 60% majority-class accuracy; balanced accuracies are all below 50%.
The software demonstration includes incorrect predictions rather than choosing a
successful case after inspecting scores. Scores are uncalibrated SVM margins, not
probabilities or diagnoses. No quantum speedup, advantage, biomarker or clinical
utility is established. Descriptive uncertainty/paired tests are in `STATISTICS.md`
and `METHOD.md`; their CV-dependence caveats must remain in submissions.

## Local evidence map (excluded from Git)

| Evidence | Location |
| --- | --- |
| Source/QC/feature/raw-case acceptance | `artifacts/tushar-acceptance-20261110/acceptance.json` |
| Explicit source-label and payload reconciliation | `artifacts/tushar-acceptance-20261110/candidate_provenance.json` |
| Preserved handoff comparison | `artifacts/tushar-acceptance-20261110/handoff_comparison.json` |
| Post-review bundle/order/OOF/repeated-run/AppTest checks | `artifacts/tushar-acceptance-20261110/integration_verification_reviewed.json` |
| Rebuilt archive/schema | `artifacts/tushar-acceptance-20261110/features/` |
| Actual scientific figures | `artifacts/tushar-acceptance-20261110/figures/` |
| OOF uncertainty with seed/replicate count | `artifacts/tushar-acceptance-20261110/statistics/` |
| Source-only submission snapshots and hashes | `artifacts/submission-20261110/` |
| Explicit-selection browser downloads/video verification | `artifacts/tushar-acceptance-20261110/provenance-demo-v4/browser_video_verification.json` |

Participant-level JSON/CSV, recordings, local paths, model bundles and generated
results remain local. Only aggregate results and generic reproduction instructions
belong in the public repository. No Kaggle credential or participant upload was needed.

## Reproduce the complete acceptance workflow

Install from README first. Obtain the allowed raw/source files using `SFARI.md`;
the Git repository intentionally cannot ship them. Keep the old screened manifest
and feature archive to compare with the repeated QC/rebuild. From the repository root:

```powershell
python scripts/run_recorded_acceptance.py --manifest data/local/sfari/candidates.csv --config configs/sfari_rest.yaml --source data/local/sfari --previous-manifest data/local/sfari/qc/ORIGINAL_SCREEN/recordings.csv --previous-features artifacts/features/subject_features.npz --output artifacts/NEW_ACCEPTANCE --results results
```

The command refuses an existing output directory, verifies source files, recreates
QC/features, and stops before evaluation if full configuration, dataset/data kind,
cohort, labels, recording membership or features differ. It writes a
protocol freeze and chooses cases before running models. It never edits the active
workspace profile or the old audits. A cohort mismatch requires scientific review,
not bypassing the check to improve performance.

```powershell
python -m asd_eeg.evaluation.statistics --run results/RUN_ID --output artifacts/NEW_STATS.json --seed 42 --replicates 10000
neuroq predict --subject SUBJECT_ID --bundle results/RUN_ID/fold_FOLD_INDEX/quantum.joblib --output artifacts/NEW_PREDICTION.json
python -m streamlit run app/main.py --browser.gatherUsageStats false
```

Use the recorded run's `folds.csv` for `SUBJECT_ID` and zero-based `FOLD_INDEX`.
Do not choose a different fold's model or pass an uploaded/untrusted joblib bundle.
The app selects the subject's correct held-out bundle automatically. Its fold label
is one-based for presentation; the bundle folder is zero-based.

`scripts/verify_recorded_integration.py` checks all saved bundles and raw-data app
cases. It explicitly uses supplied `--run`, `--manifest` and `--features` paths in its
AppTest workspace, including custom results directories; it does not change the
active profile. `--case` may be repeated.
Optional `--repeat-run` checks a deterministic rerun without interpreting its wall
clock time as reproducible. Existing verification output files are preserved.

To rehearse/record the actual laptop browser workflow, use a case JSON and protocol
freeze from the same acceptance directory. The preparer verifies them against OOF,
original feature fingerprint and session provenance, then writes a new **ignored**
private wrapper. Start the compatible local app first:

```powershell
python scripts/prepare_recorded_demo.py --run results/RUN_ID --manifest data/local/sfari/qc/SCREEN/recordings.csv --config configs/sfari_rest.yaml --features artifacts/ACCEPTANCE/features/subject_features.npz --case-json artifacts/ACCEPTANCE/case_0_quantum.json --freeze artifacts/ACCEPTANCE/protocol_freeze.json --output artifacts/NEW_DEMO
playwright-cli -s=tushar-acceptance run-code --filename=artifacts/NEW_DEMO/record_demo.private.js
```

Do not run the public JavaScript directly or copy the private wrapper into `scripts/`.
It explicitly selects paths/run/model/subject, checks actual OOF/metrics download
hashes and exported case provenance/features/score, and only then describes the case.
Browser recordings and downloads remain local; they are not included in public Git.

The completed post-review laptop demo is
`Demo AVI (after completion of Project)/Tushar_NeuroQ_Recorded_Demo_Verified_Laptop_20261110.avi`
with a same-stem MP4 fallback. Both are 1440×900, 64.17 seconds and 1,540 fully
decoded frames. The recording contains actual local study IDs/results visible in
the interface; handle it as local research material, not a public promotional asset.
FFmpeg decoding and sampled-frame inspection passed; native-player playback and
audio narration are not asserted. Original submission documents remain unchanged;
revision-suffixed Word/PPT/PDF copies and final package hashes are tracked locally.

## Remaining boundaries

1. Dataset/preprocessing owners must accept the source/QC protocol and resolve the
   older acquisition omissions. The two SFARI IDs/configurations remain distinct.
2. TUOS SET/FDT and source label/exclusion evidence are unavailable here; its old
   performance and uncertainty remain unverified.
3. Age/sex/duration imbalance, simple fixed QC, common-reference/volume-conduction
   effects, small cohort and no independent/external validation limit research claims.
4. Revised submission files need team/faculty content approval; software completion
   cannot supply approvals, invent strong findings or establish clinical utility.

See `STATUS.md` for the current submission checklist and `REPRODUCTION.md` for
fresh-install/archive evidence and platform limitations.
