# Integration audit and remaining work

**Audit date:** November 10, 2026. **Conclusion:** the software modules are integrated
and tested; the recorded-data results and final submission are not yet verified.
Reported experiments on a teammate's machine are not assumed absent everywhere,
but their supporting artifacts were not available for this audit.

## Branch review

| Remote branch | Reviewed head | Contribution | Integration |
| --- | --- | --- | --- |
| `aryan` | `8f77df8` | Graph/window feature archive, schema, validation | Already in main through PR #1 |
| `mahavir` | `0543d8b` | Frozen subject folds, OOF-regenerable metrics, runner | Already in main through PR #2 |
| `shivam` | `f4c667c` | EEG/connectivity/kernel/result plots, perturbation | Already in main through PR #3 |
| `tushar/sfari-integration` | `55d0874` | Hash-verified SFARI acquisition, BIDS/QC, shared workspace | Already in main through PR #4 |
| `sahil` | `7d4d341` | TUOS configuration/handoff, SFARI alternative, scripts/docs | Integrated with reconciliations below |

The reviewed main baseline was `626ffae`. Only Sahil had commits not yet reachable
from it (11 commits). Existing local integration branches contain earlier handoffs
already reachable from these branches. No separate remote Vivek or Pranav branch
was present; their assigned responsibilities still need acceptance review.
Integration preserves merge ancestry and all feature branch tips; no force-push,
branch deletion, or history rewriting is required.

## Verified implementation

- One CLI/Streamlit inference function and the same evaluation runner.
- Explicit diagnosis manifests and subject/session grouping; recorded and synthetic
  data cannot mix. No runtime diagnosis inference from filenames.
- NPZ/SET/EDF/BDF/FIF loaders, volts calibration, bad-channel handling, and ordered
  BIDS channel type/status checks before EEG reference and QC.
- Label-free extraction of 28 ordered features, validated archive-specific schemas,
  median/IQR graphs, used-window edge values, and recording provenance.
- Training-only scaling, bounded angles, feature-matched RBF/amplitude/quantum/
  reassigned models, deterministic quantum simulation, and full train-reference order.
- Frozen shared subject folds, duplicate/cohort/fold/leakage validation, OOF metrics,
  fold-wise AUC, complete/failed run status, and reloadable inference bundles.
- Held-out inference guards, dataset/configuration/archive compatibility, training-
  derived edge sensitivity, plots, and JSON export.
- Empty-workspace Streamlit navigation and the labelled synthetic end-to-end check.

### Validation on the integrated tree

- Windows, Python **3.13.9**: **244 tests passed**.
- `python -m ruff check .`: clean.
- `neuroq self-check`: passed; generated outputs remain ignored and **synthetic**.
- Wheel: built, installed to a separate temporary target, imported from that target,
  and passed its synthetic self-check using existing runtime dependencies. This
  verifies packaged code, not a fully fresh dependency environment.
- Four Bash scripts: syntax checked with Git Bash, not installed/executed/downloaded.
- Existing active recorded feature archive: loaded and validated against its manifest
  and configuration after integration; existing workspace/profile was not modified.

These checks are not clinical validation, a full clean-environment installation, or
cross-platform results. CI is configured for Python 3.13 on Windows and Ubuntu;
confirm the new merge's workflow results separately. Python 3.11/3.12 installation
and the extracted submission archive still need explicit reproduction.

## Reconciliations included in the merge

1. Preserved TUOS defaults and **both** SFARI window protocols without silently
   changing a scientific setting:

   | Config | Nodes | Mains | Maximum windows | Config ID |
   | --- | --- | --- | --- | --- |
   | `configs/default.yaml` (TUOS) | F7/Fz/P1/O2 | 50 Hz | 18 | `7e45d19e662f0f49` |
   | `configs/sfari.yaml` (Sahil) | F3/F4/P3/P4 | 60 Hz | 18 | `3ee65dd70d7d1ed7` |
   | `configs/sfari_rest.yaml` (Tushar) | F3/F4/P3/P4 | 60 Hz | 20 | `7bcd185ff8336878` |

   These artifacts/configuration IDs are distinct. A maximum is **not** an exact
   minimum duration. Added a regression check for the two SFARI variants.
2. Extended incoming Windows cross-drive QC support to workspace activation through
   a shared path helper and tests, including UNC/different-drive failure handling.
   Kept unique QC directories collision-safe rather than allowing audit overwrite.
3. Kept populated manifests/ledgers and original notes as ignored local snapshots
   at `data/local/branch-handoff-20261110/` on the integrating machine. Restored a
   public header-only manifest and ignore rules. No raw EEG/models/results were added.
   Original personal paths still exist in branch history; a separate coordinated
   cleanup would be needed to remove historical disclosures. No credentials were
   identified by this review; this was not a comprehensive secret-history scan.
4. Corrected the method: amplitude is a full-feature fidelity kernel, not a 16-power
   RBF; the graph circuit uses interleaved RY/RZ/IsingZZ re-uploading, not only RZ;
   SVM scores are not probabilities; caps do not enforce exactly 18/20 windows.
5. Marked inherited result/statistical claims unverified, corrected reproduction
   commands/local paths and text encoding, and identified the nested
   `neuroq-connect/` directory as a historical standalone snapshot rather than the
   maintained package. It is retained, not deleted or installed over the root package.
6. Documented that direct S3/HBN convenience scripts do not replace hash-verified,
   snapshot-pinned, explicitly labelled acquisition and QC.

## Recorded-data evidence available here

| Item | Observed evidence | Limit |
| --- | --- | --- |
| Active SFARI source | Candidate/QC manifests and local payloads | Existing source audit needs owner acceptance |
| Active SFARI QC | 528 input recordings, 103 subjects; 448 included recordings, 100 subjects (60 ASD, 40 Control) | Different from Sahil's handoff |
| Active SFARI features | `[100, 28]`, config `7bcd185ff8336878`, compatible cohort/recording IDs | Used windows range **3-20**, not identical duration |
| Sahil TUOS handoff | 56 ledger rows; 45 eligible subjects (26 ASD, 19 Control) | No TUOS SET payloads in this checkout; labels/QC not reproduced |
| Sahil SFARI handoff | 136 ledger rows; 438 recordings for 100 subjects | Private WSL paths, different dataset ID/protocol; acquisition/QC discrepancy unresolved |
| Completed recorded experiments | **No `summary.json` under `results/`** at audit time | No available OOF/bundles to verify reported performance or run a real held-out demo |

Sahil's reported TUOS/SFARI metrics remain attributed historical claims in the
handoff/method notes. No bootstrap/Wilson/McNemar implementation or supporting
completed run was supplied here. Some reported SFARI balanced accuracies do not
match the stated sensitivity/specificity arithmetic. Do not print those metrics
as verified results, infer equivalence from nonsignificance, or confuse independent
within-cohort CV with zero-shot cross-dataset transfer. Current inference explicitly
guards dataset compatibility, and the node maps differ across cohorts.

## Remaining work, in priority order

| Priority | Owner | Required next handoff / acceptance |
| --- | --- | --- |
| Blocking | Sahil + Tushar + Vivek | Reconcile 528/448 versus 517/438 SFARI selection/QC, source snapshot and dataset IDs; approve one final recorded cohort/protocol before evaluation. Preserve both original audits. |
| Blocking for TUOS | Sahil | Supply permitted TUOS SET/FDT payloads, source group definitions/label mapping, prior-cleaning notes, and a reproducible 56-to-45 exclusion audit. Create and validate local paths; do not infer labels from numbering/prefixes alone. |
| Blocking | Aryan + Sahil | Reproduce final archive/schema/graphs and reconcile manifest/subject/session/window counts. State variable duration, or prospectively define and document a new equal-duration protocol without tuning against held-out scores. |
| Blocking | Mahavir + Vivek | Supply or run a complete four-model recorded-data evaluation on identical frozen folds; retain configuration/source IDs, OOF predictions, fold AUC, versions, timing, and trusted fold bundles. Regenerate every reported metric from OOF. |
| High | Mahavir + Pranav | Supply reproducible statistical scripts and seeds/replicate counts if confidence intervals or paired tests will be reported; otherwise remove unsupported claims from final submissions. |
| High | Vivek | Review actual circuit/encoding, PSD/equivalence/reassignment mathematics, train-reference ordering, and write the model section distinguishing this adaptation from the source paper. |
| High | Tushar + Shivam | Run a real held-out subject through both CLI and app, compare to saved OOF prediction, export actual circuit/connectivity/confusion/sensitivity figures and a case study with fold/run/config provenance. |
| Blocking for submission | Pranav + all section owners | Update report/PPT/SRS to the implemented method and verified cohort/results; record and verify a playable AVI, then extract the submission archive and reproduce installation/demo independently. |

## Submission-material mismatch

Local files outside Git were inspected for content, not edited:

- `PPT(in pptx formate)/ASD_EEG_Diagnosis_Group8_v5 (1).pptx`: 31-slide
  literature/design presentation still centered on KAU, CNN-LSTM, Grad-CAM, and
  planned experiments (especially slides 19-27).
- `SRS and Design Report/SRS_and_Design_Report (2) (4) (1).docx`: original
  KAU/CNN-LSTM design/requirements, not an implementation report for this repository.
- `Report (Springer Format)/ASD_EEG_SRS_Design_Report_Springer.pdf`: 23-page
  SRS/design proposal with KAU, CNN-LSTM/softmax, ICA/ASR, and Grad-CAM; its
  requirements/architecture conflict with the implemented pipeline. It explicitly
  claims no experimental results and is not the final results paper.
- Literature-survey workbooks and research-paper folders exist, but their full
  contents/citations were not re-audited in this integration.
- The demo AVI folder has **no file**; no final submission archive was identified
  in the submission-material folders inspected.

Preserve old proposals as historical work if needed, but submit revised documents
with the actual four-model method, limitations, source/config/run IDs, and verified
results. Existing paper accuracies must never be attributed to this implementation.
