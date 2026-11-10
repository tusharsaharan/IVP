# Integration audit and remaining work

**Updated:** November 10, 2026. **Conclusion:** modules are integrated; Tushar's
unchanged local SFARI experiment and real held-out CLI/app workflow are reproduced.
All models have accuracy 46–50%, below the 60% majority-class baseline; no quantum
advantage or clinical utility is established. Source/QC owner approval and unavailable
TUOS evidence remain open. Inherited teammate metrics remain unverified.

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

The merge `ec87cfb5b982cc88b858fbf9bb5c5ac7839553e8` was pushed to `main`;
PR #5 is merged. Its Windows/Ubuntu CI completed successfully:
<https://github.com/tusharsaharan/IVP/actions/runs/42756207107>.
That CI result applies to the merge, not automatically to later sprint edits.

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

### Initial merge validation (historical baseline)

- Windows, Python **3.13.9**: **244 tests passed**.
- `python -m ruff check .`: clean.
- `neuroq self-check`: passed; generated outputs remain ignored and **synthetic**.
- Wheel: built, installed to a separate temporary target, imported from that target,
  and passed its synthetic self-check using existing runtime dependencies. This
  verifies packaged code, not a fully fresh dependency environment.
- Four Bash scripts: syntax checked with Git Bash, not installed/executed/downloaded.
- Existing active recorded feature archive: loaded and validated against its manifest
  and configuration after integration; existing workspace/profile was not modified.

These initial checks were not fresh dependency installation or clinical validation.
The sprint below adds fresh-install and recorded checks; Python 3.11/3.12 and
independent cross-platform reproduction remain untested.

### Completion-sprint verification

- Windows/Python 3.13.9 after review fixes: **582 passed, 1 skipped**; Ruff clean. The skip requires
  Windows symlink creation privileges.
- Safe app run/bundle selection validates cohort/configuration and OOF metrics;
  damaged artifacts fail clearly. CLI/app keep the shared inference function.
- Strict atomic JSON; score/window/feature provenance; persistent circuit/sensitivity
  output with PNG/CSV downloads.
- All 20 recorded bundles passed training-only scaler/reference and original
  training-column ordering checks. Saved labels/scores match OOF.
- A deterministic repeat matched folds, OOF labels/scores and fold AUC. Two cases
  selected before evaluation matched raw features/labels/scores in all four models.
  Both quantum case predictions were incorrect and remain visible in the demo.
- Refreshed AppTest passed actual quantum inference, six finite sensitivities,
  strict JSON and numerical-angle circuit PNG on both cases. Local report:
  `artifacts/tushar-acceptance-20261110/integration_verification_reviewed.json`.
- Fresh README and extracted noneditable installs passed dependencies/import/CLI and
  synthetic checks. One installed-CLI recorded case also matched OOF using original
  local inputs in place, without copying research data into the ZIP.
- Initial extracted tests exposed an invalid requirement for Git metadata in a ZIP.
  The test now accepts documented `unknown` and still checks caller-directory
  independence. Refreshed archive evidence belongs in `REPRODUCTION.md`.
- Post-review actual browser workflow completed with explicit workspace/run/model/case
  selection, OOF/metrics download hash checks and prediction provenance/feature/score
  agreement. Circuit PNG matches reviewed AppTest; six sensitivities match raw evidence.
- The verified laptop AVI and MP4 are 1440×900, 64.17 seconds, 1,540 fully decoded
  frames each. FFmpeg full decoding and sampled-frame inspection passed; no human or
  native-player playback is claimed. Earlier concise/partial recordings are preserved
  as historical local evidence, not the selected final demo.
- Regenerate and verify the exact source-only delivery snapshot separately. Its
  hashes, installation/test logs and package contents stay in ignored local evidence;
  passing an earlier snapshot does not validate subsequent changes.

Independent review findings were resolved without changing the saved experiment:
public participant details removed; the private browser wrapper explicitly selects
and verifies its run/case; full protocol/provenance comparisons run before evaluation;
AppTest uses supplied/custom paths; whole-cohort session membership is enforced.
New run summaries preserve optional original feature metadata. Negative tests cover
protocol-only drift, same-name path decoys, changed/unselected sessions, missing
provenance and mismatched embedded metadata. Legacy fixtures explicitly omit the new
field to continue exercising original-acceptance recovery. No human-owner approval
or raw-payload equality is inferred from session identifiers alone.

### Project-only follow-up: workspace recovery

- Windows/Python 3.13.9: **615 tests passed, 1 skipped** in 44.19 seconds; Ruff and
  `pip check` passed. The explicitly synthetic end-to-end self-check also passed.
- Added 33 isolated regression cases for CLI help/recovery with malformed profiles,
  fail-closed data commands, valid-default/explicit-path precedence, blank paths,
  invalid activation inputs, and failed atomic saves with temporary-file cleanup.
- `use-workspace` can now repair a damaged profile without first reading it.
  Profile activation validates inputs before atomic replacement; no silent dataset
  fallback is introduced into data commands.
- CI is configured for Windows/Ubuntu on Python 3.11, 3.12 and 3.13, including
  dependency checks, CLI help and a synthetic end-to-end self-check. Results of an
  earlier CI run do not automatically validate this expanded matrix or later code.
- Numerical settings, shared inference, subject grouping, labels, feature/reference
  order, bundles and the frozen recorded experiment are unchanged. Outstanding
  dataset evidence and scientific owner approvals remain listed below.

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
| Active SFARI source | 528 payloads and 2,116 metadata files rehashed; explicit labels/payload grouping match; no cross-subject duplicate hashes | Scientific source/QC approval remains required |
| Active SFARI QC | Repeated 528 inputs/103 subjects to identical 448 recordings/100 subjects (60 ASD, 40 Control) | Fixed simple QC, not full clinical artifact cleaning |
| Active SFARI features | Rebuilt raw `[100, 28]`, matches prior within 1e-10, config `7bcd185ff8336878` | **3-20** used windows, not identical duration |
| Sahil TUOS handoff | 56 ledger rows; 45 eligible subjects (26 ASD, 19 Control) | No TUOS SET payloads in this checkout; labels/QC not reproduced |
| Sahil SFARI handoff | 438 runs are a subset of 448, same 100 subjects/labels; ten additional runs across three subjects | Different ID/protocol; older 517-candidate omission cause/approval unresolved |
| New recorded experiment | `run-20261110-045223-cfdc86`: complete; 400 OOF rows, five shared folds, 20 bundles; metrics regenerate | Separate Tushar 20-window experiment, not TUOS or historical Sahil metrics |

Sahil's reported TUOS/SFARI metrics remain attributed historical claims in the
handoff/method notes. A separate validated OOF-statistics module now reports
Wilson, stratified bootstrap and exact McNemar/Holm summaries for the new run
(10,000 replicates, seed 42). `STATISTICS.md` explains conditional inference and
overlapping-CV-training limitations; all six adjusted p-values are 1.0, not
equivalence. Some inherited SFARI balanced accuracies do not
match the stated sensitivity/specificity arithmetic. Do not print those metrics
as verified results, infer equivalence from nonsignificance, or confuse independent
within-cohort CV with zero-shot cross-dataset transfer. Current inference explicitly
guards dataset compatibility, and the node maps differ across cohorts.

## Remaining work, in priority order

| Priority | Owner | Required next handoff / acceptance |
| --- | --- | --- |
| Scientific approval | Sahil + Tushar + Vivek | Review repeated source/QC and known 438-of-448 subset; obtain older 517-candidate audit to explain omissions. Approve final protocol without retrospective score-driven changes. |
| Blocking for TUOS | Sahil | Supply permitted TUOS SET/FDT payloads, source group definitions/label mapping, prior-cleaning notes, and a reproducible 56-to-45 exclusion audit. Create and validate local paths; do not infer labels from numbering/prefixes alone. |
| Owner review | Aryan + Sahil | Review reproduced archive/schema/graphs and documented variable 3–20 windows. Equal duration would require a new experiment. |
| Owner review | Mahavir + Vivek | Review completed frozen four-model run and passing bundle/OOF checks; do not adopt unavailable historical metrics. |
| Implemented; interpretation review | Mahavir + Pranav | OOF statistics/seeds/replicates are complete; retain CV-dependence caveats and do not infer equivalence from p=1.0. |
| High | Vivek | Review actual circuit/encoding, PSD/equivalence/reassignment mathematics, train-reference ordering, and write the model section distinguishing this adaptation from the source paper. |
| Software handoff complete | Tushar + Shivam | Real raw-input CLI/app/OOF agreement, figures, JSON and sensitivity verified. Review captions/limits for presentation. |
| Content approval / delivery verification | Pranav + all owners | Revised Word/PPT/PDF copies describe implemented method and honest results. Laptop AVI decoding passed; source-snapshot/package hashes and extracted checks are tracked locally. Faculty/team approval remains required; no clinical/final-cohort approval implied. |

## Submission-material mismatch

The original files outside Git were preserved unchanged:

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
- The demo AVI folder was empty at the initial merge audit; current completion
  evidence does not inherit that historical observation.

Preserve old proposals as historical work if needed, but submit revised documents
with the actual four-model method, limitations, source/config/run IDs, and verified
results. Existing paper accuracies must never be attributed to this implementation.

New revision-suffixed local copies (original authors/course title retained):

- `SRS and Design Report/ASD_EEG_SRS_Design_Implementation_Revised_20261110.docx`
- `PPT(in pptx formate)/ASD_EEG_Group8_Implementation_Revised_20261110.pptx`
- `Report (Springer Format)/ASD_EEG_Implementation_SpringerStyle_Revised_20261110.pdf`

These report the new SFARI method/results and conditional uncertainty. Native
Word/PowerPoint rendering and PDF inspection are local submission QA. The PDF is
**Springer-style**, not claimed to meet an official publisher template. Final
artifact hashes and package contents stay in ignored evidence; see `TUSHAR_HANDOFF.md`.
