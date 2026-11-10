# Sahil Gupta: data and preprocessing handoff

**Role:** verified data and preprocessing lead. Contributions from `sahil` are
integrated into the root package; install and run from the repository root.
This UTF-8 handoff supersedes the original machine-local contribution notes.

## Implemented contributions

- TUOS default scalp nodes, in fixed order: F7, Fz, P1, O2.
- TUOS settings: 50 Hz notch, average EEG reference, 1-40 Hz filtering,
  resampling to 128 Hz, eight-second non-overlapping windows, minimum three
  accepted windows per recording, and **at most 18** feature-valid windows per subject.
- Montage/reference documentation in `configs/channels.yaml` and
  `configs/preprocessing.yaml`. These are documentation companions;
  `configs/default.yaml` is the executable full configuration.
- Separate SFARI 18-window configuration in `configs/sfari.yaml`, retaining
  F3/F4/P3/P4 and 60 Hz mains. Tushar's `configs/sfari_rest.yaml` remains a
  distinct 20-window protocol, not an alias.
- TUOS/SFARI eligibility handoffs and acquisition convenience scripts.
- Cross-drive handling for QC manifests, extended during integration to
  workspace activation as well.

The shared extractor, four feature-matched models, fold generator, inference
pipeline, and bundle schema remain unchanged. See `docs/CONTRACTS.md` and
`docs/METHOD.md` for the actual algorithms.

## Reported cohort handoff, not reproduced experiment evidence

TUOS source: Dickinson, Jeste, and Milne (2022), *Cortex*, 148, 139-151,
DOI `10.1016/j.cortex.2021.09.022`; Figshare `10.15131/shef.data.16840351`.

Sahil's supplied ledger reports 56 TUOS recordings, with 45 eligible subjects
(26 ASD, 19 Control). Reported exclusions: nine control recordings without
verified standard-electrode mapping, one ASD recording missing Fz, and one ASD
recording with fewer than three clean windows. Internal EEGLAB metadata reportedly
resolves diagnostic mismatches in disk numbering. Verify that interpretation
against the published dataset's group definitions; do not turn file/header
prefixes alone into an automatic diagnostic-label rule. The runtime loader does
not derive labels.

Sahil's SFARI ledger reports 136 participants audited, 103 candidates,
517 candidate runs, and 100 retained participants (60 ASD, 40 Control) across
438 runs. The separately available Tushar local audit instead has 528 candidates
and 448 retained runs across 100 participants under the 20-window protocol.
These are **different handoffs**: reconcile source selection, download completeness,
QC exclusions, dataset IDs (`SFARI` versus `ds006780`), and configuration IDs before
comparing results. Do not silently discard the additional recordings.

Populated manifests and original notes are preserved locally under ignored
`data/local/branch-handoff-20261110/` on the integrating machine and in existing
branch history. A fresh clone intentionally ships no participant lists or private
paths. Relocate and revalidate source evidence before using a historical manifest.

## Historical performance claims requiring verification

The original notes report these accuracies; they are retained here as claims,
**not verified repository results or numbers ready for submission**:

| Model | Reported TUOS accuracy | Reported SFARI accuracy |
| --- | --- | --- |
| RBF-SVM | 55.56% | 50.00% |
| Amplitude kernel | 55.56% | 45.00% |
| Graph-conditioned quantum kernel | 48.89% | 44.00% |
| Reassigned-edge control | 55.56% | 45.00% |

Reported TUOS runs were `run-20261109-122602-2b9104` (20-window exploratory
configuration `88ec53f9967f98e1`) and `run-20261109-160931-0cf4a1` (18-window
configuration `7e45d19e662f0f49`). Their OOF predictions, bundles, and complete
run summaries are absent from this checkout. No supporting SFARI completed run
was supplied here. Original bootstrap intervals/McNemar claims are not produced
by the shared runner, and some reported SFARI balanced accuracies do not match
the stated sensitivity/specificity arithmetic. Obtain the source predictions and
statistical scripts, recompute metrics, and retain run/config/source IDs before
using any of these claims. Independent within-cohort CV is not zero-shot transfer
or proof of clinical/montage invariance.

## Reproduce after receiving the actual data

Use an explicit authoritative TUOS manifest at `data/local/tuos/recordings.csv`;
its paths must resolve from that directory and each SET/FDT pair must stay together.

```powershell
python -m pytest
python -m ruff check .
neuroq use-workspace --manifest data/local/tuos/recordings.csv --config configs/default.yaml --features artifacts/tuos/subject_features.npz --results results/tuos
neuroq validate
neuroq features
neuroq evaluate
python -m streamlit run app/main.py
```

Check saved `window_counts` before claiming every subject contributes 144 seconds.
The cap does not require 18 usable windows: subjects with 3-17 remain eligible.

For SFARI, follow `docs/SFARI.md`. Select `configs/sfari.yaml` and distinct
`artifacts/sfari-18/` / `results/sfari-18/` paths only if reproducing the 18-window
protocol; keep the 20-window workspace separate. Restart Streamlit after changing
profiles. Use a subject's actual saved test-fold bundle, not a hard-coded fold or
historical subject ID. The app is a research workspace, not clinical diagnostic software.

## Remaining handoff responsibilities

- **Sahil:** permitted TUOS payloads, authoritative group mapping, reproducible
  inclusion/exclusion audit, prior-cleaning notes, and SFARI count reconciliation.
- **Aryan:** validated archives, exact window counts, graph fields, and feature schema.
- **Vivek/Mahavir:** reviewed circuit/protocol, completed runs, bundles, recomputed
  OOF metrics, and any explicitly implemented uncertainty/statistical procedure.
- **Tushar/Shivam/Pranav:** real held-out demo, figures/captions, updated report/PPT,
  playable AVI, and an independently reproduced submission archive.

See `docs/STATUS.md` for the integrated evidence-based checklist.
