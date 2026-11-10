# Recording manifests

`recordings.csv` is a header-only contract template, not an acquired cohort.
Keep populated manifests, eligibility ledgers, raw recordings, and local paths under
ignored `data/local/` or `data/raw/`. Do not commit participant lists.

Each recording needs an explicit dataset ID, stable subject ID, unique recording ID,
EEG path, binary label (Control=0, ASD=1), authoritative label source, and data kind.
Paths are absolute or relative to the **manifest's own directory**, not the checkout.
Never derive diagnoses from file numbering, unverified filename prefixes, or scores.

- TUOS: obtain the permitted SET/FDT payloads and Sahil's source-based label/exclusion
  audit, relocate paths into a local manifest, then follow `README_SAHIL.md`.
- SFARI: `neuroq prepare-sfari` creates a hash-verified local candidate manifest;
  `neuroq screen` creates a new inclusion/exclusion ledger. Follow `docs/SFARI.md`.

The November 10 integration preserved incoming populated manifests and original notes
in `data/local/branch-handoff-20261110/` on the integrating machine. Those snapshots
are historical handoffs, not portable ready-to-run manifests. Moving a manifest
requires resolving its EEG paths again and verifying payloads/labels/QC. Existing
local workspace profiles were not changed. Historical branch commits still contain
their original files; this cleanup does not rewrite Git history.
