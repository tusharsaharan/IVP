# Independent installation and source-submission verification

Verification date: **November 10, 2026**. This document distinguishes software
reproduction from recorded-cohort research evidence. No raw EEG, populated local
manifest, existing bundle, or recorded result is copied into the source submission.
The maintained installation root is this repository, not the historical nested
`neuroq-connect/` project.

## Regenerate the latest source snapshot

After all owners finish their edits, run from the maintained repository root:

```powershell
python scripts/verify_submission.py --smoke
```

The script uses Python's standard library to create a new timestamped ZIP and a
SHA-256 manifest under ignored `artifacts/submission-20261110/`. Existing ZIPs,
extracted directories, user data, and results are never overwritten. Without
`--smoke`, it only creates the source archive. `--output` selects a new ZIP filename;
`--temp-root` selects an explicitly approved temporary directory.

The source allowlist includes `scripts/run_recorded_acceptance.py`,
`scripts/reconcile_sfari.py`, their focused tests, and other permitted new source
files. It does not exclude the scripts directory or require these files to have
been committed first.

For `--smoke`, a separate temporary directory under the operating system's
`LOCALAPPDATA` temporary `opencode` directory on Windows holds the extracted source,
a fresh virtual environment, installation logs, dependency versions, and generated
**synthetic** check artifacts. Installation uses the extracted project with its
`dev` extra and `--no-cache-dir`; it does not borrow the checkout's virtual environment.
It then checks dependency consistency, imports the installed package with isolated
Python from outside the source tree, invokes the installed CLI, runs the explicit
synthetic self-check, and runs pytest and Ruff from the extracted tree. Inspect
`reproduction.json` and each stage's local log. A failing stage returns a nonzero exit
status and retains evidence for diagnosis; it does not claim successful reproduction.

## Archive policy and limitations

- Only designated source directories and explicit top-level project files are
  admitted. New source/test/docs files in those directories are included even before
  they are committed; ignored artifacts and arbitrary files are not swept into a ZIP.
- Git metadata/history, environments, caches, raw/local data, populated manifests,
  recordings, model bundles, results, media, binary documents, and the historical
  nested project are excluded. The public recording CSV must remain header-only.
- Source links/reparse points, oversized files, binary/non-UTF-8 content, detected
  personal home paths, private-key markers, and selected credential prefixes fail
  closed. Pattern checks are **not** a comprehensive privacy or secret audit; owners
  must still review the source and submission contents.
- Every admitted file has a byte count and SHA-256 digest. Extraction validates the
  complete manifest, source allowlist, sizes, duplicate members, path safety, and
  content hashes before writing files into a new directory.
- Snapshot metadata records creation time, checkout HEAD when Git is available,
  dirty-tree status, and `final_freeze: false`. Edits detected during collection abort
  capture. Changes after capture are not incorporated: coordinate a final freeze,
  regenerate, and repeat verification before handing in a final submission.
- This is a **source-only** ZIP, not an assertion that the final report, presentation,
  playable AVI, permitted research artifacts, or independently verified recorded
  metrics are complete. Package those deliverables separately after owner review.

## Supported Python versus actually tested Python

The project declares Python 3.11–3.13 support. The independent verification machine's
Python launcher exposed only **CPython 3.13.9, 64-bit Windows**; no additional 3.11 or
3.12 runtime was found through the launcher or command lookup. This is not evidence
that 3.11/3.12 or Linux/macOS installation succeeds. Do not reuse the Windows 3.13
version lock as cross-platform validation.

## Independent evidence

### Fresh README installation

An independently created virtual environment used the system Python, not the
existing checkout environment. The command was the README's
`python -m pip install -e ".[dev]"`, with `--no-cache-dir` added to force fresh
dependency retrieval.

| Check | Observed outcome / wall-clock cost |
| --- | --- |
| Create fresh Python 3.13.9 virtual environment | Passed; 6.40 seconds |
| Install editable project and actual development/runtime dependencies | Passed; 156.13 seconds |
| `pip check` | Passed; no broken requirements |
| Installed package import, isolated Python from outside the checkout | Passed; distribution version 0.1.0; editable source location confirmed |
| Installed `neuroq --help` | Passed |
| Installed `neuroq self-check` with explicit config/output in temporary storage | Passed; `data_kind: synthetic`; 5.12 seconds |
| Full `python -m pytest` against the concurrent checkout | 403 passed, 1 skipped, 4 warnings; pytest reports 42.24 seconds (process wall clock 49.37 seconds) |
| Full `python -m ruff check .` | Passed; 1.13 seconds |
| Focused submission verifier tests | 42 passed, 1 skipped |

The skip is the source-symlink fixture: this Windows account cannot create the
symlink. The four warnings are scikit-learn's deliberate single-class edge-case
fixtures in the statistics tests; they are not observed recorded-cohort warnings.
Other agents were adding tests concurrently, so these totals describe the tree
actually collected at that invocation, not an immutable final test count.

The installation log reports **73 wheel downloads, approximately 189.0 MB**
(sum of pip's rounded download sizes, not measured network traffic). The fresh
environment occupies **813,759,260 bytes, approximately 0.758 GiB**. Keep roughly
1 GiB per fresh environment plus installation/extraction headroom available;
`--smoke` downloads and installs a second independent dependency set rather than
reusing the first environment. Raw-data disk usage is separate and unchanged.

All **72 pinned requirements** in `requirements.lock.txt` matched the versions
resolved by this fresh range-based installation. This is a post-install version
comparison, **not** a separate execution of the locked-install command. Seaborn,
required by the project, was also installed (0.13.2). Selected installed versions:
NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, pandas 2.3.3, PennyLane 0.45.1,
MNE 1.13.2, Streamlit 1.67.0, pytest 9.2.0, and Ruff 0.17.4. Full dependency
versions and logs remain only in the approved local temporary evidence directory
`submission-reproduction-20261110/`.

### Extracted submission snapshot

The independently extracted ZIP installation is a distinct invocation and
environment, with its own archive hashes and logs. The first editable installation
does not stand in for this check.

The initial smoke-test snapshot is
`artifacts/submission-20261110/source-20261110T045125823665Z.zip`, captured at
**2026-11-10 04:51:25.993538 UTC**, with checkout HEAD
`ec87cfb5b982cc88b858fbf9bb5c5ac7839553e8` and a dirty working tree. It contains
**106 source files** plus the snapshot manifest, **530,116 source bytes**, and is
**199,184 bytes** compressed. Its SHA-256 is:

```text
a071649323e2a61b291a8778722d80c7f23eded4419d8ddacdb0d5c482d00e43
```

An independent ZIP inspection verified every listed source digest, checked the
excluded directory names, and confirmed the public CSV has no participant rows.
The archive deliberately omits other owners' local submission-work files in the
same artifact directory. Its copied documentation predates subsequent evidence
updates in this file; neither this snapshot nor its checkout commit implies that
all concurrent edits were final or committed.

| Extracted-snapshot check | Observed outcome / wall-clock cost |
| --- | --- |
| Fresh noneditable project/wheel and dependency installation | Passed; 178.06 seconds |
| `pip check` | Passed; 1.55 seconds |
| Isolated import outside extracted source; assert package lives under fresh venv | Passed; 0.05 seconds |
| Installed CLI help | Passed; 2.41 seconds |
| Installed synthetic self-check | Passed; 9.92 seconds |
| Full pytest from extracted source | **402 passed, 1 failed, 1 skipped**, 4 fixture warnings; 46.98-second process wall clock |
| Full Ruff from extracted source, run separately after pytest failure | Passed; 0.12 seconds |

The failure is `test_source_commit_is_independent_of_caller_directory` in
`tests/test_evaluation_handoffs.py:137`. It assumes the installed package must
report a 40-character Git commit. The actual installed-wheel result is `unknown`,
as the shared contract permits when no Git metadata is available. The test needed
correction to preserve caller-directory independence while accepting the
documented no-Git case. The verifier does **not** suppress, skip, or reinterpret this
failure. This initial snapshot has **not passed the complete extracted test suite**.
Regenerate and rerun after the owning integrator corrects the test and all agents
finish their source edits.

The extracted environment and evidence are retained under the approved temporary
directory `neuroq-submission-hkvr04uq/`. Its fresh environment occupies
**814,219,892 bytes**; the directory including extraction, private logs, and generated
checks occupied approximately **817.3 MB** at measurement. Full dependency versions
were captured separately after the failed pytest stage; the fresh installation
itself succeeded. These costs are additional to the first fresh README environment.

### Separate installed-package recorded inference

After the extracted installation/import/software check succeeded, the same
noneditable installed CLI was run against a **permitted original local manifest
and trusted original fold bundle**, using absolute paths and reading EEG in place.
One requested held-out prediction reproduced the expected class and saved SVM
decision score within an absolute tolerance of **1e-9**, in **4.08 seconds**.

This was a genuine **recorded-data single-case inference check**, not synthetic
evidence, a new training run, or a cohort-metric reproduction. Participant identity,
the prediction JSON, exact case score, original paths, and the detailed verification
remain in private temporary files (`recorded-prediction-private.json` and
`recorded-inference-verification-private.json`). No EEG, manifest, or model bundle
was copied into the source ZIP or uploaded. This result does not erase the separate
extracted pytest failure.

### Completion-sprint correction and refreshed verification

The no-Git assertion was corrected to accept `unknown` or a valid 40-character
lowercase commit while still requiring caller-directory independence. No test was
skipped or suppressed to hide the initial failure. The current checkout passes
**582 tests, 1 skipped** on Windows/Python 3.13.9 after review fixes; Ruff is clean. The one skip
still requires Windows symlink creation privileges.

The source allowlist now includes JavaScript browser-reproduction scripts alongside
Python/shell sources. These scripts do not embed participant identities or private
paths. Raw browser downloads/videos remain outside the public source archive.

The first corrected source snapshot `source-20261110T050954835193Z.zip` (111 files,
SHA-256 `344b5ecd71c2e2bbcf1a3b24e311cf7e704c2ce99ec2d66a5ce208b19b6898a3`)
passed every extracted stage: install 200.66 s, pip-check 1.49 s, isolated installed
import 0.05 s, CLI help 2.17 s, synthetic self-check 10.65 s, full pytest 37.11 s,
Ruff 0.14 s and version capture 0.43 s. This verifies that snapshot only. Subsequent
independent review found protocol/session/privacy gaps; it is not the final reviewed
source tree. Its local evidence remains in `neuroq-submission-excwrdbp/`.

After the review fixes, regenerate the source-only archive and repeat all extracted
stages on that exact snapshot. The final ZIP's SHA-256 and stage outcomes belong in
its external local manifest/evidence, avoiding an impossible self-referential archive
hash embedded in source. No raw data, populated manifest, bundle or recorded result
is included. Final source freeze/delivery status is not inferred from earlier snapshots.

### Scientific boundary

The synthetic self-check generates 12 synthetic subjects, builds the numeric
features, evaluates all four models on shared three-fold subject partitions, reloads
a quantum bundle, and compares held-out inference with the saved prediction.
Its class labels use the same generative process and establish no EEG autism
classification performance. No synthetic accuracy/AUC is reported here.

This verification did not rerun recorded acquisition, cohort QC, cohort feature
extraction, recorded cross-validation, or recorded metrics. The installed CLI did
recompute features and inference for the explicitly permitted held-out case above;
no complete app/AVI demonstration was performed by this verifier. Consult the
separately reviewed recorded-data acceptance evidence and dataset-owner approvals;
do not treat these fresh installs as evidence for any historical performance claim.
The installed CLI's self-check was invoked outside the source tree with an explicit
configuration path: the wheel alone does not contain repository-root configuration
or the Streamlit application. The source ZIP supplies those files for the README's
run-from-root workflow.
