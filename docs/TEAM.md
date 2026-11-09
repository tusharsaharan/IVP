# Team handoffs for the submission sprint

All modules already have a working starting implementation. Extend and validate them rather than creating incompatible parallel pipelines. Tushar owns integration; Vivek coordinates model and interface decisions. Read `CONTRACTS.md` before editing.

## Vivek G: quantum method

**Own:** `src/asd_eeg/models/`, `src/asd_eeg/training/`.

- Review the four-qubit encoding order, input-dependent gates, and train-reference caching.
- Verify fidelity PSD, normalization, edge sensitivity, and amplitude/classical equivalence using `tests/test_quantum.py`.
- Check the reassignment control preserves node identity and edge-weight distributions.
- Supply the actual circuit diagram, kernel equation, assumptions, and method subsection.
- Coordinate any bundle change with Tushar and Dodiya.

**Done:** deterministic model fitting, reloadable bundles, mathematical checks, and a clear distinction from the reference paper.

## Sahil Gupta: data and preprocessing

**Own:** `data/metadata/recordings.csv`, `src/asd_eeg/data/`, `src/asd_eeg/preprocessing/`, preprocessing/channel configuration.

- First handoff: one readable recording, electrode metadata, and an authoritative label source.
- Freeze the available dataset. KAU is intended primary; TUOS requires verified labels and montage mapping before use.
- Check subjects versus sessions, references, conditions, sampling rates, companion files, and downloaded payloads.
- Populate the manifest and explicit exclusions. The starter stops on unusable recordings rather than silently removing them.
- Adapt preprocessing/QC to the real data and document any prior cleaning.

**Done:** `neuroq validate` passes, subject counts reconcile, one subject preprocesses, and label provenance is documented.

## Dodiya Mahavirsinh: evaluation and baselines

**Own:** `src/asd_eeg/evaluation/`, `src/asd_eeg/baselines/`.

- Freeze the subject-wise protocol and model settings before reading performance.
- Run all four comparisons on identical partitions; retain training-only transforms.
- Inspect model bundles and out-of-fold predictions for subject leakage.
- Export the authoritative result table, confusion matrices, fold AUC, runtime, and experimental limitations.
- If time permits, add paired subject uncertainty; document precisely what was run.

**Done:** a complete run whose metrics regenerate from saved predictions and whose subject counts match the manifest.

## Aryan Pandey: regional EEG features

**Own:** `src/asd_eeg/features/`, feature configuration.

- Agree on available node channels with Sahil. The shipped F3/F4/P3/P4 mapping is not yet verified for KAU.
- Review power normalization, envelope connectivity, filter boundaries, and window aggregation.
- Check finite values, symmetric graphs, deterministic channel ordering, and the fixed 28-feature contract.
- Hand Dodiya the archive, Vivek a numeric example, Tushar the shared extractor, and Shivam graph summaries.

**Done:** repeatable real-subject features, a documented feature schema, and identical extraction at training and inference.

## Tushar Saharan: integration and application

**Own:** `app/`, `src/asd_eeg/inference/`, CLI, packaging, shared contracts.

- Keep the application and CLI on the same inference function.
- Integrate verified data and completed results without hard-coded personal paths.
- Check model configuration restoration and exact training-reference ordering.
- Demonstrate a held-out subject, graph/circuit, score, saved results, sensitivity, and JSON export.
- Rehearse the complete sequence before recording the AVI.

**Done:** another teammate can install, launch, reproduce a held-out analysis, and export it.

## Shivam Mall: explanations and figures

**Own:** `src/asd_eeg/explainability/`, `src/asd_eeg/visualization/`.

- Review signal, connectivity, and confusion figures on real inputs.
- Produce a readable actual circuit figure using `make_qnode` and PennyLane drawing.
- Validate edge perturbations against training-derived references and preserve class-score direction.
- Add a held-out case study with its run, fold, node map, and explanation limits.
- Export figures and short captions for Pranav.

**Done:** plots are generated from real artifacts, integrate into the app, and are readable in the report/PPT.

## Pranav Karhale: verification and submission

**Own:** `tests/`, `docs/`, final report, presentation, AVI assembly.

- Install in a clean environment and follow README commands exactly.
- Run the test suite and labelled software check; report failures to the owning member.
- Recompute selected metrics and compare dataset counts across the app, report, and slides.
- Collect each member's technical subsection and citations; replace the old main CNN-LSTM architecture with the implemented method.
- Record a real inference workflow, verify the AVI container/playback, and check the extracted submission archive.

**Done:** reproducible instructions, consistent artifacts, an actual playable AVI, and a complete final archive.

## Integration schedule

1. First hour: verified dataset decision, shared node mapping, and fixed contracts.
2. First working handoff: one real subject from loading through features.
3. Next: real fold predictions and saved bundles.
4. Then: result figures, app integration, and each member's report section.
5. Final two to three hours: freeze code/results, reproduce the demo, assemble and check the submission.

Each handoff includes code, one example command, configuration, input/output shapes, and known limitations. Do not replace missing experiments with invented numbers.
