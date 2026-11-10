# Demonstration script

## Purpose

This script guides the recorder through the AVI demonstration workflow. The recording should run approximately 5–8 minutes and cover every key capability: EEG inspection, quantum analysis, model comparison, explanation, and export.

## Before recording

Checklist — complete every item before launching the recording application.

- [ ] Run `python -m pytest` and confirm all tests pass.
- [ ] Run `python -m ruff check .` and confirm no lint errors.
- [ ] Confirm `data/metadata/recordings.csv` contains real, verified recordings.
- [ ] Confirm `artifacts/features/subject_features.npz` exists and was built from the same manifest.
- [ ] Confirm at least one completed run exists under `results/`.
- [ ] Identify a held-out subject (use `results/RUN_ID/folds.csv` to find fold 0's test subject).
- [ ] Confirm `neuroq predict --manifest ... --subject HELD_OUT_ID --bundle results/RUN_ID/fold_0/quantum.joblib` exits with code 0.
- [ ] Confirm the CLI decision score matches the saved OOF decision score in `oof_predictions.csv`.
- [ ] Confirm the dataset files are accessible at the configured paths.
- [ ] Close other applications to reduce screen noise.

> [!WARNING]
> If real data is not yet available, record the software-check screen instead and state clearly in the AVI and report that real-data experiments are unfinished. Never present synthetic results as real-data results.

## Launch

```powershell
python -m streamlit run app/main.py
```

The application opens at http://localhost:8501. Narrate every action clearly.

## Recording sequence

### Step 1 — Workspace overview (30 s)

Open the application. Show the **Workspace** section.

Narrate:
- Dataset name and data kind (recorded vs. synthetic).
- Number of subjects and class counts.
- The completed run folder selected.

### Step 2 — EEG recording inspection (60 s)

Navigate to **Recording**.

- Select one of the verified local recordings.
- Show: sampling rate, channel list, duration, QC window counts.
- Zoom into a short EEG segment showing recognisable oscillatory activity.
- Note: the recording is loaded with the same preprocessing pipeline used during training.

### Step 3 — Quantum analysis and inference (90 s)

Navigate to **Quantum analysis**.

- Select the completed run and the held-out subject identified in the checklist.
- Show the EEG connectivity graph (four-node, six-edge layout).
- Show the actual PennyLane circuit diagram.
- Click **Run analysis**.
- Show:
  - Predicted class (Control or ASD).
  - Decision score (label it clearly as a score, not a probability).
  - Fold provenance (held-out fold number and run ID).
  - Model name (Connectivity quantum kernel).

Narrate: "This subject was excluded from training in fold N. The inference uses only the saved model bundle."

### Step 4 — Connection sensitivity (60 s)

Scroll to **Connection sensitivity** in the analysis panel.

- Click **Compute sensitivity**.
- Show the bar chart: six connections, score change per connection.
- Narrate: "Each bar shows how much the decision score changes when that connection's median and variability are replaced with training-data reference values. This is model sensitivity, not a causal biomarker."

### Step 5 — Model comparison results (60 s)

Navigate to **Results**.

- Show the model comparison table: RBF-SVM, Amplitude kernel, Connectivity quantum kernel, Reassigned connections.
- Show balanced accuracy, sensitivity, specificity, macro-F1 for each model.
- Show confusion matrices.
- Show mean fold AUC per model.

Narrate: "These numbers come from the saved `metrics.json` and are derived from `oof_predictions.csv`. They were not typed manually."

### Step 6 — Export (30 s)

Navigate to **Export**.

- Click **Export prediction JSON**.
- Show the saved file path and open the JSON to confirm it contains `subject_id`, `predicted_label`, `decision_score`, and `provenance`.
- Optionally export a figure.

### Step 7 — CLI parity (30 s, optional but recommended)

Switch to a terminal.

```powershell
neuroq predict --manifest data/metadata/recordings.csv --subject HELD_OUT_ID --bundle results/RUN_ID/fold_0/quantum.joblib --output artifacts/prediction.json
```

Show that `predicted_label` and `decision_score` match the application output.

## Narration guidelines

- State the dataset name, cohort size, and date at the start.
- When showing decision scores, say "decision score" or "SVM margin", never "probability" or "confidence".
- When showing perturbation results, say "model sensitivity" or "score change", never "causal biomarker" or "diagnostic feature".
- When showing results, attribute them to the saved run folder and OOF CSV, not to the reference paper.
- If a step produces a warning or empty state, explain why (e.g., "this fold has only one class so AUC is undefined").

## AVI production

Record using screen-capture software. Export to AVI format. Do not rename an MP4 file to `.avi`; the container format must be correct. Verify playback in a media player on a different machine before including in the submission archive.

Suggested resolution: 1920 × 1080. Suggested bitrate: sufficient for legible text in the application.

## Submission-archive check

After recording:

1. Package the archive (source code, `configs/`, `docs/`, `results/RUN_ID/`, `artifacts/features/`, the AVI, the report PDF, and the PPT).
2. Extract the archive into a new folder on a different path.
3. Create a fresh virtual environment and follow the README install steps exactly.
4. Run `neuroq self-check` to verify software integrity.
5. Confirm the AVI opens and plays correctly.
6. Confirm subject counts, class mapping, and key metrics in the report and PPT match `summary.json` and `model_comparison.csv`.

> [!CAUTION]
> Do not include raw EEG recordings in the archive if the dataset licence prohibits redistribution. Include only processed results, model bundles, and configuration files.

