# Demonstration and final verification

## Before recording

- Run `python -m pytest` and `python -m ruff check .`.
- Verify the real manifest, feature configuration, and completed experiment folder.
- Select a subject whose saved fold bundle excludes it from training.
- Confirm the CLI prediction matches the application and saved out-of-fold prediction.
- Keep the dataset accessible at the configured paths.

## Suggested sequence

1. Launch with `python -m streamlit run app/main.py`.
2. Show the workspace state and verified dataset.
3. Inspect one EEG recording and its channel information.
4. Open Analysis, choose the completed run and held-out subject.
5. Run analysis. Show the score, fold provenance, graph, and actual circuit.
6. Compute connection sensitivity and explain what it measures.
7. Show the saved model comparison and confusion matrix.
8. Export JSON and a figure.

If only synthetic verification is available, record the Software check screen and explicitly state that real-data experiments are unfinished.

## Submission checks

- Report and PPT use the same dataset count, class mapping, method, and run as the app.
- Printed accuracy and AUC come from saved outputs, not the reference paper.
- Document the quantum simulator, version, and absence of physical-hardware execution.
- Include source, configuration, instructions, and permitted artifacts. Check recording redistribution terms before including a dataset.
- Produce a real AVI file if required, then test playback on the submission machine.
- Extract the final archive into a different folder and follow the documented launch steps.
