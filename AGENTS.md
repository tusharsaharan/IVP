# Project instructions

Read `README.md`, `docs/CONTRACTS.md`, and the relevant owner section in `docs/TEAM.md` before editing.

- Keep the CLI and Streamlit application on the shared inference pipeline.
- Preserve subject grouping, training-only fitting, feature order, and model-reference order.
- Never infer diagnosis labels from filenames or mix synthetic checks with recorded-data results.
- Review dataset-specific defaults before running real experiments; the shipped node map is provisional.
- Keep raw EEG, local model bundles, generated results, credentials, and personal file paths out of Git.
- Follow the light academic interface in `PRODUCT.md` and `DESIGN.md`: clear text, no emojis, no decorative animation.
- Coordinate changes to shared contracts and dependency files with Tushar and Vivek.
- Verify changes with `python -m pytest` and `python -m ruff check .`.
- Record scientific limitations and configuration changes in `docs/METHOD.md`.
