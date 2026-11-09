# NeuroQ-Connect interface

## Scene

Students and an evaluator use a laptop in a normally lit classroom, checking recordings and experimental outputs under a submission deadline. A light, quiet interface keeps tables and signal plots readable.

## Direction

Clean academic research workspace, confirmed by the user. Restrained color: warm off-white canvas, slightly darker neutral sidebar, dark ink, and muted blue for selected controls and primary actions.

## Tokens

- Canvas: `#f8f9f7` / `oklch(0.98 0.003 120)`.
- Panel: `#f1f3f1` / `oklch(0.96 0.004 150)`.
- Ink: `#243342`.
- Secondary text: `#526273`.
- Accent: `#335e83`.
- Border: `#dce2e5`.
- Typography: system sans-serif; fixed readable sizes; body copy below 75 characters per line.
- Layout: a narrow navigation sidebar, generous page heading, flat sections, and native tables. Avoid nested cards.

## Components and content

Use Streamlit native buttons, select boxes, expanders, forms, and tables. Visible field labels are required. Keep error messages specific and actionable. Synthetic checks must carry an explicit software-check label. Model scores are decision scores, not probabilities. No emojis, gradients, ornamental graphics, or decorative animation.
