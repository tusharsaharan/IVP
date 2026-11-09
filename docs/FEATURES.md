# Subject feature handoff

The shared extractor is `asd_eeg.features.subject_features.extract_subject_features`. It returns one `SubjectFeatures` record per subject and is used unchanged by training, evaluation, and saved-model inference.

## Shapes and order

For `q` configured nodes, `X` has width `4*q + 2*(q*(q-1)/2)`. With the default four nodes this is exactly 28.

| Slice | Contents | Shape |
| --- | --- | --- |
| `0:4*q` | Node-major band power: node 0 delta/theta/alpha/beta, then node 1 | `[4*q]` |
| `4*q:4*q+e` | Edge connectivity medians | `[e]` |
| `4*q+e:4*q+2*e` | Edge connectivity IQRs | `[e]` |

`e = q*(q-1)/2`. For four nodes the edge order is:

```text
(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
```

The configured node order and edge order are authoritative. Never sort or reorder arrays after extraction.

## Definitions

Band power is the median across used windows of `log10(relative power)`. Relative power for each electrode is the integrated Welch PSD inside a half-open band divided by the integrated 1-30 Hz PSD. Electrode values are averaged within a node before the log transform.

Connectivity is the Pearson correlation between two regional log alpha-envelope series inside one accepted window. Regional envelopes average the electrodes assigned to each node. Each edge contributes its median across used windows and its interquartile range.

## Aggregation across recordings

Windows accepted by preprocessing are pooled across a subject's recordings in `recording_id` order. If the pool exceeds `max_windows`, deterministic evenly spaced windows are selected. Powers use the median across used windows; connectivity uses the median and the 75th-minus-25th percentile IQR. Labels are never inputs.

## Archive fields

`build_features` writes a compressed NPZ with no object arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `X` | `[subjects, 28]` | Feature matrix |
| `y` | `[subjects]` | 0 = Control, 1 = ASD |
| `subject_ids` | `[subjects]` | Stable participant IDs |
| `metadata` | scalar JSON | Configuration, provenance, feature names, schema, QC |
| `connectivity_median` | `[subjects, q, q]` | Symmetric median graph, diagonal one |
| `connectivity_iqr` | `[subjects, q, q]` | Symmetric variability graph, diagonal zero |
| `window_counts` | `[subjects]` | Used windows per subject |
| `accepted_window_counts` | `[subjects]` | Feature-valid windows before the cap |
| `window_connectivity` | `[used_windows, 6]` | Edge correlations for each used window |
| `window_subject_index` | `[used_windows]` | Feature row for each window |
| `window_recording_ids` | `[used_windows]` | Recording source for each window |

The schema describes every feature index, source node or edge, units, and aggregation. It must exactly match the archive configuration and feature order. The default `subject_features.npz` uses the sibling `feature_schema.json`; other archives use `<archive-stem>.feature_schema.json`, so multiple configurations can share a folder safely. Use `feature_schema_path` from `asd_eeg.features.cache` to locate an archive's sidecar. Legacy shared sidecars for differently named archives are not authoritative; their embedded schema and configured feature names are validated instead.

Before saving or evaluation, validation checks graph ranges and agreement with `X`, reconciles window counts and subject indices, and verifies that saved windows reproduce the connectivity medians and IQRs. Older schema-1 archives without the optional graph/window arrays remain usable.

## Commands

```powershell
neuroq features --manifest data/metadata/recordings.csv --config configs/default.yaml --output artifacts/features/subject_features.npz
```

This writes `artifacts/features/subject_features.npz` and `artifacts/features/feature_schema.json`.

For a synthetic software check only, `neuroq self-check` creates a separate archive under `artifacts/software-check/`. Synthetic values must never be reported as cohort features or research results.

## Python handoff

```python
from asd_eeg.features.cache import load_features

dataset = load_features("artifacts/features/subject_features.npz")
subject_features = dataset.X[i]             # [28]
subject_graph = dataset.connectivity_median[i]  # [4, 4]
window_edges = dataset.window_connectivity[dataset.window_subject_index == i]  # [windows, 6]
```

The configured `F3/F4/P3/P4` node map is provisional until Sahil confirms the dataset montage. Missing channels fail with an explicit error rather than being guessed.
