# Shared contracts

## Recording manifest

`data/metadata/recordings.csv` contains one row per recording. Required columns:

| Column | Meaning |
| --- | --- |
| dataset | One dataset identifier per manifest |
| subject_id | Stable participant ID; repeated across that person's sessions |
| recording_id | Unique recording ID |
| eeg_path | Absolute path, or path relative to the manifest's folder |
| label | 0 = Control; 1 = ASD |
| label_source | Authoritative provenance for the diagnosis mapping |
| data_kind | `recorded` or `synthetic`; cannot mix within a manifest |

The reader checks paths and conflicting metadata. It cannot establish whether supplied diagnoses are authoritative. Sahil verifies provenance before adding rows. SET/FDT pairs remain together. OpenNeuro annex pointers must be replaced by recording payloads.

## Module boundaries

```python
load_recording(manifest_row) -> EEGRecording
preprocess_recording(recording, config) -> ProcessedRecording
extract_subject_features(processed_recordings, config) -> SubjectFeatures
build_features(manifest_path, config, output_path) -> FeatureDataset
encode_states(angle_features, config, reassign=False) -> complex ndarray
fidelity_kernel(left_states, right_states) -> float ndarray
fit_model(X_train, y_train, config, model_name) -> FittedModel
run_evaluation(feature_dataset, output_directory) -> run_directory
analyze_subject(manifest_path, subject_id, bundle_path) -> prediction dict
```

`EEGRecording.data` is `[channels, samples]` in **volts**. NPZ input requires `data`, scalar `sfreq`, and a string array `channels`. Object arrays are unsupported. MNE readers calibrate to volts. Loaders exclude channels marked bad in MNE metadata.

Preprocessing returns continuous filtered data and accepted, non-overlapping window starts. It does not apply population-fitted normalization. All sessions supplied to feature extraction must belong to one subject and dataset.

## Feature order

For q nodes, width is `4*q + 2*(q*(q-1)/2)`. For q=4 it is 28.

1. Node-major power: node 0 delta/theta/alpha/beta, then node 1, and so on.
2. Edge median correlations in `(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)` order.
3. Edge IQRs in the same order.

The configured node sequence is authoritative. Channel matching ignores case but does not guess BioSemi/standard-electrode equivalences. Bands must be ordered delta, theta, alpha, beta. Changing feature definitions requires new features and trained bundles.

Feature archives use numeric arrays `X`, `y`, string `subject_ids`, and JSON `metadata`. They retain configuration, its hash, feature names, provenance, and quality counts. No feature extraction function consumes labels. Schema-1 archives may also contain `connectivity_median`, `connectivity_iqr`, `window_counts`, `accepted_window_counts`, `window_connectivity`, `window_subject_index`, and string `window_recording_ids`. The schema describes every feature's index, source node or edge, units, and aggregation, and must exactly match the configured feature definitions and order. `subject_features.npz` uses the sibling `feature_schema.json`; other archives use `<archive-stem>.feature_schema.json`. Older archives without graph arrays or an embedded schema remain readable, using their original configured feature names. An unrelated shared sidecar is not authoritative for differently named archives. Exact shapes and a team handoff are in [FEATURES.md](FEATURES.md).

Validation applies before saving, loading, and evaluation. Connectivity medians lie in `[-1,1]` and IQRs in `[0,2]`. Available graphs must match the corresponding `X` columns. Available window tables must cover every subject, reconcile with used-window counts, and reproduce the feature medians/IQRs.

## Model input

Every model first applies a StandardScaler fitted on its training fold. Then `angles = (pi/2) * tanh(z/2)`. The RBF and amplitude comparisons receive the same bounded features as the quantum method. The circuit bandwidth is applied inside quantum state preparation.

States have `[subjects, 2**q]` shape. The precomputed-kernel SVM requires all training-reference columns in their original order. Do not sort or truncate reference states to support vectors without implementing a compatible adapter.

## Bundle schema 1

Joblib contains a `FittedModel`, plus metadata: model name, fold/run IDs, dataset/data kind, config hash, feature fingerprint, training and held-out IDs, class mapping, and versions. The fitted object contains its scaler, SVM, reference angles/states, raw training medians, and full Config.

Load only project-generated, trusted bundles; joblib uses Python object serialization. The application selects local bundles created by this workspace.

Inference uses the bundle configuration, verifies the dataset, and blocks training subjects. Ground truth is displayed from evaluation metadata only after prediction. Positive SVM decision scores favor ASD; negative scores favor Control. These scores are not probabilities.

## Results

A run is usable only when `summary.json` has `status: complete`. Failed runs preserve their error and are not displayed as results. Each subject has one out-of-fold prediction per model. `folds.csv` assigns each subject to its test fold. AUC is computed per fold and then averaged because SVM margins from separate models need not share a scale.

`make_subject_folds` validates one nonempty string ID and an explicit binary label per subject, and generates partitions once for all models. `summarize_oof` requires subject IDs, fold indices, models, true/predicted labels, and finite decision scores. It rejects duplicate subject/model rows, conflicting labels, or differing cohort/fold assignments. All classification metrics and the mean, population standard deviation, and defined-fold count for AUC regenerate from `oof_predictions.csv`; wall-clock runtime does not. Undefined single-class fold AUCs are blank in CSV and excluded from summaries. The run records the source checkout commit, or `unknown` for installations without Git metadata.
