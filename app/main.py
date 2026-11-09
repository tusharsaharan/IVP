"""Light research workspace. Run: python -m streamlit run app/main.py"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from asd_eeg.config import load_config
from asd_eeg.data.loaders import load_recording
from asd_eeg.data.manifest import read_manifest
from asd_eeg.evaluation.run_experiments import run_evaluation
from asd_eeg.explainability.perturbation import connection_sensitivity
from asd_eeg.features.cache import build_features, load_features
from asd_eeg.inference.compatibility import load_bundle
from asd_eeg.inference.pipeline import analyze_subject
from asd_eeg.models.quantum_circuit import circuit_text
from asd_eeg.self_check import run_self_check
from asd_eeg.training.model import MODELS
from asd_eeg.visualization.plots import confusion_figure, connectivity_figure, signal_figure
from asd_eeg.workspace import check_feature_workspace, workspace_paths

ROOT = Path(os.environ.get("NEUROQ_ROOT", Path(__file__).resolve().parents[1])).resolve()
PAGES = ["Workspace", "Datasets", "Analysis", "Experiments", "Team", "Software check"]
TEAM = [
    ("Vivek G", "Quantum models", "models/, training/", "Validate circuits, kernels, and saved model bundles."),
    ("Sahil Gupta", "Data and preprocessing", "data/, preprocessing/", "Verify labels, channel mapping, recording quality, and cohort eligibility."),
    ("Dodiya Mahavirsinh", "Evaluation", "evaluation/, baselines/", "Freeze subject folds, compare models, and export reproducible metrics."),
    ("Aryan Pandey", "EEG features", "features/", "Review power, connectivity, window aggregation, and the feature schema."),
    ("Tushar Saharan", "Integration", "inference/, app/", "Connect modules, check compatibility, and prepare the application demo."),
    ("Shivam Mall", "Explanations and figures", "explainability/, visualization/", "Review perturbation sensitivity and produce scientific figures."),
    ("Pranav Karhale", "Verification and submission", "tests/, docs/", "Verify installation and assemble the report, presentation, and AVI."),
]

st.set_page_config(page_title="NeuroQ-Connect", layout="wide", initial_sidebar_state="expanded")
try:
    PATHS = workspace_paths(ROOT)
except (OSError, ValueError) as exc:
    st.error(f"Could not read data/local/workspace.json: {exc}")
    st.stop()
st.markdown("""
<style>
:root { --paper: oklch(0.98 0.003 120); --ink: #243342; --muted: #526273; --rule: #dce2e5; }
.stApp { background: var(--paper); }
.block-container { max-width: 1180px; padding-top: 2.8rem; padding-bottom: 3rem; }
h1 { font-size: 2.5rem !important; font-weight: 650 !important; letter-spacing: -0.045em; }
h2 { font-size: 1.4rem !important; font-weight: 600 !important; letter-spacing: -0.02em; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }
p, li { line-height: 1.6; }
[data-testid="stSidebar"] { background: #eef1f0; border-right: 1px solid var(--rule); }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
.eyebrow { color: var(--muted); font-size: .74rem; font-weight: 600; letter-spacing: .12em; margin-bottom: .4rem; }
.lede { color: var(--muted); font-size: 1.05rem; max-width: 70ch; margin-bottom: 1.8rem; }
.pipeline { display: flex; gap: 1.5rem; padding: 1.15rem 0; border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule); margin: .5rem 0 1.8rem; flex-wrap: wrap; }
.pipeline span { flex: 1; min-width: 130px; font-size: .9rem; color: var(--ink); }
.pipeline small { display: block; color: var(--muted); margin-bottom: .2rem; font-size: .73rem; }
.stButton button, .stDownloadButton button { border-radius: 6px; min-height: 2.65rem; }
.stButton button:focus-visible, .stDownloadButton button:focus-visible { outline: 2px solid #335e83; outline-offset: 3px; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
hr { border-color: var(--rule); }
@media (max-width: 700px) { .block-container { padding: 4rem 1rem 2rem; } h1 { font-size: 2rem !important; } }
</style>
""", unsafe_allow_html=True)


def local_path(value: str) -> Path:
    return (ROOT / value).resolve()


def navigate(page: str) -> None:
    st.session_state["page"] = page


def heading(title: str, description: str) -> None:
    st.markdown('<div class="eyebrow">GROUP 8 / RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="lede">{description}</p>', unsafe_allow_html=True)


def show_figure(fig, filename: str | None = None) -> None:
    st.pyplot(fig, width="stretch")
    if filename:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        st.download_button("Download figure", buffer.getvalue(), filename, "image/png", key=filename)
    plt.close(fig)


def completed_runs() -> list[tuple[Path, dict]]:
    runs = []
    for path in sorted(local_path(PATHS["results"]).glob("run-*/summary.json"), reverse=True):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("status") == "complete" and summary.get("data_kind") == "recorded":
                runs.append((path.parent, summary))
        except (OSError, ValueError):
            continue
    return runs


def manifest_state() -> tuple[pd.DataFrame | None, str | None]:
    try:
        return read_manifest(manifest_path), None
    except (OSError, ValueError) as exc:
        return None, str(exc)


with st.sidebar:
    st.markdown("### NeuroQ-Connect")
    st.caption("EEG classification research")
    page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.caption("LOCAL WORKSPACE")
    st.write("Classical quantum-circuit simulation")
    st.caption("Verified recordings and saved experiments stay on this computer.")
    with st.expander("Local paths"):
        st.text_input("Manifest CSV", PATHS["manifest"], key="manifest_path")
        st.text_input("Pipeline configuration", PATHS["config"], key="config_path")
        st.text_input("Feature archive", PATHS["features"], key="features_path")
    st.caption("Group 8 · Course project")

manifest_path = local_path(st.session_state["manifest_path"])
feature_path = local_path(st.session_state["features_path"])
try:
    config = load_config(local_path(st.session_state["config_path"]))
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"Could not read the pipeline configuration: {exc}")
    st.write("Correct the configuration path under Local paths, then reload.")
    st.stop()


def workspace_page() -> None:
    heading("NeuroQ-Connect", "Prepare EEG data and compare quantum and classical models.")
    st.markdown("""<div class="pipeline">
      <span><small>01 / DATA</small>Verify recordings</span>
      <span><small>02 / FEATURES</small>Measure connectivity</span>
      <span><small>03 / MODELS</small>Encode and classify</span>
      <span><small>04 / EVALUATION</small>Review held-out results</span>
    </div>""", unsafe_allow_html=True)
    table, error = manifest_state()
    runs = completed_runs()
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.subheader("Start with verified recordings")
        st.write("Add your dataset manifest, confirm the electrode mapping, and build one feature row per subject. The same pipeline powers the experiments and recording analysis.")
        st.button("Set up dataset", type="primary", on_click=navigate, args=("Datasets",))
        st.markdown("#### Workspace status")
        st.dataframe(pd.DataFrame([
            {"Component": "Recording manifest", "State": f"{len(table)} recordings / {table.subject_id.nunique()} subjects" if table is not None else "Needs verified recordings"},
            {"Component": "Subject features", "State": "Archive available" if feature_path.is_file() else "Not built"},
            {"Component": "Recorded-data experiments", "State": f"{len(runs)} completed runs" if runs else "No completed runs"},
        ]), hide_index=True, width="stretch")
        if error and "no recordings" not in error:
            st.caption(error)
    with right:
        st.subheader("Current configuration")
        st.write(f"**{len(config.nodes)} EEG nodes · {config.n_features} subject features**")
        st.write(f"{config.circuit.passes} encoding passes, exact statevector simulation.")
        st.caption("Node mapping must match the verified dataset before evaluation.")
        st.dataframe(pd.DataFrame([{"Node": n.name, "Channels": ", ".join(n.channels)} for n in config.nodes]),
                     hide_index=True, width="stretch")
    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Working without the dataset?")
        st.write("Run the separate synthetic software check to verify feature extraction, all four models, and saved-model inference.")
        st.button("Open software check", on_click=navigate, args=("Software check",))
    with right:
        st.subheader("Build together")
        st.write("Each member has an owned module, a shared interface, and a concrete handoff. Start with the team guide before changing a contract.")
        st.button("View team ownership", on_click=navigate, args=("Team",))


def datasets_page() -> None:
    heading("Datasets", "Connect verified recordings and inspect their signal before building features.")
    st.caption(f"Manifest: {manifest_path}")
    table, error = manifest_state()
    if table is None:
        st.subheader("Add your recording manifest")
        st.write("Fill the CSV with one row per recording. Include the subject ID, a verified label source, and the EEG path. Relative paths are resolved from the CSV's folder.")
        st.code("dataset,subject_id,recording_id,eeg_path,label,label_source,data_kind", language="text")
        st.write("Use label **0** for Control, **1** for ASD, and data_kind **recorded** for real EEG. Keep each subject's sessions under the same subject ID.")
        st.download_button("Download blank manifest", "dataset,subject_id,recording_id,eeg_path,label,label_source,data_kind\n",
                           "recordings.csv", "text/csv")
        if st.button("Validate manifest", type="primary"):
            st.error(error)
        st.caption("Supported continuous recordings: EEGLAB SET, EDF, BDF, FIF, and numeric NPZ. SET companion files must stay beside the recording.")
        return
    st.success(f"Manifest checked: {table.subject_id.nunique()} subjects across {len(table)} recordings.")
    if table.data_kind.iloc[0] == "synthetic":
        st.info("This manifest contains synthetic inputs for software verification.")
    st.dataframe(table.drop(columns="eeg_path"), hide_index=True, width="stretch")
    inspect, build = st.tabs(["Inspect a recording", "Build subject features"])
    with inspect:
        recording_id = st.selectbox("Recording", table.recording_id.tolist())
        if st.button("Load recording", type="primary"):
            try:
                recording = load_recording(table[table.recording_id == recording_id].iloc[0])
                st.write(f"**{recording.sfreq:g} Hz** · **{len(recording.channels)} channels** · "
                         f"**{recording.data.shape[1] / recording.sfreq:.1f} seconds**")
                st.caption("Raw signal preview. Preprocessing and quality checks run during feature extraction.")
                show_figure(signal_figure(recording))
                st.write("Channels: " + ", ".join(recording.channels))
            except (OSError, ValueError) as exc:
                st.error(str(exc))
    with build:
        st.write("The configuration controls channel mapping, filtering, windows, and quality rules. Feature extraction stops with a clear error if a recording cannot meet them.")
        st.caption(f"Output: {feature_path}")
        if st.button("Build features", type="primary"):
            try:
                with st.status("Loading and preprocessing recordings...", expanded=True) as status:
                    st.write("Computing band power and windowed connectivity for each subject.")
                    dataset = build_features(manifest_path, config, feature_path)
                    status.update(label="Subject features saved", state="complete", expanded=False)
                st.success(f"Saved {dataset.X.shape[0]} subjects with {dataset.X.shape[1]} features each.")
            except (OSError, ValueError, KeyError) as exc:
                st.error(str(exc))
        st.button("Open experiments", on_click=navigate, args=("Experiments",))


def display_results(run: Path, summary: dict) -> None:
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(run / "model_comparison.csv")
    st.caption(f"{summary['dataset']} · {summary['subjects']} subjects · {summary['folds']} subject-wise folds · Seed {summary['seed']}")
    st.write(summary["protocol"] + ".")
    columns = ["model", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "mean_fold_auc"]
    optional = {"std_fold_auc": "Fold AUC SD", "fold_auc_count": "Defined AUC folds"}
    columns.extend(column for column in optional if column in comparison)
    display = comparison[columns].copy()
    display["model"] = display.model.map(MODELS)
    display.columns = ["Model", "Balanced accuracy", "Macro-F1", "Sensitivity", "Specificity", "Mean fold AUC"] + [optional[c] for c in columns[6:]]
    formatting = {c: "{:.3f}" for c in display.columns[1:]}
    if "Defined AUC folds" in display:
        formatting["Defined AUC folds"] = "{:.0f}"
    st.dataframe(display.style.format(formatting, na_rep="Undefined"), hide_index=True, width="stretch")
    st.caption("AUC is computed within each held-out fold, then averaged. Decision scores are not pooled across differently fitted SVMs.")
    st.caption("Fold AUC SD is descriptive spread, not a confidence interval. This is a research comparison, not a clinical diagnostic tool.")
    selected = st.selectbox("Confusion matrix model", list(MODELS), format_func=MODELS.get)
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        show_figure(confusion_figure(metrics[selected]["confusion_matrix"]), "confusion-matrix.png")
    with right:
        st.subheader("Experiment record")
        st.write(f"Configuration: `{summary['config_id']}`")
        st.write(f"Run: `{summary['run_id']}`")
        st.write(f"Control subjects: {summary['class_counts']['Control']}")
        st.write(f"ASD subjects: {summary['class_counts']['ASD']}")
        st.download_button("Download predictions", (run / "oof_predictions.csv").read_bytes(), "oof_predictions.csv", "text/csv")
        st.download_button("Download metrics", (run / "metrics.json").read_bytes(), "metrics.json", "application/json")


def experiments_page() -> None:
    heading("Experiments", "Compare all four models on the same held-out subjects.")
    st.subheader("Run the configured protocol")
    st.write("The starter uses fixed settings and training-fold-only transformations. Every run saves its folds, predictions, configuration, and model bundles.")
    ready = False
    if feature_path.is_file():
        try:
            dataset = load_features(feature_path)
            check_feature_workspace(dataset, manifest_path, config)
            ready = dataset.metadata["data_kind"] == "recorded"
            st.caption(f"Feature archive: {len(dataset.y)} subjects · {dataset.X.shape[1]} features · {dataset.metadata['dataset']}")
            if not ready:
                st.info("Use Software check for synthetic inputs. This page displays recorded-data experiments.")
        except (OSError, ValueError, KeyError) as exc:
            st.error(str(exc))
    else:
        st.write("Build a subject feature archive from Datasets to enable evaluation.")
    if st.button("Run four-model comparison", type="primary", disabled=not ready):
        try:
            with st.status("Evaluating subject-wise folds...", expanded=True) as status:
                st.write("Fitting the RBF, amplitude, quantum, and reassigned-connection models.")
                run_evaluation(dataset, local_path(PATHS["results"]))
                status.update(label="Experiment saved", state="complete", expanded=False)
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
    st.divider()
    st.subheader("Saved results")
    runs = completed_runs()
    if not runs:
        st.write("No completed recorded-data experiments yet. Results appear here after evaluation succeeds.")
        return
    index = st.selectbox("Experiment run", range(len(runs)), format_func=lambda i: runs[i][0].name)
    display_results(*runs[index])


def analysis_page() -> None:
    heading("Analysis", "Inspect a held-out subject using the model from its test fold.")
    runs = completed_runs()
    if not runs:
        st.subheader("A saved fold model is needed")
        st.write("Complete an experiment first. The workspace will match each selected subject to the fold where it was held out.")
        st.button("Open experiments", type="primary", on_click=navigate, args=("Experiments",))
        return
    index = st.selectbox("Experiment", range(len(runs)), format_func=lambda i: runs[i][0].name)
    run, summary = runs[index]
    folds = pd.read_csv(run / "folds.csv", dtype={"subject_id": str})
    left, right = st.columns(2)
    with left:
        subject = st.selectbox("Held-out subject", folds.subject_id.tolist())
    with right:
        model_name = st.selectbox("Model", list(MODELS), index=2, format_func=MODELS.get)
    fold = int(folds[folds.subject_id == subject].iloc[0].fold)
    bundle_path = run / f"fold_{fold}" / f"{model_name}.joblib"
    st.caption(f"Uses fold {fold + 1}. The saved pipeline settings and training-reference order are restored automatically.")
    key = (str(manifest_path), str(bundle_path), subject)
    if st.button("Analyze subject", type="primary"):
        try:
            with st.status("Processing the held-out recording...", expanded=True) as status:
                st.write("Applying the saved pipeline and checking subject provenance.")
                result = analyze_subject(manifest_path, subject, bundle_path)
                status.update(label="Analysis complete", state="complete", expanded=False)
            st.session_state["analysis"] = (key, result)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            st.error(str(exc))
    cached = st.session_state.get("analysis")
    if not cached or cached[0] != key:
        return
    result = cached[1]
    bundle = load_bundle(bundle_path)
    st.divider()
    st.subheader(f"Prediction: {result['predicted_class']}")
    st.write(f"Decision score: **{result['decision_score']:.4f}**. Positive favors ASD; negative favors Control.")
    st.caption(f"{result['window_count']} accepted windows · {result['processing_seconds']:.2f} seconds · Held-out subject verified")
    verified = int(folds[folds.subject_id == subject].iloc[0].label)
    st.write(f"Verified dataset label: **{'ASD' if verified else 'Control'}**")
    graph_tab, circuit_tab, explanation_tab, details_tab = st.tabs(["Connectivity", "Circuit", "Sensitivity", "Record"])
    with graph_tab:
        show_figure(connectivity_figure(result["connectivity"], result["node_names"]), "connectivity.png")
        st.caption("Sensor-level statistical connectivity, summarized across accepted EEG windows.")
    with circuit_tab:
        if model_name == "quantum":
            angles = bundle["model"].angles(np.array(result["feature_vector"])[None, :])[0]
            st.code(circuit_text(angles, bundle["model"].config), language="text")
            st.caption("Actual gate angles for this subject. Rotations encode activity; IsingZZ gates encode connections.")
        else:
            st.write("Select Connectivity quantum kernel to inspect its subject-specific circuit.")
    with explanation_tab:
        st.write("Replace each connection's median and variability with the training subjects' reference values, then measure the change in the decision score.")
        if st.button("Compute connection sensitivity"):
            sensitivity = connection_sensitivity(bundle, np.array(result["feature_vector"]))
            frame = pd.DataFrame(sensitivity)
            st.dataframe(frame, hide_index=True, width="stretch")
            st.bar_chart(frame.set_index("connection")[["score_change"]], horizontal=True, color="#335e83")
            st.caption("This describes model sensitivity to a feature change, not a causal explanation of autism.")
    with details_tab:
        st.json({k: v for k, v in result.items() if k not in {"feature_vector", "connectivity"}})
    st.download_button("Export analysis JSON", json.dumps(result, indent=2), "prediction.json", "application/json")


def team_page() -> None:
    heading("Team", "Clear ownership and stable interfaces for parallel implementation.")
    st.dataframe(pd.DataFrame([{ "Member": row[0], "Responsibility": row[1], "Owned paths": row[2]} for row in TEAM]),
                 hide_index=True, width="stretch")
    st.caption("Python package paths are relative to src/asd_eeg. Full handoff details are in docs/TEAM.md.")
    for name, role, paths, task in TEAM:
        with st.expander(f"{name}: {role}"):
            st.write(task)
            st.code(paths, language="text")
    st.subheader("Shared contracts")
    st.write("EEG arrays use channels by samples in volts. Feature rows represent subjects. Labels are 0 for Control and 1 for ASD. Model bundles preserve channel settings, training-reference order, and held-out IDs.")
    st.code("python -m pytest\npython -m ruff check .", language="shell")
    st.write("Read CONTRIBUTING.md and docs/CONTRACTS.md before changing a shared interface.")


def software_page() -> None:
    heading("Software check", "Verify the complete pipeline using explicitly synthetic EEG.")
    st.subheader("A reproducibility check, not an experiment")
    st.write("This creates 12 generated recordings with artificial labels, runs all four models, reloads a fold bundle, and checks a held-out prediction. Its output stays outside the recorded-data results.")
    st.caption("Synthetic labels do not represent clinical diagnoses. No accuracy from this check is presented as research evidence.")
    if st.button("Run software check", type="primary"):
        try:
            with st.status("Checking the complete pipeline...", expanded=True) as status:
                st.write("Generating signals, extracting features, evaluating folds, and reloading inference.")
                st.session_state["software_check"] = run_self_check(config, ROOT / "artifacts/software-check")
                status.update(label="Software check passed", state="complete", expanded=False)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            st.error(str(exc))
    result = st.session_state.get("software_check")
    if result:
        st.success("Feature extraction, four-model evaluation, and held-out inference completed.")
        st.json(result)
        st.download_button("Download check record", json.dumps(result, indent=2), "software-check.json", "application/json")
    st.subheader("Run it from the terminal")
    st.code("neuroq self-check", language="shell")


{"Workspace": workspace_page, "Datasets": datasets_page, "Analysis": analysis_page,
 "Experiments": experiments_page, "Team": team_page, "Software check": software_page}[page]()
