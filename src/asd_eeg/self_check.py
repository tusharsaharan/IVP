"""Explicitly synthetic integration check. Never mixed with recorded-cohort results."""

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from asd_eeg.config import Config
from asd_eeg.evaluation.run_experiments import run_evaluation
from asd_eeg.features.cache import build_features
from asd_eeg.inference.export import export_prediction
from asd_eeg.inference.pipeline import analyze_subject


def create_synthetic_manifest(config: Config, output: Path, subjects: int = 12) -> Path:
    raw = output / "recordings"
    raw.mkdir(parents=True, exist_ok=True)
    channels = tuple(c for n in config.nodes for c in n.channels)
    fs = config.preprocessing.target_sfreq
    duration = config.preprocessing.window_seconds * max(4, config.preprocessing.min_windows + 1)
    t = np.arange(round(fs * duration)) / fs
    rng = np.random.default_rng(42)
    rows = []
    for index in range(subjects):
        sid = f"synthetic-{index + 1:03d}"
        waves = []
        # Identical generative process across labels. No biological interpretation is implied.
        for _ in channels:
            envelope = 1 + .25 * np.sin(2 * np.pi * .4 * t + rng.uniform(0, 6))
            wave = envelope * 16e-6 * np.sin(2 * np.pi * rng.uniform(8, 12) * t + rng.uniform(0, 6))
            wave += 7e-6 * np.sin(2 * np.pi * rng.uniform(2, 6) * t)
            waves.append(wave + rng.normal(0, 4e-6, len(t)))
        np.savez_compressed(raw / f"{sid}.npz", data=np.asarray(waves), sfreq=np.array(fs),
                            channels=np.array(channels))
        rows.append({"dataset": "synthetic-software-check", "subject_id": sid,
                     "recording_id": f"{sid}-rest", "eeg_path": f"recordings/{sid}.npz",
                     "label": index % 2, "label_source": "Generated labels for software verification only",
                     "data_kind": "synthetic"})
    manifest = output / "recordings.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def run_self_check(config: Config, output: str | Path = "artifacts/software-check") -> dict:
    root = Path(output).resolve() / f"check-{uuid4().hex[:8]}"
    config = replace(config, evaluation=replace(config.evaluation, folds=3))
    manifest = create_synthetic_manifest(config, root)
    dataset = build_features(manifest, config, root / "features.npz")
    run = run_evaluation(dataset, root / "runs")
    fold_table = pd.read_csv(run / "folds.csv", dtype={"subject_id": str})
    held_out = str(fold_table[fold_table.fold == 0].iloc[0].subject_id)
    prediction = analyze_subject(manifest, held_out, run / "fold_0" / "quantum.joblib")
    oof = pd.read_csv(run / "oof_predictions.csv")
    expected = oof[(oof.subject_id == held_out) & (oof.model == "quantum")].iloc[0]
    if prediction["predicted_label"] != int(expected.predicted_label) or not np.isclose(
        prediction["decision_score"], expected.decision_score, atol=1e-10
    ):
        raise ValueError("Reloaded inference differs from the saved held-out prediction.")
    export_prediction(prediction, root / "prediction.json")
    return {"status": "passed", "data_kind": "synthetic", "directory": str(root),
            "manifest": str(manifest), "run": str(run), "held_out_subject": held_out,
            "note": "Software verification only. These are not research results."}
