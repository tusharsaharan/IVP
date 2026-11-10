"""Prepare an ignored browser wrapper bound to a frozen, preselected held-out case.

The public JavaScript contains no participant selections or results. This preparer
reads trusted local evidence, verifies it against OOF and the feature archive, and
writes private settings/code into a NEW ignored artifacts directory. It does not
change workspace profiles, experiments, or model settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from asd_eeg.config import load_config
from asd_eeg.data.manifest import read_manifest
from asd_eeg.features.cache import load_features
from asd_eeg.inference.artifacts import check_run_workspace, read_results, read_run
from asd_eeg.training.model import MODELS
from asd_eeg.workspace import check_feature_workspace


def read_json(path: Path) -> dict:
    def reject(value):
        raise ValueError(f"Local evidence contains a nonfinite JSON value: {value}.")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path, run: Path, manifest: Path, config_path: Path, features: Path,
            case_json: Path, freeze: Path, output: Path, url: str) -> Path:
    root, run, output = root.resolve(), run.resolve(), output.resolve()
    if not output.is_relative_to(root / "artifacts") or output == root / "artifacts":
        raise ValueError("Private demo settings must be in a new ignored artifacts subdirectory.")
    if url not in {"http://localhost:8501/", "http://127.0.0.1:8501/"}:
        raise ValueError("The demo must use the local Streamlit app, not a remote endpoint.")
    config = load_config(config_path)
    table = read_manifest(manifest)
    dataset = load_features(features)
    check_feature_workspace(dataset, manifest, config)
    summary, folds = read_run(run)
    check_run_workspace(summary, folds, table, config, run, provenance_root=root / "artifacts")
    metrics, _ = read_results(run)
    expected = read_json(case_json)
    frozen = read_json(freeze)
    for key in ("config_id", "dataset", "data_kind"):
        if (summary[key] != dataset.metadata[key]
                or expected[key] != summary[key]):
            raise ValueError(f"Frozen run/archive/case {key} do not agree.")
    if summary["data_kind"] != "recorded" or expected["provenance"] != "held-out":
        raise ValueError("The browser demo requires a real held-out case, not a synthetic result.")
    subject = expected["subject_id"]
    if (frozen["config_id"] != summary["config_id"]
            or subject not in frozen["cases_selected_before_evaluation"]
            or frozen["feature_reproduction"]["archive_sha256"] != digest(features)):
        raise ValueError("Case/config/features differ from the pre-evaluation protocol freeze.")
    if expected["run_id"] != run.name or expected["model"] != "quantum":
        raise ValueError("Select a quantum case from the explicitly supplied run.")
    row = folds[folds.subject_id.eq(subject)]
    if len(row) != 1 or int(row.iloc[0].fold) != expected["fold"]:
        raise ValueError("The held-out case does not belong to its frozen fold.")
    oof = pd.read_csv(run / "oof_predictions.csv", dtype={"subject_id": str})
    prediction = oof[oof.subject_id.eq(subject) & oof.model.eq("quantum")]
    if (len(prediction) != 1
            or int(prediction.iloc[0].predicted_label) != expected["predicted_label"]
            or int(prediction.iloc[0].true_label) != expected["evaluation_truth_after_prediction"]
            or not np.isclose(prediction.iloc[0].decision_score, expected["decision_score"],
                              atol=1e-9, rtol=1e-9)):
        raise ValueError("Private case prediction/truth differs from authoritative OOF.")
    index = dataset.subject_ids.tolist().index(subject)
    if not np.allclose(expected["feature_vector"], dataset.X[index], atol=1e-9, rtol=1e-9):
        raise ValueError("Private case raw-input features differ from the frozen archive.")
    # This fingerprint uses the evaluation runner's unchanged existing contract.
    fingerprint = hashlib.sha256(dataset.X.tobytes() + dataset.y.tobytes()
                                 + "\n".join(dataset.subject_ids).encode()).hexdigest()
    if summary["feature_fingerprint"] != fingerprint:
        raise ValueError("The supplied run was not evaluated from this feature archive.")
    low, high = (100 * f([value["accuracy"] for value in metrics.values()]) for f in (min, max))
    majority = 100 * max(summary["class_counts"].values()) / summary["subjects"]
    caption = (f"Accuracy {low:g}–{high:g}%; majority-class baseline {majority:g}%. "
               "No quantum advantage or clinical utility established.")
    def local(path):
        path = Path(path).resolve()
        return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)

    settings = {"url": url, "output": local(output), "expected": expected,
                "workspace": {"Manifest CSV": local(manifest),
                              "Pipeline configuration": local(config_path),
                              "Feature archive": local(features), "Results directory": local(run.parent)},
                "subjects": summary["subjects"], "recordings": len(table),
                "preview_recording": sorted(table[table.subject_id.eq(subject)].recording_id)[0],
                "model_label": MODELS["quantum"], "aggregate_caption": caption,
                "oof_sha256": digest(run / "oof_predictions.csv"),
                "metrics_sha256": digest(run / "metrics.json")}
    source = (root / "scripts/record_tushar_demo.js").read_text(encoding="utf-8")
    output.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(settings, allow_nan=False)
    (output / "demo_settings.json").write_text(encoded, encoding="utf-8")
    wrapper = output / "record_demo.private.js"
    wrapper.write_text(f"async page => (\n{source}\n)(page, {encoded})\n", encoding="utf-8")
    return wrapper


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "config", "features", "case-json", "freeze", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--url", default="http://localhost:8501/")
    args = parser.parse_args(argv)
    wrapper = prepare(Path(__file__).resolve().parents[1], args.run, args.manifest, args.config,
                      args.features, args.case_json, args.freeze, args.output, args.url)
    print(f"Private wrapper ready: {wrapper.name} in {wrapper.parent.name}. Keep it local.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
