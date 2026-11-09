from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from asd_eeg.config import Config
from asd_eeg.contracts import FeatureDataset
from asd_eeg.evaluation.metrics import classification_metrics
from asd_eeg.training.model import MODELS, fit_model


def software_versions() -> dict:
    names = ["numpy", "scipy", "scikit-learn", "pennylane", "mne", "streamlit"]
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in names}}


def run_evaluation(dataset: FeatureDataset, output: str | Path) -> Path:
    dataset.validate()
    config = Config.from_dict(dataset.metadata["config"])
    if dataset.metadata["config_id"] != config.id or dataset.X.shape[1] != config.n_features:
        raise ValueError("Feature configuration does not match the evaluation input.")
    counts = np.bincount(dataset.y.astype(int), minlength=2)
    folds = min(config.evaluation.folds, int(counts.min()))
    if folds < 2:
        raise ValueError("Need at least two verified subjects per class for subject-wise evaluation.")
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    run = Path(output).resolve() / run_id
    run.mkdir(parents=True)
    digest = hashlib.sha256(dataset.X.tobytes() + dataset.y.tobytes()
                            + "\n".join(dataset.subject_ids).encode()).hexdigest()
    versions = software_versions()
    summary = {
        "schema_version": 1, "run_id": run_id, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.metadata["dataset"], "data_kind": dataset.metadata["data_kind"],
        "subjects": len(dataset.y), "class_counts": {"Control": int(counts[0]), "ASD": int(counts[1])},
        "folds": folds, "requested_folds": config.evaluation.folds, "seed": config.evaluation.seed,
        "config_id": config.id, "feature_fingerprint": digest,
        "protocol": "Fixed settings; stratified subject-wise cross-validation; no hyperparameter search",
        "versions": versions,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    splitter = StratifiedKFold(folds, shuffle=True, random_state=config.evaluation.seed)
    predictions, fold_rows, auc_rows = [], [], []
    timings = {name: 0.0 for name in MODELS}
    try:
        for fold, (train, test) in enumerate(splitter.split(dataset.X, dataset.y)):
            train_ids, test_ids = dataset.subject_ids[train].tolist(), dataset.subject_ids[test].tolist()
            if set(train_ids) & set(test_ids):
                raise ValueError("Subject leakage detected in fold generation.")
            fold_path = run / f"fold_{fold}"
            fold_path.mkdir()
            fold_rows.extend({"subject_id": sid, "fold": fold, "label": int(label)}
                             for sid, label in zip(test_ids, dataset.y[test], strict=True))
            for name in MODELS:
                started = time.perf_counter()
                model = fit_model(dataset.X[train], dataset.y[train], config, name)
                predicted, scores = model.predict(dataset.X[test])
                timings[name] += time.perf_counter() - started
                bundle = {"schema_version": 1, "model": model,
                          "metadata": {"model": name, "fold": fold, "run_id": run_id,
                                       "dataset": summary["dataset"], "data_kind": summary["data_kind"],
                                       "config_id": config.id, "feature_fingerprint": digest,
                                       "train_subject_ids": train_ids, "test_subject_ids": test_ids,
                                       "class_mapping": {0: "Control", 1: "ASD"},
                                       "versions": versions}}
                joblib.dump(bundle, fold_path / f"{name}.joblib")
                for sid, label, pred, score in zip(test_ids, dataset.y[test], predicted, scores, strict=True):
                    predictions.append({"subject_id": sid, "fold": fold, "model": name,
                                        "true_label": int(label), "predicted_label": int(pred),
                                        "decision_score": float(score)})
                auc_rows.append({"model": name, "fold": fold,
                                 "auc": float(roc_auc_score(dataset.y[test], scores))})
        frame = pd.DataFrame(predictions)
        frame.to_csv(run / "oof_predictions.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(run / "folds.csv", index=False)
        pd.DataFrame(auc_rows).to_csv(run / "fold_auc.csv", index=False)
        metrics = {}
        for name in MODELS:
            rows = frame[frame.model == name]
            metrics[name] = classification_metrics(rows.true_label, rows.predicted_label)
            metrics[name]["mean_fold_auc"] = float(np.mean([r["auc"] for r in auc_rows if r["model"] == name]))
            metrics[name]["fit_predict_seconds"] = timings[name]
        (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        pd.DataFrame([{ "model": name, **{k: v for k, v in value.items() if k != "confusion_matrix"}}
                      for name, value in metrics.items()]).to_csv(run / "model_comparison.csv", index=False)
        summary["status"] = "complete"
    except Exception as exc:
        summary.update(status="failed", error=str(exc))
        raise
    finally:
        (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run
