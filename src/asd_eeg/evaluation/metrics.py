import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "subjects": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": np.asarray(matrix).tolist(),
    }


def fold_auc(y_true, decision_scores) -> float | None:
    """ROC-AUC for one fold, or None when the test fold holds one class."""
    labels = np.asarray(y_true)
    scores = np.asarray(decision_scores, dtype=float)
    if labels.ndim != 1 or not len(labels) or scores.shape != labels.shape:
        raise ValueError("Fold labels and scores must be nonempty aligned vectors.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Fold AUC needs binary labels and finite decision scores.")
    if len(set(labels.tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def summarize_oof(predictions: pd.DataFrame) -> dict:
    """Recompute pooled metrics purely from an oof_predictions frame.

    Expects one row per subject/model with subject_id, fold, model,
    true_label, predicted_label, and decision_score. Models must share
    the same subjects, labels, and fold assignments.
    Scores are uncalibrated SVM decision scores, never pooled across folds
    for AUC: per-fold AUC is averaged only over folds where it is defined.
    """
    required = {"subject_id", "fold", "model", "true_label", "predicted_label", "decision_score"}
    if predictions.empty or not required.issubset(predictions.columns):
        raise ValueError(f"OOF predictions need columns: {sorted(required)}.")
    predictions = predictions.copy()
    for column in ("subject_id", "model"):
        if predictions[column].isna().any():
            raise ValueError(f"OOF predictions need nonempty {column} values.")
        predictions[column] = predictions[column].astype(str)
        if predictions[column].str.strip().eq("").any():
            raise ValueError(f"OOF predictions need nonempty {column} values.")
    for column in ("true_label", "predicted_label", "fold", "decision_score"):
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"OOF {column} values must be finite numeric values.")
        if column.endswith("label") and not np.isin(values, [0, 1]).all():
            raise ValueError("OOF labels must be 0 (Control) or 1 (ASD).")
        if column == "fold" and (np.any(values < 0) or np.any(values != np.floor(values))):
            raise ValueError("OOF folds must be nonnegative integer indices.")
        predictions[column] = values
    if predictions.duplicated(["subject_id", "model"]).any():
        raise ValueError("Each subject must have exactly one OOF prediction per model.")
    if predictions.groupby("subject_id")["true_label"].nunique().max() > 1:
        raise ValueError("OOF predictions have conflicting subject labels across models.")
    reference = None
    for _, rows in predictions.groupby("model"):
        assignments = set(rows[["subject_id", "fold", "true_label"]].itertuples(index=False,
                                                                                name=None))
        if reference is None:
            reference = assignments
        elif assignments != reference:
            raise ValueError("All models must use identical OOF subjects and fold assignments.")
    summary = {}
    for name, rows in predictions.groupby("model"):
        metrics = classification_metrics(rows.true_label, rows.predicted_label)
        aucs = [
            auc
            for _, fold_rows in rows.groupby("fold")
            if (auc := fold_auc(fold_rows.true_label, fold_rows.decision_score)) is not None
        ]
        metrics["mean_fold_auc"] = float(np.mean(aucs)) if aucs else None
        metrics["std_fold_auc"] = float(np.std(aucs)) if aucs else None
        metrics["fold_auc_count"] = len(aucs)
        summary[name] = metrics
    return summary
