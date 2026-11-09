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
    if len(set(np.asarray(y_true).tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, decision_scores))


def summarize_oof(predictions: pd.DataFrame) -> dict:
    """Recompute pooled metrics purely from an oof_predictions frame.

    Expects columns: model, true_label, predicted_label, decision_score.
    Scores are uncalibrated SVM decision scores, never pooled across folds
    for AUC: per-fold AUC is averaged only over folds where it is defined.
    """
    required = {"model", "true_label", "predicted_label", "decision_score"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"OOF predictions need columns: {sorted(required)}.")
    summary = {}
    for name, rows in predictions.groupby("model"):
        metrics = classification_metrics(rows.true_label, rows.predicted_label)
        aucs = [
            auc
            for _, fold_rows in rows.groupby("fold")
            if (auc := fold_auc(fold_rows.true_label, fold_rows.decision_score)) is not None
        ]
        metrics["mean_fold_auc"] = float(np.mean(aucs)) if aucs else None
        metrics["fold_auc_count"] = len(aucs)
        summary[name] = metrics
    return summary
