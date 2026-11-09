"""Owner: Dodiya Mahavirsinh. Frozen subject-wise splits for evaluation.

One row per subject; all recordings of a subject stay together because the
caller passes one label per subject. Splits are stratified, shuffled with
``random_state=42``, and identical for every model: the runner generates them
once and reuses them for all four comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 42
MAX_FOLDS = 5


def make_subject_folds(
    subject_ids: np.ndarray,
    y: np.ndarray,
    n_splits: int = MAX_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Assign each subject to exactly one test fold.

    Returns a frame with columns ``subject_id``, ``fold``, ``label``.
    Raises if subject IDs repeat, a class is missing, or fewer than two
    subjects per class exist.
    """
    subjects = np.asarray(subject_ids)
    labels = np.asarray(y)
    if subjects.ndim != 1 or not len(subjects) or subjects.shape != labels.shape:
        raise ValueError("Subject IDs and labels must be nonempty aligned vectors.")
    if pd.isna(subjects).any() or not all(isinstance(subject, str) for subject in subjects):
        raise ValueError("Subject IDs must be nonempty strings.")
    subjects = subjects.astype(str)
    if np.any(np.char.strip(subjects.astype(str)) == ""):
        raise ValueError("Subject IDs must be nonempty strings.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 (Control) or 1 (ASD); do not coerce diagnosis labels.")
    labels = labels.astype(int)
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError("Request at least two integer folds.")
    if len(set(subjects.tolist())) != len(subjects):
        raise ValueError("Feature rows must represent unique subjects.")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Need both Control (0) and ASD (1) subjects.")
    counts = np.bincount(labels, minlength=2)
    folds = min(n_splits, int(counts.min()))
    if folds < 2:
        raise ValueError("Need at least two verified subjects per class.")
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    fold_of = np.empty(len(subjects), dtype=int)
    for fold, (_, test) in enumerate(splitter.split(subjects, labels)):
        fold_of[test] = fold
    frame = pd.DataFrame({"subject_id": subjects, "fold": fold_of, "label": labels})
    # Every subject tested exactly once; train/test disjoint by construction.
    if (not frame.groupby("subject_id").size().eq(1).all()
            or set(frame["fold"]) != set(range(folds))):
        raise ValueError("Subject fold assignment is incomplete or contains duplicate subjects.")
    return frame


def save_folds(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the folds table. The caller picks the directory.

    Real runs use ``results/<run_id>/``; labelled software checks use
    their separate ``artifacts/software-check/<check_id>/runs/`` folder.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
