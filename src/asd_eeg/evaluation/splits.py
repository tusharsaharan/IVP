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
    subjects = np.asarray(subject_ids, dtype=str)
    labels = np.asarray(y, dtype=int)
    if subjects.shape != labels.shape:
        raise ValueError("Subject IDs and labels must align.")
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
    assert frame.groupby("subject_id").size().eq(1).all()
    assert set(frame["fold"]) == set(range(folds))
    return frame


def save_folds(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the folds table. The caller picks the directory.

    Real runs use their ``results/<run_id>/`` folder; synthetic smoke runs
    use ``results/_synthetic_smoke/``. Never write synthetic folds to
    ``results/`` proper.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
