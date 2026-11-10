import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "reconcile_sfari", Path(__file__).resolve().parents[1] / "scripts/reconcile_sfari.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def frames():
    candidates = pd.DataFrame([
        {"subject_id": "a", "recording_id": "r1", "label": 0, "dataset": "ds006780"},
        {"subject_id": "a", "recording_id": "r2", "label": 0, "dataset": "ds006780"},
        {"subject_id": "b", "recording_id": "r3", "label": 1, "dataset": "ds006780"},
    ])
    right = candidates.iloc[[0, 2]].copy()
    right["dataset"] = "SFARI"
    return candidates.copy(), right, candidates


def test_reconciliation_preserves_different_identifiers_and_locates_omissions():
    report = module.reconcile(*frames())
    assert report["subject_sets_match"]
    assert report["left_only_recordings"] == ["r2"]
    assert report["right_only_recordings"] == []
    assert report["right"]["dataset"] == "SFARI"


def test_reconciliation_rejects_label_mismatch():
    left, right, candidates = frames()
    right.loc[right.subject_id.eq("b"), "label"] = 0
    with pytest.raises(ValueError, match="conflicts"):
        module.reconcile(left, right, candidates)


def test_reconciliation_rejects_unacquired_recordings():
    left, right, candidates = frames()
    right.loc[right.recording_id.eq("r3"), "recording_id"] = "unacquired"
    with pytest.raises(ValueError, match="absent"):
        module.reconcile(left, right, candidates)
