"""Compare preserved SFARI handoffs without accessing private EEG paths or merging cohorts."""

import argparse
import json
from pathlib import Path

import pandas as pd


def reconcile(left: pd.DataFrame, right: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    for frame in (left, right, candidates):
        required = {"subject_id", "recording_id", "label", "dataset"}
        if frame.empty or not required.issubset(frame):
            raise ValueError("Each manifest needs nonempty subject/recording/label/dataset metadata.")
        if frame.recording_id.duplicated().any() or frame[list(required)].isna().any().any():
            raise ValueError("Manifest identifiers must be unique and metadata complete.")
        if not frame.label.isin([0, 1]).all() or frame.dataset.nunique() != 1:
            raise ValueError("Each manifest needs one dataset and explicit binary labels.")
        if frame.groupby("subject_id").label.nunique().gt(1).any():
            raise ValueError("Subject labels conflict within a manifest.")
    all_ids = set(candidates.recording_id)
    if not set(left.recording_id).issubset(all_ids) or not set(right.recording_id).issubset(all_ids):
        raise ValueError("A screened handoff contains recordings absent from the frozen candidates.")
    source = candidates.set_index("recording_id")
    for frame in (left, right):
        expected = source.loc[frame.recording_id, ["subject_id", "label"]].reset_index(drop=True)
        actual = frame[["subject_id", "label"]].reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError("A handoff subject/label mapping conflicts with source candidates.")
    left_ids, right_ids = set(left.recording_id), set(right.recording_id)
    only_left, only_right = sorted(left_ids - right_ids), sorted(right_ids - left_ids)
    affected = sorted(set(source.loc[only_left + only_right, "subject_id"]))
    return {
        "status": "comparison_complete_not_owner_approval",
        "left": {"dataset": str(left.dataset.iloc[0]), "recordings": len(left),
                 "subjects": int(left.subject_id.nunique())},
        "right": {"dataset": str(right.dataset.iloc[0]), "recordings": len(right),
                  "subjects": int(right.subject_id.nunique())},
        "subject_sets_match": set(left.subject_id) == set(right.subject_id),
        "all_subject_labels_match_candidates": True,
        "left_only_recordings": only_left, "right_only_recordings": only_right,
        "affected_subjects": [
            {"subject_id": subject, "candidate_recordings": int(candidates.subject_id.eq(subject).sum()),
             "left_recordings": int(left.subject_id.eq(subject).sum()),
             "right_recordings": int(right.subject_id.eq(subject).sum())}
            for subject in affected
        ],
        "limits": [
            "Different dataset identifiers/configuration IDs remain incompatible; no normalization applied.",
            "The maximum subject feature-window cap does not affect recording-level QC eligibility.",
            "The older pre-QC 517-run manifest/audit is needed to attribute its 11-run shortfall.",
            "This comparison does not establish why the older handoff omitted each recording.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "candidates", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    report = reconcile(*(pd.read_csv(path, dtype={"subject_id": str, "recording_id": str})
                         for path in (args.left, args.right, args.candidates)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"report": str(args.output), "left_only": len(report["left_only_recordings"]),
                      "right_only": len(report["right_only_recordings"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
