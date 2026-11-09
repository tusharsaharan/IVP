"""Record label-independent QC exclusions before evaluation, using the shared preprocessor."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from asd_eeg.config import Config
from asd_eeg.data.loaders import load_recording
from asd_eeg.data.manifest import read_manifest
from asd_eeg.preprocessing.pipeline import preprocess_recording


def screen_manifest(manifest: str | Path, config: Config, output: str | Path) -> dict:
    table = read_manifest(manifest)
    run = Path(output).resolve() / (
        datetime.now(timezone.utc).strftime("screen-%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
    run.mkdir(parents=True)
    audit, accepted = [], []
    for _, row in table.iterrows():
        record = {"subject_id": row.subject_id, "recording_id": row.recording_id}
        try:
            processed = preprocess_recording(load_recording(row), config)
            audit.append({**record, "status": "included", "reason": "Passed fixed recording QC",
                          "quality": processed.quality})
            accepted.append(row.to_dict())
        except (OSError, ValueError, RuntimeError) as exc:
            audit.append({**record, "status": "excluded", "reason": str(exc)})
    included = pd.DataFrame(accepted, columns=table.columns)
    excluded_subjects = sorted(set(table.subject_id) - set(included.subject_id))
    eligible_path = run / "recordings.csv"
    included["eeg_path"] = [Path(os.path.relpath(path, run)).as_posix()
                            for path in included.eeg_path]
    included.to_csv(eligible_path, index=False)
    (run / "recording_qc.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in row.items() if k != "quality"} for row in audit]).to_csv(
        run / "recording_qc.csv", index=False)
    counts = included.drop_duplicates("subject_id").label.value_counts().to_dict()
    summary = {"status": "complete" if counts.get(0, 0) >= 2 and counts.get(1, 0) >= 2 else "failed",
               "dataset": str(table.dataset.iloc[0]), "data_kind": str(table.data_kind.iloc[0]),
               "config_id": config.id, "config": config.to_dict(),
               "input_recordings": len(table), "input_subjects": int(table.subject_id.nunique()),
               "included_recordings": len(included), "included_subjects": int(included.subject_id.nunique()),
               "class_counts": {"Control": int(counts.get(0, 0)), "ASD": int(counts.get(1, 0))},
               "excluded_subjects": excluded_subjects, "manifest": str(eligible_path),
               "selection": "All recordings passing fixed label-independent QC; no outcome selection"}
    (run / "screen_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["status"] == "failed":
        raise ValueError(f"QC leaves fewer than two subjects per class. Audit retained in {run}.")
    return summary
