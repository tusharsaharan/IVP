"""CLI/pipeline agreement, missing-file errors, shape errors, runner entry."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asd_eeg.cli import main as cli_main
from asd_eeg.inference.compatibility import load_bundle
from asd_eeg.inference.pipeline import analyze_subject


def test_cli_and_pipeline_agree(completed_check, tmp_path):
    output = tmp_path / "prediction.json"
    bundle_path = (
        Path(completed_check["run"]) / "fold_0/quantum.joblib"
    )
    subject = completed_check["held_out_subject"]
    code = cli_main([
        "predict", "--manifest", completed_check["manifest"],
        "--subject", subject, "--bundle", str(bundle_path),
        "--output", str(output),
    ])
    assert code == 0
    import json

    via_cli = json.loads(output.read_text())
    via_pipeline = analyze_subject(
        completed_check["manifest"], subject, bundle_path
    )
    assert via_cli["predicted_label"] == via_pipeline["predicted_label"]
    assert via_cli["decision_score"] == pytest.approx(
        via_pipeline["decision_score"], abs=1e-10
    )


def test_missing_bundle_and_subject_errors_are_clear(completed_check):
    bundle_path = Path(completed_check["run"]) / "fold_0/quantum.joblib"
    with pytest.raises(FileNotFoundError, match="[Bb]undle not found"):
        load_bundle(bundle_path.parent / "absent.joblib")
    with pytest.raises(ValueError, match="not present in this manifest"):
        analyze_subject(
            completed_check["manifest"], "no-such-subject", bundle_path
        )


def test_incompatible_feature_shape_rejected(completed_check):
    bundle = load_bundle(Path(completed_check["run"]) / "fold_0/quantum.joblib")
    model = bundle["model"]
    bad_width = model.config.n_features + 1
    with pytest.raises(ValueError, match="do not match|match the fitted"):
        model.predict(np.zeros((1, bad_width)))


def test_runner_module_entry_point(completed_check, tmp_path):
    from asd_eeg.evaluation.run_experiments import main as run_main

    out = tmp_path / "smoke-results"
    code = run_main([
        "--features", str(Path(completed_check["directory"]) / "features.npz"),
        "--output", str(out),
    ])
    assert code == 0
    runs = [p for p in out.iterdir() if p.is_dir()]
    assert len(runs) == 1
    summary = pd.read_json(runs[0] / "summary.json", typ="series")
    assert summary["status"] == "complete"
