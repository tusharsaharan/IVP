import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from asd_eeg.cli import main
from asd_eeg.features.cache import load_features
from asd_eeg.inference.compatibility import check_subject, load_bundle
from asd_eeg.inference.pipeline import analyze_subject
from asd_eeg.training.model import MODELS


def test_folds_and_round_trip_inference(completed_check):
    run = Path(completed_check["run"])
    predictions = pd.read_csv(run / "oof_predictions.csv")
    assert set(predictions.model) == set(MODELS)
    assert predictions.groupby(["subject_id", "model"]).size().eq(1).all()
    bundle_path = run / "fold_0/quantum.joblib"
    bundle = load_bundle(bundle_path)
    meta = bundle["metadata"]
    assert not set(meta["train_subject_ids"]) & set(meta["test_subject_ids"])
    subject = meta["test_subject_ids"][0]
    actual = analyze_subject(completed_check["manifest"], subject, bundle_path)
    expected = predictions[(predictions.subject_id == subject) & (predictions.model == "quantum")].iloc[0]
    assert actual["predicted_label"] == expected.predicted_label
    assert actual["decision_score"] == pytest.approx(expected.decision_score, abs=1e-10)
    dataset = load_features(Path(completed_check["directory"]) / "features.npz")
    train = np.isin(dataset.subject_ids, meta["train_subject_ids"])
    np.testing.assert_allclose(bundle["model"].scaler.mean_, dataset.X[train].mean(axis=0))


def test_subject_provenance_is_enforced(completed_check):
    bundle = load_bundle(Path(completed_check["run"]) / "fold_0/quantum.joblib")
    meta = bundle["metadata"]
    with pytest.raises(ValueError, match="used for training"):
        check_subject(bundle, meta["train_subject_ids"][0], meta["dataset"], False)
    with pytest.raises(ValueError, match="held-out fold"):
        check_subject(bundle, "new-person", meta["dataset"])
    assert check_subject(bundle, "new-person", meta["dataset"], False) == "new-subject"
    with pytest.raises(ValueError, match="dataset differs"):
        check_subject(bundle, meta["test_subject_ids"][0], "different-dataset")


def test_tampered_bundle_rejected(completed_check, tmp_path):
    bundle = load_bundle(Path(completed_check["run"]) / "fold_0/quantum.joblib")
    bundle["metadata"]["config_id"] = "wrong-config"
    path = tmp_path / "invalid.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="inconsistent"):
        load_bundle(path)


def test_cli_inference_matches_shared_pipeline(completed_check, tmp_path, capsys):
    output = tmp_path / "prediction.json"
    bundle_path = Path(completed_check["run"]) / "fold_0/quantum.joblib"
    code = main(["predict", "--manifest", completed_check["manifest"],
                 "--subject", completed_check["held_out_subject"], "--bundle", str(bundle_path),
                 "--output", str(output)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["data_kind"] == "synthetic"
    prediction = json.loads(output.read_text())
    assert prediction["provenance"] == "held-out"


def test_cli_empty_manifest_is_actionable(tmp_path, capsys):
    manifest = tmp_path / "empty.csv"
    manifest.write_text("dataset,subject_id,recording_id,eeg_path,label,label_source,data_kind\n")
    assert main(["validate", "--manifest", str(manifest)]) == 1
    assert "no recordings" in capsys.readouterr().err
