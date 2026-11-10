"""Software-only fixtures for conditional OOF uncertainty and paired tests."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from asd_eeg.config import Config, load_config
from asd_eeg.evaluation.metrics import classification_metrics, fold_auc, summarize_oof
from asd_eeg.evaluation.splits import make_subject_folds
from asd_eeg.evaluation.statistics import (
    analyze_oof,
    exact_mcnemar,
    holm_adjust,
    load_complete_run,
    main,
    wilson_interval,
    write_statistics,
)
from asd_eeg.training.model import MODELS


@pytest.fixture
def stats_config():
    raw = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml").to_dict()
    raw["evaluation"].update(folds=3, seed=17)
    return Config.from_dict(raw)


@pytest.fixture
def folds(stats_config):
    return make_subject_folds(np.array(["008", "001", "NA", "004", "005", "006", "007", "002"]),
                              np.array([0, 0, 0, 1, 1, 1, 1, 1]),
                              stats_config.evaluation.folds, stats_config.evaluation.seed)


@pytest.fixture
def oof(folds):
    predictions = {
        "rbf": [0, 1, 0, 1, 0, 1, 0, 1],
        "amplitude": [0, 0, 1, 1, 0, 0, 1, 1],
        "quantum": [0, 1, 0, 1, 0, 1, 0, 1],  # Identical predictions to RBF.
        "reassigned": [1, 1, 0, 0, 1, 1, 1, 0],
    }
    return pd.concat([
        folds.rename(columns={"label": "true_label"}).assign(
            model=model, predicted_label=predictions[model],
            decision_score=np.array(predictions[model], dtype=float) * 2 - 1,
        ) for model in MODELS
    ], ignore_index=True)


@pytest.fixture
def complete_run(tmp_path, stats_config, folds, oof):
    run = tmp_path / "run-software-fixture"
    run.mkdir()
    summary = {
        "schema_version": 1, "run_id": run.name, "status": "complete", "dataset": "software-fixture",
        "data_kind": "synthetic", "subjects": 8, "class_counts": {"Control": 3, "ASD": 5},
        "folds": 3, "requested_folds": 3, "seed": 17, "config_id": stats_config.id,
        "feature_fingerprint": "fixture-only", "git_commit": "unknown",
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "config.json").write_text(json.dumps(stats_config.to_dict()), encoding="utf-8")
    folds.to_csv(run / "folds.csv", index=False)
    oof.to_csv(run / "oof_predictions.csv", index=False)
    metrics = summarize_oof(oof)
    for value in metrics.values():
        value["fit_predict_seconds"] = 0.1
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame([
        {"model": model, **{key: value for key, value in values.items() if key != "confusion_matrix"}}
        for model, values in metrics.items()
    ]).to_csv(run / "model_comparison.csv", index=False)
    aucs = []
    for (model, fold), rows in oof.groupby(["model", "fold"]):
        aucs.append({"model": model, "fold": fold, "auc": fold_auc(rows.true_label, rows.decision_score)})
        bundle = run / f"fold_{fold}" / f"{model}.joblib"
        bundle.parent.mkdir(exist_ok=True)
        # Deliberately not a deserializable bundle: this tool must never load it.
        bundle.write_bytes(b"software fixture; existence check only")
    pd.DataFrame(aucs).to_csv(run / "fold_auc.csv", index=False)
    return run


def test_wilson_known_interval_and_boundaries():
    result = wilson_interval(5, 10)
    assert result["lower"] == pytest.approx(0.236593090512564)
    assert result["upper"] == pytest.approx(0.763406909487436)
    assert wilson_interval(0, 10)["lower"] == pytest.approx(0)
    assert wilson_interval(10, 10)["upper"] == pytest.approx(1)
    assert wilson_interval(5, 10, confidence=0.9)["lower"] > result["lower"]


@pytest.mark.parametrize("successes,trials", [(1, 0), (-1, 10), (11, 10), (0.5, 10),
                                             (True, 10), (1, 2.5), (1, np.inf)])
def test_wilson_rejects_invalid_counts(successes, trials):
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)


def test_correct_class_denominators_and_balanced_accuracy(oof):
    result = analyze_oof(oof, seed=71, replicates=100)
    assert result["class_counts"] == {"Control": 3, "ASD": 5}
    for model, values in result["models"].items():
        rows = oof[oof.model == model]
        expected = classification_metrics(rows.true_label, rows.predicted_label)
        metrics = values["metrics"]
        assert metrics["accuracy"]["wilson"]["trials"] == 8
        assert metrics["sensitivity"]["wilson"]["trials"] == 5
        assert metrics["specificity"]["wilson"]["trials"] == 3
        assert metrics["accuracy"]["wilson"]["successes"] == int((rows.true_label == rows.predicted_label).sum())
        for name in ("accuracy", "balanced_accuracy", "sensitivity", "specificity"):
            assert metrics[name]["estimate"] == expected[name]
        assert "wilson" not in metrics["balanced_accuracy"]
        assert metrics["balanced_accuracy"]["estimate"] == pytest.approx(
            (metrics["sensitivity"]["estimate"] + metrics["specificity"]["estimate"]) / 2,
        )


def test_reproducible_shared_stratified_subject_resampling(oof):
    seed, replicates = 987, 101
    actual = analyze_oof(oof, seed=seed, replicates=replicates)
    assert analyze_oof(oof.sample(frac=1, random_state=91), seed=seed, replicates=replicates) == actual
    rng = np.random.Generator(np.random.PCG64(seed))
    models = sorted(MODELS)
    rows = oof[oof.model == models[0]].sort_values("subject_id")
    labels = rows.true_label.to_numpy()
    strata = [np.flatnonzero(labels == label) for label in (0, 1)]
    draws = {model: {metric: [] for metric in actual["models"][model]["metrics"]} for model in models}
    for _ in range(replicates):
        indices = np.concatenate([stratum[rng.integers(len(stratum), size=len(stratum))] for stratum in strata])
        assert np.bincount(labels[indices]).tolist() == [3, 5]
        for model in models:
            predicted = oof[oof.model == model].sort_values("subject_id").predicted_label.to_numpy()
            metrics = classification_metrics(labels[indices], predicted[indices])
            for metric in draws[model]:
                draws[model][metric].append(metrics[metric])
    for model in models:
        for metric, values in draws[model].items():
            interval = actual["models"][model]["metrics"][metric]["bootstrap"]
            np.testing.assert_allclose([interval["lower"], interval["upper"]],
                                       np.quantile(values, [0.025, 0.975], method="linear"))
    for pair in actual["pairs"]:
        for metric, values in pair["paired_differences"].items():
            deltas = np.array(draws[pair["model_a"]][metric]) - draws[pair["model_b"]][metric]
            np.testing.assert_allclose([values["bootstrap"]["lower"], values["bootstrap"]["upper"]],
                                       np.quantile(deltas, [0.025, 0.975], method="linear"))


def test_seed_and_replicates_are_required(oof):
    with pytest.raises(TypeError):
        analyze_oof(oof)


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"seed": True}, {"seed": 0.5},
                                     {"replicates": 1}, {"replicates": True}, {"replicates": 10.5},
                                     {"confidence": 0}, {"confidence": 1}, {"confidence": np.nan},
                                     {"confidence": np.inf}, {"confidence": True}, {"confidence": "0.95"}])
def test_invalid_resampling_options(oof, kwargs):
    options = {"seed": 42, "replicates": 20, **kwargs}
    with pytest.raises(ValueError):
        analyze_oof(oof, **options)


def test_all_pairs_holm_and_identical_predictions(oof):
    result = analyze_oof(oof, seed=42, replicates=20)
    assert len(result["pairs"]) == 6
    p_values = [pair["mcnemar"]["p_value"] for pair in result["pairs"]]
    assert [pair["mcnemar"]["p_value_holm"] for pair in result["pairs"]] == holm_adjust(p_values)
    pair = next(pair for pair in result["pairs"] if {pair["model_a"], pair["model_b"]} == {"quantum", "rbf"})
    assert pair["mcnemar"]["discordant_subjects"] == 0
    assert pair["mcnemar"]["p_value"] == pair["mcnemar"]["p_value_holm"] == 1
    for values in pair["paired_differences"].values():
        assert values["estimate"] == values["bootstrap"]["lower"] == values["bootstrap"]["upper"] == 0
    assert result["multiplicity"]["intervals"].startswith("pointwise")


@pytest.mark.parametrize("a_only,b_only", [(0, 0), (8, 0), (3, 1), (4, 4), (1, 12)])
def test_exact_two_sided_mcnemar(a_only, b_only):
    a = [1] * a_only + [0] * b_only + [1, 0]
    b = [0] * a_only + [1] * b_only + [1, 0]
    result = exact_mcnemar(a, b)
    discordant = a_only + b_only
    expected = binomtest(a_only, discordant, 0.5, alternative="two-sided").pvalue if discordant else 1
    assert result["p_value"] == expected
    assert result["p_value"] == exact_mcnemar(b, a)["p_value"]
    assert result["a_correct_b_wrong"] == a_only
    assert result["a_wrong_b_correct"] == b_only
    assert result["both_correct"] == result["both_wrong"] == 1
    if (a_only, b_only) == (8, 0):
        assert result["p_value"] == 2 / 256  # Exact, not asymptotic chi-squared.


@pytest.mark.parametrize("a,b", [([], []), ([1, 0], [1]), ([np.nan], [0]), ([2], [0]), ([[1]], [[0]])])
def test_mcnemar_rejects_invalid_pairs(a, b):
    with pytest.raises(ValueError):
        exact_mcnemar(a, b)


def test_holm_known_values_ties_and_original_order():
    assert holm_adjust([0.03, 0.01, 0.04, 0.2]) == pytest.approx([0.09, 0.04, 0.09, 0.2])
    assert holm_adjust([0.01, 0.01, 0.5]) == pytest.approx([0.03, 0.03, 0.5])
    assert holm_adjust([0.8, 0.9]) == [1, 1]
    assert holm_adjust([]) == []
    assert holm_adjust([0.2]) == [0.2]


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-0.1], [1.1], [[0.2]]])
def test_holm_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        holm_adjust(values)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "fold", "label", "score", "nonbinary", "blank"])
def test_shared_oof_validation_is_not_bypassed(oof, problem):
    frame = oof.copy()
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "missing":
        frame = frame.iloc[1:]
    elif problem == "fold":
        frame.loc[0, "fold"] = (int(frame.loc[0, "fold"]) + 1) % 3
    elif problem == "label":
        frame.loc[0, "true_label"] = 1
    elif problem == "score":
        frame.loc[0, "decision_score"] = np.inf
    elif problem == "nonbinary":
        frame.loc[0, "predicted_label"] = 2
    else:
        frame.loc[0, "subject_id"] = " "
    with pytest.raises(ValueError):
        analyze_oof(frame, seed=42, replicates=20)


def test_need_both_classes_and_multiple_folds(oof):
    with pytest.warns(UserWarning, match="single label"), pytest.raises(ValueError, match="two OOF subjects per class"):
        analyze_oof(oof.assign(true_label=0, predicted_label=0), seed=42, replicates=20)
    with pytest.raises(ValueError, match="two held-out folds"):
        analyze_oof(oof.assign(fold=0), seed=42, replicates=20)


def test_score_offsets_do_not_enter_uncertainty(oof):
    result = analyze_oof(oof, seed=42, replicates=20)
    modified = oof.copy()
    modified.loc[modified.fold == 1, "decision_score"] += 1e9
    assert analyze_oof(modified, seed=42, replicates=20) == result
    assert all("auc" not in metric for values in result["models"].values() for metric in values["metrics"])


def test_complete_run_validates_and_preserves_string_ids(complete_run):
    frame, source = load_complete_run(complete_run)
    assert {"001", "NA"}.issubset(frame.subject_id)
    assert source["data_kind"] == "synthetic"
    for name, digest in source["sha256"].items():
        assert digest == hashlib.sha256((complete_run / name).read_bytes()).hexdigest()
    assert "bundle existence only" in source["validation"]


def test_actual_software_check_run_is_compatible(completed_check):
    frame, source = load_complete_run(completed_check["run"])
    assert set(frame.model) == set(MODELS)
    assert source["data_kind"] == "synthetic"
    result = analyze_oof(frame, seed=42, replicates=20)
    assert result["subjects"] == frame.subject_id.nunique()


@pytest.mark.parametrize("filename", ["summary.json", "config.json", "folds.csv", "oof_predictions.csv",
                                      "metrics.json", "fold_auc.csv", "model_comparison.csv"])
def test_complete_run_refuses_missing_artifacts(complete_run, filename):
    (complete_run / filename).unlink()
    with pytest.raises(OSError):
        load_complete_run(complete_run)


@pytest.mark.parametrize("field,value", [("status", "running"), ("status", "failed"), ("subjects", 7),
                                        ("class_counts", {"Control": 5, "ASD": 3}), ("folds", 2),
                                        ("config_id", "different"), ("seed", 19), ("requested_folds", 5),
                                        ("data_kind", "mixed"), ("schema_version", 2),
                                        ("schema_version", True), ("subjects", 8.0), ("folds", True),
                                        ("class_counts", {"Control": 3, "ASD": 5, "unknown": 0})])
def test_complete_run_refuses_mismatched_summary(complete_run, field, value):
    path = complete_run / "summary.json"
    summary = json.loads(path.read_text())
    summary[field] = value
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        load_complete_run(complete_run)


@pytest.mark.parametrize("filename,problem", [
    ("oof_predictions.csv", "duplicate"), ("oof_predictions.csv", "missing_subject_all_models"),
    ("oof_predictions.csv", "missing_model"), ("oof_predictions.csv", "nonfinite"),
    ("folds.csv", "duplicate"), ("folds.csv", "missing"), ("folds.csv", "changed_fold"),
    ("folds.csv", "changed_label"), ("folds.csv", "nonfinite"),
    ("fold_auc.csv", "duplicate"), ("fold_auc.csv", "missing"),
    ("fold_auc.csv", "wrong_auc"), ("fold_auc.csv", "nonfinite"),
    ("model_comparison.csv", "duplicate"), ("model_comparison.csv", "missing"),
    ("model_comparison.csv", "wrong_accuracy"), ("model_comparison.csv", "nonfinite"),
])
def test_complete_run_refuses_corrupt_tables(complete_run, filename, problem):
    path = complete_run / filename
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if problem == "duplicate":
        table = pd.concat([table, table.iloc[:1]])
    elif problem == "missing":
        table = table.iloc[1:]
    elif problem == "missing_subject_all_models":
        table = table[table.subject_id != table.iloc[0].subject_id]
    elif problem == "missing_model":
        table = table[table.model != "quantum"]
    elif problem == "changed_fold":
        table.loc[0, "fold"] = str((int(table.loc[0, "fold"]) + 1) % 3)
    elif problem == "changed_label":
        table.loc[0, "label"] = "1"
    elif problem == "wrong_auc":
        table.loc[0, "auc"] = "0.123"
    elif problem == "wrong_accuracy":
        table.loc[0, "accuracy"] = "0.123"
    else:
        column = {"oof_predictions.csv": "decision_score", "folds.csv": "fold",
                  "fold_auc.csv": "auc", "model_comparison.csv": "fit_predict_seconds"}[filename]
        table.loc[0, column] = "inf"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_complete_run(complete_run)


def test_complete_run_replays_frozen_folds_even_if_changed_in_both_tables(complete_run):
    for filename in ("folds.csv", "oof_predictions.csv"):
        path = complete_run / filename
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        table["fold"] = (pd.to_numeric(table.fold) + 1) % 3
        table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="frozen stratified partition"):
        load_complete_run(complete_run)


@pytest.mark.parametrize("filename", ["summary.json", "metrics.json", "config.json"])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e999"])
def test_nonfinite_json_is_rejected(complete_run, filename, value):
    path = complete_run / filename
    text = path.read_text()
    path.write_text(text[:-1] + ', "unexpected_number": ' + value + '}')
    with pytest.raises(ValueError, match="nonfinite"):
        load_complete_run(complete_run)


def test_json_duplicate_keys_and_csv_duplicate_headers_are_rejected(complete_run):
    path = complete_run / "summary.json"
    original = path.read_text()
    path.write_text(original[:-1] + ', "status": "complete"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_complete_run(complete_run)
    path.write_text(original)
    path = complete_run / "oof_predictions.csv"
    text = path.read_text()
    header, body = text.split("\n", 1)
    path.write_text(header + ",model\n" + body)
    with pytest.raises(ValueError, match="unique columns"):
        load_complete_run(complete_run)


@pytest.mark.parametrize("problem", ["changed", "missing_metric", "missing_runtime", "negative_runtime", "missing_model"])
def test_corrupt_saved_metrics_rejected(complete_run, problem):
    path = complete_run / "metrics.json"
    metrics = json.loads(path.read_text())
    if problem == "changed":
        metrics["rbf"]["accuracy"] = 0.123
    elif problem == "missing_metric":
        del metrics["rbf"]["sensitivity"]
    elif problem == "missing_runtime":
        del metrics["rbf"]["fit_predict_seconds"]
    elif problem == "negative_runtime":
        metrics["rbf"]["fit_predict_seconds"] = -1
    else:
        del metrics["quantum"]
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError):
        load_complete_run(complete_run)


@pytest.mark.parametrize("empty", [False, True])
def test_complete_run_needs_bundles_but_does_not_load_them(complete_run, empty):
    path = complete_run / "fold_0/quantum.joblib"
    if empty:
        path.write_bytes(b"")
    else:
        path.unlink()
    with pytest.raises(ValueError, match="nonempty"):
        load_complete_run(complete_run)


def test_json_output_is_reproducible_finite_and_never_overwritten(complete_run, tmp_path):
    original = {path: path.read_bytes() for path in complete_run.rglob("*") if path.is_file()}
    first, second = tmp_path / "statistics-one.json", tmp_path / "statistics-two.json"
    assert write_statistics(complete_run, first, seed=99, replicates=20) == first
    write_statistics(complete_run, second, seed=99, replicates=20)
    assert first.read_bytes() == second.read_bytes()
    result = json.loads(first.read_text())
    json.dumps(result, allow_nan=False)
    assert result["bootstrap"]["seed"] == 99
    assert result["bootstrap"]["replicates"] == 20
    assert result["source"]["data_kind"] == "synthetic"
    assert "statistics.py" in result["software"]["source_sha256"]
    assert all(path.read_bytes() == contents for path, contents in original.items())
    with pytest.raises(FileExistsError, match="overwrite"):
        write_statistics(complete_run, first, seed=1, replicates=20)
    assert first.read_bytes() == second.read_bytes()


def test_exclusive_write_refuses_a_file_created_during_analysis(complete_run, tmp_path, monkeypatch):
    from asd_eeg.evaluation import statistics

    output = tmp_path / "racing-output.json"
    original_analyze = statistics.analyze_oof

    def create_file_then_analyze(*args, **kwargs):
        output.write_text("another writer's content", encoding="utf-8")
        return original_analyze(*args, **kwargs)

    monkeypatch.setattr(statistics, "analyze_oof", create_file_then_analyze)
    with pytest.raises(FileExistsError):
        write_statistics(complete_run, output, seed=42, replicates=20)
    assert output.read_text() == "another writer's content"


def test_cli_command_and_error_exit(complete_run, tmp_path, capsys):
    output = tmp_path / "new/subdirectory/statistics.json"
    args = ["--run", str(complete_run), "--output", str(output), "--seed", "29", "--replicates", "20"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["replicates"] == 20
    assert main(args) == 1
    assert "overwrite" in capsys.readouterr().err
    absent_output = tmp_path / "absent.json"
    assert main(["--run", str(tmp_path / "missing-run"), "--output", str(absent_output)]) == 1
    assert not absent_output.exists()
    assert "Error:" in capsys.readouterr().err


def test_module_entry_point(complete_run, tmp_path):
    output = tmp_path / "module-statistics.json"
    completed = subprocess.run(
        [sys.executable, "-m", "asd_eeg.evaluation.statistics", "--run", str(complete_run),
         "--output", str(output), "--seed", "7", "--replicates", "20"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    assert json.loads(completed.stdout)["statistics"] == str(output)
    assert json.loads(output.read_text())["bootstrap"]["seed"] == 7
