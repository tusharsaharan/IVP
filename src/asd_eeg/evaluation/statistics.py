"""Conditional uncertainty and paired comparisons of frozen subject OOF labels.

No models are fitted or bundles deserialized. These summaries do not account for
dependence induced by overlapping CV training sets; see docs/STATISTICS.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from asd_eeg.config import Config
from asd_eeg.evaluation.metrics import fold_auc, summarize_oof
from asd_eeg.evaluation.splits import make_subject_folds

METRICS = ("accuracy", "balanced_accuracy", "sensitivity", "specificity")
LIMITATIONS = [
    "Conditional on the frozen cohort, QC, labels, folds, settings and fitted OOF predictions.",
    "Subjects, not windows, sessions, scores or folds, are resampled within true-label strata; "
    "observed class counts are fixed and subjects are assumed exchangeable within each class.",
    "Models share each resample. No fitting, threshold tuning or uncertainty from retraining "
    "or choosing folds is included.",
    "CV training sets overlap, so OOF errors can be dependent. Bootstrap, Wilson and exact "
    "McNemar calculations are descriptive/conditional; nominal coverage and type-I error "
    "are not guaranteed for CV-based algorithm comparisons.",
    "McNemar tests paired correctness (accuracy), not balanced accuracy, sensitivity, "
    "specificity, score ranking or equivalence; nonsignificance does not establish equivalence.",
    "Holm correction covers all model pairs in this analysis only, not earlier runs, "
    "protocol exploration or other outcomes.",
    "No pooled cross-fold decision-score or AUC bootstrap is performed; uncalibrated SVM "
    "scores from independently fitted folds need not share a scale.",
    "No independent/external validation, clinical validity, causal biomarker or quantum "
    "advantage is established. Synthetic inputs verify software only.",
]


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _confidence(value) -> float:
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError("confidence must be a finite number strictly between 0 and 1.")
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("confidence must be a finite number strictly between 0 and 1.")
    return value


def wilson_interval(successes: int, trials: int, *, confidence: float = 0.95) -> dict:
    """Two-sided Wilson score interval, without continuity correction."""
    successes = _integer(successes, "successes")
    trials = _integer(trials, "trials", 1)
    confidence = _confidence(confidence)
    if successes > trials:
        raise ValueError("successes cannot exceed trials.")
    z = float(norm.isf((1 - confidence) / 2))
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials**2))
    radius /= denominator
    return {"successes": successes, "trials": trials, "estimate": proportion,
            "lower": float(max(0, center - radius)), "upper": float(min(1, center + radius)),
            "confidence": confidence, "method": "Wilson score; no continuity correction"}


def holm_adjust(p_values) -> list[float]:
    """Holm step-down family-wise correction, returned in original pair order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite vector in [0, 1].")
    if not len(values):
        return []
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted.tolist()


def exact_mcnemar(correct_a, correct_b) -> dict:
    """Exact two-sided conditional binomial test of matched binary correctness."""
    a, b = np.asarray(correct_a), np.asarray(correct_b)
    if a.ndim != 1 or not len(a) or a.shape != b.shape:
        raise ValueError("Matched correctness must be nonempty aligned vectors.")
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Matched correctness must be binary and finite.")
    a, b = a.astype(bool), b.astype(bool)
    a_only = int(np.count_nonzero(a & ~b))
    b_only = int(np.count_nonzero(~a & b))
    discordant = a_only + b_only
    p_value = float(binomtest(a_only, discordant, p=0.5, alternative="two-sided").pvalue) \
        if discordant else 1.0
    return {"subjects": len(a), "both_correct": int(np.count_nonzero(a & b)),
            "both_wrong": int(np.count_nonzero(~a & ~b)), "a_correct_b_wrong": a_only,
            "a_wrong_b_correct": b_only, "discordant_subjects": discordant,
            "p_value": p_value, "method": "exact two-sided McNemar (binomial p=0.5)"}


def _normalized_oof(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Reuse the authoritative validator before any resampling or matching.
    summary = summarize_oof(predictions)
    frame = predictions.copy()
    for column in ("subject_id", "model"):
        frame[column] = frame[column].astype(str)
    for column in ("true_label", "predicted_label", "fold"):
        frame[column] = pd.to_numeric(frame[column]).astype(int)
    frame["decision_score"] = pd.to_numeric(frame["decision_score"]).astype(float)
    return frame.sort_values(["model", "subject_id"]).reset_index(drop=True), summary


def _percentile(samples: np.ndarray, confidence: float) -> dict:
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail], method="linear")
    return {"lower": float(lower), "upper": float(upper), "confidence": confidence,
            "method": "stratified subject percentile bootstrap; linear quantiles"}


def analyze_oof(predictions: pd.DataFrame, *, seed: int, replicates: int,
                confidence: float = 0.95) -> dict:
    """Analyze aligned frozen OOF labels; callers must supply seed and replicates.

    The run CLI additionally validates the complete saved run. This in-memory
    function cannot establish provenance or that inputs really are held out.
    """
    seed = _integer(seed, "seed")
    replicates = _integer(replicates, "replicates", 2)
    confidence = _confidence(confidence)
    frame, point_metrics = _normalized_oof(predictions)
    models = sorted(point_metrics)
    first = frame[frame.model == models[0]]
    labels = first.true_label.to_numpy()
    strata = [np.flatnonzero(labels == label) for label in (0, 1)]
    n0, n1 = map(len, strata)
    if min(n0, n1) < 2:
        raise ValueError("Need at least two OOF subjects per class for subject uncertainty.")
    if first.fold.nunique() < 2:
        raise ValueError("OOF uncertainty needs at least two held-out folds.")
    correct = np.column_stack([
        frame[frame.model == model].predicted_label.to_numpy() == labels for model in models
    ])
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty((replicates, len(models), len(METRICS)), dtype=float)
    for replicate in range(replicates):
        # Draw whole matched subject vectors; retain true-label class counts.
        negative = strata[0][rng.integers(n0, size=n0)]
        positive = strata[1][rng.integers(n1, size=n1)]
        tn = correct[negative].sum(axis=0)
        tp = correct[positive].sum(axis=0)
        samples[replicate, :, 0] = (tn + tp) / (n0 + n1)
        samples[replicate, :, 1] = (tn / n0 + tp / n1) / 2
        samples[replicate, :, 2] = tp / n1
        samples[replicate, :, 3] = tn / n0
    model_results = {}
    for index, model in enumerate(models):
        point = point_metrics[model]
        tn, fp = point["confusion_matrix"][0]
        fn, tp = point["confusion_matrix"][1]
        model_results[model] = {
            "subjects": len(labels), "confusion_matrix": [[tn, fp], [fn, tp]],
            "metrics": {metric: {"estimate": point[metric],
                                 "bootstrap": _percentile(samples[:, index, j], confidence)}
                        for j, metric in enumerate(METRICS)},
        }
        for metric, successes, trials in (("accuracy", tn + tp, len(labels)),
                                          ("sensitivity", tp, n1), ("specificity", tn, n0)):
            model_results[model]["metrics"][metric]["wilson"] = wilson_interval(
                successes, trials, confidence=confidence,
            )
    pairs = []
    for a, b in combinations(range(len(models)), 2):
        pair = {"model_a": models[a], "model_b": models[b],
                "difference_direction": "model_a minus model_b",
                "mcnemar": exact_mcnemar(correct[:, a], correct[:, b]),
                "paired_differences": {
                    metric: {"estimate": point_metrics[models[a]][metric] - point_metrics[models[b]][metric],
                             "bootstrap": _percentile(samples[:, a, j] - samples[:, b, j], confidence)}
                    for j, metric in enumerate(METRICS)
                }}
        pairs.append(pair)
    for pair, adjusted in zip(pairs, holm_adjust([p["mcnemar"]["p_value"] for p in pairs]), strict=True):
        pair["mcnemar"].update(p_value_holm=adjusted, family_size=len(pairs))
    return {
        "schema_version": 1, "analysis": "conditional frozen subject OOF label analysis",
        "subjects": len(labels), "class_counts": {"Control": n0, "ASD": n1},
        "folds": sorted(first.fold.unique().tolist()),
        "bootstrap": {"seed": seed, "replicates": replicates, "confidence": confidence,
                      "rng": "numpy.random.Generator(PCG64)", "strata": "true_label (0, 1)",
                      "subject_order": "lexicographic subject_id within true-label strata",
                      "shared_resamples": True, "refit": False},
        "multiplicity": {"method": "Holm step-down", "family": "all unordered model pairs",
                         "tests": len(pairs), "outcome": "paired correctness (accuracy)",
                         "intervals": "pointwise, not simultaneous or multiplicity-adjusted"},
        "models": model_results, "pairs": pairs, "limitations": LIMITATIONS.copy(),
    }


def _json_object(payload: bytes, name: str) -> dict:
    def reject_constant(value):
        raise ValueError(f"{name} contains nonfinite JSON value {value}.")

    def unique_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key}.")
            result[key] = value
        return result

    value = json.loads(payload.decode("utf-8"), parse_constant=reject_constant,
                       object_pairs_hook=unique_keys)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object.")
    _finite_json(value, name)
    return value


def _finite_json(value, name: str) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _finite_json(child, name)
    elif isinstance(value, list):
        for child in value:
            _finite_json(child, name)
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{name} contains nonfinite values.")


def _csv_frame(payload: bytes, name: str, columns: set[str]) -> pd.DataFrame:
    text = payload.decode("utf-8")
    header = next(csv.reader(io.StringIO(text)), [])
    if len(header) != len(set(header)) or not columns.issubset(header):
        raise ValueError(f"{name} needs unique columns including {sorted(columns)}.")
    # Preserve IDs such as 001 and NA; numeric fields are validated explicitly.
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def _same(actual, expected, name: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not set(expected).issubset(actual):
            raise ValueError(f"{name} is incomplete or mismatched.")
        for key, value in expected.items():
            _same(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{name} is incomplete or mismatched.")
        for a, b in zip(actual, expected, strict=True):
            _same(a, b, name)
    elif isinstance(expected, (float, int)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not np.isfinite(actual):
            raise ValueError(f"{name} must be finite numeric values.")
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{name} differs from regenerated OOF values.")
    elif actual != expected:
        raise ValueError(f"{name} is incomplete or mismatched.")


def load_complete_run(run: str | Path) -> tuple[pd.DataFrame, dict]:
    """Validate run tables/configuration without reading EEG or loading bundles.

    Bundle existence is checked, not serialized content or train-only fitting.
    The source file hashes bind the exact table/configuration snapshot analyzed.
    """
    from asd_eeg.training.model import MODELS

    run = Path(run)
    names = ("summary.json", "config.json", "folds.csv", "oof_predictions.csv",
             "metrics.json", "fold_auc.csv", "model_comparison.csv")
    payloads = {name: (run / name).read_bytes() for name in names}
    summary = _json_object(payloads["summary.json"], "summary.json")
    if summary.get("status") != "complete":
        raise ValueError("Statistics require summary.json status: complete.")
    for key in ("run_id", "dataset", "config_id", "feature_fingerprint", "git_commit"):
        if not isinstance(summary.get(key), str) or not summary[key].strip():
            raise ValueError(f"summary.json needs nonempty {key}.")
    if (_integer(summary.get("schema_version"), "summary.schema_version", 1) != 1
            or summary.get("data_kind") not in {"recorded", "synthetic"}):
        raise ValueError("Unsupported summary schema or data kind.")
    for key, minimum in (("subjects", 4), ("folds", 2), ("requested_folds", 2), ("seed", 0)):
        _integer(summary.get(key), f"summary.{key}", minimum)
    if not isinstance(summary.get("class_counts"), dict) or set(summary["class_counts"]) != {"Control", "ASD"}:
        raise ValueError("summary.class_counts must contain exactly Control and ASD.")
    for count in summary["class_counts"].values():
        _integer(count, "summary.class_counts", 2)
    config = Config.from_dict(_json_object(payloads["config.json"], "config.json"))
    _same(summary.get("config_id"), config.id, "summary.config_id")
    _same(summary.get("seed"), config.evaluation.seed, "summary.seed")
    _same(summary.get("requested_folds"), config.evaluation.folds, "summary.requested_folds")
    predictions = _csv_frame(payloads["oof_predictions.csv"], "oof_predictions.csv",
                             {"subject_id", "model", "fold", "true_label", "predicted_label", "decision_score"})
    frame, regenerated = _normalized_oof(predictions)
    if set(regenerated) != set(MODELS):
        raise ValueError("A complete run needs every configured model on the same OOF cohort.")
    folds = _csv_frame(payloads["folds.csv"], "folds.csv", {"subject_id", "fold", "label"})
    # Reuse OOF validation for uniqueness, finite labels/folds and cohort matching.
    fold_oof = folds.rename(columns={"label": "true_label"}).assign(
        model="folds", predicted_label=0, decision_score=0,
    )
    checked_folds, _ = _normalized_oof(fold_oof)
    first = frame[frame.model == sorted(regenerated)[0]]
    actual_assignments = checked_folds[["subject_id", "fold", "true_label"]].to_numpy().tolist()
    _same(actual_assignments, first[["subject_id", "fold", "true_label"]].to_numpy().tolist(),
          "folds.csv cohort/labels/folds")
    # The CSV preserves original feature-subject order, needed to replay frozen folds.
    expected_folds = make_subject_folds(folds.subject_id.to_numpy(),
                                       pd.to_numeric(folds.label).to_numpy(),
                                       config.evaluation.folds, config.evaluation.seed)
    _same(checked_folds.fold.tolist(), expected_folds.sort_values("subject_id").fold.tolist(),
          "folds.csv frozen stratified partition")
    counts = first.true_label.value_counts()
    _same(summary.get("subjects"), len(first), "summary.subjects")
    _same(summary.get("class_counts"), {"Control": int(counts[0]), "ASD": int(counts[1])},
          "summary.class_counts")
    _same(summary.get("folds"), first.fold.nunique(), "summary.folds")
    saved_metrics = _json_object(payloads["metrics.json"], "metrics.json")
    if set(saved_metrics) != set(regenerated):
        raise ValueError("metrics.json model set is incomplete or mismatched.")
    _same(saved_metrics, regenerated, "metrics.json")
    for model in regenerated:
        if set(saved_metrics[model]) != set(regenerated[model]) | {"fit_predict_seconds"}:
            raise ValueError("metrics.json has incomplete or mismatched metric fields.")
        runtime = saved_metrics[model].get("fit_predict_seconds")
        if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or runtime < 0:
            raise ValueError("metrics.json needs finite nonnegative fit_predict_seconds.")
    auc_frame = _csv_frame(payloads["fold_auc.csv"], "fold_auc.csv", {"model", "fold", "auc"})
    expected_auc = {(model, fold): rows for (model, fold), rows in frame.groupby(["model", "fold"])}
    seen = set()
    for row in auc_frame.itertuples(index=False):
        fold = float(row.fold)
        if not np.isfinite(fold) or fold < 0 or fold != np.floor(fold):
            raise ValueError("fold_auc.csv needs finite integer folds.")
        key = (row.model, int(fold))
        if key in seen or key not in expected_auc:
            raise ValueError("fold_auc.csv contains duplicate or mismatched model/fold rows.")
        seen.add(key)
        rows = expected_auc[key]
        expected = fold_auc(rows.true_label, rows.decision_score)
        actual = None if row.auc == "" else float(row.auc)
        _same(actual, expected, "fold_auc.csv.auc")
    if seen != set(expected_auc):
        raise ValueError("fold_auc.csv is incomplete.")
    table = _csv_frame(payloads["model_comparison.csv"], "model_comparison.csv", {"model"})
    if table.model.duplicated().any() or set(table.model) != set(regenerated):
        raise ValueError("model_comparison.csv model rows are duplicate, incomplete or mismatched.")
    for row in table.to_dict(orient="records"):
        for key, expected in saved_metrics[row["model"]].items():
            if key == "confusion_matrix":
                continue
            if key not in row:
                raise ValueError(f"model_comparison.csv is missing {key}.")
            actual = None if row[key] == "" else float(row[key])
            _same(actual, expected, f"model_comparison.csv.{key}")
    for model, fold in expected_auc:
        bundle = run / f"fold_{fold}" / f"{model}.joblib"
        if not bundle.is_file() or bundle.stat().st_size == 0:
            raise ValueError(f"Complete run is missing a nonempty fold {fold} {model} bundle.")
    source = {key: summary[key] for key in ("run_id", "dataset", "data_kind", "config_id",
                                           "feature_fingerprint", "git_commit", "seed", "requested_folds")}
    source["sha256"] = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    source["validation"] = "complete status; config, frozen folds, cohort, all models, metrics and AUC tables; bundle existence only"
    return frame, source


def write_statistics(run: str | Path, output: str | Path, *, seed: int, replicates: int,
                     confidence: float = 0.95) -> Path:
    """Write one JSON analysis file exclusively; never overwrite an existing path."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing statistics output: {output}")
    frame, source = load_complete_run(run)
    result = analyze_oof(frame, seed=seed, replicates=replicates, confidence=confidence)
    result["source"] = source
    result["software"] = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn")},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), Path(__file__).with_name("metrics.py"),
                                       Path(__file__).with_name("splits.py"))},
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OOF-only conditional uncertainty and paired comparisons.")
    parser.add_argument("--run", required=True, help="Completed evaluation run directory.")
    parser.add_argument("--output", required=True, help="New JSON file; existing paths are refused.")
    parser.add_argument("--seed", type=int, default=42, help="PCG64 bootstrap seed (default: 42).")
    parser.add_argument("--replicates", type=int, default=10000, help="Subject bootstrap draws (default: 10000).")
    parser.add_argument("--confidence", type=float, default=0.95, help="Pointwise interval level (default: 0.95).")
    args = parser.parse_args(argv)
    try:
        path = write_statistics(args.run, args.output, seed=args.seed,
                                replicates=args.replicates, confidence=args.confidence)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"statistics": str(path), "seed": args.seed, "replicates": args.replicates}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
