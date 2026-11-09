import argparse
import json
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="neuroq", description="NeuroQ-Connect research commands")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate paths, subject IDs, and explicit labels")
    validate.add_argument("--manifest", default="data/metadata/recordings.csv")
    features = commands.add_parser("features", help="Build label-free subject features")
    features.add_argument("--manifest", default="data/metadata/recordings.csv")
    features.add_argument("--config", default="configs/default.yaml")
    features.add_argument("--output", default="artifacts/features/subject_features.npz")
    evaluate = commands.add_parser("evaluate", help="Run fixed-setting subject-wise comparisons")
    evaluate.add_argument("--features", default="artifacts/features/subject_features.npz")
    evaluate.add_argument("--output", default="results")
    predict = commands.add_parser("predict", help="Analyze one held-out or explicitly new subject")
    predict.add_argument("--manifest", default="data/metadata/recordings.csv")
    predict.add_argument("--subject", required=True)
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--output", default="artifacts/prediction.json")
    predict.add_argument("--allow-new-subject", action="store_true")
    check = commands.add_parser("self-check", help="Run a labelled synthetic software check")
    check.add_argument("--config", default="configs/default.yaml")
    check.add_argument("--output", default="artifacts/software-check")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "validate":
            from asd_eeg.data.manifest import read_manifest
            table = read_manifest(args.manifest)
            result = {"recordings": len(table), "subjects": table.subject_id.nunique(),
                      "dataset": table.dataset.iloc[0], "data_kind": table.data_kind.iloc[0]}
        elif args.command == "features":
            from asd_eeg.config import load_config
            from asd_eeg.features.cache import build_features
            dataset = build_features(args.manifest, load_config(args.config), args.output)
            result = {"output": str(Path(args.output).resolve()), "shape": list(dataset.X.shape)}
        elif args.command == "evaluate":
            from asd_eeg.evaluation.run_experiments import run_evaluation
            from asd_eeg.features.cache import load_features
            result = {"run": str(run_evaluation(load_features(args.features), args.output))}
        elif args.command == "predict":
            from asd_eeg.inference.export import export_prediction
            from asd_eeg.inference.pipeline import analyze_subject
            prediction = analyze_subject(args.manifest, args.subject, args.bundle, not args.allow_new_subject)
            result = {"output": str(export_prediction(prediction, args.output)), **prediction}
        else:
            from asd_eeg.config import load_config
            from asd_eeg.self_check import run_self_check
            result = run_self_check(load_config(args.config), args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
