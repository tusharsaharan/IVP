import argparse
import json
import sys
from pathlib import Path

from asd_eeg.workspace import workspace_paths


def parser() -> argparse.ArgumentParser:
    paths = workspace_paths(Path.cwd())
    root = argparse.ArgumentParser(prog="neuroq", description="NeuroQ-Connect research commands")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate paths, subject IDs, and explicit labels")
    validate.add_argument("--manifest", default=paths["manifest"])
    features = commands.add_parser("features", help="Build label-free subject features")
    features.add_argument("--manifest", default=paths["manifest"])
    features.add_argument("--config", default=paths["config"])
    features.add_argument("--output", default=paths["features"])
    evaluate = commands.add_parser("evaluate", help="Run fixed-setting subject-wise comparisons")
    evaluate.add_argument("--features", default=paths["features"])
    evaluate.add_argument("--output", default=paths["results"])
    predict = commands.add_parser("predict", help="Analyze one held-out or explicitly new subject")
    predict.add_argument("--manifest", default=paths["manifest"])
    predict.add_argument("--subject", required=True)
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--output", default="artifacts/prediction.json")
    predict.add_argument("--allow-new-subject", action="store_true")
    check = commands.add_parser("self-check", help="Run a labelled synthetic software check")
    check.add_argument("--config", default="configs/default.yaml")
    check.add_argument("--output", default="artifacts/software-check")
    prepare = commands.add_parser("prepare-sfari", help="Plan or retrieve verified SFARI resting EEG")
    prepare.add_argument("--source", required=True, help="Unmodified ds006780 metadata clone")
    prepare.add_argument("--output", default="data/local/sfari")
    prepare.add_argument("--download", action="store_true", help="Without this flag, only show sizes")
    prepare.add_argument("--workers", type=int, default=4)
    screen = commands.add_parser("screen", help="Write explicit fixed-QC recording/cohort exclusions")
    screen.add_argument("--manifest", default=paths["manifest"])
    screen.add_argument("--config", default=paths["config"])
    screen.add_argument("--output", default="data/local/screening")
    use = commands.add_parser("use-workspace", help="Activate shared local CLI/application paths")
    use.add_argument("--manifest", required=True)
    use.add_argument("--config", required=True)
    use.add_argument("--features", default="artifacts/features/subject_features.npz")
    use.add_argument("--results", default="results")
    return root


def main(argv=None) -> int:
    try:
        args = parser().parse_args(argv)
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
            from asd_eeg.config import load_config
            from asd_eeg.evaluation.run_experiments import run_evaluation
            from asd_eeg.features.cache import load_features
            from asd_eeg.workspace import PROFILE, check_feature_workspace
            dataset = load_features(args.features)
            paths = workspace_paths(Path.cwd())
            if ((Path.cwd() / PROFILE).is_file()
                    and Path(args.features).resolve() == Path(paths["features"]).resolve()):
                check_feature_workspace(dataset, paths["manifest"], load_config(paths["config"]))
            result = {"run": str(run_evaluation(dataset, args.output))}
        elif args.command == "predict":
            from asd_eeg.inference.export import export_prediction
            from asd_eeg.inference.pipeline import analyze_subject
            prediction = analyze_subject(args.manifest, args.subject, args.bundle, not args.allow_new_subject)
            result = {"output": str(export_prediction(prediction, args.output)), **prediction}
        elif args.command == "prepare-sfari":
            from asd_eeg.data.sfari import prepare_sfari
            result = prepare_sfari(args.source, args.output, args.download, args.workers)
        elif args.command == "screen":
            from asd_eeg.config import load_config
            from asd_eeg.data.screening import screen_manifest
            result = screen_manifest(args.manifest, load_config(args.config), args.output)
        elif args.command == "use-workspace":
            from asd_eeg.workspace import activate_workspace
            result = activate_workspace(Path.cwd(), args.manifest, args.config,
                                        args.features, args.results)
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
