"""Evidence-conditioned verification: train, evaluate and export."""

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from fmfc.config import load_config
from fmfc.environment import collect_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    try:
        package_version = version("fmfc")
    except PackageNotFoundError:
        package_version = "source-checkout"
    parser.add_argument("--version", action="version", version=package_version)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("environment", help="Print observed Python and optional Torch details")
    config_parser = commands.add_parser("check-config", help="Load and display a YAML mapping")
    config_parser.add_argument("path", type=Path)
    for command in ("train-local", "train-centralized", "train-federated", "matrix"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument("--config", type=Path, required=True)
        if command != "matrix":
            command_parser.add_argument(
                "--limit", type=int,
                help="Maximum samples per client in train/dev; default: use all samples",
            )
        if command == "train-local":
            command_parser.add_argument("--dataset")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument("--split", choices=["dev", "test"], default="dev")
    evaluation.add_argument("--final", action="store_true")
    evaluation.add_argument(
        "--ablation", choices=["all", "both", "text", "image", "table", "shuffle_image", "drop_image", "drop_table"]
    )
    freeze = commands.add_parser("freeze-selection")
    freeze.add_argument("--run-dir", type=Path, required=True)
    export = commands.add_parser("export-results")
    export.add_argument("--runs-root", type=Path, default=Path("outputs"))
    export.add_argument("--output", type=Path, default=Path("reports/experiments"))
    args = parser.parse_args(argv)
    if args.command == "environment":
        report = collect_environment()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["torch"]["status"] == "error" else 0
    try:
        if args.command == "check-config":
            print(yaml.safe_dump(load_config(args.path), sort_keys=False), end="")
            return 0
        if args.command == "export-results":
            from fmfc.experiments.exports import export_results

            print(json.dumps(export_results(args.runs_root, args.output), indent=2))
            return 0
        from fmfc.experiments.runner import evaluate_run, freeze_selection, run

        if args.command == "evaluate":
            print(
                json.dumps(
                    evaluate_run(args.run_dir, args.split, args.final, args.ablation), indent=2
                )
            )
        elif args.command == "freeze-selection":
            freeze_selection(args.run_dir)
            print("Dev selection frozen; test evaluation requires --final")
        elif args.command == "matrix":
            from fmfc.experiments.matrix import run_matrix

            print(
                json.dumps([str(path) for path in run_matrix(load_config(args.config))], indent=2)
            )
        else:
            config = load_config(args.config)
            config["kind"] = args.command.removeprefix("train-")
            if args.command == "train-local" and args.dataset:
                config["local_client"] = args.dataset
            if args.limit is not None:
                config["limit"] = args.limit
            print(run(config, args.command.removeprefix("train-")))
    except (OSError, ValueError, RuntimeError, yaml.YAMLError, ModuleNotFoundError) as exc:
        parser.error(str(exc))
    return 0
