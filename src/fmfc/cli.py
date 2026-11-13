"""Evidence-conditioned verification: train, explain and export."""

import argparse
import json
from pathlib import Path

import yaml


def load_config(path: str | Path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config


def apply_overrides(config: dict, items: list[str]) -> dict:
    """Apply ``key.path=value`` overrides; values are parsed as YAML (write 0.0001, not 1e-4)."""
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Override must look like key.path=value: {item!r}")
        *parents, leaf = key.split(".")
        node = config
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override inside a non-mapping value: {item!r}")
        node[leaf] = yaml.safe_load(value)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train-local", "train-centralized", "train-federated", "matrix"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument("--config", type=Path, required=True)
        if command != "matrix":
            command_parser.add_argument(
                "overrides", nargs="*", metavar="key.path=value",
                help="Config overrides parsed as YAML, e.g. run_id=e4_fedavg_s42 seed=42 "
                "local_client=mocheg fl.prox_mu=0.001",
            )
    explanation = commands.add_parser(
        "explain", help="Explain best_model.pt using evidence removal"
    )
    explanation.add_argument("--config", type=Path, required=True)
    explanation.add_argument("--checkpoint", type=Path, required=True)
    explanation.add_argument("--dataset", required=True)
    explanation.add_argument("--split", choices=["train", "dev"], default="dev")
    explanation.add_argument(
        "--record-id", help="Exact record ID, for example mocheg:123"
    )
    explanation.add_argument("--limit", type=int, default=1)
    explanation.add_argument("--head", choices=["global", "original"], default="global")
    explanation.add_argument("--top-k", type=int, default=1)
    explanation.add_argument(
        "--output", type=Path, help="Optional JSON export; default: print only"
    )
    evaluation = commands.add_parser(
        "evaluate", help="Re-score best_model.pt of a finished run (B2)"
    )
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument("--split", choices=["train", "dev", "test"], default="test")
    evaluation.add_argument(
        "--ablation",
        choices=["all", "text", "image", "table", "shuffle_image", "drop_image", "drop_table"],
        help="Evaluation-time ablation; default: the run's own setting",
    )
    evaluation.add_argument(
        "--config", type=Path, help="Default: <run-dir>/config_used.yaml"
    )
    export = commands.add_parser("export-results")
    export.add_argument("--runs-root", type=Path, default=Path("outputs"))
    export.add_argument("--output", type=Path, default=Path("reports/experiments"))
    args = parser.parse_args(argv)
    try:
        if args.command == "export-results":
            from fmfc.experiments.exports import export_results

            print(json.dumps(export_results(args.runs_root, args.output), indent=2))
            return 0
        if args.command == "evaluate":
            from fmfc.experiments.runner import evaluate_run

            print(evaluate_run(args.run_dir, args.split, args.ablation, args.config))
            return 0
        if args.command == "explain":
            from fmfc.experiments.explain import explain_checkpoint

            result = explain_checkpoint(
                load_config(args.config),
                args.checkpoint,
                args.dataset,
                split=args.split,
                limit=args.limit,
                record_id=args.record_id,
                head=args.head,
                top_k=args.top_k,
            )
            content = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(content + "\n", encoding="utf-8")
                print(args.output)
            else:
                print(content)
            return 0
        from fmfc.experiments.runner import run

        if args.command == "matrix":
            from fmfc.experiments.matrix import run_matrix

            print(
                json.dumps(
                    [str(path) for path in run_matrix(load_config(args.config))],
                    indent=2,
                )
            )
        else:
            config = apply_overrides(load_config(args.config), args.overrides)
            config["kind"] = args.command.removeprefix("train-")
            print(run(config, config["kind"]))
    except (
        OSError,
        ValueError,
        RuntimeError,
        yaml.YAMLError,
        ModuleNotFoundError,
    ) as exc:
        parser.error(str(exc))
    return 0
