"""Export observed dev results; absent runs never become zero-valued measurements."""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _config_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Observed F1 must be numeric or null")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Observed F1 must be finite and in [0, 1]")
    return float(value)


def _client_dataset(datasets: dict[str, str], client_id: str) -> str | None:
    """Resolve a client's dataset identity without equating different pseudo-clients."""
    dataset_id = client_id
    if client_id not in datasets and "_pseudo_" in client_id:
        dataset_id, suffix = client_id.rsplit("_pseudo_", 1)
        if not suffix.isdigit():
            return None
    value = datasets.get(dataset_id)
    return value if isinstance(value, str) and value.strip() else None


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_results(runs_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Write run/client CSVs and paired-seed summaries from existing dev artifacts.

    Negative transfer is defined only with exactly one local baseline sharing a
    declared comparison_id, seed, client, dataset config and model config.
    A single seed has an undefined sample standard deviation, emitted as blank.
    No test file or raw example is read by this function.
    """
    root, output = Path(runs_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    clients: list[dict[str, Any]] = []
    limitations: list[str] = []
    for metric_path in sorted(root.glob("*/metrics_dev.json")):
        run_dir = metric_path.parent
        metrics = _json(metric_path)
        config_path = run_dir / "config_resolved.yaml"
        config = (
            yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        )
        if not isinstance(config, dict):
            raise ValueError(f"Expected configuration mapping: {config_path}")
        algorithm = config.get("algorithm", config.get("fl", {}).get("algorithm", "unknown"))
        if config.get("kind") in {"local", "centralized"}:
            algorithm = config["kind"]
        seed, comparison = config.get("seed"), config.get("comparison_id")
        synthetic = metrics.get("synthetic", False)
        dataset_specs = config.get("datasets", [])
        dataset_map = {spec["id"]: _config_key(spec) for spec in dataset_specs}
        dataset_config = _config_key(dataset_specs) if dataset_specs else None
        model_config = _config_key(
            {
                "model": config.get("model", {}),
                "encoders": config.get("encoders"),
                "ablation": config.get("ablation"),
                "primary_cohort": config.get("evaluation", {}).get("primary_cohort"),
                "training_mode": config.get("training", {}).get("mode"),
                "global_weight": config.get("training", {}).get("global_weight"),
                "private_weight": config.get("training", {}).get("private_weight"),
            }
        )
        client_metrics = metrics.get("clients")
        if not isinstance(client_metrics, dict):
            raise ValueError(f"Dev metrics must contain a clients mapping: {metric_path}")
        local_client = None
        if algorithm == "local":
            local_client = config.get("local_client")
            if local_client is None and len(client_metrics) == 1:
                local_client = next(iter(client_metrics))
        observed = []
        total = 0
        for client_id, report in sorted(client_metrics.items()):
            binary = report["binary"]
            score = _score(binary.get("macro_f1"))
            count = binary["sample_count"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("Observed sample_count must be a nonnegative integer")
            if (count == 0) != (score is None):
                raise ValueError("F1 must be null exactly when no eligible samples were evaluated")
            if score is not None:
                observed.append(score)
            total += count
            client_dataset = _client_dataset(dataset_map, client_id)
            if client_dataset is None:
                limitations.append(
                    f"{run_dir.name}/{client_id}: client dataset config absent; "
                    "negative transfer undefined"
                )
            clients.append(
                {
                    "run_id": run_dir.name,
                    "algorithm": algorithm,
                    "seed": seed,
                    "comparison_id": comparison,
                    "synthetic": synthetic,
                    "client_id": client_id,
                    "binary_sample_count": count,
                    "binary_macro_f1": score,
                    "class_support": json.dumps(binary.get("class_support")),
                    "strict_text_image_count": report.get("cohort", {}).get("strict_text_image"),
                    "negative_transfer": None,
                    "dataset_config": dataset_config,
                    "client_dataset_config": client_dataset,
                    "model_config": model_config,
                }
            )
        run = {
            "run_id": run_dir.name,
            "algorithm": algorithm,
            "local_client": local_client,
            "seed": seed,
            "comparison_id": comparison,
            "synthetic": synthetic,
            "client_count": len(client_metrics),
            "binary_sample_count": total,
            "mean_client_binary_macro_f1": statistics.mean(observed) if observed else None,
            "worst_client_binary_macro_f1": min(observed) if observed else None,
            "clients_with_binary_metrics": len(observed),
            "dataset_config": dataset_config,
            "model_config": model_config,
        }
        communication_path = run_dir / "communication.json"
        traffic = _json(communication_path) if communication_path.exists() else {}
        for direction in ("uplink_tensor_bytes", "downlink_tensor_bytes"):
            run[direction] = traffic.get(direction)
        runs.append(run)
        if comparison is None or dataset_config is None:
            limitations.append(
                f"{run_dir.name}: no paired negative transfer without comparison_id/dataset config"
            )
        if seed is None:
            limitations.append(f"{run_dir.name}: seed unknown; excluded from multi-seed summary")

    def pairing(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            row.get(key)
            for key in (
                "comparison_id",
                "seed",
                "synthetic",
                "client_id",
                "client_dataset_config",
                "model_config",
            )
        )

    baselines: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in clients:
        if (
            row["algorithm"] == "local"
            and row["comparison_id"] is not None
            and row["client_dataset_config"]
        ):
            baselines[pairing(row)].append(row)
    for row in clients:
        references = baselines.get(pairing(row), [])
        if row["algorithm"] != "local" and len(references) == 1:
            reference = references[0]["binary_macro_f1"]
            if row["binary_macro_f1"] is not None and reference is not None:
                row["negative_transfer"] = row["binary_macro_f1"] - reference

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run["seed"] is not None:
            key = tuple(
                run[key]
                for key in (
                    "algorithm",
                    "local_client",
                    "comparison_id",
                    "synthetic",
                    "dataset_config",
                    "model_config",
                )
            )
            grouped[key].append(run)
    summaries = []
    for key, members in grouped.items():
        seeds = [member["seed"] for member in members]
        if len(set(seeds)) != len(seeds):
            limitations.append(f"{key[0]}: duplicate seeds; ambiguous group omitted from summary")
            continue
        summary = dict(
            zip(
                (
                    "algorithm",
                    "local_client",
                    "comparison_id",
                    "synthetic",
                    "dataset_config",
                    "model_config",
                ),
                key,
                strict=True,
            )
        )
        summary.update({"seed_count": len(seeds), "seeds": json.dumps(sorted(seeds))})
        for metric in ("mean_client_binary_macro_f1", "worst_client_binary_macro_f1"):
            values = [member[metric] for member in members if member[metric] is not None]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else None
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(summary)

    results_path, clients_path, summary_path = (
        output / "results.csv",
        output / "client_results.csv",
        output / "summary.csv",
    )
    _write_csv(
        results_path,
        [
            "run_id",
            "algorithm",
            "local_client",
            "seed",
            "comparison_id",
            "synthetic",
            "client_count",
            "binary_sample_count",
            "clients_with_binary_metrics",
            "mean_client_binary_macro_f1",
            "worst_client_binary_macro_f1",
            "uplink_tensor_bytes",
            "downlink_tensor_bytes",
            "dataset_config",
            "model_config",
        ],
        runs,
    )
    _write_csv(
        clients_path,
        [
            "run_id",
            "algorithm",
            "seed",
            "comparison_id",
            "synthetic",
            "client_id",
            "binary_sample_count",
            "binary_macro_f1",
            "class_support",
            "strict_text_image_count",
            "negative_transfer",
            "client_dataset_config",
            "dataset_config",
            "model_config",
        ],
        clients,
    )
    _write_csv(
        summary_path,
        [
            "algorithm",
            "local_client",
            "comparison_id",
            "synthetic",
            "seed_count",
            "seeds",
            "mean_client_binary_macro_f1_mean",
            "mean_client_binary_macro_f1_std",
            "worst_client_binary_macro_f1_mean",
            "worst_client_binary_macro_f1_std",
            "dataset_config",
            "model_config",
        ],
        summaries,
    )
    limitations_path = output / "limitations.json"
    limitations_path.write_text(
        json.dumps({"limitations": limitations}, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "run_count": len(runs),
        "client_row_count": len(clients),
        "paths": {
            "results": str(results_path),
            "clients": str(clients_path),
            "summary": str(summary_path),
            "limitations": str(limitations_path),
        },
    }
