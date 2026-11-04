"""Run explicitly selected algorithms/seeds with paired configurations."""

import copy
from pathlib import Path
from typing import Any

from fmfc.experiments.runner import run


def run_matrix(config: dict[str, Any]) -> list[Path]:
    base = config["base"]
    seeds = config.get("seeds", [42, 43, 44])
    algorithms = config.get("algorithms", ["fedavg", "fedprox", "fedadam", "evidence_aware"])
    folders = []
    for seed in seeds:
        for algorithm in algorithms:
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            cfg.setdefault("fl", {})["algorithm"] = algorithm
            cfg["comparison_id"] = config.get("comparison_id", "paired-v1")
            cfg.pop("run_id", None)
            folders.append(run(cfg))
    return folders
