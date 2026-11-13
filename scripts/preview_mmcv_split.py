"""M1 preview: compare the MMCV split strategies before enabling one (nothing is written).

Usage (from the repo root):
    python -B scripts/preview_mmcv_split.py [--config configs/runs/fl_configs.yaml]

Prints, per strategy (grouped = current, grouped_stratified = M1), the claims per split
by modality combination, hop group and label, and checks that no claim, text, image or
table content is in two splits and that the same seed gives the same split.
To enable M1 afterwards, add `strategy: grouped_stratified` under the MMCV `split:`
block of every run config; MMCV dev/test then change, so earlier MMCV results stop
being comparable (Gate D).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmfc.data.mmcv import claim_hops, get_dataset  # noqa: E402
from fmfc.experiments.runner import _mmcv_keys, _mmcv_splits, _mmcv_stratum  # noqa: E402

SPLITS = ("train", "dev", "test")


def name(stratum):
    (text, image, table), hop = stratum
    combination = "+".join(n for n, flag in (("T", text), ("I", image), ("Tab", table)) if flag)
    return combination, "3-4" if hop == 3 else str(hop)


def report(parts, hops):
    rows = {split: [name(_mmcv_stratum(row, hops)) for row in parts[split]] for split in SPLITS}
    for title, index in (("combination", 0), ("hop", 1)):
        counts = {split: Counter(row[index] for row in rows[split]) for split in SPLITS}
        print(f"  {title:<12}" + "".join(f"{split:>8}" for split in SPLITS) + "   dev% test%")
        for key in sorted(set().union(*counts.values())):
            values = [counts[split][key] for split in SPLITS]
            total = sum(values)
            print(f"  {key:<12}" + "".join(f"{value:>8}" for value in values)
                  + f"   {100 * values[1] / total:4.1f} {100 * values[2] / total:5.1f}")
    labels = {split: Counter(str(row[3]) for row in parts[split]) for split in SPLITS}
    print("  labels      " + "  ".join(f"{split} {dict(labels[split])}" for split in SPLITS))
    owner: dict = {}
    leaks = 0
    for split in SPLITS:
        for row in parts[split]:
            for key in _mmcv_keys(row):
                leaks += owner.setdefault(key, split) != split
    print(f"  sizes {[len(parts[split]) for split in SPLITS]}; contents in two splits: {leaks}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/runs/fl_configs.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = next(spec for spec in cfg["datasets"] if spec["id"] == "mmcv")
    subset = spec.get("subset", "all")
    claims = get_dataset(spec["path"], subset=subset, mmqa_path=spec.get("mmqa_path"))
    hops = claim_hops(spec["path"], subset)
    for strategy in ("grouped", "grouped_stratified"):
        policy = {**spec["split"], "strategy": strategy}
        parts = _mmcv_splits(claims, policy, hops)
        again = _mmcv_splits(claims, policy, hops)
        same = all([r[4] for r in parts[s]] == [r[4] for r in again[s]] for s in SPLITS)
        print(f"\n== {strategy} (ratios {policy['ratios']}, seed {policy['seed']}); "
              f"same seed, same split: {same}")
        report(parts, hops)


if __name__ == "__main__":
    main()
