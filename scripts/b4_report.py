"""B4 diagnostic report: apply the single-seed decision rule to finished runs (dev only).

Usage (from the repo root), after the runs and evaluations in docs/scenarios_C_E.md (B4):
    python -B scripts/b4_report.py --r0 outputs/e4_central_s42 \
        --r1 outputs/b4_r1_claimonly_central_s42 --r2 outputs/b4_r2_central_s42

Reads from R0: metrics_dev.json, metrics_train.json (B1b), and metrics_dev_shuffle_image.json,
metrics_dev_text.json, metrics_dev_image.json written by scripts/evaluate.py; metrics_dev.json
from R1 (claim-only) and R2 (trained without images and tables). Decisions use dev only, never
test. "Not different" means a gap below --margin (default 0.02, one seed); train F1 counts as
clearly above dev from --overfit-gap (default 0.10).
"""

import argparse
import json
from pathlib import Path

COLUMNS = ("mocheg_3class_macro_f1", "mmcv_binary_macro_f1")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def majority_f1(metrics: dict, column: str) -> float | None:
    """Macro-F1 of always predicting the most frequent class: s / (1 + s) / classes * 2."""
    client = column.split("_", 1)[0]
    report = metrics["clients"].get(client, {})
    node = report["original"].get(client) if "3class" in column else report.get("binary")
    if not node or not node["sample_count"]:
        return None
    share = max(node["class_support"]) / node["sample_count"]
    return 2 * share / (1 + share) / len(node["class_support"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--r0", type=Path, required=True)
    parser.add_argument("--r1", type=Path, help="claim_only run")
    parser.add_argument("--r2", type=Path, help="run trained with ablation=text")
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--overfit-gap", type=float, default=0.10,
                        help="train minus dev F1 that counts as clearly higher (B1b)")
    args = parser.parse_args()
    runs = {
        "R0": load(args.r0 / "metrics_dev.json"),
        "R0-shuf": load(args.r0 / "metrics_dev_shuffle_image.json"),
        "R0-noimg": load(args.r0 / "metrics_dev_text.json"),
        "R0-notext": load(args.r0 / "metrics_dev_image.json"),
        "R1": load(args.r1 / "metrics_dev.json") if args.r1 else None,
        "R2": load(args.r2 / "metrics_dev.json") if args.r2 else None,
        "R0-train": load(args.r0 / "metrics_train.json"),
    }
    if runs["R0"] is None:
        raise SystemExit(f"{args.r0}/metrics_dev.json not found")
    value = {name: (m or {}).get("selection_f1", {}) for name, m in runs.items()}
    print(f"{'run':<10}" + "".join(f"{c:>26}" for c in COLUMNS))
    for name in runs:
        cells = [value[name].get(c) for c in COLUMNS]
        print(f"{name:<10}" + "".join(f"{'-' if v is None else f'{v:.4f}':>26}" for v in cells))
    margin = args.margin
    for column in COLUMNS:
        f1 = {name: value[name].get(column) for name in runs}
        r0 = f1["R0"]
        if r0 is None:
            continue

        def gap(name: str) -> float | None:
            return None if f1[name] is None else r0 - f1[name]

        print(f"\n{column} (R0 {r0:.4f}, margin {margin}):")
        rules = []
        if gap("R0-shuf") is not None and gap("R2") is not None:
            if abs(gap("R0-shuf")) < margin and abs(gap("R2")) < margin:
                rules.append("R0 ~ R0-shuf and R0 ~ R2: the image branch adds almost nothing -> C2b")
            if gap("R0-shuf") >= margin and gap("R1") is not None and gap("R1") >= margin:
                rules.append("R0 >> R0-shuf and R0 >> R1: evidence is used -> Gate G")
        if gap("R1") is not None and gap("R1") < margin:
            rules.append("R0 - R1 small: mostly claim-only signal, weak evidence use -> C4, C5c")
        majority = majority_f1(runs["R0"], column)
        if f1["R0-notext"] is not None and majority is not None and f1["R0-notext"] < majority + margin:
            rules.append(f"image-only F1 {f1['R0-notext']:.4f} ~ majority-class {majority:.4f}: "
                         "image branch weak on its own -> C2b")
        train = f1["R0-train"]
        if train is not None:
            if train - r0 >= args.overfit_gap:
                rules.append(f"train {train:.4f} >> dev: overfitting or a shortcut -> no C7")
            elif train - r0 < margin and gap("R1") is not None and gap("R1") >= margin:
                rules.append(f"train {train:.4f} ~ dev and R1 below R0: underfitting, frozen "
                             "features may be the bottleneck -> C5c, then C7")
        missing = [name for name in runs if f1[name] is None]
        for rule in rules or ["no rule matched"]:
            print(f"  - {rule}")
        if missing:
            print(f"  (missing: {', '.join(missing)})")


if __name__ == "__main__":
    main()
