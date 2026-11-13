"""Local/centralized/federated runners over feature batches."""

import copy
import csv
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from fmfc.data.crosstrans import CrossTransCollator
from fmfc.data.schema import EvidenceItem, VerificationRecord
from fmfc.environment import resolve_device
from fmfc.federated.client import ClientState
from fmfc.federated.server import ServerState
from fmfc.model.claim_only import ClaimOnlyVerifier
from fmfc.model.crosstrans import CrossTransVerifier
from fmfc.model.encoders import (
    FrozenCLIPImageEncoder,
    FrozenCLIPTextEncoder,
    FrozenTokenTextEncoder,
)
from fmfc.model.verifier import Verifier
from fmfc.seeds import seed_everything
from fmfc.training.evaluation import evaluate
from fmfc.training.local import make_optimizer, presence_log_priors, train_epoch

DEFAULTS = {
    "model": {"hidden_dim": 256},
    "training": {
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "mode": "binary_only",
        "global_weight": 1.0,
        "private_weight": 0.0,
        "patience": 0,
        "reshuffle_records": False,
        "max_grad_norm": 0.0,
        "warmup_steps": 0,
        "logit_adjustment": 0.0,
        "image_aux_weight": 0.0,
    },
    "fl": {
        "algorithm": "fedavg",
        "rounds": 3,
        "local_epochs": 1,
        "weighting": "sample",
        "prox_mu": 0.01,
        "server_learning_rate": 0.01,
        # D1: optimizer steps per client per round; an int for every client or a
        # {client: steps} mapping; None = no limit.
        "max_local_steps": None,
        # D5: keep fusion_mlp.gates and every LayerNorm on each client; aggregate the
        # rest. Federated runs only; same parameters, only what is sent changes.
        "local_gates_norms": False,
    },
    "evaluation": {
        "primary_cohort": "all_eligible",
        "train_metrics": False,  # B1b: also score the selected checkpoint on train
    },
}
TRAIN_KEYS = (
    "mode",
    "global_weight",
    "private_weight",
    "max_grad_norm",
    "warmup_steps",
    "logit_adjustment",
    "image_aux_weight",
)
GLOBAL_LABELS = {
    "supported": "SUPPORT",
    "refuted": "REFUTE",
    "SUPPORT": "SUPPORT",
    "REFUTE": "REFUTE",
}
# Checkpoint selection (user decision 2026-11-13): MOCHEG uses the 3-class macro-F1 of its
# private head (supported/refuted/NEI); every other dataset uses binary macro-F1.
SELECTION_HEAD = {"mocheg": "original"}


def _json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def resolve_config(config: dict[str, Any], kind: str = "federated") -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    # No config file sets these two; the matrix runner passes no kind.
    cfg.setdefault("kind", kind)
    cfg.setdefault("ablation", "all")
    for section, defaults in DEFAULTS.items():
        cfg[section] = {**defaults, **cfg.get(section, {})}
    if cfg.get("limit") is not None:
        raise ValueError("limit was removed: runs always use the full train/dev splits")
    return cfg


def build_model(cfg: dict[str, Any], private_labels: dict[str, int]) -> Verifier:
    """CrossTrans by default; `model.architecture: claim_only` builds the B3 claim-only control."""
    args = {key: value for key, value in cfg["model"].items() if key != "architecture"}
    if cfg["model"].get("architecture", "crosstrans") == "claim_only":
        return ClaimOnlyVerifier(**args, private_labels=private_labels)
    return CrossTransVerifier(**args, private_labels=private_labels)


# ----------------------------------------------------------------------------- data


def _mmcv_keys(row: tuple) -> list[tuple[str, str]]:
    """Contents that may not appear in two splits: claim, text, image path, table."""
    claim, texts, images, _, _, tables = row
    keys = [("claim", " ".join(claim.casefold().split()))]
    keys.extend(
        ("text", " ".join(text.casefold().split()))
        for text in texts
        if isinstance(text, str) and text.strip()
    )
    keys.extend(("image", path.replace("\\", "/").casefold()) for path in images)
    keys.extend(
        ("table", json.dumps(table, sort_keys=True, ensure_ascii=False))
        for table in tables
    )
    return keys


def _mmcv_stratum(row: tuple, hops: list[int]) -> tuple[tuple[bool, ...], int]:
    """M1 stratum: modality combination (as records see it) x hop group {1, 2, 3-4}."""
    _, texts, images, _, claim_id, tables = row
    combination = (
        any(isinstance(text, str) and text.strip() for text in texts),
        any(isinstance(path, str) and path.strip() for path in images),
        bool(tables),
    )
    return combination, min(hops[claim_id], 3)


def _mmcv_groups(claims: list) -> list[list]:
    """Union-find: claims that share any content key end up in the same group."""
    parent = list(range(len(claims)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict = {}
    for index, row in enumerate(claims):
        for key in _mmcv_keys(row):
            if key in seen:
                parent[root(index)] = root(seen[key])
            else:
                seen[key] = index
    groups: dict[int, list] = {}
    for index, row in enumerate(claims):
        groups.setdefault(root(index), []).append(row)
    return list(groups.values())


def _mmcv_splits(
    claims: list,
    policy: dict[str, Any],
    hops: list[int] | None = None,
) -> dict[str, list]:
    ratios = policy.get("ratios", [])
    strategy = policy.get("strategy", "grouped")
    if strategy not in {"grouped", "grouped_stratified"}:
        raise ValueError("split.strategy must be grouped or grouped_stratified")
    if strategy == "grouped_stratified" and hops is None:
        raise ValueError("split.strategy grouped_stratified needs hop counts")
    claims = sorted(claims, key=lambda row: row[4])
    ordered = _mmcv_groups(claims)
    random.Random(policy.get("seed")).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    parts: list[list] = [[], [], []]
    if strategy == "grouped":
        targets = [len(claims) * ratio for ratio in ratios]
        for group in ordered:
            target = max(range(3), key=lambda i: targets[i] - len(parts[i]))
            parts[target].extend(group)
    else:
        # M1: iterative stratification (Sechidis et al., 2011) at group level. A group goes to
        # the split where most of its claims still fit the per-stratum target: each claim that
        # fits counts +1, each claim beyond the target -1. Labels are not used. (A relative-need
        # score instead sent large groups to dev: dev 1,662 claims, T+Tab 3-4 hop 24% in dev.)
        strata = {row[4]: _mmcv_stratum(row, hops) for row in claims}
        totals: dict = {}
        for stratum in strata.values():
            totals[stratum] = totals.get(stratum, 0) + 1
        need = {stratum: [n * ratio for ratio in ratios] for stratum, n in totals.items()}
        assigned = {stratum: [0, 0, 0] for stratum in totals}
        for group in ordered:
            counts: dict = {}
            for row in group:
                counts[strata[row[4]]] = counts.get(strata[row[4]], 0) + 1
            target = max(
                range(3),
                key=lambda i: sum(
                    2 * min(count, max(need[stratum][i] - assigned[stratum][i], 0))
                    - count
                    for stratum, count in counts.items()
                ),
            )
            parts[target].extend(group)
            for stratum, count in counts.items():
                assigned[stratum][target] += count
    return {
        name: sorted(rows, key=lambda row: row[4])
        for name, rows in zip(("train", "dev", "test"), parts, strict=True)
    }


def _encoders(cfg: dict[str, Any], cache: dict) -> dict[str, Any]:
    """Frozen encoders, loaded once per cache; also sets the feature sizes in cfg["model"]."""
    if "encoders" not in cache:
        specs = cfg["encoders"]
        device = {"device": cfg["device"]}
        cache["encoders"] = {  # built in this order: text, vision, evidence, claim CLIP
            "text": FrozenTokenTextEncoder(**{**specs["text"], **device}),
            "vision": FrozenCLIPImageEncoder(**{**specs["vision"], **device}),
            "evidence": FrozenTokenTextEncoder(**{**specs["evidence"], **device}),
            # C3: the claim's CLIP text embedding must share the image encoder's checkpoint.
            "claim_clip": (
                FrozenCLIPTextEncoder(**{**specs["vision"], **device})
                if cfg["model"].get("clip_similarity")
                else None
            ),
        }
    encoders = cache["encoders"]
    cfg["model"].update(
        text_dim=encoders["text"].output_dim,
        image_dim=encoders["vision"].output_dim,
        evidence_dim=encoders["evidence"].output_dim,
        # Tables are serialized and encoded by the same frozen evidence encoder.
        table_dim=encoders["evidence"].output_dim if cfg["use_table_evidence"] else None,
    )
    return encoders


def _dataset_splits(spec: dict[str, Any], use_tables: bool) -> tuple[dict, dict]:
    """Rows per split (MOCHEG: official splits; MMCV: grouped split) and the label map."""
    if spec["id"] == "mocheg":
        from fmfc.data.mocheg import get_dataset

        parts = dict(
            zip(("train", "dev", "test"), get_dataset(spec["path"]), strict=True)
        )
    else:
        from fmfc.data.mmcv import claim_hops, get_dataset

        subset = spec.get("subset", "all")
        claims = get_dataset(spec["path"], subset=subset, mmqa_path=spec.get("mmqa_path"))
        stratified = spec["split"].get("strategy") == "grouped_stratified"
        hops = claim_hops(spec["path"], subset) if stratified else None
        parts = _mmcv_splits(claims, spec["split"], hops)
    if not use_tables and any(
        len(row) > 5 and row[5] for rows in parts.values() for row in rows
    ):
        raise ValueError(
            f"{spec['id']} has table evidence; set use_table_evidence: true "
            "(tables are never silently discarded)"
        )
    vocabulary = sorted({str(row[3]) for rows in parts.values() for row in rows})
    return parts, {label: index for index, label in enumerate(vocabulary)}


def _record(dataset: str, split: str, row: tuple) -> VerificationRecord:
    claim, texts, images, label, claim_id = row[:5]
    tables = row[5] if len(row) > 5 else ()
    record_id = f"{dataset}:{claim_id}"
    evidence = [
        EvidenceItem(
            f"{record_id}:{kind}:{index}",
            kind,
            record_id,
            **{"text" if kind == "text" else "image_path": value},
        )
        for kind, values in (("text", texts), ("image", images))
        for index, value in enumerate(values)
        # Pandas represents missing text as NaN; never encode it as "nan".
        if isinstance(value, str) and value.strip()
    ]
    evidence.extend(
        EvidenceItem(f"{record_id}:table:{index}", "table", record_id, table=table)
        for index, table in enumerate(tables)
    )
    return VerificationRecord(
        record_id=record_id,
        dataset_id=dataset,
        split=split,
        claim_text=claim,
        evidence=tuple(evidence),
        original_label=str(label),
        global_label=GLOBAL_LABELS.get(label),
    )


def _real_batches(
    cfg: dict[str, Any], split: str, cache: dict | None = None
) -> tuple[dict, dict]:
    """Deferred batches per client and label maps; frozen encoding happens per batch."""
    cache = {} if cache is None else cache
    specs = cfg.get("datasets", [])
    if cfg["kind"] == "local":
        selected = cfg.get("local_client", specs[0]["id"])
        specs = [spec for spec in specs if spec["id"] == selected]
    encoders = _encoders(cfg, cache)
    budgets = {
        "max_text_evidence": cfg.get("max_text_evidence", 4),
        "max_image_evidence": cfg.get("max_image_evidence", 2),
    }
    if cfg["use_table_evidence"]:
        budgets["max_table_evidence"] = cfg.get("max_table_evidence", 2)
        budgets["max_table_tokens"] = cfg.get("max_table_tokens", 1024)
    size = cfg["training"]["batch_size"]
    clients, labels = {}, {}
    for spec in specs:
        dataset = spec["id"]
        if dataset not in cache:
            cache[dataset] = _dataset_splits(spec, cfg["use_table_evidence"])
        parts, labels[dataset] = cache[dataset]
        rows = sorted(parts[split], key=lambda row: row[4])
        records = [_record(dataset, split, row) for row in rows]
        collator = CrossTransCollator(
            encoders["text"],
            encoders["evidence"],
            encoders["vision"],
            labels[dataset],
            **budgets,
            cache_dir=cfg.get("feature_cache_dir"),
            claim_clip_encoder=encoders["claim_clip"],
        )
        clients[dataset] = [
            collator.defer(records[start : start + size], cfg["ablation"], cfg["seed"])
            for start in range(0, len(records), size)
        ]
    return clients, labels


def _reshuffled_batches(batches: list[dict], size: int) -> list[dict]:
    """Re-chunk shuffled records so batch composition changes every epoch/round.

    Batches of one client share one collator (one label map), so datasets never mix.
    """
    if not batches:
        return []
    first = batches[0]
    records = [record for batch in batches for record in batch["_records"]]
    random.shuffle(records)
    return [
        first["_collator"].defer(
            records[start : start + size], first["_ablation"], first["_seed"]
        )
        for start in range(0, len(records), size)
    ]


# ----------------------------------------------------------------------- evaluation


def _selection_column(client: str) -> str:
    kind = "3class" if SELECTION_HEAD.get(client) == "original" else "binary"
    return f"{client}_{kind}_macro_f1"


def _metrics(
    models: dict[str, Verifier],
    data: dict,
    cfg: dict,
    *,
    progress_desc: str = "Val",
) -> dict[str, Any]:
    reports = {
        client: evaluate(
            models[client],
            batches,
            device=cfg["device"],
            progress_desc=f"{progress_desc} | {client}",
        )
        for client, batches in data.items()
    }
    training = cfg["training"]
    use_private = (
        training["mode"] == "shared_private" and training["private_weight"] > 0
    )
    for report in reports.values():
        report["all_eligible_binary"] = report["binary"]
        if cfg["evaluation"]["primary_cohort"] == "strict_text_image":
            report["binary"] = report["strict_binary"]
        # Same weighting as the training loss, on raw logits (no logit adjustment / prox / aux).
        loss = report["loss"]
        loss["total"] = training["global_weight"] * (loss["global"] or 0.0) + (
            training["private_weight"] * (loss["private"] or 0.0) if use_private else 0.0
        )
    values = [
        report["binary"]["macro_f1"]
        for report in reports.values()
        if report["binary"]["macro_f1"] is not None
    ]
    selection = {
        client: (
            report["original"].get(client, {}).get("macro_f1")
            if SELECTION_HEAD.get(client) == "original"
            else report["binary"]["macro_f1"]
        )
        for client, report in reports.items()
    }
    complete = bool(selection) and None not in selection.values()
    return {
        "clients": reports,
        "selection_rule": "mean over clients of macro-F1: MOCHEG 3-class (private head), "
        "other datasets binary (global head)",
        "selection_f1": {
            _selection_column(client): value for client, value in selection.items()
        },
        "selection_score": (
            sum(selection.values()) / len(selection) if complete else None
        ),
        "mean_client_binary_macro_f1": sum(values) / len(values) if values else None,
        "worst_client_binary_macro_f1": min(values) if values else None,
        "selection_split": "dev",
        "protocol": "evidence-conditioned verification",
        "ablation": cfg["ablation"],
        "primary_cohort": cfg["evaluation"]["primary_cohort"],
        "real_client_count": len(cfg["datasets"]),
        "evidence_sufficiency_unknown_after_ablation": cfg["ablation"]
        not in {"all", "both"},
        "original_head_trained": use_private,
    }


def _scored(models: dict, data: dict, cfg: dict, split: str, step: int) -> dict:
    """Metrics of the dev-selected checkpoint on one split."""
    metrics = _metrics(models, data, cfg, progress_desc=f"{split} | selected step {step}")
    metrics.update(selected_step=step, evaluation_split=split)
    return metrics


# ------------------------------------------------------------------------- training


def _build_clients(
    cfg: dict, labels: dict, train: dict
) -> tuple[Verifier, dict, dict]:
    """Federated: one model per client, with only its own private head.
    Local/centralized: one pooled model and optimizer, shared by every client key."""
    initial = build_model(cfg, {key: len(value) for key, value in labels.items()})
    initial = initial.to(cfg["device"])
    if cfg["kind"] == "federated" and cfg["fl"]["local_gates_norms"]:
        initial.local_gates_norms = True  # before the client copies and the server state
    for dataset, head in initial.private_heads.items():
        # Dataset-specific initialization must not depend on client ordering.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                cfg["seed"] + sum((i + 1) * ord(char) for i, char in enumerate(dataset))
            )
            head.reset_parameters()
    optimizer_args = {
        key: cfg["training"][key] for key in ("learning_rate", "weight_decay")
    }
    if cfg["kind"] != "federated":
        pooled = ClientState("pooled", initial, make_optimizer(initial, **optimizer_args))
        return initial, {client: initial for client in train}, {"pooled": pooled}
    models = {client: copy.deepcopy(initial) for client in train}
    for client, model in models.items():
        own = {name for batch in train[client] for name in batch["dataset_ids"]}
        for name in [name for name in model.private_heads if name not in own]:
            del model.private_heads[name]
    clients = {
        client: ClientState(client, model, make_optimizer(model, **optimizer_args))
        for client, model in models.items()
    }
    return initial, models, clients


def _log_priors(cfg: dict, train: dict) -> dict:
    """A6 priors, from each client's own train records; keys carry the dataset."""
    if not cfg["training"]["logit_adjustment"]:
        return {}
    if cfg["ablation"] not in {"all", "both"}:
        raise ValueError("Logit-adjustment priors assume unablated training evidence")
    return {client: presence_log_priors(batches) for client, batches in train.items()}


def _federated_round(
    cfg: dict,
    train: dict,
    clients: dict,
    server: ServerState,
    kwargs: dict,
    priors: dict,
    desc: str,
) -> dict:
    """Broadcast, local training on every client, aggregation, broadcast again."""
    fl, size = cfg["fl"], cfg["training"]["batch_size"]
    updates, stats = [], {}
    for client_id, client in clients.items():
        client.broadcast(server.shared_state())
        batches = train[client_id]
        batches = (
            _reshuffled_batches(batches, size)
            if cfg["training"]["reshuffle_records"]
            else random.sample(batches, len(batches))
        )
        update = client.train(
            batches,
            epochs=fl["local_epochs"],
            **kwargs,
            log_priors=priors.get(client_id),
            progress_desc=desc,
            prox_mu=fl["prox_mu"] if fl["algorithm"] == "fedprox" else 0.0,
            # D1: one budget for every client, or a {client: steps} mapping.
            max_steps=(
                fl["max_local_steps"].get(client_id)
                if isinstance(fl["max_local_steps"], dict)
                else fl["max_local_steps"]
            ),
        )
        updates.append(update)
        stats[client_id] = {
            "optimizer_steps": update.optimizer_steps,
            "sample_count": update.sample_count,
            "loss": update.metrics["loss"],
        }
    shared = server.aggregate(updates)
    for client in clients.values():
        client.broadcast(shared)
    return stats


def _pooled_epoch(
    cfg: dict, train: dict, pooled: ClientState, kwargs: dict, priors: dict, desc: str
) -> dict:
    """One epoch over the batches of every loaded dataset (local or centralized)."""
    size = cfg["training"]["batch_size"]
    batches = [
        batch
        for client_batches in train.values()
        for batch in (
            _reshuffled_batches(client_batches, size)
            if cfg["training"]["reshuffle_records"]
            else client_batches
        )
    ]
    return train_epoch(
        pooled.model,
        random.sample(batches, len(batches)),
        pooled.optimizer,
        **kwargs,
        log_priors={key: value for part in priors.values() for key, value in part.items()},
        progress_desc=desc,
    )


def _history_row(step: int, metrics: dict, stats: dict) -> dict:
    return {
        "step": step,
        "selection_score": metrics["selection_score"],
        **metrics["selection_f1"],
        "mean_client_binary_macro_f1": metrics["mean_client_binary_macro_f1"],
        "worst_client_binary_macro_f1": metrics["worst_client_binary_macro_f1"],
        # Train loss: mean training objective of the epoch (includes logit adjustment,
        # FedProx and C6 terms when on). "pooled" = all training data of a local or
        # centralized run.
        **{f"{name}_train_loss": value["loss"] for name, value in stats.items()},
        # Dev loss: same weighting on raw logits; private = MOCHEG 3-class head.
        **{
            f"{client}_dev_loss" if part == "total" else f"{client}_dev_{part}_loss": (
                report["loss"][part]
            )
            for client, report in metrics["clients"].items()
            for part in ("total", "global", "private")
        },
        "optimizer_steps": json.dumps(
            {name: value["optimizer_steps"] for name, value in stats.items()}
        ),
    }


def _run_folder(config: dict, cfg: dict) -> Path:
    """Create outputs/<run_id> and save the config as given (overrides applied) in it."""
    run_id = cfg.get("run_id")
    if run_id is None:
        run_id = f"{cfg['kind']}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    folder = Path(cfg["output_root"]) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "config_used.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {**config, "kind": cfg["kind"], "run_id": run_id},
            stream,
            sort_keys=False,
            allow_unicode=True,
        )
    return folder


def run(config: dict[str, Any], kind: str = "federated") -> Path:
    cfg = resolve_config(config, kind)
    cfg["device"] = resolve_device(cfg["device"])
    seed_everything(cfg["seed"])
    torch.set_num_threads(1)
    cache: dict = {}
    train, labels = _real_batches(cfg, "train", cache)
    dev, _ = _real_batches(cfg, "dev", cache)
    # Test is evaluated once, on the checkpoint selected on dev; it never drives selection.
    test, _ = _real_batches(cfg, "test", cache)
    cache.clear()
    federated = cfg["kind"] == "federated"
    initial, models, clients = _build_clients(cfg, labels, train)
    fl = cfg["fl"]
    server = ServerState(
        initial.shared_state(),
        initial.parameter_groups(),
        algorithm=fl["algorithm"],
        weighting=fl["weighting"],
        learning_rate=fl["server_learning_rate"],
    )
    kwargs = {key: cfg["training"][key] for key in TRAIN_KEYS}
    if kwargs["image_aux_weight"] and not cfg["model"].get("image_aux_head"):
        raise ValueError("training.image_aux_weight requires model.image_aux_head: true")
    kwargs["device"] = cfg["device"]
    priors = _log_priors(cfg, train)
    folder = _run_folder(config, cfg)

    target = fl["rounds"] if federated else cfg["training"]["epochs"]
    label = "Round" if federated else "Epoch"
    history, best, stale, selected = [], -1.0, 0, None
    for step in range(1, target + 1):
        desc = f"Train | {label} {step}/{target}"
        if federated:
            stats = _federated_round(cfg, train, clients, server, kwargs, priors, desc)
        else:
            pooled = clients["pooled"]
            stats = {"pooled": _pooled_epoch(cfg, train, pooled, kwargs, priors, desc)}
        metrics = _metrics(models, dev, cfg, progress_desc=f"Val | {label} {step}/{target}")
        if metrics["selection_score"] is None:
            raise ValueError("No dev examples for the checkpoint selection metric")
        history.append(_history_row(step, metrics, stats))
        if metrics["selection_score"] > best:
            best, stale = metrics["selection_score"], 0
            selected = {
                "step": step,
                "shared": server.shared_state(),
                "clients": {name: client.state_dict() for name, client in clients.items()},
            }
        else:
            stale += 1
        if cfg["training"]["patience"] and stale >= cfg["training"]["patience"]:
            break

    for name, client in clients.items():  # restore the dev-selected checkpoint
        client.load_state_dict(selected["clients"][name])
        if federated:
            client.broadcast(selected["shared"])
    torch.save(
        {
            "step": selected["step"],
            "score": best,
            "models": {name: state["model"] for name, state in selected["clients"].items()},
        },
        folder / "best_model.pt",
    )
    splits = {"dev": dev, "test": test}
    if cfg["evaluation"]["train_metrics"]:  # B1b: tells underfitting from overfitting
        splits["train"] = train
    combined = {}
    for split, data in splits.items():
        combined[split] = _scored(models, data, cfg, split, selected["step"])
        _json(folder / f"metrics_{split}.json", combined[split])
    _json(folder / "metrics.json", combined)
    with (folder / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    if federated:
        _json(
            folder / "communication.json",
            {
                **server.communication,
                "scope": "shared tensor payload only; excludes serialization/transport overhead",
            },
        )
        _json(folder / "round_history.json", server.history)
    return folder


def evaluate_run(
    run_dir: str | Path,
    split: str = "test",
    ablation: str | None = None,
    config: str | Path | None = None,
) -> Path:
    """B2: re-score the dev-selected best_model.pt of a finished run; selection is untouched.

    Without ``ablation`` the run's own setting is used and metrics_<split>.json is written;
    with one, metrics_<split>_<ablation>.json. An existing file is replaced.
    """
    folder = Path(run_dir)
    path = Path(config) if config is not None else folder / "config_used.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if ablation is not None:
        raw["ablation"] = ablation
    cfg = resolve_config(raw, raw.get("kind", "federated"))
    cfg["device"] = resolve_device(cfg["device"])
    data, labels = _real_batches(cfg, split)
    payload = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
    models = {}
    for client in data:
        if cfg["kind"] == "federated":  # each client model keeps only its own private head
            heads, state = {client: len(labels[client])}, payload["models"][client]
        else:  # local / centralized: one pooled model with a head per loaded dataset
            heads = {name: len(value) for name, value in labels.items()}
            state = payload["models"]["pooled"]
        model = build_model(cfg, heads)
        model.load_state_dict(state, strict=True)
        models[client] = model.to(cfg["device"])
    output = folder / f"metrics_{split}{'' if ablation is None else f'_{ablation}'}.json"
    _json(output, _scored(models, data, cfg, split, payload["step"]))
    return output
