"""Local/centralized/federated runners over feature batches."""

import copy
import csv
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from fmfc.config import load_config
from fmfc.data.crosstrans import CrossTransCollator
from fmfc.data.schema import EvidenceItem, VerificationRecord
from fmfc.environment import resolve_device
from fmfc.experiments.ablations import ablate_batch
from fmfc.federated.client import ClientState
from fmfc.federated.server import ServerState
from fmfc.model.crosstrans import CrossTransVerifier
from fmfc.model.encoders import (
    FrozenCLIPImageEncoder,
    FrozenTokenTextEncoder,
)
from fmfc.seeds import seed_everything
from fmfc.training.evaluation import evaluate
from fmfc.training.local import make_optimizer, presence_log_priors, train_epoch


def _json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def resolve_config(config: dict[str, Any], kind: str = "federated") -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.setdefault("kind", kind)
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "cpu")
    cfg.setdefault("output_root", "outputs")
    cfg.setdefault("ablation", "all")
    cfg.setdefault(
        "use_table_evidence", cfg.get("model", {}).get("table_dim") is not None
    )
    for section, defaults in {
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
        },
        "fl": {
            "algorithm": "fedavg",
            "rounds": 3,
            "local_epochs": 1,
            "weighting": "sample",
            "prox_mu": 0.01,
            "server_learning_rate": 0.01,
        },
        "evaluation": {
            "selection_split": "dev",
            "final_split": "test",
            "primary_cohort": "all_eligible",
        },
    }.items():
        cfg[section] = {**defaults, **cfg.get(section, {})}
    if cfg.get("limit") is not None:
        raise ValueError("limit was removed: runs always use the full train/dev splits")
    cfg["algorithm"] = (
        cfg["kind"] if cfg["kind"] != "federated" else cfg["fl"]["algorithm"]
    )
    return cfg


def _mmcv_splits(claims: list, policy: dict[str, Any]) -> dict[str, list]:
    ratios = policy.get("ratios", [])
    seed = policy.get("seed")

    claims = sorted(claims, key=lambda row: row[4])
    parent = list(range(len(claims)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, (claim, texts, images, _, _, tables) in enumerate(claims):
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
        for key in keys:
            if key in seen:
                parent[root(index)] = root(seen[key])
            else:
                seen[key] = index
    groups: dict[int, list] = {}
    for index, row in enumerate(claims):
        groups.setdefault(root(index), []).append(row)
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    parts: list[list] = [[], [], []]
    targets = [len(claims) * ratio for ratio in ratios]
    for group in ordered:
        target = max(range(3), key=lambda i: targets[i] - len(parts[i]))
        parts[target].extend(group)
    return {
        name: sorted(rows, key=lambda row: row[4])
        for name, rows in zip(("train", "dev", "test"), parts, strict=True)
    }


def _real_batches(
    cfg: dict[str, Any],
    split: str,
    cache: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Adapt the supplied tuple loaders; frozen encoding is deferred to each batch."""
    specs = cfg.get("datasets", [])
    if cfg["kind"] == "local":
        selected = cfg.get("local_client", specs[0]["id"])
        specs = [spec for spec in specs if spec["id"] == selected]
    if cache is None:
        cache = {}
    if "encoders" not in cache:
        encoders = cfg.get("encoders", {})
        cache["encoders"] = (
            FrozenTokenTextEncoder(**{**encoders["text"], "device": cfg["device"]}),
            FrozenCLIPImageEncoder(**{**encoders["vision"], "device": cfg["device"]}),
        )
        cache["evidence_encoder"] = FrozenTokenTextEncoder(
            **{**encoders["evidence"], "device": cfg["device"]}
        )
    text_encoder, vision_encoder = cache["encoders"]
    cfg["model"]["text_dim"] = text_encoder.output_dim
    cfg["model"]["image_dim"] = vision_encoder.output_dim
    cfg["model"]["evidence_dim"] = cache["evidence_encoder"].output_dim
    # Tables are serialized and encoded by the same frozen evidence encoder.
    cfg["model"]["table_dim"] = (
        cache["evidence_encoder"].output_dim if cfg["use_table_evidence"] else None
    )
    clients, labels, sources = {}, {}, {}
    for spec in specs:
        dataset = spec["id"]
        if dataset not in cache:
            if dataset == "mocheg":
                from fmfc.data.mocheg import get_dataset

                parts = dict(
                    zip(
                        ("train", "dev", "test"), get_dataset(spec["path"]), strict=True
                    )
                )
                metadata = {"split_protocol": "supplied train/val/test"}
            else:
                from fmfc.data.mmcv import get_dataset

                claims = get_dataset(
                    spec["path"],
                    subset=spec.get("subset", "all"),
                    mmqa_path=spec.get("mmqa_path"),
                )
                parts = _mmcv_splits(claims, spec["split"])
                metadata = {
                    "split_protocol": "grouped claim/text-content/image-path/table-content v2",
                    "split_policy": spec["split"],
                    "table_claims": sum(bool(row[5]) for row in claims),
                    "subset": spec.get("subset", "all"),
                    "mmqa_path": spec.get("mmqa_path"),
                }
            if not cfg["use_table_evidence"] and any(
                len(row) > 5 and row[5] for rows in parts.values() for row in rows
            ):
                raise ValueError(
                    f"{dataset} has table evidence; set use_table_evidence: true "
                    "(tables are never silently discarded)"
                )
            vocabulary = sorted(
                {str(row[3]) for rows in parts.values() for row in rows}
            )
            label_map = {label: index for index, label in enumerate(vocabulary)}
            cache[dataset] = parts, label_map, metadata
        parts, label_map, metadata = cache[dataset]
        rows = sorted(parts[split], key=lambda row: row[4])
        global_map = {
            "supported": "SUPPORT",
            "refuted": "REFUTE",
            "SUPPORT": "SUPPORT",
            "REFUTE": "REFUTE",
        }
        records = []
        for row in rows:
            claim, texts, images, label, claim_id = row[:5]
            tables = row[5] if len(row) > 5 else ()
            record_id = f"{dataset}:{claim_id}"
            evidence = []
            for kind, values in (("text", texts), ("image", images)):
                for index, value in enumerate(values):
                    # Pandas represents missing text as NaN; never encode it as "nan".
                    if isinstance(value, str) and value.strip():
                        content = {"text" if kind == "text" else "image_path": value}
                        evidence.append(
                            EvidenceItem(
                                f"{record_id}:{kind}:{index}",
                                kind,
                                record_id,
                                **content,
                            )
                        )
            evidence.extend(
                EvidenceItem(
                    f"{record_id}:table:{index}", "table", record_id, table=table
                )
                for index, table in enumerate(tables)
            )
            records.append(
                VerificationRecord(
                    record_id=record_id,
                    dataset_id=dataset,
                    split=split,
                    claim_text=claim,
                    evidence=tuple(evidence),
                    original_label=str(label),
                    global_label=global_map.get(label),
                )
            )
        size = cfg["training"]["batch_size"]
        budgets = {
            "max_text_evidence": cfg.get("max_text_evidence", 4),
            "max_image_evidence": cfg.get("max_image_evidence", 2),
        }
        if cfg["use_table_evidence"]:
            budgets["max_table_evidence"] = cfg.get("max_table_evidence", 2)
            budgets["max_table_tokens"] = cfg.get("max_table_tokens", 1024)
        collator = CrossTransCollator(
            text_encoder,
            cache["evidence_encoder"],
            vision_encoder,
            label_map,
            **budgets,
            cache_dir=cfg.get("feature_cache_dir"),
        )
        clients[dataset] = [
            collator.defer(records[start : start + size], cfg["ablation"], cfg["seed"])
            for start in range(0, len(records), size)
        ]
        labels[dataset] = label_map
        sources[dataset] = {
            "path": str(spec["path"]),
            **metadata,
            "split_sizes": {name: len(part) for name, part in parts.items()},
            "encoded_samples": len(records),
            "split": split,
            "encoders": {
                "text": text_encoder.fingerprint,
                "vision": vision_encoder.fingerprint,
            },
        }
    return clients, labels, sources


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


def _metrics(
    models: dict[str, CrossTransVerifier],
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
    for report in reports.values():
        report["all_eligible_binary"] = report["binary"]
        if cfg["evaluation"]["primary_cohort"] == "strict_text_image":
            report["binary"] = report["strict_binary"]
    values = [
        report["binary"]["macro_f1"]
        for report in reports.values()
        if report["binary"]["macro_f1"] is not None
    ]
    return {
        "clients": reports,
        "mean_client_binary_macro_f1": sum(values) / len(values) if values else None,
        "worst_client_binary_macro_f1": min(values) if values else None,
        "selection_split": "dev",
        "protocol": "evidence-conditioned verification",
        "ablation": cfg["ablation"],
        "primary_cohort": cfg["evaluation"]["primary_cohort"],
        "real_client_count": len(cfg["datasets"]),
        "pseudo_client_count": cfg.get("pseudo_clients", 1),
        "evidence_sufficiency_unknown_after_ablation": cfg["ablation"]
        not in {"all", "both"},
        "original_head_trained": cfg["training"]["mode"] == "shared_private"
        and cfg["training"]["private_weight"] > 0,
    }


def run(config: dict[str, Any], kind: str = "federated") -> Path:
    cfg = resolve_config(config, kind)
    cfg["device"] = resolve_device(cfg["device"])
    seed_everything(cfg["seed"])
    torch.set_num_threads(1)
    started = time.perf_counter()
    real_cache: dict = {}
    train, labels, _ = _real_batches(cfg, "train", real_cache)
    dev, dev_labels, _ = _real_batches(cfg, "dev", real_cache)
    real_cache.clear()
    if labels != dev_labels:
        raise ValueError("Training and dev label vocabularies differ")
    if cfg["kind"] == "local" and len(train) != 1:
        selected = cfg.get("local_client", next(iter(train)))
        train, dev = {selected: train[selected]}, {selected: dev[selected]}
    model_args = {
        **cfg["model"],
        "private_labels": {key: len(value) for key, value in labels.items()},
    }
    initial = CrossTransVerifier(**model_args).to(cfg["device"])
    for dataset, head in initial.private_heads.items():
        # Dataset-specific initialization must not depend on client ordering.
        with torch.random.fork_rng(devices=[]):
            private_seed = cfg["seed"] + sum(
                (index + 1) * ord(char) for index, char in enumerate(dataset)
            )
            torch.manual_seed(private_seed)
            head.reset_parameters()
    models = {client: copy.deepcopy(initial) for client in train}
    if cfg["kind"] == "federated":
        for client, model in models.items():
            local_datasets = {
                name for batch in train[client] for name in batch["dataset_ids"]
            }
            for name in list(model.private_heads):
                if name not in local_datasets:
                    del model.private_heads[name]
    optimizer_args = {
        key: cfg["training"][key] for key in ("learning_rate", "weight_decay")
    }
    clients = {
        client: ClientState(client, model, make_optimizer(model, **optimizer_args))
        for client, model in models.items()
    }
    if cfg["kind"] in {"local", "centralized"}:
        for client in models:
            models[client] = initial
        clients = {
            "pooled": ClientState(
                "pooled", initial, make_optimizer(initial, **optimizer_args)
            )
        }
    fl = cfg["fl"]
    server = ServerState(
        initial.shared_state(),
        initial.parameter_groups(),
        algorithm=fl["algorithm"],
        weighting=fl["weighting"],
        learning_rate=fl["server_learning_rate"],
    )
    history, step, best, stale = [], 0, -1.0, 0
    # run_id = cfg.get("run_id") or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = (
        f'{kind}_{datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")}'
        if cfg.get("run_id") is None
        else cfg["run_id"]
    )
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    folder = Path(cfg["output_root"]) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    target = fl["rounds"] if cfg["kind"] == "federated" else cfg["training"]["epochs"]
    train_kwargs = {
        key: cfg["training"][key]
        for key in (
            "mode",
            "global_weight",
            "private_weight",
            "max_grad_norm",
            "warmup_steps",
            "logit_adjustment",
        )
    }
    train_kwargs["device"] = cfg["device"]
    batch_size, reshuffle = (
        cfg["training"]["batch_size"],
        cfg["training"]["reshuffle_records"],
    )
    priors = {}
    if cfg["training"]["logit_adjustment"]:
        if cfg["ablation"] not in {"all", "both"}:
            raise ValueError(
                "Logit-adjustment priors assume unablated training evidence"
            )
        # Each client's own train records; keys carry the dataset, so pooling is safe.
        priors = {
            client: presence_log_priors(batches) for client, batches in train.items()
        }
    selected_state = None
    step_label = "Round" if cfg["kind"] == "federated" else "Epoch"
    while step < target:
        stats = {}
        if cfg["kind"] == "federated":
            updates = []
            for client_id, client in clients.items():
                client.broadcast(server.shared_state())
                batches = (
                    _reshuffled_batches(train[client_id], batch_size)
                    if reshuffle
                    else random.sample(train[client_id], len(train[client_id]))
                )
                update = client.train(
                    batches,
                    epochs=fl["local_epochs"],
                    **train_kwargs,
                    log_priors=priors.get(client_id),
                    progress_desc=f"Train | Round {step + 1}/{target}",
                    prox_mu=fl["prox_mu"] if fl["algorithm"] == "fedprox" else 0.0,
                )
                updates.append(update)
                stats[client_id] = {
                    "optimizer_steps": update.optimizer_steps,
                    "sample_count": update.sample_count,
                }
            shared = server.aggregate(updates)
            for client in clients.values():
                client.broadcast(shared)
        else:
            pooled = [
                batch
                for batches in train.values()
                for batch in (
                    _reshuffled_batches(batches, batch_size) if reshuffle else batches
                )
            ]
            stats["pooled"] = train_epoch(
                initial,
                random.sample(pooled, len(pooled)),
                clients["pooled"].optimizer,
                **train_kwargs,
                log_priors={
                    key: value
                    for part in priors.values()
                    for key, value in part.items()
                },
                progress_desc=f"Train | {cfg['kind']} | Epoch {step + 1}/{target}",
            )
        step += 1
        metrics = _metrics(
            models,
            dev,
            cfg,
            progress_desc=f"Val | {step_label} {step}/{target}",
        )
        score = metrics["mean_client_binary_macro_f1"]
        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        history.append(
            {
                "step": step,
                "mean_client_binary_macro_f1": score,
                "worst_client_binary_macro_f1": metrics["worst_client_binary_macro_f1"],
                "optimizer_steps": json.dumps(
                    {k: v["optimizer_steps"] for k, v in stats.items()}
                ),
            }
        )
        if improved:
            selected_state = {
                "step": step,
                "server": server.state_dict(),
                "clients": {
                    key: client.state_dict() for key, client in clients.items()
                },
            }
        patience = cfg["training"]["patience"]
        if patience and stale >= patience:
            break
    _load_selected(clients, selected_state, cfg)
    torch.save(
        {
            "step": selected_state["step"],
            "score": best,
            "models": {
                client: payload["model"]
                for client, payload in selected_state["clients"].items()
            },
        },
        folder / "best_model.pt",
    )
    metrics = _metrics(
        models,
        dev,
        cfg,
        progress_desc=f"Val | Selected {step_label.lower()} {selected_state['step']}",
    )
    metrics["selected_step"] = selected_state["step"]
    _json(folder / "metrics_dev.json", metrics)
    _json(folder / "metrics.json", {"dev": metrics})
    with (folder / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    if cfg["kind"] == "federated":
        _json(
            folder / "communication.json",
            {
                **server.communication,
                "scope": "shared tensor payload only; excludes serialization/transport overhead",
            },
        )
        _json(folder / "round_history.json", server.history)
    return folder


def _load_selected(clients: dict, state: dict, cfg: dict) -> None:
    for name, client in clients.items():
        client.load_state_dict(state["clients"][name])
        if cfg["kind"] == "federated":
            client.broadcast(state["server"]["shared"])


def freeze_selection(run_dir: str | Path) -> None:
    folder = Path(run_dir)
    if not (folder / "selection.json").is_file():
        raise ValueError(
            "freeze-selection requires selection.json from an older run; "
            "new runs save results and best model weights only"
        )
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    _json(
        folder / "selection_frozen.json",
        {
            "selected_step": selection["selected_step"],
            "frozen_at": datetime.now(UTC).isoformat(),
        },
    )


def evaluate_run(
    run_dir: str | Path,
    split: str = "dev",
    final: bool = False,
    ablation: str | None = None,
) -> dict:
    folder = Path(run_dir)
    if (
        not (folder / "config_resolved.yaml").is_file()
        or not (folder / "checkpoints" / "selected.pt").is_file()
    ):
        raise ValueError(
            "Re-evaluation requires config and checkpoint files from an older run; "
            "new runs save results and best model weights only"
        )
    cfg = load_config(folder / "config_resolved.yaml")
    if split not in {"dev", "test"}:
        raise ValueError("Evaluation split must be dev or test")
    if split == "test":
        if ablation:
            raise ValueError(
                "Choose and preregister test ablations before final evaluation"
            )
        frozen = folder / "selection_frozen.json"
        if not final or not frozen.is_file():
            raise ValueError("Test is locked; freeze selection and pass --final")
        if (folder / "metrics_test.json").exists():
            raise ValueError("Final test evaluation already exists")
    elif final:
        raise ValueError("--final applies only to test")
    state = torch.load(
        folder / "checkpoints" / "selected.pt", map_location="cpu", weights_only=True
    )
    data, labels, _ = _real_batches(cfg, split)
    if cfg["kind"] == "local":
        selected = cfg.get("local_client", next(iter(data)))
        data = {selected: data[selected]}
    models = {}
    for name in data:
        payload = state["clients"][name if cfg["kind"] == "federated" else "pooled"]
        local_labels = {
            key: len(value)
            for key, value in labels.items()
            if any(
                param.startswith(f"private_heads.{key}.") for param in payload["model"]
            )
        }
        model = CrossTransVerifier(**cfg["model"], private_labels=local_labels)
        model.load_state_dict(payload["model"])
        if cfg["kind"] == "federated":
            model.load_shared_state(state["server"]["shared"])
        models[name] = model
        if ablation:
            data[name] = [
                (
                    {**batch, "_ablation": ablation}
                    if "_collator" in batch
                    else ablate_batch(batch, ablation, cfg["seed"])[0]
                )
                for batch in data[name]
            ]
    metrics = _metrics(
        models, data, cfg, progress_desc="Test" if split == "test" else "Val"
    )
    metrics["evaluation_split"] = split
    metrics["ablation"] = ablation or cfg["ablation"]
    metrics["robustness_only"] = metrics["ablation"] in {
        "shuffle_image",
        "drop_image",
        "drop_table",
    }
    metrics["evidence_sufficiency_unknown_after_ablation"] = metrics[
        "ablation"
    ] not in {"all", "both"}
    target = (
        f"metrics_{split}.json" if not ablation else f"ablation_{split}_{ablation}.json"
    )
    _json(folder / target, metrics)
    return metrics
