"""Local/centralized/federated runners over feature batches."""

import copy
import csv
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fmfc.config import load_config
from fmfc.data.collator import FeatureCollator
from fmfc.data.crosstrans import CrossTransCollator
from fmfc.data.schema import EvidenceItem, VerificationRecord
from fmfc.environment import resolve_device
from fmfc.experiments.ablations import ablate_batch
from fmfc.federated.client import ClientState
from fmfc.federated.server import ServerState
from fmfc.model.crosstrans import CrossTransVerifier
from fmfc.model.encoders import (
    FrozenCLIPImageEncoder,
    FrozenTextEncoder,
    FrozenTokenTextEncoder,
    FrozenVisionEncoder,
)
from fmfc.model.verifier import Verifier
from fmfc.seeds import seed_everything
from fmfc.training.evaluation import evaluate
from fmfc.training.local import make_optimizer, train_epoch


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_config(config: dict[str, Any], kind: str = "federated") -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.setdefault("kind", kind)
    cfg.setdefault("run_type", "real")
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "cpu")
    cfg.setdefault("output_root", "outputs")
    cfg.setdefault("ablation", "all")
    cfg.setdefault("model_type", "verifier_v1")
    cfg.setdefault("use_table_evidence", cfg.get("model", {}).get("table_dim") is not None)
    for section, defaults in {
        "model": {"text_dim": 8, "image_dim": 8, "hidden_dim": 16},
        "training": {
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "mode": "binary_only",
            "global_weight": 1.0,
            "private_weight": 0.0,
            "patience": 0,
        },
        "fl": {
            "algorithm": "fedavg",
            "rounds": 3,
            "local_epochs": 1,
            "weighting": "sample",
            "prox_mu": 0.01,
            "server_learning_rate": 0.01,
        },
        "synthetic": {
            "clients": 2,
            "train_samples": 16,
            "dev_samples": 8,
            "mixed_modalities": False,
        },
        "evaluation": {
            "selection_split": "dev",
            "final_split": "test",
            "primary_cohort": "all_eligible",
        },
    }.items():
        cfg[section] = {**defaults, **cfg.get(section, {})}
    limit = cfg.get("limit")
            
    cfg["algorithm"] = cfg["kind"] if cfg["kind"] != "federated" else cfg["fl"]["algorithm"]
    return cfg


def _synthetic_batches(cfg: dict[str, Any], split: str) -> tuple[dict, dict, dict]:
    """Generate feature tensors, never benchmark rows/media or verified evidence."""
    count = cfg["synthetic"][f"{split}_samples"]
    if cfg.get("limit") is not None and split in {"train", "dev"}:
        count = min(count, cfg["limit"])
    clients, labels = {}, {}
    dt, di = cfg["model"]["text_dim"], cfg["model"]["image_dim"]
    de = cfg["model"].get("evidence_dim", dt)
    table_dim = cfg["model"].get("table_dim")
    if cfg["use_table_evidence"]:
        table_dim = table_dim or dt
        cfg["model"]["table_dim"] = table_dim
    for index in range(cfg["synthetic"]["clients"]):
        client = f"synthetic_{index}"
        labels[client] = {"SUPPORT": 0, "REFUTE": 1, "NEI": 2}
        rng = torch.Generator().manual_seed(cfg["seed"] + index * 100 + (split != "train") * 50000)
        y = torch.arange(count) % 2
        text = torch.randn(count, 1, de, generator=rng) * 0.1
        image = torch.randn(count, 1, di, generator=rng) * 0.1
        text[:, 0, 0] += 1 - 2 * y
        image[:, 0, 0] += 1 - 2 * y
        text_mask = torch.ones(count, 1, dtype=torch.bool)
        image_mask = torch.ones(count, 1, dtype=torch.bool)
        if cfg["synthetic"]["mixed_modalities"]:
            text_mask[::3] = False
            image_mask[1::3] = False
        full = {
            "claim_features": torch.randn(count, dt, generator=rng) * 0.1,
            "text_features": text,
            "image_features": image,
            "text_mask": text_mask,
            "image_mask": image_mask,
            "global_labels": y,
            "original_labels": y.clone(),
            "dataset_ids": [client] * count,
            "record_ids": [f"synthetic:{client}:{split}:{i}" for i in range(count)],
        }
        if table_dim is not None:
            table = torch.randn(count, 1, table_dim, generator=rng) * 0.1
            table[:, 0, 0] += 1 - 2 * y
            table_mask = torch.zeros(count, 1, dtype=torch.bool)
            # The first client has no tables; the other exercises table-only inputs.
            if index % 2:
                table_mask[::3] = True
                if cfg["synthetic"]["mixed_modalities"]:
                    text_mask[::3] = False
                    image_mask[::3] = False
            full["table_features"], full["table_mask"] = table, table_mask
        batches = []
        for start in range(0, count, cfg["training"]["batch_size"]):
            end = start + cfg["training"]["batch_size"]
            batch = {key: value[start:end] for key, value in full.items()}
            batches.append(ablate_batch(batch, cfg["ablation"], cfg["seed"])[0])
        clients[client] = batches
    return clients, labels, {"synthetic_generator": "label-signal-features-v1"}


def _mmcv_splits(claims: list, policy: dict[str, Any]) -> dict[str, list]:
    """Assign whole connected claim/evidence groups, without consulting labels."""
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
    for index, (claim, texts, images, _, _) in enumerate(claims):
        keys = [("claim", " ".join(claim.casefold().split()))]
        keys.extend(("text", " ".join(text.casefold().split())) for text in texts
                    if isinstance(text, str) and text.strip())
        keys.extend(("image", path.replace("\\", "/").casefold()) for path in images)
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
    return {name: sorted(rows, key=lambda row: row[4])
            for name, rows in zip(("train", "dev", "test"), parts, strict=True)}


def _real_batches(
    cfg: dict[str, Any], split: str, cache: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Adapt the supplied tuple loaders; limit records before frozen encoding."""
    specs = cfg.get("datasets", [])
    if cfg["kind"] == "local":
        selected = cfg.get("local_client", specs[0]["id"])
        specs = [spec for spec in specs if spec["id"] == selected]
    if cache is None:
        cache = {}
    if "encoders" not in cache:
        encoders = cfg.get("encoders", {})
        if cfg["model_type"] == "crosstrans_mfc":
            cache["encoders"] = (
                FrozenTokenTextEncoder(**{**encoders["text"], "device": cfg["device"]}),
                FrozenCLIPImageEncoder(**{**encoders["vision"], "device": cfg["device"]}),
            )
            cache["evidence_encoder"] = FrozenTokenTextEncoder(
                **{**encoders["evidence"], "device": cfg["device"]}
            )
        else:
            cache["encoders"] = (
                FrozenTextEncoder(**{**encoders["text"], "device": cfg["device"]}),
                FrozenVisionEncoder(**{**encoders["vision"], "device": cfg["device"]}),
            )
    text_encoder, vision_encoder = cache["encoders"]
    cfg["model"]["text_dim"] = text_encoder.output_dim
    cfg["model"]["image_dim"] = vision_encoder.output_dim
    if cfg["model_type"] == "crosstrans_mfc":
        cfg["model"]["evidence_dim"] = cache["evidence_encoder"].output_dim
    clients, labels, sources = {}, {}, {}
    for spec in specs:
        dataset = spec["id"]
        if dataset not in cache:
            if dataset == "mocheg":
                from fmfc.data.mocheg import get_dataset

                parts = dict(zip(("train", "dev", "test"), get_dataset(spec["path"]), strict=True))
                metadata = {"split_protocol": "supplied train/val/test"}
            else:
                from fmfc.data.mmcv import get_dataset

                if not spec.get("split"):
                    raise ValueError("MMCV requires an explicit grouped split policy in datasets[].split")
                claims, table_claims = get_dataset(
                    spec["path"], subset=spec.get("subset", "all"), mmqa_path=spec.get("mmqa_path")
                )
                parts = _mmcv_splits(claims, spec["split"])
                metadata = {
                    "split_protocol": "grouped claim/text-content/image-path v1",
                    "split_policy": spec["split"],
                    "quarantined_table_claims": len(table_claims),
                    "subset": spec.get("subset", "all"),
                    "mmqa_path": spec.get("mmqa_path"),
                }
            vocabulary = sorted({str(row[3]) for rows in parts.values() for row in rows})
            label_map = {label: index for index, label in enumerate(vocabulary)}
            cache[dataset] = parts, label_map, metadata
        parts, label_map, metadata = cache[dataset]
        rows = sorted(parts[split], key=lambda row: row[4])
        if cfg.get("limit") is not None and split in {"train", "dev"}:
            rows = rows[:cfg["limit"]]
        global_map = {"supported": "SUPPORT", "refuted": "REFUTE",
                      "SUPPORT": "SUPPORT", "REFUTE": "REFUTE"}
        records = []
        for claim, texts, images, label, claim_id in rows:
            record_id = f"{dataset}:{claim_id}"
            evidence = []
            for kind, values in (("text", texts), ("image", images)):
                for index, value in enumerate(values):
                    # Pandas represents missing text as NaN; never encode it as "nan".
                    if isinstance(value, str) and value.strip():
                        content = {"text" if kind == "text" else "image_path": value}
                        evidence.append(EvidenceItem(
                            f"{record_id}:{kind}:{index}", kind, record_id, **content
                        ))
            records.append(VerificationRecord(
                record_id=record_id, dataset_id=dataset, split=split, claim_text=claim,
                evidence=tuple(evidence), original_label=str(label), global_label=global_map.get(label),
            ))
        size = cfg["training"]["batch_size"]
        budgets = {
            "max_text_evidence": cfg.get("max_text_evidence", 4),
            "max_image_evidence": cfg.get("max_image_evidence", 2),
        }
        if cfg["model_type"] == "crosstrans_mfc":
            collator = CrossTransCollator(
                text_encoder, cache["evidence_encoder"], vision_encoder, label_map, **budgets,
            )
            clients[dataset] = [
                collator.defer(records[start:start + size], cfg["ablation"], cfg["seed"])
                for start in range(0, len(records), size)
            ]
        else:
            collator = FeatureCollator(
                text_encoder, vision_encoder, label_map,
                text_encoder.output_dim, vision_encoder.output_dim, **budgets,
            )
            clients[dataset] = [
                ablate_batch(collator(records[start:start + size]), cfg["ablation"], cfg["seed"])[0]
                for start in range(0, len(records), size)
            ]
        labels[dataset] = label_map
        sources[dataset] = {
            "path": str(spec["path"]), **metadata,
            "split_sizes": {name: len(part) for name, part in parts.items()},
            "encoded_samples": len(records), "split": split,
            "encoders": {"text": text_encoder.fingerprint, "vision": vision_encoder.fingerprint},
        }
    return clients, labels, sources


def _data(cfg: dict[str, Any], split: str, real_cache: dict | None = None) -> tuple[dict, dict, dict]:
    data, labels, sources = (
        _synthetic_batches(cfg, split)
        if cfg["run_type"] == "synthetic"
        else _real_batches(cfg, split, real_cache)
    )
    limit = cfg.get("limit")
    if limit is None or split not in {"train", "dev"}:
        return data, labels, sources
    limited = {}
    for client, batches in data.items():
        limited[client] = []
        remaining = limit
        for batch in batches:
            if remaining == 0:
                break
            count = len(batch["global_labels"])
            if count > remaining:
                batch = {
                    key: value[:remaining] if (
                        isinstance(value, torch.Tensor) and value.ndim > 0 and len(value) == count
                        or isinstance(value, (list, tuple)) and len(value) == count
                    ) else value
                    for key, value in batch.items()
                }
            limited[client].append(batch)
            remaining -= min(count, remaining)
    return limited, labels, sources


def _metrics(models: dict[str, Verifier], data: dict, cfg: dict) -> dict[str, Any]:
    reports = {
        client: evaluate(models[client], batches, device=cfg["device"])
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
        "synthetic": cfg["run_type"] == "synthetic",
        "selection_split": "dev",
        "protocol": "evidence-conditioned verification",
        "ablation": cfg["ablation"],
        "primary_cohort": cfg["evaluation"]["primary_cohort"],
        "real_client_count": 0 if cfg["run_type"] == "synthetic" else len(cfg["datasets"]),
        "pseudo_client_count": cfg.get("pseudo_clients", 1),
        "evidence_sufficiency_unknown_after_ablation": cfg["ablation"] not in {"all", "both"},
        "original_head_trained": cfg["training"]["mode"] == "shared_private"
        and cfg["training"]["private_weight"] > 0,
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state.get("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint requires compatible CUDA hardware")
        torch.cuda.set_rng_state_all(state["cuda"])


def run(config: dict[str, Any], kind: str = "federated") -> Path:
    cfg = resolve_config(config, kind)
    cfg["device"] = resolve_device(cfg["device"])
    seed_everything(cfg["seed"])
    torch.set_num_threads(1)
    started = time.perf_counter()
    real_cache: dict = {}
    train, labels, _ = _data(cfg, "train", real_cache)
    dev, dev_labels, _ = _data(cfg, "dev", real_cache)
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
    model_class = CrossTransVerifier if cfg["model_type"] == "crosstrans_mfc" else Verifier
    initial = model_class(**model_args).to(cfg["device"])
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
            local_datasets = {name for batch in train[client] for name in batch["dataset_ids"]}
            for name in list(model.private_heads):
                if name not in local_datasets:
                    del model.private_heads[name]
    optimizer_args = {key: cfg["training"][key] for key in ("learning_rate", "weight_decay")}
    clients = {
        client: ClientState(client, model, make_optimizer(model, **optimizer_args))
        for client, model in models.items()
    }
    if cfg["kind"] in {"local", "centralized"}:
        for client in models:
            models[client] = initial
        clients = {
            "pooled": ClientState("pooled", initial, make_optimizer(initial, **optimizer_args))
        }
    fl = cfg["fl"]
    server = ServerState(
        initial.shared_state(),
        initial.parameter_groups(),
        algorithm=fl["algorithm"],
        weighting=fl["weighting"],
        learning_rate=fl["server_learning_rate"],
        allow_experimental_conflict=fl.get("allow_experimental_conflict", False),
        conflict_beta=fl.get("conflict_beta", 0.5),
    )
    history, step, best, stale = [], 0, -1.0, 0
    if cfg.get("resume_from"):
        resume = torch.load(cfg["resume_from"], map_location="cpu", weights_only=True)
        server.load_state_dict(resume["server"])
        for name, client in clients.items():
            client.load_state_dict(resume["clients"][name])
        step, best, stale, history = (
            resume["step"],
            resume["best"],
            resume["stale"],
            resume["history"],
        )
        _restore_rng(resume["rng"])
    run_id = cfg.get("run_id") or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    folder = Path(cfg["output_root"]) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    target = fl["rounds"] if cfg["kind"] == "federated" else cfg["training"]["epochs"]
    train_kwargs = {
        key: cfg["training"][key] for key in ("mode", "global_weight", "private_weight")
    }
    train_kwargs["device"] = cfg["device"]
    selected_state = None
    while step < target:
        stats = {}
        if cfg["kind"] == "federated":
            updates = []
            for client_id, client in clients.items():
                client.broadcast(server.shared_state())
                batches = random.sample(train[client_id], len(train[client_id]))
                update = client.train(
                    batches,
                    epochs=fl["local_epochs"],
                    **train_kwargs,
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
            pooled = [batch for batches in train.values() for batch in batches]
            stats["pooled"] = train_epoch(
                initial,
                random.sample(pooled, len(pooled)),
                clients["pooled"].optimizer,
                **train_kwargs,
            )
        step += 1
        metrics = _metrics(models, dev, cfg)
        score = metrics["mean_client_binary_macro_f1"]
        if score is None:
            raise ValueError("No binary-eligible dev examples; selection is undefined")
        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        history.append(
            {
                "step": step,
                "mean_client_binary_macro_f1": score,
                "worst_client_binary_macro_f1": metrics["worst_client_binary_macro_f1"],
                "optimizer_steps": json.dumps({k: v["optimizer_steps"] for k, v in stats.items()}),
            }
        )
        if improved:
            # Keep the selected state independent of later training updates.
            selected_state = {
                "step": step,
                "server": server.state_dict(),
                "clients": {key: client.state_dict() for key, client in clients.items()},
            }
        patience = cfg["training"]["patience"]
        if patience and stale >= patience:
            break
    if selected_state is None:
        # A resumed run may not improve on its already selected dev checkpoint.
        old_selected = Path(cfg["resume_from"]).with_name("selected.pt")
        if not old_selected.is_file():
            raise ValueError("Resume requires the matching selected.pt checkpoint")
        selected_state = torch.load(old_selected, map_location="cpu", weights_only=True)
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
    metrics = _metrics(models, dev, cfg)
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
    _json(folder / "timing.json", {"wall_seconds": time.perf_counter() - started})
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
    run_dir: str | Path, split: str = "dev", final: bool = False, ablation: str | None = None
) -> dict:
    folder = Path(run_dir)
    if (not (folder / "config_resolved.yaml").is_file()
            or not (folder / "checkpoints" / "selected.pt").is_file()):
        raise ValueError(
            "Re-evaluation requires config and checkpoint files from an older run; "
            "new runs save results and best model weights only"
        )
    cfg = load_config(folder / "config_resolved.yaml")
    if split not in {"dev", "test"}:
        raise ValueError("Evaluation split must be dev or test")
    if split == "test":
        if ablation:
            raise ValueError("Choose and preregister test ablations before final evaluation")
        frozen = folder / "selection_frozen.json"
        if not final or not frozen.is_file():
            raise ValueError("Test is locked; freeze selection and pass --final")
        if (folder / "metrics_test.json").exists():
            raise ValueError("Final test evaluation already exists")
        if cfg["run_type"] == "synthetic":
            raise ValueError("Synthetic smoke runs have no final benchmark test split")
    elif final:
        raise ValueError("--final applies only to test")
    state = torch.load(
        folder / "checkpoints" / "selected.pt", map_location="cpu", weights_only=True
    )
    data, labels, _ = _data(cfg, split)
    if cfg["kind"] == "local":
        selected = cfg.get("local_client", next(iter(data)))
        data = {selected: data[selected]}
    models = {}
    for name in data:
        payload = state["clients"][name if cfg["kind"] == "federated" else "pooled"]
        local_labels = {
            key: len(value)
            for key, value in labels.items()
            if any(param.startswith(f"private_heads.{key}.") for param in payload["model"])
        }
        model_class = CrossTransVerifier if cfg.get("model_type") == "crosstrans_mfc" else Verifier
        model = model_class(**cfg["model"], private_labels=local_labels)
        model.load_state_dict(payload["model"])
        if cfg["kind"] == "federated":
            model.load_shared_state(state["server"]["shared"])
        models[name] = model
        if ablation:
            data[name] = [
                {**batch, "_ablation": ablation} if "_collator" in batch
                else ablate_batch(batch, ablation, cfg["seed"])[0]
                for batch in data[name]
            ]
    metrics = _metrics(models, data, cfg)
    metrics["evaluation_split"] = split
    metrics["ablation"] = ablation or cfg["ablation"]
    metrics["robustness_only"] = metrics["ablation"] in {"shuffle_image", "drop_image", "drop_table"}
    metrics["evidence_sufficiency_unknown_after_ablation"] = metrics["ablation"] not in {"all", "both"}
    target = f"metrics_{split}.json" if not ablation else f"ablation_{split}_{ablation}.json"
    _json(folder / target, metrics)
    return metrics
