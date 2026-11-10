"""Post-hoc evidence removal explanations for an unchanged CrossTrans model."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from fmfc.data.crosstrans import CrossTransCollator
from fmfc.data.schema import VerificationRecord
from fmfc.environment import resolve_device
from fmfc.model.crosstrans import CrossTransVerifier


@torch.no_grad()
def explain_record(
    model: CrossTransVerifier, collator: CrossTransCollator, record: VerificationRecord,
    *, head: str = "global", top_k: int = 1,
) -> dict[str, Any]:
    """Keep the original predicted class fixed in every probability difference.

    Remove raw evidence and re-encode, rather than masking contextual states
    which can still contain information about the removed text. Freeze evidence
    budgets first, so omitted items never backfill a removed evidence slot.
    """
    if head not in {"global", "original"} or top_k < 1:
        raise ValueError("head must be global/original and top_k must be positive")
    if any(item.kind not in {"text", "image"} for item in record.evidence):
        raise ValueError("CrossTrans explanations support text/image evidence only")
    if len({item.evidence_id for item in record.evidence}) != len(record.evidence):
        raise ValueError("Evidence IDs must be unique within a record")
    selected = []
    counts = {"text": 0, "image": 0}
    budgets = {"text": collator.max_text, "image": collator.max_image}
    for item in record.evidence:
        if budgets[item.kind] is None or counts[item.kind] < budgets[item.kind]:
            selected.append(item)
            counts[item.kind] += 1
    if not selected:
        raise ValueError(f"{record.record_id} has no evidence to explain")
    selected_ids = tuple(item.evidence_id for item in selected)
    labels = (["SUPPORT", "REFUTE"] if head == "global" else
              sorted(collator.labels, key=collator.labels.__getitem__))
    device = next(model.parameters()).device
    cache: dict[tuple[str, ...], list[float]] = {}

    def predict(keep: tuple[str, ...]) -> list[float]:
        if keep not in cache:
            row = replace(record, evidence=tuple(e for e in selected if e.evidence_id in keep))
            batch = collator([row])
            # Labels/metadata are not inputs to the explanation forward pass.
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()
                     if key not in {"global_labels", "original_labels", "truncation"}}
            output = model(batch)
            logits = (output["global_logits"] if head == "global" else
                      output["private_logits"][record.dataset_id])
            if logits.shape != (1, len(labels)) or not torch.isfinite(logits).all():
                raise ValueError("Invalid logits or label vocabulary for explanation")
            cache[keep] = logits[0].softmax(-1).cpu().tolist()
        return cache[keep]

    def truncated(encoder: Any, text: str) -> bool:
        if encoder.max_length is None:  # the collator splits instead of truncating
            return False
        tokens = encoder.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        return len(tokens) > encoder.max_length

    was_training = model.training
    model.eval()
    try:
        base = predict(selected_ids)
        target = max(range(len(base)), key=base.__getitem__)

        def removal(removed: set[str]) -> dict[str, Any]:
            probabilities = predict(tuple(key for key in selected_ids if key not in removed))
            return {
                "target_probability_without": probabilities[target],
                "delta_probability": base[target] - probabilities[target],
                "prediction_without": labels[max(range(len(labels)), key=probabilities.__getitem__)],
            }

        evidence = [
            {**item.to_dict(), **removal({item.evidence_id})} for item in selected
        ]
        evidence.sort(key=lambda item: -item["delta_probability"])
        modality = {
            kind: removal({item.evidence_id for item in selected if item.kind == kind})
            for kind in counts if counts[kind]
        }
        rationale = tuple(item["evidence_id"] for item in evidence
                          if item["delta_probability"] > 0)[:top_k]
        faithfulness: dict[str, Any] = {
            "top_k": top_k, "rationale_ids": list(rationale),
            "status": "computed" if rationale else "no_positive_evidence",
        }
        if rationale:
            keep = tuple(key for key in selected_ids if key in rationale)
            faithfulness.update({
                "comprehensiveness": removal(set(rationale))["delta_probability"],
                "sufficiency": base[target] - predict(keep)[target],
            })
        separator = collator.evidence_encoder.tokenizer.sep_token or "[SEP]"
        content = f" {separator} ".join(e.text for e in selected if e.kind == "text")
        return {
            "record_id": record.record_id, "dataset_id": record.dataset_id,
            "split": record.split, "claim_text": record.claim_text,
            "head": head, "method": "leave_one_evidence_out",
            "perturbation": "remove raw evidence and re-encode; preserve selected-item order",
            "prediction": labels[target], "target_class": labels[target],
            "probabilities": dict(zip(labels, base, strict=True)),
            "evidence": evidence, "modality": modality,
            "no_evidence": removal(set(selected_ids)), "faithfulness": faithfulness,
            "excluded_by_budget": [e.evidence_id for e in record.evidence
                                   if e.evidence_id not in selected_ids],
            "token_truncation": {
                "claim": truncated(collator.claim_encoder, record.claim_text),
                "text_evidence": bool(content) and truncated(collator.evidence_encoder, content),
            },
            "forward_passes": len(cache),
        }
    finally:
        model.train(was_training)


def explain_checkpoint(
    config: dict[str, Any], checkpoint: str | Path, dataset: str, *, split: str = "dev",
    limit: int = 1, record_id: str | None = None, head: str = "global", top_k: int = 1,
) -> dict[str, Any]:
    """Load current best_model.pt weights, using the supplied training config.

    A federated checkpoint selects the requested client's private head. This
    diagnostic deliberately exposes train/dev only, leaving final test locked.
    """
    from fmfc.experiments.runner import _real_batches, resolve_config
    from fmfc.seeds import seed_everything

    if split not in {"train", "dev"} or limit < 1 or top_k < 1:
        raise ValueError("Use train/dev with positive limit and top_k")
    if head not in {"global", "original"}:
        raise ValueError("head must be global or original")
    cfg = resolve_config(config)
    # Removal re-encodes evidence subsets under the same record_id; a cache would
    # return the full-evidence features instead.
    cfg.pop("feature_cache_dir", None)
    if dataset not in {spec["id"] for spec in cfg.get("datasets", [])}:
        raise ValueError(f"Dataset {dataset!r} is not in the supplied config")
    if cfg["ablation"] not in {"all", "both"}:
        raise ValueError("Explain requires a config using all supplied text/image evidence")
    if head == "original" and (cfg["training"]["mode"] != "shared_private"
                               or cfg["training"]["private_weight"] <= 0):
        raise ValueError("Original-label explanations require a trained private head")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    models = payload.get("models", {})
    key = "pooled" if "pooled" in models else dataset
    if key not in models:
        raise ValueError(f"Checkpoint has no pooled or {dataset!r} model")
    state = models[key]
    prefix, suffix = "private_heads.", ".weight"
    private_labels = {name[len(prefix):-len(suffix)]: value.shape[0]
                      for name, value in state.items()
                      if name.startswith(prefix) and name.endswith(suffix)}
    if dataset not in private_labels:
        raise ValueError(f"Checkpoint has no private head for {dataset!r}")
    cfg["device"] = str(resolve_device(cfg["device"]))
    cfg["kind"], cfg["local_client"] = "local", dataset
    cfg["training"]["batch_size"] = 1
    seed_everything(cfg["seed"])
    batches, label_maps, _ = _real_batches(cfg, split)
    if len(label_maps[dataset]) != private_labels[dataset]:
        raise ValueError("Dataset label vocabulary size differs from the checkpoint")
    rows = [(raw["_records"][0], raw["_collator"]) for raw in batches[dataset]
            if not record_id or raw["_records"][0].record_id == record_id]
    if not rows:
        raise ValueError(f"No matching records for {dataset!r} in {split}")
    model = CrossTransVerifier(**cfg["model"], private_labels=private_labels).to(cfg["device"])
    model.load_state_dict(state, strict=True)
    explanations = [explain_record(model, collator, row, head=head, top_k=top_k)
                    for row, collator in rows[:limit]]
    return {
        "protocol": "evidence-conditioned verification",
        "checkpoint": str(checkpoint), "selected_step": payload.get("step"),
        "explanations": explanations,
    }
