"""Local optimization with counts measured on the actual eligible batches."""

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from fmfc.model.losses import compute_loss
from fmfc.model.verifier import Verifier

PRESENCE = ("text", "image", "table")
PriorKey = tuple[str, tuple[bool, ...]]


def presence_log_priors(batches: Iterable[Mapping[str, Any]]) -> dict[PriorKey, dict[str, Tensor]]:
    """Laplace-smoothed log p_train(label | dataset, evidence-presence pattern).

    Reads the raw records of deferred batches, so nothing is encoded. In FL each
    client computes this from its own records; it never reaches the server.
    Records without any evidence are never trained, so they are not counted.
    """
    counts: dict[PriorKey, dict[str, Tensor]] = {}
    for batch in batches:
        if "_records" not in batch:
            raise ValueError("Logit-adjustment priors need deferred batches with raw records")
        labels = batch["_collator"].labels
        for record, global_label in zip(batch["_records"], batch["global_labels"].tolist(),
                                        strict=True):
            pattern = tuple(any(item.kind == kind for item in record.evidence) for kind in PRESENCE)
            if not any(pattern):
                continue
            entry = counts.setdefault((record.dataset_id, pattern), {
                "global": torch.zeros(2), "original": torch.zeros(len(labels)),
            })
            if global_label != -100:
                entry["global"][global_label] += 1
            entry["original"][labels[record.original_label]] += 1
    return {
        key: {head: ((value + 1) / (value.sum() + len(value))).log() for head, value in entry.items()}
        for key, entry in counts.items()
    }


def _adjust_logits(
    outputs: Mapping[str, Any], batch: Mapping[str, Any],
    log_priors: Mapping[PriorKey, Mapping[str, Tensor]], tau: float,
) -> dict[str, Any]:
    """Logit-adjusted loss (Menon et al., ICLR 2021) per (dataset, presence pattern).

    Only the training loss sees tau * log prior; evaluation keeps raw logits, so a
    label prior tied to which modalities are present is not learned as evidence.
    """
    rows = len(batch["dataset_ids"])
    flags = [
        batch[f"{kind}_mask"].any(1).cpu().tolist() if f"{kind}_mask" in batch else [False] * rows
        for kind in PRESENCE
    ]
    keys = list(zip(batch["dataset_ids"], zip(*flags), strict=True))
    missing = sorted({key for key in keys if key not in log_priors})
    if missing:
        raise ValueError(f"No training label prior for {missing}")
    logits = outputs["global_logits"]
    shift = torch.stack([log_priors[key]["global"] for key in keys]).to(logits)
    private = {
        dataset: values + tau * torch.stack([
            log_priors[keys[index]]["original"]
            for index in outputs["private_indices"][dataset].tolist()
        ]).to(values)
        for dataset, values in outputs["private_logits"].items()
    }
    adjusted = {**outputs, "global_logits": logits + tau * shift, "private_logits": private}
    if "image_logits" in outputs:  # C6 head: same prior, so it cannot learn "image => SUPPORT"
        adjusted["image_logits"] = outputs["image_logits"] + tau * shift.to(outputs["image_logits"])
    return adjusted


def to_device(batch: Mapping[str, Any], device: str | torch.device) -> dict[str, Any]:
    if "_collator" in batch:
        from fmfc.experiments.ablations import ablate_batch

        batch = ablate_batch(
            batch["_collator"](batch["_records"]), batch["_ablation"], batch["_seed"],
        )[0]
    return {key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in batch.items()}


def train_epoch(
    model: Verifier, batches: Iterable[Mapping[str, Any]], optimizer: torch.optim.Optimizer, *,
    device: str | torch.device = "cpu", mode: str = "binary_only", global_weight: float = 1.0,
    private_weight: float = 1.0, prox_mu: float = 0.0,
    prox_reference: Mapping[str, Tensor] | None = None,
    progress_desc: str = "Train", max_grad_norm: float = 0.0, warmup_steps: int = 0,
    logit_adjustment: float = 0.0,
    log_priors: Mapping[PriorKey, Mapping[str, Tensor]] | None = None,
    image_aux_weight: float = 0.0,
) -> dict[str, Any]:
    if mode not in {"binary_only", "shared_private"}:
        raise ValueError("mode must be binary_only or shared_private")
    if global_weight < 0 or private_weight < 0:
        raise ValueError("Loss weights must be nonnegative")
    if logit_adjustment < 0:
        raise ValueError("logit_adjustment must be nonnegative")
    if logit_adjustment and log_priors is None:
        raise ValueError("Logit adjustment requires training label priors")
    if max_grad_norm < 0:
        raise ValueError("max_grad_norm must be nonnegative")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if prox_mu < 0:
        raise ValueError("prox_mu must be nonnegative")
    if image_aux_weight < 0:
        raise ValueError("image_aux_weight must be nonnegative")
    if prox_mu and prox_reference is None:
        raise ValueError("FedProx requires a shared reference checkpoint")
    if prox_reference is not None and set(prox_reference) != set(model.parameter_groups()):
        raise ValueError("Proximal reference must contain exactly the shared parameters")
    model.to(device).train()
    parameter_groups = model.parameter_groups()
    counts = {group: 0 for group in set(parameter_groups.values())}
    steps = samples = binary_count = 0
    loss_sum = 0.0
    progress = tqdm(batches, desc=progress_desc, unit="batch", dynamic_ncols=True)
    for raw_batch in progress:
        batch = to_device(raw_batch, device)
        text_present = batch["text_mask"].any(1)
        image_present = batch["image_mask"].any(1)
        usable = text_present | image_present
        table_present = batch.get("table_mask")
        table_present = (table_present.any(1) if table_present is not None
                         else torch.zeros_like(text_present))
        usable |= table_present
        binary = (batch["global_labels"] != -100) & usable & (global_weight > 0)
        private = ((batch["original_labels"] != -100) & usable
                   & (mode == "shared_private") & (private_weight > 0))
        active = binary | private
        if not active.any():
            continue
        batch_counts = {
            "claim_text": int(active.sum()),
            "text_evidence": int((active & text_present).sum()),
            "image_evidence": int((active & image_present).sum()),
            "table_evidence": int((active & table_present).sum()),
            "joint_fusion": int((active & text_present & image_present).sum()),
            "fusion": int(active.sum()),
            "global_head": int(binary.sum()),
        }
        # Exclude unsupervised rows before forward: inactive evidence branches
        # must not acquire zero-valued gradients and drift under weight decay.
        active_indices = active.nonzero(as_tuple=True)[0].cpu().tolist()
        batch = {
            key: value[active] if isinstance(value, Tensor)
            else [value[index] for index in active_indices]
            if key in {"dataset_ids", "record_ids"} else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        if logit_adjustment:
            outputs = _adjust_logits(outputs, batch, log_priors, logit_adjustment)
        objective = compute_loss(
            outputs, batch, mode=mode, global_weight=global_weight, private_weight=private_weight,
            image_aux_weight=image_aux_weight,
        )["loss"]
        if prox_mu:
            penalty = sum(
                (parameter - prox_reference[name].to(device)).square().sum()
                for name, parameter in model.named_parameters()
                if name in parameter_groups and batch_counts[parameter_groups[name]] > 0
            )
            objective = objective + (prox_mu / 2) * penalty
        if not torch.isfinite(objective):
            raise ValueError("Non-finite training objective")
        objective.backward()
        for name, parameter in model.named_parameters():
            if name in parameter_groups and batch_counts[parameter_groups[name]] == 0:
                parameter.grad = None
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.grad is not None], max_grad_norm
            )
        if warmup_steps > 0:
            # Counters live in param_groups, so they persist in optimizer.state_dict()
            # across FL rounds and best-checkpoint restores.
            for group in optimizer.param_groups:
                group.setdefault("base_lr", group["lr"])
                group["warmup_step"] = group.get("warmup_step", 0) + 1
                group["lr"] = group["base_lr"] * min(1.0, group["warmup_step"] / warmup_steps)
        optimizer.step()
        steps += 1
        samples += int(active.sum())
        binary_count += int(binary.sum())
        for group in counts:
            counts[group] += batch_counts[group]
        loss_sum += float(objective.detach())
        progress.set_postfix(loss=f"{loss_sum / steps:.4f}", refresh=False)
    return {"loss": loss_sum / max(steps, 1), "optimizer_steps": steps, "sample_count": samples,
            "binary_eligible_count": binary_count, "group_counts": counts}


def make_optimizer(model: nn.Module, learning_rate: float = 1e-4, weight_decay: float = 0.0):
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, weight_decay=weight_decay,
    )
