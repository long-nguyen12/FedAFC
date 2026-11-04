"""Local optimization with counts measured on the actual eligible batches."""

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from fmfc.model.losses import compute_loss
from fmfc.model.verifier import Verifier


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
) -> dict[str, Any]:
    if mode not in {"binary_only", "shared_private"}:
        raise ValueError("mode must be binary_only or shared_private")
    if global_weight < 0 or private_weight < 0:
        raise ValueError("Loss weights must be nonnegative")
    if prox_mu < 0:
        raise ValueError("prox_mu must be nonnegative")
    if prox_mu and prox_reference is None:
        raise ValueError("FedProx requires a shared reference checkpoint")
    if prox_reference is not None and set(prox_reference) != set(model.parameter_groups()):
        raise ValueError("Proximal reference must contain exactly the shared parameters")
    model.to(device).train()
    parameter_groups = model.parameter_groups()
    counts = {group: 0 for group in set(parameter_groups.values())}
    steps = samples = binary_count = 0
    loss_sum = 0.0
    for raw_batch in batches:
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
        objective = compute_loss(
            outputs, batch, mode=mode, global_weight=global_weight, private_weight=private_weight
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
        optimizer.step()
        steps += 1
        samples += int(active.sum())
        binary_count += int(binary.sum())
        for group in counts:
            counts[group] += batch_counts[group]
        loss_sum += float(objective.detach())
    return {"loss": loss_sum / max(steps, 1), "optimizer_steps": steps, "sample_count": samples,
            "binary_eligible_count": binary_count, "group_counts": counts}


def make_optimizer(model: nn.Module, learning_rate: float = 1e-4, weight_decay: float = 0.0):
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, weight_decay=weight_decay,
    )
