"""Only compatible shared deltas and optimization exposure counts cross this boundary."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

GROUPS = frozenset({
    "claim_text", "text_evidence", "image_evidence", "table_evidence",
    "joint_fusion", "fusion", "global_head",
})


@dataclass(frozen=True)
class ModelUpdate:
    client_id: str
    delta: dict[str, Tensor]
    sample_count: int
    group_counts: dict[str, int]
    parameter_groups: dict[str, str]
    optimizer_steps: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


def clone_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def validate_shared(state: Mapping[str, Tensor], parameter_groups: Mapping[str, str]) -> None:
    if not state or set(state) != set(parameter_groups):
        raise ValueError("Shared parameter names must match an exhaustive group partition")
    for name, tensor in state.items():
        if "private" in name.lower() or "frozen_encoder" in name.lower():
            raise ValueError(f"Private heads/frozen encoders cannot be shared: {name}")
        if parameter_groups[name] not in GROUPS:
            raise ValueError(f"Unknown shared parameter meaning for {name}")
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
            raise ValueError(f"Shared parameter must be a floating tensor: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite shared parameter: {name}")


def _count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def validate_updates(
    state: Mapping[str, Tensor],
    updates: Sequence[ModelUpdate],
    parameter_groups: Mapping[str, str],
) -> list[ModelUpdate]:
    validate_shared(state, parameter_groups)
    seen: set[str] = set()
    for update in updates:
        seen.add(update.client_id)
        _count(update.sample_count, "sample_count")
        _count(update.optimizer_steps, "optimizer_steps")
        required_groups = set(parameter_groups.values())
        for group, count in update.group_counts.items():
            _count(count, f"group_counts[{group}]")
            
        for name, reference in state.items():
            delta = update.delta[name]
    return sorted(updates, key=lambda update: update.client_id)
