"""Optional P2 mechanics; disabled until train-side empirical justification exists."""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from fmfc.federated.contracts import ModelUpdate, clone_state, validate_updates
from fmfc.federated.evidence_aware import group_weights


def conflict_weights(
    updates: Sequence[ModelUpdate],
    parameter_groups: Mapping[str, str],
    group: str,
    beta: float = 0.5,
) -> dict[str, float]:
    """Multiply exposure weights by bounded agreement with their train-update mean."""
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("Conflict beta must be finite and in [0, 1]")
    base = group_weights(updates, group)
    if not base:
        return {}
    names = sorted(name for name, value in parameter_groups.items() if value == group)
    vectors = {
        update.client_id: torch.cat(
            [update.delta[name].detach().cpu().double().reshape(-1) for name in names]
        )
        for update in updates
        if update.client_id in base
    }
    reference = sum(
        (base[key] * value for key, value in vectors.items()),
        torch.zeros_like(next(iter(vectors.values()))),
    )
    norm_reference = float(reference.norm())
    scores = {}
    for client_id, vector in vectors.items():
        denominator = norm_reference * float(vector.norm())
        cosine = float(torch.dot(vector, reference)) / denominator if denominator > 1e-12 else 0.0
        cosine = max(-1.0, min(1.0, cosine))
        multiplier = max(1e-3, 1 + beta * cosine)
        scores[client_id] = base[client_id] * multiplier
    total = sum(scores.values())
    return {client_id: score / total for client_id, score in scores.items()}


def conflict_aware(
    state: Mapping[str, Tensor],
    updates: Sequence[ModelUpdate],
    parameter_groups: Mapping[str, str],
    beta: float = 0.5,
    *,
    enabled: bool = False,
) -> dict[str, Tensor]:
    if not enabled:
        raise ValueError("P2 is experimental and disabled without explicit justification")
    ordered = validate_updates(state, updates, parameter_groups)
    weights = {
        group: conflict_weights(ordered, parameter_groups, group, beta)
        for group in set(parameter_groups.values())
    }
    result = clone_state(state)
    for name in result:
        for update in ordered:
            weight = weights[parameter_groups[name]].get(update.client_id, 0.0)
            if weight:
                result[name].add_(update.delta[name].detach().cpu(), alpha=weight)
    return result
