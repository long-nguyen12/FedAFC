"""Sample-weighted FedAvg, McMahan et al. (2017), plus a uniform baseline.

Reference: https://proceedings.mlr.press/v54/mcmahan17a.html
"""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from fmfc.federated.contracts import ModelUpdate, validate_updates


def average_delta(
    state: Mapping[str, Tensor],
    updates: Sequence[ModelUpdate],
    parameter_groups: Mapping[str, str],
    weighting: str = "sample",
) -> dict[str, Tensor]:
    if weighting not in {"sample", "uniform"}:
        raise ValueError("FedAvg weighting must be sample or uniform")
    ordered = validate_updates(state, updates, parameter_groups)
    active = [update for update in ordered if update.sample_count > 0]
    weights = [update.sample_count if weighting == "sample" else 1 for update in active]
    denominator = sum(weights)
    result = {name: torch.zeros_like(tensor, device="cpu") for name, tensor in state.items()}
    if denominator:
        for update, weight in zip(active, weights, strict=True):
            for name in result:
                result[name].add_(update.delta[name].detach().cpu(), alpha=weight / denominator)
    return result


def fedavg(
    state: Mapping[str, Tensor],
    updates: Sequence[ModelUpdate],
    parameter_groups: Mapping[str, str],
    weighting: str = "sample",
) -> dict[str, Tensor]:
    delta = average_delta(state, updates, parameter_groups, weighting)
    return {name: tensor.detach().cpu() + delta[name] for name, tensor in state.items()}
