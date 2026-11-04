"""FedProx is a local objective regularizer, not an aggregation weighting rule.

Reference: https://proceedings.mlsys.org/paper_files/paper/2020/hash/
1f5fe83998a09396ebe6477d9475ba0c-Abstract.html
"""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn


def proximal_penalty(model: nn.Module, reference: Mapping[str, Tensor], mu: float) -> Tensor:
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("FedProx mu must be finite and nonnegative")
    parameters = dict(model.named_parameters())
    expected = set(model.shared_state())
    if set(reference) != expected:
        raise ValueError("FedProx reference must contain exactly the shared trainable parameters")
    first = next(iter(parameters.values()))
    penalty = torch.zeros((), device=first.device, dtype=first.dtype)
    for name in reference:
        parameter = parameters[name]
        anchor = reference[name].to(device=parameter.device, dtype=parameter.dtype)
        if anchor.shape != parameter.shape:
            raise ValueError("FedProx reference shape mismatch")
        penalty = penalty + (parameter - anchor).square().sum()
    return (mu / 2) * penalty
