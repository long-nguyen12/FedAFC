"""Shared-parameter state for verifiers over frozen-encoder features."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class Verifier(nn.Module):
    """Base class: subclasses define ``forward`` and ``parameter_groups``."""

    def shared_state(self) -> dict[str, Tensor]:
        parameters = dict(self.named_parameters())
        return {
            name: parameters[name].detach().cpu().clone()
            for name in self.parameter_groups()
        }

    def load_shared_state(self, state: Mapping[str, Tensor]) -> None:
        """A2: refuse mismatched names, shapes or dtypes instead of broadcasting silently."""
        parameters = dict(self.named_parameters())
        if set(state) != set(self.parameter_groups()):
            raise ValueError("Shared parameter names differ from this model's shared parameters")
        for name, value in state.items():
            if value.shape != parameters[name].shape or value.dtype != parameters[name].dtype:
                raise ValueError(f"Shared parameter shape/dtype mismatch: {name}")
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(parameters[name].device))
