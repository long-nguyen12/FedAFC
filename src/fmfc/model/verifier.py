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
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(parameters[name].device))
