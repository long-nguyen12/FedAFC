"""Reddi et al. FedAdam Algorithm 2, without Adam bias correction.

Delta is local minus global, hence the server adds its adaptive step. Initialize
v to tau squared; tau is added outside the square root. Reference:
https://arxiv.org/html/2003.00295v5#S3
"""

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from fmfc.federated.contracts import clone_state


class FedAdam:
    def __init__(
        self,
        state: Mapping[str, Tensor],
        learning_rate: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        if not all(math.isfinite(value) for value in (learning_rate, beta1, beta2, tau)):
            raise ValueError("FedAdam hyperparameters must be finite")
        if learning_rate < 0 or tau <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Invalid FedAdam learning rate, decay or tau")
        self.learning_rate, self.beta1, self.beta2, self.tau = learning_rate, beta1, beta2, tau
        self.m = {name: torch.zeros_like(value, device="cpu") for name, value in state.items()}
        self.v = {
            name: torch.full_like(value, tau**2, device="cpu") for name, value in state.items()
        }
        self.steps = 0

    def step(self, state: Mapping[str, Tensor], delta: Mapping[str, Tensor]) -> dict[str, Tensor]:
        if set(state) != set(self.m) or set(delta) != set(state):
            raise ValueError("FedAdam state/delta names do not match")
        result = {}
        for name, value in state.items():
            update = delta[name].detach().cpu()
            if update.shape != value.shape or update.dtype != value.dtype:
                raise ValueError("FedAdam delta shape/dtype mismatch")
            if not torch.isfinite(update).all():
                raise ValueError("FedAdam delta must be finite")
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * update
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * update.square()
            result[name] = value.detach().cpu() + self.learning_rate * self.m[name] / (
                self.v[name].sqrt() + self.tau
            )
        self.steps += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "m": clone_state(self.m),
            "v": clone_state(self.v),
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "tau": self.tau,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        for name in ("learning_rate", "beta1", "beta2", "tau"):
            if payload[name] != getattr(self, name):
                raise ValueError("FedAdam checkpoint hyperparameters differ")
        for slot in ("m", "v"):
            values = payload[slot]
            if set(values) != set(self.m):
                raise ValueError("FedAdam checkpoint parameter names differ")
            for name, reference in self.m.items():
                if values[name].shape != reference.shape or values[name].dtype != reference.dtype:
                    raise ValueError("FedAdam checkpoint shape/dtype mismatch")
                if not torch.isfinite(values[name]).all():
                    raise ValueError("FedAdam checkpoint moments must be finite")
        if any((value < 0).any() for value in payload["v"].values()):
            raise ValueError("FedAdam second moment cannot be negative")
        steps = payload["steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("FedAdam checkpoint steps must be a nonnegative integer")
        self.m, self.v, self.steps = clone_state(payload["m"]), clone_state(payload["v"]), steps
