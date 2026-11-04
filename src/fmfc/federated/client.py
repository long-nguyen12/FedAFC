"""Client-private optimizer/model state is never part of a ModelUpdate."""

import copy
from collections.abc import Iterable, Mapping
from typing import Any

from torch import Tensor, nn
from torch.optim import Optimizer

from fmfc.federated.contracts import ModelUpdate, clone_state, validate_shared


class ClientState:
    def __init__(self, client_id: str, model: nn.Module, optimizer: Optimizer) -> None:
        if not client_id:
            raise ValueError("client_id cannot be empty")
        self.client_id, self.model, self.optimizer = client_id, model, optimizer
        self._anchor: dict[str, Tensor] | None = None

    def broadcast(self, shared: Mapping[str, Tensor]) -> None:
        """Replace only shared parameters; keep this client's private heads and optimizer."""
        validate_shared(shared, self.model.parameter_groups())
        self.model.load_shared_state(shared)
        self._anchor = clone_state(shared)

    def export_update(self, stats: Mapping[str, Any]) -> ModelUpdate:
        if self._anchor is None:
            raise RuntimeError("Broadcast a shared checkpoint before local training")
        shared = self.model.shared_state()
        return ModelUpdate(
            client_id=self.client_id,
            delta={
                name: value.detach().cpu() - self._anchor[name] for name, value in shared.items()
            },
            sample_count=stats["sample_count"],
            group_counts=dict(stats["group_counts"]),
            parameter_groups=self.model.parameter_groups(),
            optimizer_steps=stats["optimizer_steps"],
            metrics={"loss": float(stats["loss"])},
        )

    def train(
        self, batches: Iterable[dict[str, Any]], epochs: int = 1, **train_kwargs: Any
    ) -> ModelUpdate:
        from fmfc.training.local import train_epoch

        if self._anchor is None:
            raise RuntimeError("Broadcast a shared checkpoint before local training")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        local_batches = list(batches)
        train_kwargs.setdefault("prox_reference", self._anchor)
        totals: dict[str, Any] = {
            "loss": 0.0,
            "sample_count": 0,
            "optimizer_steps": 0,
            "group_counts": dict.fromkeys(set(self.model.parameter_groups().values()), 0),
        }
        for _ in range(epochs):
            stats = train_epoch(self.model, local_batches, self.optimizer, **train_kwargs)
            for key in ("sample_count", "optimizer_steps"):
                totals[key] += stats[key]
            totals["loss"] += stats["loss"]
            for group, count in stats["group_counts"].items():
                totals["group_counts"][group] = totals["group_counts"].get(group, 0) + count
        totals["loss"] /= epochs
        return self.export_update(totals)

    def state_dict(self) -> dict[str, Any]:
        """For a client-local checkpoint only; never send this payload to the server."""
        return {
            "client_id": self.client_id,
            "model": clone_state(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "anchor": clone_state(self._anchor) if self._anchor is not None else None,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload["client_id"] != self.client_id:
            raise ValueError("Client checkpoint belongs to a different client")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self._anchor = clone_state(payload["anchor"]) if payload["anchor"] is not None else None
