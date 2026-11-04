"""Synchronous server with resumable shared weights and adaptive moments."""

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor

from fmfc.federated.communication import tensor_bytes
from fmfc.federated.conflict_aware import conflict_aware
from fmfc.federated.contracts import ModelUpdate, clone_state, validate_shared, validate_updates
from fmfc.federated.evidence_aware import evidence_aware, exposure_degenerates_to_fedavg
from fmfc.federated.fedadam import FedAdam
from fmfc.federated.fedavg import average_delta, fedavg


class ServerState:
    def __init__(
        self,
        shared: Mapping[str, Tensor],
        parameter_groups: Mapping[str, str],
        algorithm: str = "fedavg",
        weighting: str = "sample",
        learning_rate: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        allow_experimental_conflict: bool = False,
        conflict_beta: float = 0.5,
    ) -> None:
        validate_shared(shared, parameter_groups)
        if algorithm not in {"fedavg", "fedprox", "fedadam", "evidence_aware", "conflict_aware"}:
            raise ValueError("Unknown federated algorithm")
        if weighting not in {"sample", "uniform", "exposure", "uniform_active"}:
            raise ValueError("Unknown aggregation weighting")
        if algorithm in {"fedavg", "fedprox", "fedadam"} and weighting not in {"sample", "uniform"}:
            raise ValueError("Baseline weighting must be sample or uniform")
        if algorithm == "conflict_aware" and not allow_experimental_conflict:
            raise ValueError("P2 is disabled until explicitly justified")
        self.shared = clone_state(shared)
        self.parameter_groups = dict(parameter_groups)
        self.algorithm, self.weighting = algorithm, weighting
        self.allow_experimental_conflict = allow_experimental_conflict
        self.conflict_beta = conflict_beta
        self.adam = FedAdam(shared, learning_rate, beta1, beta2, tau)
        self.round = 0
        self.communication = {"uplink_tensor_bytes": 0, "downlink_tensor_bytes": 0}
        self.history: list[dict[str, Any]] = []

    def shared_state(self) -> dict[str, Tensor]:
        return clone_state(self.shared)

    def aggregate(self, updates: Sequence[ModelUpdate]) -> dict[str, Tensor]:
        ordered = validate_updates(self.shared, updates, self.parameter_groups)
        active = any(update.sample_count > 0 for update in ordered)
        if self.algorithm in {"fedavg", "fedprox"}:
            self.shared = fedavg(self.shared, ordered, self.parameter_groups, self.weighting)
        elif self.algorithm == "fedadam" and active:
            delta = average_delta(self.shared, ordered, self.parameter_groups, self.weighting)
            self.shared = self.adam.step(self.shared, delta)
        elif self.algorithm == "evidence_aware":
            weighting = "uniform_active" if self.weighting == "uniform_active" else "exposure"
            self.shared = evidence_aware(self.shared, ordered, self.parameter_groups, weighting)
        elif self.algorithm == "conflict_aware":
            self.shared = conflict_aware(
                self.shared,
                ordered,
                self.parameter_groups,
                self.conflict_beta,
                enabled=self.allow_experimental_conflict,
            )
        self.round += 1
        uplink = sum(tensor_bytes(update.delta) for update in ordered)
        downlink = len(ordered) * tensor_bytes(self.shared)
        self.communication["uplink_tensor_bytes"] += uplink
        self.communication["downlink_tensor_bytes"] += downlink
        self.history.append(
            {
                "round": self.round,
                "client_sample_counts": {
                    update.client_id: update.sample_count for update in ordered
                },
                "client_optimizer_steps": {
                    update.client_id: update.optimizer_steps for update in ordered
                },
                "client_group_counts": {
                    update.client_id: dict(update.group_counts) for update in ordered
                },
                "p1_equals_fedavg_by_counts": exposure_degenerates_to_fedavg(ordered),
                "uplink_tensor_bytes": uplink,
                "downlink_tensor_bytes": downlink,
            }
        )
        return self.shared_state()

    def state_dict(self) -> dict[str, Any]:
        return {
            "shared": self.shared_state(),
            "parameter_groups": dict(self.parameter_groups),
            "algorithm": self.algorithm,
            "weighting": self.weighting,
            "round": self.round,
            "adam": self.adam.state_dict(),
            "communication": dict(self.communication),
            "history": copy.deepcopy(self.history),
            "conflict_beta": self.conflict_beta,
            "allow_experimental_conflict": self.allow_experimental_conflict,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        for key in (
            "parameter_groups",
            "algorithm",
            "weighting",
            "conflict_beta",
            "allow_experimental_conflict",
        ):
            if payload[key] != getattr(self, key):
                raise ValueError(f"Server checkpoint contract mismatch: {key}")
        validate_shared(payload["shared"], self.parameter_groups)
        for name, reference in self.shared.items():
            candidate = payload["shared"][name]
            if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
                raise ValueError("Server checkpoint shared shape/dtype mismatch")
        if isinstance(payload["round"], bool) or not isinstance(payload["round"], int):
            raise ValueError("Server checkpoint round must be a nonnegative integer")
        if payload["round"] < 0:
            raise ValueError("Server checkpoint round cannot be negative")
        self.adam.load_state_dict(payload["adam"])
        self.shared, self.round = clone_state(payload["shared"]), payload["round"]
        self.communication = dict(payload["communication"])
        self.history = copy.deepcopy(payload["history"])
