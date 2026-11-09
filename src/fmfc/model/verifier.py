"""Verifier over frozen-encoder features; metadata cannot enter this interface."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from fmfc.model.evidence_attention import EvidenceAttention
from fmfc.model.fusion import MaskedFusion


class Verifier(nn.Module):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        hidden_dim: int,
        private_labels: dict[str, int],
        table_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.claim_text = nn.Linear(text_dim, hidden_dim)
        self.text_evidence = nn.ModuleDict(
            {
                "projection": nn.Linear(text_dim, hidden_dim),
                "attention": EvidenceAttention(hidden_dim),
            }
        )
        self.image_evidence = nn.ModuleDict(
            {
                "projection": nn.Linear(image_dim, hidden_dim),
                "attention": EvidenceAttention(hidden_dim),
            }
        )
        if table_dim is not None and table_dim <= 0:
            raise ValueError("Table feature dimension must be positive")
        self.table_dim = table_dim
        self.table_evidence = (
            None
            if table_dim is None
            else nn.ModuleDict(
                {
                    "projection": nn.Linear(table_dim, hidden_dim),
                    "attention": EvidenceAttention(hidden_dim),
                }
            )
        )
        self.fusion = MaskedFusion(hidden_dim, include_table=table_dim is not None)
        self.global_head = nn.Linear(hidden_dim, 2)
        self.private_heads = nn.ModuleDict(
            {
                dataset: nn.Linear(hidden_dim, count)
                for dataset, count in private_labels.items()
            }
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        claim = torch.tanh(self.claim_text(batch["claim_features"]))
        text_mask = batch["text_mask"].bool()
        image_mask = batch["image_mask"].bool()
        text_features = batch["text_features"].masked_fill(~text_mask.unsqueeze(-1), 0)
        image_features = batch["image_features"].masked_fill(
            ~image_mask.unsqueeze(-1), 0
        )
        text = self.text_evidence["attention"](
            claim,
            torch.tanh(self.text_evidence["projection"](text_features)),
            text_mask,
        )
        image = self.image_evidence["attention"](
            claim,
            torch.tanh(self.image_evidence["projection"](image_features)),
            image_mask,
        )
        supplied_table_mask = batch.get("table_mask")
        if self.table_evidence is None:
            fused = self.fusion(claim, text, image, text_mask.any(1), image_mask.any(1))
        else:
            table_mask = (
                supplied_table_mask.bool()
                if supplied_table_mask is not None
                else torch.zeros(
                    claim.shape[0], 0, dtype=torch.bool, device=claim.device
                )
            )
            table_features = (
                batch["table_features"]
                if supplied_table_mask is not None
                else claim.new_zeros(claim.shape[0], 0, self.table_dim)
            )
            table_features = table_features.masked_fill(~table_mask.unsqueeze(-1), 0)
            table = self.table_evidence["attention"](
                claim,
                torch.tanh(self.table_evidence["projection"](table_features)),
                table_mask,
            )
            fused = self.fusion(
                claim,
                text,
                image,
                text_mask.any(1),
                image_mask.any(1),
                table,
                table_mask.any(1),
            )
        indices = {
            dataset: torch.tensor(
                [
                    index
                    for index, value in enumerate(batch["dataset_ids"])
                    if value == dataset
                ],
                dtype=torch.long,
                device=claim.device,
            )
            for dataset in dict.fromkeys(batch["dataset_ids"])
        }
        return {
            "global_logits": self.global_head(fused),
            "private_logits": {
                dataset: self.private_heads[dataset](fused[index])
                for dataset, index in indices.items()
            },
            "private_indices": indices,
        }

    def parameter_groups(self) -> dict[str, str]:
        return {
            name: name.split(".", 1)[0]
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("private_heads.")
        }

    def shared_state(self) -> dict[str, Tensor]:
        parameters = dict(self.named_parameters())
        return {
            name: parameters[name].detach().cpu().clone()
            for name in self.parameter_groups()
        }

    def load_shared_state(self, state: Mapping[str, Tensor]) -> None:
        parameters = dict(self.named_parameters())
        expected = self.parameter_groups()
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(parameters[name].device))
