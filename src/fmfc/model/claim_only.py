"""B3 claim-only control: same claim features, heads and eligible rows; evidence ignored by design.

Trained and evaluated on exactly the rows CrossTrans uses (rows with evidence), so the gap
to CrossTrans measures what the evidence adds. It is a diagnostic baseline, not a path for
claims without evidence.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from fmfc.model.modules.modules import masked_mean
from fmfc.model.verifier import Verifier


class ClaimOnlyVerifier(Verifier):
    def __init__(
        self, text_dim: int, hidden_dim: int, private_labels: dict[str, int],
        cls_dropout: float = 0.4, **_unused: Any,
    ) -> None:
        nn.Module.__init__(self)
        self.claim_text = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.dropout = nn.Dropout(cls_dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.private_heads = nn.ModuleDict({
            dataset: nn.Linear(hidden_dim, count) for dataset, count in private_labels.items()
        })

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        mask = batch.get("claim_mask")
        claim: Tensor = masked_mean(
            batch["claim_features"], None if mask is None else mask.bool(), dim=1,
        )
        fused = self.dropout(self.claim_text(claim))
        indices = {
            dataset: torch.tensor(
                [i for i, value in enumerate(batch["dataset_ids"]) if value == dataset],
                dtype=torch.long, device=fused.device,
            )
            for dataset in dict.fromkeys(batch["dataset_ids"])
        }
        return {
            "global_logits": self.classifier(fused),
            "private_logits": {d: self.private_heads[d](fused[i]) for d, i in indices.items()},
            "private_indices": indices,
        }

    def parameter_groups(self, include_local: bool = False) -> dict[str, str]:
        """No client-local parameters here; ``include_local`` only matches CrossTrans."""
        meanings = {"claim_text": "claim_text", "classifier": "global_head"}
        return {
            name: meanings[name.split(".", 1)[0]]
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("private_heads.")
        }
