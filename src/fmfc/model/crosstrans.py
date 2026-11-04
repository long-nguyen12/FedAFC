"""Image-evidence adapter around unchanged CrossTrans-MFC computation blocks."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from fmfc.model._crosstrans_mfc.modules import (
    MultimodalFusionModule,
    MultiHeadGatedFusion,
    TemporalPositionalEncoding,
)
from fmfc.model.verifier import Verifier


class CrossTransVerifier(Verifier):
    """CrossTransVFC backbone with FedFC heads and shared-parameter contracts.

    Frozen token/image encoders live in the collator, outside client copies.
    The upstream fusion blocks and their parameters are used without edits.
    """

    def __init__(
        self, text_dim: int, image_dim: int, hidden_dim: int,
        private_labels: dict[str, int], evidence_dim: int | None = None,
        num_heads: int = 8, dropout: float = 0.1, cls_dropout: float = 0.4,
        table_dim: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if table_dim is not None:
            raise ValueError("CrossTrans image adapter does not accept table features")
        if (num_heads < 1 or hidden_dim % num_heads or hidden_dim % 4
                or image_dim % 2):
            raise ValueError("CrossTrans requires hidden_dim divisible by num_heads and 4, and even image_dim")
        self.hidden_dim = hidden_dim
        self.table_dim = None
        self.temporal_pe_vision = TemporalPositionalEncoding(d_model=image_dim)
        self.claim_video_trans = MultimodalFusionModule(
            text_in_dim=text_dim, img_in_dim=image_dim, d_model=hidden_dim,
            n_heads=num_heads, out_dim=hidden_dim, dropout=dropout, bidirectional=True,
        )
        self.claim_evidence_trans = MultimodalFusionModule(
            text_in_dim=text_dim, img_in_dim=evidence_dim or text_dim, d_model=hidden_dim,
            n_heads=num_heads, out_dim=hidden_dim, dropout=dropout, bidirectional=False,
        )
        self.fusion_mlp = MultiHeadGatedFusion(
            dim1=hidden_dim, dim2=hidden_dim, out_dim=hidden_dim, dropout=dropout,
        )
        self.dropout = nn.Dropout(cls_dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.private_heads = nn.ModuleDict({
            dataset: nn.Linear(hidden_dim, count) for dataset, count in private_labels.items()
        })

    def _branch(
        self, module: nn.Module, claim: Tensor, claim_mask: Tensor,
        evidence: Tensor, evidence_mask: Tensor, visual: bool = False,
    ) -> Tensor:
        # Do not pass an all-masked row into upstream cross-attention. Absent
        # evidence also must not create gradients for an inactive FL branch.
        indices = evidence_mask.any(1).nonzero(as_tuple=True)[0]
        result = claim.new_zeros(claim.shape[0], self.hidden_dim)
        if indices.numel() == 0:
            return result
        tokens = evidence[indices].masked_fill(~evidence_mask[indices].unsqueeze(-1), 0)
        if visual:
            tokens = self.temporal_pe_vision(tokens)
        fused = module(
            text_tokens=claim[indices], img_tokens=tokens,
            text_mask=claim_mask[indices], img_mask=evidence_mask[indices],
        )
        return result.index_copy(0, indices, fused)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        if "table_mask" in batch and batch["table_mask"].any():
            raise ValueError("CrossTrans image adapter does not accept table evidence")
        claim = batch["claim_features"]
        if claim.ndim == 2:  # Synthetic feature fixtures have one claim token.
            claim = claim.unsqueeze(1)
        claim_mask = batch.get("claim_mask")
        if claim_mask is None:
            claim_mask = torch.ones(claim.shape[:2], dtype=torch.bool, device=claim.device)
        claim = claim.masked_fill(~claim_mask.unsqueeze(-1), 0)
        visual = self._branch(
            self.claim_video_trans, claim, claim_mask,
            batch["image_features"], batch["image_mask"].bool(), visual=True,
        )
        evidence = self._branch(
            self.claim_evidence_trans, claim, claim_mask,
            batch["text_features"], batch["text_mask"].bool(),
        )
        fused = self.dropout(self.fusion_mlp(visual, evidence))
        indices = {
            dataset: torch.tensor(
                [i for i, value in enumerate(batch["dataset_ids"]) if value == dataset],
                dtype=torch.long, device=claim.device,
            )
            for dataset in dict.fromkeys(batch["dataset_ids"])
        }
        unknown = set(indices) - set(self.private_heads)
        if unknown:
            raise ValueError(f"No original-label private head configured for {sorted(unknown)}")
        return {
            "global_logits": self.classifier(fused),
            "private_logits": {
                dataset: self.private_heads[dataset](fused[index])
                for dataset, index in indices.items()
            },
            "private_indices": indices,
        }

    def parameter_groups(self) -> dict[str, str]:
        meanings = {
            "claim_video_trans": "image_evidence",
            "claim_evidence_trans": "text_evidence",
            "fusion_mlp": "fusion", "classifier": "global_head",
        }
        return {
            name: meanings[name.split(".", 1)[0]]
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("private_heads.")
        }
