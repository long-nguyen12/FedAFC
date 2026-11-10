"""Text/image/table evidence adapter around unchanged CrossTrans-MFC computation blocks."""

import re
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from fmfc.model.modules.modules import (
    MultimodalFusionModule,
    MultiHeadGatedFusion,
)
from fmfc.model.verifier import Verifier


class CrossTransVerifier(Verifier):
    def __init__(
        self, text_dim: int, image_dim: int, hidden_dim: int,
        private_labels: dict[str, int], evidence_dim: int | None = None,
        num_heads: int = 8, dropout: float = 0.1, cls_dropout: float = 0.4,
        table_dim: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if table_dim is not None and table_dim <= 0:
            raise ValueError("Table feature dimension must be positive")
        self.hidden_dim = hidden_dim
        self.table_dim = table_dim
        self.claim_video_trans = MultimodalFusionModule(
            text_in_dim=text_dim, img_in_dim=image_dim, d_model=hidden_dim,
            n_heads=num_heads, out_dim=hidden_dim, dropout=dropout, bidirectional=True,
        )
        self.claim_evidence_trans = MultimodalFusionModule(
            text_in_dim=text_dim, img_in_dim=evidence_dim or text_dim, d_model=hidden_dim,
            n_heads=num_heads, out_dim=hidden_dim, dropout=dropout, bidirectional=False,
        )
        # Upstream reserves `gate` but never calls it. Deleting after construction
        # keeps the RNG stream, hence every other initialization, unchanged.
        del self.claim_video_trans.gate
        del self.claim_evidence_trans.gate
        self.fusion_mlp = MultiHeadGatedFusion(
            dim1=hidden_dim, dim2=hidden_dim, out_dim=hidden_dim, dropout=dropout,
        )
        self.dropout = nn.Dropout(cls_dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.private_heads = nn.ModuleDict({
            dataset: nn.Linear(hidden_dim, count) for dataset, count in private_labels.items()
        })
        if table_dim is not None:
            # Built last, so every text/image module keeps its table-free initialization.
            # Tables reuse the unidirectional claim<-evidence block and the 2-input gate.
            self.claim_table_trans = MultimodalFusionModule(
                text_in_dim=text_dim, img_in_dim=table_dim, d_model=hidden_dim,
                n_heads=num_heads, out_dim=hidden_dim, dropout=dropout, bidirectional=False,
            )
            del self.claim_table_trans.gate
            self.table_fusion = MultiHeadGatedFusion(
                dim1=hidden_dim, dim2=hidden_dim, out_dim=hidden_dim, dropout=dropout,
            )

    _LEGACY_KEYS = re.compile(r"^claim_(video|evidence)_trans\.gate\.")

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False,
    ):
        """Accept checkpoints saved before the unused upstream gates were removed."""
        state_dict = {k: v for k, v in state_dict.items() if not self._LEGACY_KEYS.match(k)}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _branch(
        self, module: nn.Module, claim: Tensor, claim_mask: Tensor,
        evidence: Tensor, evidence_mask: Tensor, visual: bool = False,
    ) -> Tensor:
        indices = evidence_mask.any(1).nonzero(as_tuple=True)[0]
        result = claim.new_zeros(claim.shape[0], self.hidden_dim)
        if indices.numel() == 0:
            return result
        tokens = evidence[indices].masked_fill(~evidence_mask[indices].unsqueeze(-1), 0)
        fused = module(
            text_tokens=claim[indices], img_tokens=tokens,
            text_mask=claim_mask[indices], img_mask=evidence_mask[indices],
        )
        return result.index_copy(0, indices, fused)

    def _fuse_available(
        self, visual: Tensor, evidence: Tensor,
        image_present: Tensor, text_present: Tensor,
    ) -> Tensor:
        """Route missing modalities around biased projections and joint gates."""
        fusion = self.fusion_mlp
        result = visual.new_zeros(visual.shape[0], self.hidden_dim)
        both = (image_present & text_present).nonzero(as_tuple=True)[0]
        if both.numel():
            result = result.index_copy(0, both, fusion(visual[both], evidence[both]))
        for present, features, projections in (
            (image_present & ~text_present, visual, fusion.proj1),
            (text_present & ~image_present, evidence, fusion.proj2),
        ):
            indices = present.nonzero(as_tuple=True)[0]
            if indices.numel():
                heads = torch.cat(
                    [projection(features[indices]) for projection in projections], dim=-1,
                )
                fused = fusion.norm(fusion.dropout(heads))
                result = result.index_copy(0, indices, fused)
        # Rows with no evidence also bypass LayerNorm's learned bias.
        return result

    def _add_table(
        self, fused: Tensor, table: Tensor, table_present: Tensor, other_present: Tensor,
    ) -> Tensor:
        """Gate tables into the text/image result; rows without a table are unchanged."""
        fusion = self.table_fusion
        both = (table_present & other_present).nonzero(as_tuple=True)[0]
        only = (table_present & ~other_present).nonzero(as_tuple=True)[0]
        result = fused
        if both.numel():
            result = result.index_copy(0, both, fusion(fused[both], table[both]))
        if only.numel():
            heads = torch.cat([projection(table[only]) for projection in fusion.proj2], dim=-1)
            result = result.index_copy(0, only, fusion.norm(fusion.dropout(heads)))
        return result

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        table_mask = batch.get("table_mask")
        if self.table_dim is None and table_mask is not None and table_mask.any():
            raise ValueError("Table evidence requires CrossTrans built with table_dim")
        claim = batch["claim_features"]
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
        image_present = batch["image_mask"].bool().any(1)
        text_present = batch["text_mask"].bool().any(1)
        fused = self._fuse_available(visual, evidence, image_present, text_present)
        if self.table_dim is not None and table_mask is not None:
            table_mask = table_mask.bool()
            table = self._branch(
                self.claim_table_trans, claim, claim_mask, batch["table_features"], table_mask,
            )
            fused = self._add_table(fused, table, table_mask.any(1), image_present | text_present)
        fused = self.dropout(fused)
        indices = {
            dataset: torch.tensor(
                [i for i, value in enumerate(batch["dataset_ids"]) if value == dataset],
                dtype=torch.long, device=claim.device,
            )
            for dataset in dict.fromkeys(batch["dataset_ids"])
        }
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
            "fusion_mlp.proj1": "image_evidence",
            "fusion_mlp.proj2": "text_evidence",
            "fusion_mlp.gates": "joint_fusion",
            "fusion_mlp.norm": "fusion",
            "classifier": "global_head",
            "claim_table_trans": "table_evidence",
            "table_fusion": "table_evidence",
        }
        groups = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or name.startswith("private_heads."):
                continue
            prefix = name.split(".", 1)[0]
            if prefix == "fusion_mlp":
                prefix = ".".join(name.split(".")[:2])
            groups[name] = meanings[prefix]
        return groups
