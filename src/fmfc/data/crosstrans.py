"""Token-level, batch-wise encoding for the CrossTrans image adapter."""

from typing import Any

import torch

from fmfc.data.schema import VerificationRecord


class CrossTransCollator:
    def __init__(
        self, claim_encoder: Any, evidence_encoder: Any, vision_encoder: Any,
        original_label_mapping: dict[str, int], max_text_evidence: int = 4,
        max_image_evidence: int = 2,
    ) -> None:
        if min(max_text_evidence, max_image_evidence) < 1:
            raise ValueError("Evidence budgets must be positive")
        self.claim_encoder = claim_encoder
        self.evidence_encoder = evidence_encoder
        self.vision_encoder = vision_encoder
        self.labels = original_label_mapping
        self.max_text = max_text_evidence
        self.max_image = max_image_evidence

    def defer(
        self, records: list[VerificationRecord], ablation: str = "all", seed: int = 42,
    ) -> dict[str, Any]:
        """Keep raw records until to_device encodes this batch, never all tokens in RAM."""
        return {
            "_records": records, "_collator": self, "_ablation": ablation, "_seed": seed,
            "global_labels": torch.tensor([
                {"SUPPORT": 0, "REFUTE": 1, None: -100}[r.global_label] for r in records
            ]),
            "dataset_ids": [r.dataset_id for r in records],
            "record_ids": [r.record_id for r in records],
        }

    def __call__(self, records: list[VerificationRecord]) -> dict[str, Any]:
        if not records:
            raise ValueError("Cannot collate an empty batch")
        if any(item.kind == "table" for row in records for item in row.evidence):
            raise ValueError("CrossTrans image adapter does not accept table evidence")
        batch = len(records)
        texts = [[e for e in row.evidence if e.kind == "text"] for row in records]
        images = [[e for e in row.evidence if e.kind == "image"] for row in records]
        claim_features, claim_mask = self.claim_encoder.encode_tokens([
            row.claim_text for row in records
        ])
        text_indices = [i for i, items in enumerate(texts) if items]
        text_features = torch.zeros(batch, 0, self.evidence_encoder.output_dim)
        text_mask = torch.zeros(batch, 0, dtype=torch.bool)
        if text_indices:
            separator = self.evidence_encoder.tokenizer.sep_token or "[SEP]"
            content = [f" {separator} ".join(
                e.text for e in texts[i][:self.max_text]
            ) for i in text_indices]
            features, masks = self.evidence_encoder.encode_tokens(content)
            text_features = torch.zeros(batch, features.shape[1], features.shape[2])
            text_mask = torch.zeros(batch, features.shape[1], dtype=torch.bool)
            text_features[text_indices] = features
            text_mask[text_indices] = masks
        image_slots = min(self.max_image, max(len(items) for items in images))
        image_features = torch.zeros(batch, image_slots, self.vision_encoder.output_dim)
        image_mask = torch.zeros(batch, image_slots, dtype=torch.bool)
        for index, items in enumerate(images):
            selected = items[:self.max_image]
            if selected:
                features = self.vision_encoder.encode_paths([e.image_path for e in selected])
                image_features[index, :len(selected)] = features
                image_mask[index, :len(selected)] = True
        return {
            "claim_features": claim_features, "claim_mask": claim_mask,
            "text_features": text_features, "text_mask": text_mask,
            "image_features": image_features, "image_mask": image_mask,
            "global_labels": torch.tensor([
                {"SUPPORT": 0, "REFUTE": 1, None: -100}[r.global_label] for r in records
            ]),
            "original_labels": torch.tensor([self.labels[r.original_label] for r in records]),
            "dataset_ids": [r.dataset_id for r in records],
            "record_ids": [r.record_id for r in records],
            "truncation": {
                "text_evidence": sum(max(0, len(x) - self.max_text) for x in texts),
                "image_evidence": sum(max(0, len(x) - self.max_image) for x in images),
            },
        }
