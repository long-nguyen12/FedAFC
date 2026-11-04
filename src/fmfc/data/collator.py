"""Frozen feature batches: only claim/evidence content enters encoders."""

from typing import Any

from fmfc.data.schema import VerificationRecord, serialize_table


class FeatureCollator:
    def __init__(
        self,
        text_encoder: Any,
        vision_encoder: Any,
        original_label_mapping: dict[str, int],
        text_dim: int,
        image_dim: int,
        max_text_evidence: int = 4,
        max_image_evidence: int = 2,
        table_encoder: Any = None,
        table_dim: int | None = None,
        max_table_evidence: int = 2,
    ) -> None:
        if min(text_dim, image_dim, max_text_evidence, max_image_evidence) <= 0:
            raise ValueError("Feature dimensions and evidence budgets must be positive")
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.labels = original_label_mapping
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.max_text = max_text_evidence
        self.max_image = max_image_evidence
        if table_dim is not None and min(table_dim, max_table_evidence) <= 0:
            raise ValueError("Table dimension and evidence budget must be positive")
        if table_encoder is not None and table_dim is None:
            raise ValueError("table_dim is required when a table encoder is provided")
        self.table_dim = table_dim
        self.table_encoder = table_encoder if table_encoder is not None else text_encoder
        self.max_table = max_table_evidence

    def __call__(self, records: list[VerificationRecord]) -> dict[str, Any]:
        import torch

        if not records:
            raise ValueError("Cannot collate an empty batch")
        batch = len(records)
        texts = [[item for item in record.evidence if item.kind == "text"] for record in records]
        images = [[item for item in record.evidence if item.kind == "image"] for record in records]
        tables = [[item for item in record.evidence if item.kind == "table"] for record in records]
        if self.table_dim is None and any(tables):
            raise ValueError("Table evidence requires an enabled table feature branch")
        text_slots = min(self.max_text, max(len(items) for items in texts))
        image_slots = min(self.max_image, max(len(items) for items in images))
        text_features = torch.zeros(batch, text_slots, self.text_dim)
        image_features = torch.zeros(batch, image_slots, self.image_dim)
        text_mask = torch.zeros(batch, text_slots, dtype=torch.bool)
        image_mask = torch.zeros(batch, image_slots, dtype=torch.bool)
        if self.table_dim is not None:
            table_slots = min(self.max_table, max(len(items) for items in tables))
            table_features = torch.zeros(batch, table_slots, self.table_dim)
            table_mask = torch.zeros(batch, table_slots, dtype=torch.bool)
        with torch.no_grad():
            claims = torch.as_tensor(self.text_encoder.encode([r.claim_text for r in records]))
            for index, (text_items, image_items) in enumerate(zip(texts, images, strict=True)):
                selected_text = text_items[: self.max_text]
                selected_images = image_items[: self.max_image]
                if selected_text:
                    features = self.text_encoder.encode([item.text for item in selected_text])
                    text_features[index, : len(selected_text)] = torch.as_tensor(features).cpu()
                    text_mask[index, : len(selected_text)] = True
                if selected_images:
                    paths = [item.image_path for item in selected_images]
                    features = self.vision_encoder.encode_paths(paths)
                    image_features[index, : len(selected_images)] = torch.as_tensor(features).cpu()
                    image_mask[index, : len(selected_images)] = True
                if self.table_dim is not None:
                    selected_tables = tables[index][: self.max_table]
                    if selected_tables:
                        features = torch.as_tensor(self.table_encoder.encode([
                            serialize_table(item.table) for item in selected_tables
                        ])).cpu()
                        if features.shape != (len(selected_tables), self.table_dim):
                            raise ValueError("Table encoder output dimension does not match collator")
                        table_features[index, : len(selected_tables)] = features
                        table_mask[index, : len(selected_tables)] = True
        if claims.shape != (batch, self.text_dim):
            raise ValueError("Text encoder output dimension does not match collator")
        labels = {"SUPPORT": 0, "REFUTE": 1, None: -100}
        result = {
            "claim_features": claims.detach().cpu().float(),
            "text_features": text_features,
            "image_features": image_features,
            "text_mask": text_mask,
            "image_mask": image_mask,
            "global_labels": torch.tensor([labels[r.global_label] for r in records]),
            "original_labels": torch.tensor([self.labels[r.original_label] for r in records]),
            "dataset_ids": [r.dataset_id for r in records],
            "record_ids": [r.record_id for r in records],
            "truncation": {
                "text_evidence": sum(max(0, len(x) - self.max_text) for x in texts),
                "image_evidence": sum(max(0, len(x) - self.max_image) for x in images),
            },
        }
        if self.table_dim is not None:
            result["table_features"] = table_features
            result["table_mask"] = table_mask
            result["truncation"]["table_evidence"] = sum(
                max(0, len(items) - self.max_table) for items in tables
            )
        return result
