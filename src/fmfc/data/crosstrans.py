"""Token-level, batch-wise encoding for the CrossTrans text/image/table adapter."""

import os
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from fmfc.data.schema import VerificationRecord, serialize_table_rows

Row = dict[str, Tensor]


class CrossTransCollator:
    def __init__(
        self, claim_encoder: Any, evidence_encoder: Any, vision_encoder: Any,
        original_label_mapping: dict[str, int], max_text_evidence: int | None = 4,
        max_image_evidence: int | None = 2, max_table_evidence: int | None = None,
        max_table_tokens: int | None = None, cache_dir: str | Path | None = None,
        claim_clip_encoder: Any = None,
    ) -> None:
        # None means every supplied text piece / image is used.
        if any(budget is not None and budget < 1
               for budget in (max_text_evidence, max_image_evidence)):
            raise ValueError("Evidence budgets must be positive, or None to use every item")
        if (max_table_evidence is None) != (max_table_tokens is None) or (
            max_table_evidence is not None and min(max_table_evidence, max_table_tokens) < 1
        ):
            raise ValueError("Table evidence and token budgets must both be positive or unset")
        self.claim_encoder = claim_encoder
        self.evidence_encoder = evidence_encoder
        self.vision_encoder = vision_encoder
        self.labels = original_label_mapping
        self.max_text = max_text_evidence
        self.max_image = max_image_evidence
        self.max_table = max_table_evidence
        self.max_table_tokens = max_table_tokens
        # Frozen features per record. Entries are never invalidated automatically:
        # the user picks a new directory when encoders, budgets or evidence change.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # Only for model.clip_similarity; computed per batch, so caches stay valid.
        self.claim_clip_encoder = claim_clip_encoder

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
        if self.max_table is None and any(self._items(r, "table") for r in records):
            raise ValueError("Table evidence requires a table-enabled CrossTrans collator")
        rows = self._encode_rows(records) if self.cache_dir is None else self._cached_rows(records)
        result = self._assemble(records, rows)
        if self.claim_clip_encoder is not None:
            result["claim_clip"] = self.claim_clip_encoder.encode([r.claim_text for r in records])
        return result

    @staticmethod
    def _items(record: VerificationRecord, kind: str) -> list[Any]:
        return [item for item in record.evidence if item.kind == kind]

    def _encode_rows(self, records: list[VerificationRecord]) -> list[Row]:
        """Encode one batch, then keep each record's frozen features without padding."""
        rows: list[Row] = [{} for _ in records]
        features, masks = self.claim_encoder.encode_tokens([r.claim_text for r in records])
        for row, states, mask in zip(rows, features, masks, strict=True):
            row["claim"] = states[mask]
        separator = self.evidence_encoder.tokenizer.sep_token or "[SEP]"
        texts = [self._items(r, "text") for r in records]
        untruncated = self.evidence_encoder.max_length is None
        self._encode_sequences(rows, "text", [
            (index, sequence)
            for index, (record, items) in enumerate(zip(records, texts, strict=True)) if items
            for sequence in self._text_sequences(record, items[:self.max_text], separator)
        ], max_length=self._text_token_limit() if untruncated else None)
        for row, record in zip(rows, records, strict=True):
            selected = self._items(record, "image")[:self.max_image]
            if selected:
                row["image"] = self.vision_encoder.encode_paths([e.image_path for e in selected])
        if self.max_table is not None:
            # Tables are a separate token stream from the same frozen evidence encoder.
            tables = [self._items(r, "table") for r in records]
            self._encode_sequences(rows, "table", [
                (index, f" {separator} ".join(
                    serialize_table_rows(item.table) for item in items[:self.max_table]
                ))
                for index, items in enumerate(tables) if items
            ], max_length=self.max_table_tokens)
        return rows

    def _text_token_limit(self) -> int:
        """Longest sequence the evidence encoder accepts (Longformer: 4,096 tokens)."""
        config = self.evidence_encoder.model.config
        return min(self.evidence_encoder.tokenizer.model_max_length,
                   config.max_position_embeddings - 2)

    def _text_sequences(
        self, record: VerificationRecord, items: list[Any], separator: str,
    ) -> list[str]:
        """Join text pieces with the separator token.

        With a fixed ``encoders.evidence.max_length`` this is one sequence that the
        encoder truncates (previous behaviour). With ``max_length: null`` nothing is
        cut: pieces that no longer fit the encoder limit start a new sequence at a
        piece boundary, and a single piece longer than the limit raises.
        """
        joiner = f" {separator} "
        texts = [item.text for item in items]
        if self.evidence_encoder.max_length is not None:
            return [joiner.join(texts)]
        tokenizer, limit = self.evidence_encoder.tokenizer, self._text_token_limit()

        def fits(text: str) -> bool:
            return len(tokenizer(text, truncation=False)["input_ids"]) <= limit

        if fits(joiner.join(texts)):
            return [joiner.join(texts)]
        sequences, current = [], []
        for text in texts:
            if current and not fits(joiner.join([*current, text])):
                sequences.append(joiner.join(current))
                current = []
            current.append(text)
        sequences.append(joiner.join(current))
        if not all(fits(sequence) for sequence in sequences):
            raise ValueError(
                f"A text evidence piece of {record.record_id} exceeds the {limit}-token "
                "encoder limit; it would have to be truncated"
            )
        return sequences

    def _encode_sequences(
        self, rows: list[Row], key: str, contents: list[tuple[int, str]],
        max_length: int | None = None,
    ) -> None:
        """Encode (row, text) pairs; a row with several sequences keeps their states in order."""
        if not contents:
            return
        options = {} if max_length is None else {"max_length": max_length}
        features, masks = self.evidence_encoder.encode_tokens(
            [text for _, text in contents], **options,
        )
        for (index, _), states, mask in zip(contents, features, masks, strict=True):
            part = states[mask]
            rows[index][key] = torch.cat([rows[index][key], part]) if key in rows[index] else part

    def _cache_path(self, record: VerificationRecord) -> Path:
        parts = (record.dataset_id, record.split, record.record_id)
        name = [re.sub(r"[^A-Za-z0-9._-]", "_", part) for part in parts]
        return self.cache_dir / name[0] / name[1] / f"{name[2]}.pt"

    def _cached_rows(self, records: list[VerificationRecord]) -> list[Row]:
        """Reuse cached frozen features; encode and store only records not cached yet.

        Features are stored as float16 and always used at that precision, so the
        epoch that fills the cache sees the same inputs as every later epoch.
        """
        paths = [self._cache_path(record) for record in records]
        rows = [torch.load(path, weights_only=True) if path.is_file() else None for path in paths]
        missing = [index for index, row in enumerate(rows) if row is None]
        if missing:
            encoded = self._encode_rows([records[index] for index in missing])
            for index, row in zip(missing, encoded, strict=True):
                stored = {key: value.to(torch.float16) for key, value in row.items()}
                paths[index].parent.mkdir(parents=True, exist_ok=True)
                # Write then rename, so an interrupted run never leaves a partial entry.
                temporary = paths[index].with_name(f"{paths[index].name}.{os.getpid()}.tmp")
                torch.save(stored, temporary)
                os.replace(temporary, paths[index])
                rows[index] = stored
        for record, row in zip(records, rows, strict=True):
            self._check_cached(record, row)
        return [{key: value.float() for key, value in row.items()} for row in rows]

    def _check_cached(self, record: VerificationRecord, row: Row) -> None:
        """Fail instead of silently dropping evidence when a cache came from other settings."""
        expected = {"claim"} | {kind for kind in ("text", "image") if self._items(record, kind)}
        if self.max_table is not None and self._items(record, "table"):
            expected.add("table")
        images = len(self._items(record, "image")[:self.max_image])
        if set(row) != expected or (images and row["image"].shape[0] != images):
            raise ValueError(
                f"Cached features of {record.record_id} do not match this configuration; "
                "use a new feature_cache_dir"
            )

    def _assemble(self, records: list[VerificationRecord], rows: list[Row]) -> dict[str, Any]:
        """Pad per-record features back into batch tensors; padding stays zero and masked."""
        batch = len(records)

        def pad(key: str, dim: int) -> tuple[Tensor, Tensor]:
            length = max((row[key].shape[0] for row in rows if key in row), default=0)
            features = torch.zeros(batch, length, dim)
            mask = torch.zeros(batch, length, dtype=torch.bool)
            for index, row in enumerate(rows):
                if key in row:
                    size = row[key].shape[0]
                    features[index, :size] = row[key]
                    mask[index, :size] = True
            return features, mask

        claim_features, claim_mask = pad("claim", self.claim_encoder.output_dim)
        text_features, text_mask = pad("text", self.evidence_encoder.output_dim)
        image_features, image_mask = pad("image", self.vision_encoder.output_dim)
        counts = {kind: [len(self._items(r, kind)) for r in records]
                  for kind in ("text", "image", "table")}
        result = {
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
                # A None budget keeps every item, so nothing is dropped.
                "text_evidence": sum(max(0, n - (self.max_text or n)) for n in counts["text"]),
                "image_evidence": sum(max(0, n - (self.max_image or n)) for n in counts["image"]),
            },
        }
        if self.max_table is not None:
            result["table_features"], result["table_mask"] = pad(
                "table", self.evidence_encoder.output_dim,
            )
            result["truncation"]["table_evidence"] = sum(
                max(0, n - self.max_table) for n in counts["table"]
            )
            result["truncation"]["table_tokens"] = sum(
                row["table"].shape[0] >= self.max_table_tokens for row in rows if "table" in row
            )
        return result
