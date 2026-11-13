import os
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from fmfc.data.schema import VerificationRecord, serialize_table_rows

Row = dict[str, Tensor]  # one record's unpadded features: claim / text / image / table
GLOBAL_IDS = {"SUPPORT": 0, "REFUTE": 1, None: -100}


class CrossTransCollator:
    def __init__(
        self,
        claim_encoder: Any,
        evidence_encoder: Any,
        vision_encoder: Any,
        original_label_mapping: dict[str, int],
        max_table_evidence: int | None = None,
        max_table_tokens: int | None = None,
        cache_dir: str | Path | None = None,
        claim_clip_encoder: Any = None,
    ) -> None:
        if evidence_encoder.max_length is not None:
            raise ValueError(
                "encoders.evidence.max_length must be null: text is never truncated"
            )
        if (max_table_evidence is None) != (max_table_tokens is None) or (
            max_table_evidence is not None
            and min(max_table_evidence, max_table_tokens) < 1
        ):
            raise ValueError(
                "Table evidence and token budgets must both be positive or unset"
            )
        self.claim_encoder = claim_encoder
        self.evidence_encoder = evidence_encoder
        self.vision_encoder = vision_encoder
        self.labels = original_label_mapping
        self.max_table = max_table_evidence
        self.max_table_tokens = max_table_tokens
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.claim_clip_encoder = claim_clip_encoder

    def defer(
        self,
        records: list[VerificationRecord],
        ablation: str = "all",
        seed: int = 42,
    ) -> dict[str, Any]:
        return {
            "_records": records,
            "_collator": self,
            "_ablation": ablation,
            "_seed": seed,
            **self._ids(records),
        }

    def __call__(self, records: list[VerificationRecord]) -> dict[str, Any]:
        if not records:
            raise ValueError("Cannot collate an empty batch")
        if self.max_table is None and any(self._items(r)["table"] for r in records):
            raise ValueError(
                "Table evidence requires a table-enabled CrossTrans collator"
            )
        rows = (
            self._cached_rows(records) if self.cache_dir else self._encode_rows(records)
        )
        result = self._assemble(records, rows)
        if self.claim_clip_encoder is not None:
            result["claim_clip"] = self.claim_clip_encoder.encode(
                [r.claim_text for r in records]
            )
        return result

    def _items(self, record: VerificationRecord) -> dict[str, list[Any]]:
        """Encoded evidence by kind: every text piece and image, the first max_table tables."""
        items = {
            kind: [item for item in record.evidence if item.kind == kind]
            for kind in ("text", "image", "table")
        }
        items["table"] = items["table"][: self.max_table]
        return items

    # ------------------------------------------------------------------------ encoding

    def _encode_rows(self, records: list[VerificationRecord]) -> list[Row]:
        """Encode one batch, then keep each record's frozen features without padding."""
        items = [self._items(record) for record in records]
        rows: list[Row] = [{} for _ in records]
        features, masks = self.claim_encoder.encode_tokens(
            [r.claim_text for r in records]
        )
        for row, states, mask in zip(rows, features, masks, strict=True):
            row["claim"] = states[mask]
        joiner = f" {self.evidence_encoder.tokenizer.sep_token or '[SEP]'} "
        limit = self._text_token_limit()
        self._encode_sequences(
            rows,
            "text",
            [
                (index, sequence)
                for index, (record, kinds) in enumerate(
                    zip(records, items, strict=True)
                )
                if kinds["text"]
                for sequence in self._text_sequences(
                    record.record_id,
                    [item.text for item in kinds["text"]],
                    joiner,
                    limit,
                )
            ],
            limit,
        )
        for row, kinds in zip(rows, items, strict=True):
            if kinds["image"]:
                row["image"] = self.vision_encoder.encode_paths(
                    [item.image_path for item in kinds["image"]]
                )
        if self.max_table is not None:
            # Tables are a separate token stream from the same frozen evidence encoder.
            self._encode_sequences(
                rows,
                "table",
                [
                    (
                        index,
                        joiner.join(
                            serialize_table_rows(item.table) for item in kinds["table"]
                        ),
                    )
                    for index, kinds in enumerate(items)
                    if kinds["table"]
                ],
                self.max_table_tokens,
            )
        return rows

    def _text_token_limit(self) -> int:
        """Longest sequence the evidence encoder accepts (Longformer: 4,096 tokens)."""
        config = self.evidence_encoder.model.config
        return min(
            self.evidence_encoder.tokenizer.model_max_length,
            config.max_position_embeddings - 2,
        )

    def _text_sequences(
        self,
        record_id: str,
        texts: list[str],
        joiner: str,
        limit: int,
    ) -> list[str]:
        """Join the pieces; when they do not fit, start a new sequence at a piece boundary."""
        tokenizer = self.evidence_encoder.tokenizer

        def fits(text: str) -> bool:
            return len(tokenizer(text, truncation=False)["input_ids"]) <= limit

        if fits(joiner.join(texts)):  # the usual case: one sequence, one tokenizer call
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
                f"A text evidence piece of {record_id} exceeds the {limit}-token "
                "encoder limit; it would have to be truncated"
            )
        return sequences

    def _encode_sequences(
        self,
        rows: list[Row],
        key: str,
        contents: list[tuple[int, str]],
        max_length: int,
    ) -> None:
        """Encode (row, text) pairs; a row with several sequences keeps their states in order."""
        if not contents:
            return
        features, masks = self.evidence_encoder.encode_tokens(
            [text for _, text in contents],
            max_length=max_length,
        )
        for (index, _), states, mask in zip(contents, features, masks, strict=True):
            part = states[mask]
            rows[index][key] = (
                torch.cat([rows[index][key], part]) if key in rows[index] else part
            )

    # --------------------------------------------------------------------------- cache

    def _cache_path(self, record: VerificationRecord) -> Path:
        """<cache_dir>/<dataset>/<split>/<record_id>.pt, with unsafe characters as "_"."""
        dataset, split, name = (
            re.sub(r"[^A-Za-z0-9._-]", "_", part)
            for part in (record.dataset_id, record.split, record.record_id)
        )
        return self.cache_dir / dataset / split / f"{name}.pt"

    def _cached_rows(self, records: list[VerificationRecord]) -> list[Row]:
        """Reuse cached frozen features; encode and store only records not cached yet.

        Features are stored as float16 and always used at that precision, so the
        epoch that fills the cache sees the same inputs as every later epoch.
        """
        paths = [self._cache_path(record) for record in records]
        rows = [
            torch.load(path, weights_only=True) if path.is_file() else None
            for path in paths
        ]
        missing = [index for index, row in enumerate(rows) if row is None]
        encoded = (
            self._encode_rows([records[index] for index in missing]) if missing else []
        )
        for index, row in zip(missing, encoded, strict=True):
            rows[index] = {key: value.to(torch.float16) for key, value in row.items()}
            paths[index].parent.mkdir(parents=True, exist_ok=True)
            # Write then rename, so an interrupted run never leaves a partial entry.
            temporary = paths[index].with_name(f"{paths[index].name}.{os.getpid()}.tmp")
            torch.save(rows[index], temporary)
            os.replace(temporary, paths[index])
        for record, row in zip(records, rows, strict=True):
            # Fail instead of silently using features built from other evidence.
            items = self._items(record)
            expected = {"claim"} | {kind for kind, values in items.items() if values}
            if set(row) != expected or (
                items["image"] and row["image"].shape[0] != len(items["image"])
            ):
                raise ValueError(
                    f"Cached features of {record.record_id} do not match this configuration; "
                    "use a new feature_cache_dir"
                )
        return [{key: value.float() for key, value in row.items()} for row in rows]

    # ------------------------------------------------------------------------ assembly

    @staticmethod
    def _ids(records: list[VerificationRecord]) -> dict[str, Any]:
        return {
            "global_labels": torch.tensor(
                [GLOBAL_IDS[r.global_label] for r in records]
            ),
            "dataset_ids": [r.dataset_id for r in records],
            "record_ids": [r.record_id for r in records],
        }

    def _assemble(
        self, records: list[VerificationRecord], rows: list[Row]
    ) -> dict[str, Any]:
        """Pad per-record features back into batch tensors; padding stays zero and masked."""

        def pad(key: str, dim: int) -> tuple[Tensor, Tensor]:
            length = max((row[key].shape[0] for row in rows if key in row), default=0)
            features = torch.zeros(len(rows), length, dim)
            mask = torch.zeros(len(rows), length, dtype=torch.bool)
            for index, row in enumerate(rows):
                if key in row:
                    features[index, : row[key].shape[0]] = row[key]
                    mask[index, : row[key].shape[0]] = True
            return features, mask

        encoders = {
            "claim": self.claim_encoder,
            "text": self.evidence_encoder,
            "image": self.vision_encoder,
        }
        if self.max_table is not None:
            encoders["table"] = self.evidence_encoder  # serialized tables, same encoder
        result: dict[str, Any] = {}
        for key, encoder in encoders.items():
            result[f"{key}_features"], result[f"{key}_mask"] = pad(
                key, encoder.output_dim
            )
        result.update(self._ids(records))
        result["original_labels"] = torch.tensor(
            [self.labels[r.original_label] for r in records]
        )
        # A3b: tables beyond max_table_evidence, and table streams cut at max_table_tokens.
        result["truncation"] = (
            {}
            if self.max_table is None
            else {
                "table_evidence": sum(
                    max(
                        0,
                        sum(item.kind == "table" for item in r.evidence)
                        - self.max_table,
                    )
                    for r in records
                ),
                "table_tokens": sum(
                    row["table"].shape[0] >= self.max_table_tokens
                    for row in rows
                    if "table" in row
                ),
            }
        )
        return result
