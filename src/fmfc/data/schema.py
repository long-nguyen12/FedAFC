"""Versioned model input records; metadata never forms model inputs."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _keys(data: Mapping[str, Any], allowed: set[str], required: set[str]) -> None:
    unexpected = set(data) - allowed
    missing = required - set(data)
    if unexpected or missing:
        raise ValueError(
            f"Unexpected fields {sorted(unexpected)}; missing fields {sorted(missing)}"
        )


def _forbidden(value: Any) -> None:
    if isinstance(value, Mapping):
        if {"table_evidence", "claim_image"} & set(value):
            raise ValueError("Raw table_evidence and claim_image fields are not canonical evidence items")
        for item in value.values():
            _forbidden(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _forbidden(item)


def _table(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"header", "rows"}:
        raise ValueError("Table must contain exactly header and rows")
    header, rows = value["header"], value["rows"]
    if not isinstance(header, list) or not header or any(not isinstance(x, str) for x in header):
        raise ValueError("Table header must be a nonempty list of strings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Table rows must be nonempty")
    if any(not isinstance(row, list) or len(row) != len(header)
           or any(not isinstance(cell, str) for cell in row) for row in rows):
        raise ValueError("Table rows must be rectangular lists of strings")


def serialize_table(table: dict[str, Any]) -> str:
    """Encode only ordered cell/header content, with explicit row/column roles."""
    _table(table)
    lines = ["[TABLE]"]
    for index, row in enumerate(table["rows"]):
        cells = [f"[COLUMN {column} {json.dumps(header, ensure_ascii=False)}] "
                 f"{json.dumps(cell, ensure_ascii=False)}"
                 for column, (header, cell) in enumerate(zip(table["header"], row, strict=True))]
        lines.append(f"[ROW {index}] " + " ".join(cells))
    return "\n".join(lines)


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    source_id: str
    text: str | None = None
    image_path: str | None = None
    table: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _text(self.evidence_id, "evidence_id")
        _text(self.source_id, "source_id")
        if self.kind == "text":
            _text(self.text, "text")
            if self.image_path is not None or self.table is not None:
                raise ValueError("Text evidence cannot have image_path or table")
        elif self.kind == "image":
            _text(self.image_path, "image_path")
            if self.text is not None or self.table is not None:
                raise ValueError("Image evidence cannot have text or table")
        elif self.kind == "table":
            _table(self.table)
            if self.text is not None or self.image_path is not None:
                raise ValueError("Table evidence cannot have text or image_path")
        else:
            raise ValueError("Evidence kind must be text, image or table")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceItem":
        if not isinstance(data, Mapping):
            raise ValueError("Evidence must be an object")
        _keys(
            data,
            {"evidence_id", "kind", "source_id", "text", "image_path", "table"},
            {"evidence_id", "kind", "source_id"},
        )
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        values = {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "source_id": self.source_id,
            "text": self.text,
            "image_path": self.image_path,
        }
        if self.kind == "table":
            values["table"] = self.table
        return values


@dataclass(frozen=True)
class VerificationRecord:
    record_id: str
    dataset_id: str
    split: str
    claim_text: str
    evidence: tuple[EvidenceItem, ...]
    original_label: str
    global_label: str | None = None
    schema_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for name in ("record_id", "dataset_id", "claim_text", "original_label"):
            _text(getattr(self, name), name)
        _forbidden(self.metadata)
        object.__setattr__(self, "evidence", tuple(self.evidence))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerificationRecord":
        _forbidden(data)
        allowed = {
            "schema_version",
            "record_id",
            "dataset_id",
            "split",
            "claim_text",
            "evidence",
            "original_label",
            "global_label",
            "metadata",
        }
        _keys(data, allowed, allowed - {"metadata", "global_label"})
        values = dict(data)
        values["evidence"] = tuple(EvidenceItem.from_dict(item) for item in values["evidence"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "claim_text": self.claim_text,
            "evidence": [item.to_dict() for item in self.evidence],
            "original_label": self.original_label,
            "global_label": self.global_label,
            "metadata": self.metadata,
        }
