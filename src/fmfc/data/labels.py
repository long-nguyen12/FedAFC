"""Explicit label mappings, preserving unmapped original labels."""

from collections.abc import Mapping


def global_label(original: str, mapping: Mapping[str, str | None]) -> str | None:
    mapped = mapping.get(original)
    if mapped not in {None, "SUPPORT", "REFUTE"}:
        raise ValueError("Global label mapping must contain only SUPPORT, REFUTE or null")
    if original.strip().upper() in {"NEI", "NOT ENOUGH INFO", "NOT_ENOUGH_INFO"}:
        if mapped is not None:
            raise ValueError("NEI cannot map to a binary label")
    return mapped


def original_label_indices(labels: list[str]) -> dict[str, int]:
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Original labels must be a nonempty unique list")
    return {label: index for index, label in enumerate(labels)}
