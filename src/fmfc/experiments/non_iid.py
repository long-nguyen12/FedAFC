"""Deterministic pseudo-clients for engineering, never additional real datasets."""

import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any


def partition_indices(
    labels: Sequence[str | int],
    num_clients: int,
    seed: int = 42,
    groups: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    """Stratify by label while keeping each supplied family on one pseudo-client.

    This is a seeded balanced partition, not a claimed Dirichlet/non-IID model.
    Group constraints may prevent exact balance and are never broken to fill it.
    """
    if isinstance(num_clients, bool) or not isinstance(num_clients, int) or num_clients < 1:
        raise ValueError("num_clients must be a positive integer")
    if groups is not None and len(groups) != len(labels):
        raise ValueError("Family group and label lengths differ")
    families: dict[str, list[int]] = defaultdict(list)
    for index in range(len(labels)):
        families[str(index) if groups is None else groups[index]].append(index)
    if len(families) < num_clients:
        raise ValueError("Not enough independent families to create nonempty pseudo-clients")
    rng = random.Random(seed)
    by_label: dict[str, list[list[int]]] = defaultdict(list)
    for indices in families.values():
        counts = Counter(str(labels[index]) for index in indices)
        dominant = min(counts, key=lambda label: (-counts[label], label))
        by_label[dominant].append(indices)
    clients: list[list[int]] = [[] for _ in range(num_clients)]
    client_counts: list[Counter[str]] = [Counter() for _ in clients]
    for label in sorted(by_label):
        group_list = by_label[label]
        rng.shuffle(group_list)
        for indices in group_list:
            order = list(range(num_clients))
            rng.shuffle(order)
            target = min(
                order, key=lambda client: (client_counts[client][label], len(clients[client]))
            )
            clients[target].extend(indices)
            client_counts[target].update(str(labels[index]) for index in indices)
    return {f"pseudo_{index}": sorted(indices) for index, indices in enumerate(clients)}


def partition_records(
    records: Sequence[Any], num_clients: int, seed: int = 42
) -> dict[str, list[Any]]:
    """Partition canonical training records, joining shared claim/evidence families."""
    if any(record.split != "train" for record in records):
        raise ValueError("Pseudo-client construction must use training records only")
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        keys = [("claim", " ".join(record.claim_text.casefold().split()))]
        for item in record.evidence:
            keys.extend([("evidence", item.evidence_id), ("source", item.source_id)])
            if item.text:
                keys.append(("text", " ".join(item.text.casefold().split())))
            if item.image_path:
                keys.append(("image", item.image_path))
            if item.kind == "table":
                keys.append(("table", json.dumps(item.table, sort_keys=True, ensure_ascii=False)))
        for key in keys:
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    families = [str(find(index)) for index in range(len(records))]
    indices = partition_indices(
        [record.original_label for record in records], num_clients, seed, families
    )
    return {client: [records[index] for index in items] for client, items in indices.items()}
