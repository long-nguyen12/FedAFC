"""Explicit modality controls and robustness perturbations over feature batches."""

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor


def ablate_batch(
    batch: Mapping[str, Any], mode: str = "both", seed: int = 42
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode not in {"both", "all", "text", "image", "table", "shuffle_image", "drop_image", "drop_table"}:
        raise ValueError("Unknown modality ablation")
    result = {
        key: value.clone() if isinstance(value, Tensor) else copy.deepcopy(value)
        for key, value in batch.items()
    }
    result.setdefault(
        "strict_cohort_mask", batch["text_mask"].any(1) & batch["image_mask"].any(1)
    )
    changed = torch.zeros(len(result["text_mask"]), dtype=torch.bool,
                          device=result["text_mask"].device)
    removed = {
        "text": ("image", "table"), "image": ("text", "table"),
        "table": ("text", "image"), "drop_image": ("image",), "drop_table": ("table",),
    }.get(mode, ())
    for modality in removed:
        if f"{modality}_mask" in result:
            changed |= result[f"{modality}_mask"].any(1)
            result[f"{modality}_features"].zero_()
            result[f"{modality}_mask"].zero_()
    changed_rows = int(changed.sum())
    if mode == "shuffle_image":
        size = len(result["image_mask"])
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(size, generator=generator)
        if size > 1 and torch.equal(permutation, torch.arange(size)):
            permutation = permutation.roll(1)
        changed_rows = int((permutation != torch.arange(size)).sum())
        for key in ("image_features", "image_mask"):
            result[key] = result[key][permutation.to(result[key].device)]
    return result, {
        "ablation": mode,
        "robustness_only": mode in {"shuffle_image", "drop_image", "drop_table"},
        "evidence_sufficiency_unknown": mode not in {"both", "all"},
        "changed_rows": changed_rows,
        "seed": seed,
    }
