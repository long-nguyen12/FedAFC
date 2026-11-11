"""Binary-compatible and original-label objectives remain separate."""

from collections.abc import Mapping
from typing import Any

from torch import Tensor
from torch.nn import functional as F


def compute_loss(
    outputs: Mapping[str, Any], batch: Mapping[str, Any], *, mode: str = "binary_only",
    global_weight: float = 1.0, private_weight: float = 1.0, image_aux_weight: float = 0.0,
) -> dict[str, Tensor]:
    valid_evidence = batch["text_mask"].any(1) | batch["image_mask"].any(1)
    if "table_mask" in batch:
        valid_evidence |= batch["table_mask"].any(1)
    global_mask = (batch["global_labels"] != -100) & valid_evidence
    zero = outputs["global_logits"].new_zeros(())
    global_loss = (
        F.cross_entropy(outputs["global_logits"][global_mask], batch["global_labels"][global_mask])
        if global_mask.any() and global_weight > 0 else zero
    )
    private_sum = zero
    private_count = 0
    if mode == "shared_private" and private_weight > 0:
        for dataset, logits in outputs["private_logits"].items():
            index = outputs["private_indices"][dataset]
            labels = batch["original_labels"][index]
            valid = (labels != -100) & valid_evidence[index]
            if valid.any():
                private_sum = private_sum + F.cross_entropy(
                    logits[valid], labels[valid], reduction="sum"
                )
                private_count += int(valid.sum())
    private_loss = private_sum / max(private_count, 1)
    image_loss = zero
    if image_aux_weight > 0:
        if "image_logits" not in outputs:
            raise ValueError("image_aux_weight requires a model built with image_aux_head")
        # C6: image-only head, on rows that have images and a SUPPORT/REFUTE label.
        image_mask = global_mask & batch["image_mask"].any(1)
        if image_mask.any():
            image_loss = F.cross_entropy(
                outputs["image_logits"][image_mask], batch["global_labels"][image_mask]
            )
    return {
        "loss": global_weight * global_loss + private_weight * private_loss
        + image_aux_weight * image_loss,
        "global_loss": global_loss,
        "private_loss": private_loss,
        "image_loss": image_loss,
    }
