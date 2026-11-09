"""Class-complete macro F1 and confusion matrices on eligible examples."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from tqdm import tqdm

from fmfc.model.verifier import Verifier
from fmfc.training.local import to_device


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> dict[str, Any]:
    if len(targets) != len(predictions):
        raise ValueError("Target/prediction lengths differ")
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        if not 0 <= target < num_classes or not 0 <= prediction < num_classes:
            raise ValueError("Class index outside the declared label space")
        confusion[target][prediction] += 1
    supports = [sum(row) for row in confusion]
    f1 = []
    for index in range(num_classes):
        tp = confusion[index][index]
        denominator = supports[index] + sum(row[index] for row in confusion)
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {
        "sample_count": len(targets), "macro_f1": sum(f1) / num_classes if targets else None,
        "confusion_matrix": confusion, "class_support": supports,
        "class_coverage": sum(count > 0 for count in supports) / num_classes,
        "per_class_f1": f1,
    }


@torch.inference_mode()
def evaluate(
    model: Verifier, batches: Iterable[Mapping[str, Any]], *, device: str = "cpu",
    progress_desc: str = "Val",
) -> dict[str, Any]:
    model.to(device).eval()
    targets: list[int] = []
    predictions: list[int] = []
    strict_targets: list[int] = []
    strict_predictions: list[int] = []
    original: dict[str, tuple[list[int], list[int]]] = {}
    seen = eligible = strict = table_count = 0
    has_table_mask = False
    progress = tqdm(batches, desc=progress_desc, unit="batch", dynamic_ncols=True)
    for raw_batch in progress:
        batch = to_device(raw_batch, device)
        output = model(batch)
        text = batch["text_mask"].any(1)
        image = batch["image_mask"].any(1)
        strict_membership = batch.get("strict_cohort_mask", text & image).bool()
        usable = text | image
        if "table_mask" in batch:
            has_table_mask = True
            table = batch["table_mask"].any(1)
            usable |= table
            table_count += int(table.sum())
        valid = usable & (batch["global_labels"] != -100)
        seen += len(usable)
        eligible += int(usable.sum())
        strict += int(strict_membership.sum())
        targets.extend(batch["global_labels"][valid].cpu().tolist())
        predictions.extend(output["global_logits"][valid].argmax(1).cpu().tolist())
        strict_valid = valid & strict_membership
        strict_targets.extend(batch["global_labels"][strict_valid].cpu().tolist())
        strict_predictions.extend(output["global_logits"][strict_valid].argmax(1).cpu().tolist())
        for dataset, logits in output["private_logits"].items():
            indices = output["private_indices"][dataset]
            labels = batch["original_labels"][indices]
            valid_private = usable[indices] & (labels != -100)
            actual, predicted = original.setdefault(dataset, ([], []))
            actual.extend(labels[valid_private].cpu().tolist())
            predicted.extend(logits[valid_private].argmax(1).cpu().tolist())
        progress.set_postfix(samples=seen, refresh=False)
    return {
        "binary": classification_metrics(targets, predictions, 2),
        "strict_binary": classification_metrics(strict_targets, strict_predictions, 2),
        "original": {
            dataset: classification_metrics(
                actual, predicted, model.private_heads[dataset].out_features
            )
            for dataset, (actual, predicted) in original.items()
        },
        "cohort": {
            "seen": seen, "usable_evidence": eligible, "strict_text_image": strict,
            **({"table_evidence": table_count} if has_table_mask else {}),
        },
    }
