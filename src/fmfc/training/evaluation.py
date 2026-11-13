"""Class-complete macro F1 and confusion matrices on eligible examples."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch.nn import functional as F
from tqdm import tqdm

from fmfc.model.verifier import Verifier
from fmfc.training.local import to_device


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> dict[str, Any]:
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target][prediction] += 1
    supports = [sum(row) for row in confusion]
    f1 = []
    for index in range(num_classes):
        tp = confusion[index][index]
        denominator = supports[index] + sum(row[index] for row in confusion)
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {
        "sample_count": len(targets), "macro_f1": sum(f1) / num_classes if targets else None,
        # Accuracy equals micro-F1 for single-label classification.
        "accuracy": sum(confusion[i][i] for i in range(num_classes)) / len(targets) if targets else None,
        "confusion_matrix": confusion, "class_support": supports,
        "class_coverage": sum(count > 0 for count in supports) / num_classes,
        "per_class_f1": f1,
    }


def _by_modality(
    targets: Sequence[int], predictions: Sequence[int], patterns: Sequence[str], num_classes: int,
) -> dict[str, Any]:
    """B1: metrics per evidence pattern the model saw; group sample counts sum to the total."""
    groups: dict[str, tuple[list[int], list[int]]] = {}
    for target, prediction, pattern in zip(targets, predictions, patterns, strict=True):
        actual, predicted = groups.setdefault(pattern, ([], []))
        actual.append(target)
        predicted.append(prediction)
    return {pattern: classification_metrics(actual, predicted, num_classes)
            for pattern, (actual, predicted) in sorted(groups.items())}


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
    original: dict[str, tuple[list[int], list[int], list[str]]] = {}
    patterns: list[str] = []  # B1: evidence pattern of each binary row, aligned with targets
    truncation: dict[str, int] = {}  # A3b: items/tables the collator dropped or cut
    seen = eligible = strict = table_count = 0
    has_table_mask = False
    # Cross-entropy on raw logits (no logit adjustment), averaged over the same rows as training.
    loss_sum = {"global": 0.0, "private": 0.0}
    loss_rows = {"global": 0, "private": 0}
    progress = tqdm(batches, desc=progress_desc, unit="batch", dynamic_ncols=True)
    for raw_batch in progress:
        batch = to_device(raw_batch, device)
        output = model(batch)
        text = batch["text_mask"].any(1)
        image = batch["image_mask"].any(1)
        strict_membership = batch.get("strict_cohort_mask", text & image).bool()
        usable = text | image
        table = torch.zeros_like(text)
        if "table_mask" in batch:
            has_table_mask = True
            table = batch["table_mask"].any(1)
            usable |= table
            table_count += int(table.sum())
        for key, value in batch.get("truncation", {}).items():
            truncation[key] = truncation.get(key, 0) + int(value)
        row_patterns = [
            "+".join(name for name, present in zip(("T", "I", "Tab"), flags) if present)
            for flags in zip(text.tolist(), image.tolist(), table.tolist())
        ]
        valid = usable & (batch["global_labels"] != -100)
        seen += len(usable)
        eligible += int(usable.sum())
        strict += int(strict_membership.sum())
        targets.extend(batch["global_labels"][valid].cpu().tolist())
        predictions.extend(output["global_logits"][valid].argmax(1).cpu().tolist())
        patterns.extend(row_patterns[row] for row in valid.nonzero(as_tuple=True)[0].tolist())
        if valid.any():
            loss_sum["global"] += float(F.cross_entropy(
                output["global_logits"][valid], batch["global_labels"][valid], reduction="sum"))
            loss_rows["global"] += int(valid.sum())
        strict_valid = valid & strict_membership
        strict_targets.extend(batch["global_labels"][strict_valid].cpu().tolist())
        strict_predictions.extend(output["global_logits"][strict_valid].argmax(1).cpu().tolist())
        for dataset, logits in output["private_logits"].items():
            indices = output["private_indices"][dataset]
            labels = batch["original_labels"][indices]
            valid_private = usable[indices] & (labels != -100)
            actual, predicted, private_patterns = original.setdefault(dataset, ([], [], []))
            actual.extend(labels[valid_private].cpu().tolist())
            predicted.extend(logits[valid_private].argmax(1).cpu().tolist())
            private_patterns.extend(
                row_patterns[row] for row, keep in
                zip(indices.tolist(), valid_private.tolist(), strict=True) if keep
            )
            if valid_private.any():
                loss_sum["private"] += float(F.cross_entropy(
                    logits[valid_private], labels[valid_private], reduction="sum"))
                loss_rows["private"] += int(valid_private.sum())
        progress.set_postfix(samples=seen, refresh=False)
    return {
        "binary": classification_metrics(targets, predictions, 2),
        "strict_binary": classification_metrics(strict_targets, strict_predictions, 2),
        "original": {
            dataset: classification_metrics(
                actual, predicted, model.private_heads[dataset].out_features
            )
            for dataset, (actual, predicted, _) in original.items()
        },
        "binary_by_modality": _by_modality(targets, predictions, patterns, 2),
        "original_by_modality": {
            dataset: _by_modality(
                actual, predicted, groups, model.private_heads[dataset].out_features
            )
            for dataset, (actual, predicted, groups) in original.items()
        },
        "cohort": {
            "seen": seen, "usable_evidence": eligible, "strict_text_image": strict,
            **({"table_evidence": table_count} if has_table_mask else {}),
        },
        "truncation": truncation,
        # global: binary head CE; private: original-label head CE (MOCHEG 3-class).
        "loss": {head: loss_sum[head] / loss_rows[head] if loss_rows[head] else None
                 for head in loss_sum},
    }
