"""Candidate P1: average each shared group only over its actual active exposures."""

from collections.abc import Mapping, Sequence

from torch import Tensor

from fmfc.federated.contracts import ModelUpdate, clone_state, validate_updates


def group_weights(
    updates: Sequence[ModelUpdate], group: str, weighting: str = "exposure"
) -> dict[str, float]:
    if weighting not in {"exposure", "uniform_active"}:
        raise ValueError("Group weighting must be exposure or uniform_active")
    active = [update for update in updates if update.group_counts.get(group, 0) > 0]
    counts = [update.group_counts[group] if weighting == "exposure" else 1 for update in active]
    total = sum(counts)
    return {update.client_id: count / total for update, count in zip(active, counts, strict=True)}


def evidence_aware(
    state: Mapping[str, Tensor],
    updates: Sequence[ModelUpdate],
    parameter_groups: Mapping[str, str],
    weighting: str = "exposure",
) -> dict[str, Tensor]:
    ordered = validate_updates(state, updates, parameter_groups)
    weights = {
        group: group_weights(ordered, group, weighting) for group in set(parameter_groups.values())
    }
    result = clone_state(state)
    for name in result:
        group = parameter_groups[name]
        for update in ordered:
            weight = weights[group].get(update.client_id, 0.0)
            if weight:
                result[name].add_(update.delta[name].detach().cpu(), alpha=weight)
    return result


def exposure_degenerates_to_fedavg(updates: Sequence[ModelUpdate]) -> bool:
    """True when every active group has the same client ratios as sample weighting.

    All-zero groups are excluded from this sufficient condition. This reports a
    mathematical control, never empirical evidence that P1 improves a benchmark.
    """
    active = [update for update in updates if update.sample_count > 0]
    if not active:
        return True
    groups = set(active[0].parameter_groups.values())
    reference = active[0]
    for group in groups:
        count = reference.group_counts.get(group, 0)
        if count <= 0:
            return False
        for update in active[1:]:
            if update.group_counts.get(group, 0) * reference.sample_count != (
                count * update.sample_count
            ):
                return False
    return True
