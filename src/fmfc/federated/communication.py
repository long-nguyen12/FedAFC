"""Tensor payload sizes; serialized metadata overhead is reported separately."""

from collections.abc import Mapping

from torch import Tensor


def tensor_bytes(state: Mapping[str, Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())
