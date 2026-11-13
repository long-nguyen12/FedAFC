"""Resolve the compute device without requiring PyTorch to be installed."""

import importlib
from types import ModuleType


def _load_torch() -> ModuleType | None:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return None
        raise


def resolve_device(device: str = "auto") -> str:
    """Resolve CPU/CUDA explicitly; an absent optional Torch permits CPU mode."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cpu":
        return "cpu"
    try:
        torch = _load_torch()
        cuda_available = torch is not None and bool(torch.cuda.is_available())
    except Exception as exc:
        raise RuntimeError(f"Cannot inspect PyTorch: {type(exc).__name__}: {exc}") from exc
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but PyTorch with available CUDA is required")
    return "cuda" if cuda_available else "cpu"
