"""Inspect the active interpreter without installing or downloading anything."""

import importlib
import importlib.metadata
import platform
import sys
from types import ModuleType
from typing import Any


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


def collect_environment() -> dict[str, Any]:
    """Return observed versions, including conflicting distribution metadata."""
    names = ("pip", "setuptools", "wheel", "numpy", "PyYAML", "pytest", "ruff", "torch",
             "transformers", "Pillow")
    versions: dict[str, list[str]] = {name: [] for name in names}
    canonical_names = {name.lower(): name for name in names}
    for distribution in importlib.metadata.distributions():
        name = canonical_names.get(distribution.metadata.get("Name", "").lower())
        if name is not None:
            versions[name].append(distribution.version)
    for installed_versions in versions.values():
        installed_versions.sort()

    torch_report: dict[str, Any]
    try:
        torch = _load_torch()
        if torch is None:
            torch_report = {"status": "missing"}
        else:
            cuda_available = bool(torch.cuda.is_available())
            device_count = int(torch.cuda.device_count())
            torch_report = {
                "status": "available",
                "version": str(torch.__version__),
                "module_path": str(torch.__file__),
                "cuda_build": torch.version.cuda,
                "cuda_available": cuda_available,
                "cuda_device_count": device_count,
                "cuda_devices": [
                    torch.cuda.get_device_name(index) for index in range(device_count)
                ] if cuda_available else [],
            }
    except Exception as exc:
        torch_report = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "packages": versions,
        "torch": torch_report,
    }
