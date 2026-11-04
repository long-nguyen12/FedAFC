"""Checkpoints include client-private optimizer and RNG state for local resume."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
    *, metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np_state = np.random.get_state()
    torch.save({
        "checkpoint_version": 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metadata": metadata or {}, "python_rng": random.getstate(),
        "numpy_rng": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }, target)


def load_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
    *, restore_rng: bool = True,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if state.get("checkpoint_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        if state["optimizer"] is None:
            raise ValueError("Checkpoint has no optimizer state")
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng:
        random.setstate(state["python_rng"])
        np_state = state["numpy_rng"]
        np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            if not torch.cuda.is_available():
                raise RuntimeError("Cannot exactly resume a CUDA RNG checkpoint without CUDA")
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["metadata"]
