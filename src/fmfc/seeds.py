"""Seed available random generators; no model or dataset is loaded."""

import importlib
import random

import numpy as np


def seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
