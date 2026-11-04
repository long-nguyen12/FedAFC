"""Load YAML configuration without assuming a dataset schema."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a nonempty YAML mapping; paths inside it remain as written."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not config:
        raise ValueError("Configuration must be a nonempty YAML mapping")
    if any(not isinstance(key, str) for key in config):
        raise ValueError("Configuration keys must be strings")
    return config
