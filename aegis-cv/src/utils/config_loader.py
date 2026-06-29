"""
Configuration loader — reads YAML config files and provides typed access.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config/settings.yaml") -> dict[str, Any]:
    """Load and return the YAML configuration as a dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    _validate(config)
    return config


def load_bins_config(config_path: str = "config/bins_map.yaml") -> dict[str, Any]:
    """Load the bin geofence configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Bins config not found: {path.resolve()}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _validate(config: dict) -> None:
    """Basic sanity checks on required config sections."""
    required_sections = ["hardware", "vision", "fsm", "performance"]
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required config section: '{section}'")
