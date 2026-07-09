"""
Tracker Registry
================
A simple service-locator so the pipeline can instantiate any registered
hand-tracking backend by name from config.
"""

from __future__ import annotations

import logging
from typing import Type

from .base_hand_tracker import BaseHandTracker

logger = logging.getLogger("aegis.trackers")


class TrackerRegistry:
    """Central registry of hand-tracker backends."""

    _backends: dict[str, Type[BaseHandTracker]] = {}

    @classmethod
    def register(cls, name: str, tracker_class: Type[BaseHandTracker]) -> None:
        """Register a backend under the given name."""
        cls._backends[name.lower()] = tracker_class
        logger.debug("Registered hand tracker backend: %s", name)

    @classmethod
    def create(cls, name: str, config: dict) -> BaseHandTracker:
        """Instantiate a registered backend by name."""
        key = name.lower()
        if key not in cls._backends:
            available = ", ".join(cls._backends.keys()) or "(none)"
            raise ValueError(
                f"Unknown hand tracker backend '{name}'. "
                f"Available: {available}"
            )
        tracker = cls._backends[key](config)
        tracker.load_model()
        return tracker

    @classmethod
    def available(cls) -> list[str]:
        """List all registered backend names."""
        return list(cls._backends.keys())
