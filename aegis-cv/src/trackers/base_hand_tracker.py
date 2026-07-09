"""
Abstract Hand Tracker Interface
================================
Defines the contract every hand-tracking backend must satisfy.
New backends just subclass BaseHandTracker and register themselves.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


# ── Data structures ──────────────────────────────────────────

@dataclass
class HandLandmark:
    """A single named keypoint on a detected hand."""
    name: str               # e.g. "index_tip", "wrist", "thumb_tip"
    x: float                # pixel x-coordinate
    y: float                # pixel y-coordinate
    z: float = 0.0          # depth (if available)
    confidence: float = 1.0


@dataclass
class HandDetection:
    """One detected hand with all its landmarks."""
    hand_id: int
    handedness: str                         # "left" | "right" | "unknown"
    landmarks: list[HandLandmark] = field(default_factory=list)
    bounding_box: tuple[float, float, float, float] | None = None  # x1,y1,x2,y2

    def get_landmark(self, name: str) -> HandLandmark | None:
        """Look up a landmark by name (e.g. 'index_tip')."""
        for lm in self.landmarks:
            if lm.name == name:
                return lm
        return None

    def get_point(self, name: str) -> tuple[float, float] | None:
        """Convenience: return (x, y) for the named landmark."""
        lm = self.get_landmark(name)
        return (lm.x, lm.y) if lm else None


# ── Abstract base class ─────────────────────────────────────

class BaseHandTracker(abc.ABC):
    """
    Plug-and-play interface for any hand tracking backend.

    To add a new backend:
      1. Subclass BaseHandTracker.
      2. Implement load_model() and detect().
      3. Register it:  TrackerRegistry.register("my_backend", MyTracker)
    """

    def __init__(self, config: dict):
        self._config = config
        self._max_hands: int = config.get("max_hands", 2)
        self._confidence: float = config.get("confidence_threshold", 0.5)

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load / initialize the underlying model or pipeline."""
        ...

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        """
        Run hand detection on a single BGR frame.

        Returns a list of HandDetection (may be empty if no hands visible).
        """
        ...

    def release(self) -> None:
        """Optional cleanup hook (close handles, free GPU memory, etc.)."""
        pass
