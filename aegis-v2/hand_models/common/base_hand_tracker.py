"""
Abstract Hand Tracker Interface
================================
Every hand-tracking backend must implement this contract.
This ensures all models are hot-swappable in the integration pipeline.

To add a new backend:
  1. Create a new folder under hand-models/ (e.g. hand-models/my-model/)
  2. Subclass BaseHandTracker
  3. Implement load_model() and detect()
  4. Register it in hand-models/common/registry.py
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


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
    is_grabbing: bool = False
    grab_score: float = 0.0
    confidence: float = 1.0

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

    @property
    def center(self) -> tuple[float, float] | None:
        """Return the center of the bounding box, or centroid of landmarks."""
        if self.bounding_box:
            x1, y1, x2, y2 = self.bounding_box
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        if self.landmarks:
            xs = [lm.x for lm in self.landmarks]
            ys = [lm.y for lm in self.landmarks]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        return None

    @property
    def area(self) -> float:
        """Approximate hand area from bounding box (pixels^2)."""
        if self.bounding_box:
            x1, y1, x2, y2 = self.bounding_box
            return abs((x2 - x1) * (y2 - y1))
        return 0.0


class BaseHandTracker(abc.ABC):
    """
    Plug-and-play interface for any hand tracking backend.

    All hand models (MediaPipe, YOLO-hand, custom) must implement this.
    """

    def __init__(self, config: dict):
        self._config = config
        self._max_hands: int = config.get("max_hands", 2)
        self._confidence: float = config.get("confidence_threshold", 0.5)

    @property
    def name(self) -> str:
        """Human-readable name for this backend."""
        return self.__class__.__name__

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load / initialize the underlying model or pipeline."""
        ...

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        """
        Run hand detection on a single BGR frame.
        Returns a list of HandDetection (may be empty).
        """
        ...

    def release(self) -> None:
        """Optional cleanup hook (close handles, free GPU memory, etc.)."""
        pass
