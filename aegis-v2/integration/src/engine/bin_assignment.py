"""
Bin Assignment Engine
=====================
Given bin geofences (from cv-models) and hand detections (from hand-models),
determines which bin each hand is currently reaching into.

This is the core integration logic: it combines outputs from both
model pipelines to produce bin assignments for kinetic gating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("aegis.engine.assignment")


@dataclass
class BinEvent:
    """Result of assigning one hand to a bin for a single frame."""
    hand_id: int
    handedness: str
    bin_id: Optional[str]
    bin_label: Optional[str]
    hand_point: tuple[float, float]
    hand_area: float
    confidence: float
    method: str                     # "point_in_polygon" | "nearest_centroid" | "area_overlap"


@dataclass
class BinRegion:
    """A bin region defined by a bounding box, with an optional segmentation
    polygon. ``polygon`` is a list of [x, y] points (image pixels) when the
    detector produced a mask; ``None`` for box-only / manual-layout bins."""
    bin_id: str
    label: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    confidence: float = 0.0
    polygon: Optional[list] = None

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


class BinAssignmentEngine:
    """
    Stateless engine: call assign() every frame.
    Determines which bin each hand is in based on hand position and bin boundaries.
    """

    def __init__(self, config: dict):
        self._method: str = config.get("method", "point_in_polygon")
        self._keypoint: str = config.get("hand_keypoint", "index_tip")
        self._overlap_threshold: float = config.get("overlap_threshold", 0.3)
        self._bins: list[BinRegion] = []

    def set_bin_map(self, bins: list[BinRegion]) -> None:
        self._bins = bins
        logger.info("Bin map set: %d region(s)", len(bins))

    def set_bin_map_from_geofences(self, geofences: dict) -> None:
        """Build bin map from geofence dict (as produced by BinDetector)."""
        self._bins = [
            BinRegion(
                bin_id=bid, label=bid,
                x_min=c["x_min"], x_max=c["x_max"],
                y_min=c["y_min"], y_max=c["y_max"],
                confidence=c.get("confidence", 0.0),
                polygon=c.get("polygon"),
            )
            for bid, c in geofences.items()
        ]
        logger.info("Bin map set from geofences: %d region(s)", len(self._bins))

    def assign(self, hands: list) -> list[BinEvent]:
        """
        For each detected hand, determine which bin it is in.

        Uses the configured method:
          - point_in_polygon: check if hand keypoint is inside bin bounds
          - nearest_centroid: assign to closest bin center
          - area_overlap: estimate overlap between hand bbox and bin bbox
        """
        events: list[BinEvent] = []

        for hand in hands:
            point = hand.get_point(self._keypoint)
            if point is None:
                # Fallback to center if specific keypoint missing
                point = hand.center
            if point is None:
                continue

            hand_area = getattr(hand, "area", 0.0)

            if self._method == "area_overlap":
                event = self._assign_overlap(hand, point, hand_area)
            elif self._method == "nearest_centroid":
                event = self._assign_nearest(hand, point, hand_area)
            else:
                event = self._assign_pip(hand, point, hand_area)

            events.append(event)

        return events

    def _assign_pip(self, hand, point: tuple[float, float], hand_area: float) -> BinEvent:
        """Point-in-polygon: is the hand keypoint inside a bin's bounding box?"""
        px, py = point
        for b in self._bins:
            if b.x_min <= px <= b.x_max and b.y_min <= py <= b.y_max:
                return BinEvent(
                    hand_id=hand.hand_id, handedness=hand.handedness,
                    bin_id=b.bin_id, bin_label=b.label,
                    hand_point=point, hand_area=hand_area,
                    confidence=b.confidence, method="point_in_polygon",
                )
        return BinEvent(
            hand_id=hand.hand_id, handedness=hand.handedness,
            bin_id=None, bin_label=None,
            hand_point=point, hand_area=hand_area,
            confidence=0.0, method="point_in_polygon",
        )

    def _assign_nearest(self, hand, point: tuple[float, float], hand_area: float) -> BinEvent:
        """Nearest centroid fallback."""
        if not self._bins:
            return BinEvent(
                hand_id=hand.hand_id, handedness=hand.handedness,
                bin_id=None, bin_label=None,
                hand_point=point, hand_area=hand_area,
                confidence=0.0, method="nearest_centroid",
            )
        px, py = point
        best = min(self._bins, key=lambda b: np.hypot(px - b.centroid[0], py - b.centroid[1]))
        return BinEvent(
            hand_id=hand.hand_id, handedness=hand.handedness,
            bin_id=best.bin_id, bin_label=best.label,
            hand_point=point, hand_area=hand_area,
            confidence=best.confidence, method="nearest_centroid",
        )

    def _assign_overlap(self, hand, point: tuple[float, float], hand_area: float) -> BinEvent:
        """Estimate overlap between hand bounding box and bin regions."""
        bbox = getattr(hand, "bounding_box", None)
        if bbox is None:
            return self._assign_pip(hand, point, hand_area)

        hx1, hy1, hx2, hy2 = bbox
        best_bin = None
        best_overlap = 0.0

        for b in self._bins:
            # Intersection
            ix1 = max(hx1, b.x_min)
            iy1 = max(hy1, b.y_min)
            ix2 = min(hx2, b.x_max)
            iy2 = min(hy2, b.y_max)
            if ix1 < ix2 and iy1 < iy2:
                overlap_area = (ix2 - ix1) * (iy2 - iy1)
                overlap_ratio = overlap_area / max(hand_area, 1.0)
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_bin = b

        if best_bin and best_overlap >= self._overlap_threshold:
            return BinEvent(
                hand_id=hand.hand_id, handedness=hand.handedness,
                bin_id=best_bin.bin_id, bin_label=best_bin.label,
                hand_point=point, hand_area=hand_area,
                confidence=best_bin.confidence, method="area_overlap",
            )

        return BinEvent(
            hand_id=hand.hand_id, handedness=hand.handedness,
            bin_id=None, bin_label=None,
            hand_point=point, hand_area=hand_area,
            confidence=0.0, method="area_overlap",
        )
