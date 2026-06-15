"""
Bin Assignment Engine
=====================
Given bin geofences (from cv-models) and hand detections (from hand-models),
determines which bin each hand is currently reaching into.

This is the core integration logic: it combines outputs from both
model pipelines to produce bin assignments, which the dashboard uses to
signal whether a hand is hovering the right or wrong bin.
"""

from __future__ import annotations

import logging
import math
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
        gate_cfg = config.get("occlusion_gate", {}) or {}
        self._gate_enabled: bool = gate_cfg.get("enabled", True)
        self._bins: list[BinRegion] = []
        # Grid structure for the occlusion gate, recomputed on every bin-map
        # change. Empty until a multi-row bin map is set.
        self._top_rows: set[int] = set()
        self._bottom_bins: list[BinRegion] = []
        self._global_occ_y: float = 0.0

    def set_bin_map(self, bins: list[BinRegion]) -> None:
        self._bins = bins
        self._recompute_grid_structure()
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
        self._recompute_grid_structure()
        logger.info("Bin map set from geofences: %d region(s)", len(self._bins))

    @staticmethod
    def _bin_row(bin_id: str) -> int:
        """Row index from 'bin_{row}_{col}'. -1 when unparseable."""
        parts = bin_id.split("_")
        try:
            return int(parts[-2])
        except (ValueError, IndexError):
            return -1

    def _recompute_grid_structure(self) -> None:
        """Precompute the top rows, bottom-row bins (sorted by x), and the
        global bottom-band rim used by the occlusion gate. Inert (everything
        empty) for single-row or unparseable layouts."""
        rows = {self._bin_row(b.bin_id) for b in self._bins}
        rows.discard(-1)
        if len(rows) < 2:
            self._top_rows = set()
            self._bottom_bins = []
            self._global_occ_y = 0.0
            return
        bottom_row = max(rows)
        self._top_rows = {r for r in rows if r < bottom_row}
        self._bottom_bins = sorted(
            (b for b in self._bins if self._bin_row(b.bin_id) == bottom_row),
            key=lambda b: b.x_min,
        )
        self._global_occ_y = min(b.y_min for b in self._bottom_bins)

    def _occlusion_anchor(self, hand, frame_shape):
        """Most-proximal reliably-available landmark for the gate: the wrist if
        finite and (when frame_shape is given) in-frame, else the centroid of the
        finite MCP knuckles. Returns (x, y) or None."""
        wrist = hand.get_landmark("wrist")
        if wrist is not None and math.isfinite(wrist.x) and math.isfinite(wrist.y):
            in_frame = True
            if frame_shape is not None:
                fh, fw = frame_shape[0], frame_shape[1]
                in_frame = (0 <= wrist.x <= fw) and (0 <= wrist.y <= fh)
            if in_frame:
                return (wrist.x, wrist.y)

        pts = []
        for name in ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"):
            lm = hand.get_landmark(name)
            if lm is not None and math.isfinite(lm.x) and math.isfinite(lm.y):
                pts.append((lm.x, lm.y))
        if not pts:
            return None
        n = len(pts)
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)

    def _bottom_bin_at(self, x: float):
        """Bottom-row bin whose x-range contains x, or None."""
        for b in self._bottom_bins:
            if b.x_min <= x <= b.x_max:
                return b
        return None

    def _apply_occlusion_gate(self, hand, event, frame_shape):
        """Reject an extrapolated fingertip that fell into a top bin while the
        hand's proximal anchor sits at/below the bottom-band rim. Reassign to the
        bottom bin beneath, or suppress when there is clearly a bottom reach but
        no bin to assign to. Returns the (possibly rewritten) event."""
        if not self._gate_enabled or event.bin_id is None or not self._top_rows:
            return event
        if self._bin_row(event.bin_id) not in self._top_rows:
            return event

        anchor = self._occlusion_anchor(hand, frame_shape)
        if anchor is None:
            return event
        ax, ay = anchor

        below = self._bottom_bin_at(ax)
        if below is None:
            below = self._bottom_bin_at(event.hand_point[0])

        # Per-column rim (primary): anchor at/below the bin's cut-off rim.
        if below is not None and ay >= below.y_min:
            return BinEvent(
                hand_id=event.hand_id, handedness=event.handedness,
                bin_id=below.bin_id, bin_label=below.label,
                hand_point=event.hand_point, hand_area=event.hand_area,
                confidence=below.confidence, method="occlusion_gate",
            )

        # Global fallback: clearly a bottom reach but no bin beneath → suppress.
        if below is None and ay >= self._global_occ_y:
            logger.info(
                "Occlusion gate suppressed false top hit %s (anchor x=%.1f off all bottom bins)",
                event.bin_id, ax,
            )
            return BinEvent(
                hand_id=event.hand_id, handedness=event.handedness,
                bin_id=None, bin_label=None,
                hand_point=event.hand_point, hand_area=event.hand_area,
                confidence=0.0, method="occlusion_gate",
            )

        return event

    def assign(self, hands: list, frame_shape=None) -> list[BinEvent]:
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

            event = self._apply_occlusion_gate(hand, event, frame_shape)
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
