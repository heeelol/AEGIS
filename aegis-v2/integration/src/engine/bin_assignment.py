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

    # MCP knuckle landmarks, the primary anchor for the occlusion gate (the
    # wrist is the fallback). The knuckles track which bin the hand is in.
    _MCP_NAMES = ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp")

    def __init__(self, config: dict):
        self._method: str = config.get("method", "point_in_polygon")
        self._keypoint: str = config.get("hand_keypoint", "index_tip")
        self._overlap_threshold: float = config.get("overlap_threshold", 0.3)
        self._vote_keypoints: list[str] = config.get(
            "vote_keypoints", ["index_tip", "middle_tip"]
        )
        self._vote_confidence_floor: float = config.get("vote_confidence_floor", 0.5)
        self._bins: list[BinRegion] = []

        # Occlusion gate (see docs/superpowers/specs/2026-06-15-occlusion-gate-design.md)
        self._gate_enabled: bool = config.get("occlusion_gate", {}).get("enabled", True)
        # Grid structure precomputed from the bin map by _recompute_grid_structure.
        self._bottom_row: Optional[int] = None
        self._top_rows: set[int] = set()
        self._bottom_bins: list[BinRegion] = []
        self._global_occ_y: Optional[float] = None

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

    def assign(self, hands: list, frame_shape: Optional[tuple] = None) -> list[BinEvent]:
        """
        For each detected hand, determine which bin it is in.

        Uses the configured method:
          - point_in_polygon: check if hand keypoint is inside bin bounds
          - nearest_centroid: assign to closest bin center
          - area_overlap: estimate overlap between hand bbox and bin bbox

        ``frame_shape`` is the camera frame's ``(h, w, ...)`` and is used only by
        the occlusion gate's in-frame check on the wrist. When omitted the gate
        still runs but skips that check (falls back to the MCP centroid anchor).
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
            elif self._method == "finger_vote":
                event = self._assign_vote(hand, hand_area, frame_shape)
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

    # ── Finger-vote assignment ───────────────────────────────

    def _usable_vote_tips(self, hand, frame_shape):
        """Configured vote fingertips that are finite, confident, and in-frame.

        Drops a tip when its landmark is missing, non-finite, below
        ``vote_confidence_floor``, or (when ``frame_shape`` is given) outside the
        frame. Returns a list of (x, y).
        """
        h = w = None
        if frame_shape is not None:
            h, w = frame_shape[0], frame_shape[1]
        tips = []
        for name in self._vote_keypoints:
            lm = hand.get_landmark(name)
            if lm is None:
                continue
            if not (math.isfinite(lm.x) and math.isfinite(lm.y)):
                continue
            if lm.confidence < self._vote_confidence_floor:
                continue
            if w is not None and not (0 <= lm.x <= w and 0 <= lm.y <= h):
                continue
            tips.append((lm.x, lm.y))
        return tips

    def _bin_containing(self, point):
        """First bin whose axis-aligned bounds contain ``point``, or None."""
        px, py = point
        for b in self._bins:
            if b.x_min <= px <= b.x_max and b.y_min <= py <= b.y_max:
                return b
        return None

    @staticmethod
    def _interiority(point, b):
        """Distance from ``point`` to the nearest edge of bin ``b`` (deeper = larger)."""
        px, py = point
        return min(px - b.x_min, b.x_max - px, py - b.y_min, b.y_max - py)

    def _vote_event(self, hand, b, point, hand_area):
        """Build a finger_vote BinEvent for bin ``b`` (None → no-match event)."""
        return BinEvent(
            hand_id=hand.hand_id, handedness=hand.handedness,
            bin_id=(b.bin_id if b else None), bin_label=(b.label if b else None),
            hand_point=point, hand_area=hand_area,
            confidence=(b.confidence if b else 0.0), method="finger_vote",
        )

    def _assign_vote(self, hand, hand_area, frame_shape):
        """Index+middle fingertip voting (tiebreak/centroid added in later tasks).

        Each usable tip inside a bin casts a vote. A single inside tip → its bin.
        No tip inside any bin → no match (centroid fallback added in Task 3).
        No usable tips → hand center.
        """
        tips = self._usable_vote_tips(hand, frame_shape)
        if not tips:
            point = hand.center
            if point is None:
                return self._vote_event(hand, None, (0.0, 0.0), hand_area)
            return self._vote_event(hand, self._bin_containing(point), point, hand_area)

        inside = []
        for t in tips:
            b = self._bin_containing(t)
            if b is not None:
                inside.append((t, b))

        if not inside:
            cx = sum(t[0] for t in tips) / len(tips)
            cy = sum(t[1] for t in tips) / len(tips)
            centroid = (cx, cy)
            return self._vote_event(hand, self._bin_containing(centroid), centroid, hand_area)

        inside.sort(key=lambda tb: (-self._interiority(tb[0], tb[1]), tb[1].bin_id))
        tip, b = inside[0]
        return self._vote_event(hand, b, tip, hand_area)

    # ── Occlusion gate ───────────────────────────────────────

    @staticmethod
    def _parse_row_col(bin_id: str) -> tuple[Optional[int], Optional[int]]:
        """Parse ``bin_{row}_{col}`` → (row, col); (None, None) if it doesn't fit."""
        try:
            _, row, col = bin_id.split("_")
            return int(row), int(col)
        except (AttributeError, ValueError):
            return None, None

    def _recompute_grid_structure(self) -> None:
        """Precompute the bottom-row geometry the occlusion gate needs.

        Bottom row = the largest ``row`` over ``bin_{row}_{col}`` ids; top rows are
        everything above it; ``_bottom_bins`` are the bottom-row regions sorted by
        ``x_min`` (the "beneath the anchor" lookup); ``_global_occ_y`` is the
        highest bottom rim — the fallback occlusion line. A single-row (or
        unparseable) layout leaves ``_top_rows`` empty, making the gate inert.
        """
        rows = [r for r, _ in (self._parse_row_col(b.bin_id) for b in self._bins)
                if r is not None]
        if not rows:
            self._bottom_row = None
            self._top_rows = set()
            self._bottom_bins = []
            self._global_occ_y = None
            return
        self._bottom_row = max(rows)
        self._top_rows = {r for r in rows if r < self._bottom_row}
        self._bottom_bins = sorted(
            (b for b in self._bins
             if self._parse_row_col(b.bin_id)[0] == self._bottom_row),
            key=lambda b: b.x_min,
        )
        self._global_occ_y = min((b.y_min for b in self._bottom_bins), default=None)

    def _occlusion_anchor(
        self, hand, frame_shape: Optional[tuple]
    ) -> Optional[tuple[float, float]]:
        """The landmark that best reports which bin the hand is reaching into.

        The MCP knuckle centroid (index/middle/ring/pinky) when any knuckle is
        finite; the wrist only as a fallback. The knuckles sit at the base of
        the fingers, so they track where the hand actually is. The wrist is too
        proximal — during a genuine reach *over* the shelf lip into a top bin it
        trails back down over the bottom band, even though the hand is up top.
        Anchoring on the wrist there misfires the gate on a legitimate top pick
        (it reads the bottom bin), so the wrist is consulted only when no
        knuckle is available. None when neither is usable.
        """
        xs, ys = [], []
        for name in self._MCP_NAMES:
            lm = hand.get_landmark(name)
            if lm is not None and math.isfinite(lm.x) and math.isfinite(lm.y):
                xs.append(lm.x)
                ys.append(lm.y)
        if xs:
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        wrist = hand.get_landmark("wrist")
        if wrist is not None and math.isfinite(wrist.x) and math.isfinite(wrist.y):
            in_frame = True
            if frame_shape is not None:
                h, w = frame_shape[0], frame_shape[1]
                in_frame = (0 <= wrist.x <= w) and (0 <= wrist.y <= h)
            if in_frame:
                return (wrist.x, wrist.y)
        return None

    def _bottom_bin_at(self, x: float) -> Optional[BinRegion]:
        """The bottom-row bin whose x-extent contains ``x``, or None."""
        for b in self._bottom_bins:
            if b.x_min <= x <= b.x_max:
                return b
        return None

    def _apply_occlusion_gate(
        self, hand, event: BinEvent, frame_shape: Optional[tuple]
    ) -> BinEvent:
        """Reject a fingertip extrapolated under the shelf into a top bin.

        Fires only when the assigned bin is top-row and the hand's proximal anchor
        (wrist/knuckles) sits at or below the bottom-bin rim — physically the arm
        must be reaching into the bottom bin beneath. Reassigns to that bottom bin;
        if the anchor is below the global line but under no bottom bin (an angled
        reach with no target), suppresses the event instead. Otherwise unchanged.
        """
        if not self._gate_enabled or event.bin_id is None or not self._top_rows:
            return event
        row, _ = self._parse_row_col(event.bin_id)
        if row not in self._top_rows:
            return event

        anchor = self._occlusion_anchor(hand, frame_shape)
        if anchor is None:
            return event  # cannot judge
        ax, ay = anchor

        bottom = self._bottom_bin_at(ax)
        if bottom is not None:
            if ay >= bottom.y_min:
                logger.debug("Occlusion gate: %s -> %s (anchor below rim)",
                             event.bin_id, bottom.bin_id)
                return BinEvent(
                    hand_id=event.hand_id, handedness=event.handedness,
                    bin_id=bottom.bin_id, bin_label=bottom.label,
                    hand_point=event.hand_point, hand_area=event.hand_area,
                    confidence=bottom.confidence, method="occlusion_gate",
                )
            return event  # genuine top reach (anchor above the rim)

        # No bottom bin beneath the anchor. A clear bottom reach with no target
        # gets suppressed rather than left as a false top hit.
        if self._global_occ_y is not None and ay >= self._global_occ_y:
            logger.debug("Occlusion gate: suppressing %s (bottom reach, no target)",
                         event.bin_id)
            return BinEvent(
                hand_id=event.hand_id, handedness=event.handedness,
                bin_id=None, bin_label=None,
                hand_point=event.hand_point, hand_area=event.hand_area,
                confidence=0.0, method="occlusion_gate",
            )
        return event
