"""
Overlay UI
==========
OpenCV-based real-time visualization drawn on every camera frame.

Renders:
  - Bin boundary rectangles with color-coded fill (active vs inactive)
  - Hand skeleton (21 landmarks + connections)
  - Status bar with the bin each hand is hovering over
  - FPS counter
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from integration.src.engine.bin_assignment import BinEvent, BinRegion

# Up to 12 distinct bin colors (BGR)
_BIN_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 50),  (50, 200, 150),  (150, 50, 200),
    (200, 200, 200), (128, 0, 0),     (0, 128, 0),
]

# Hand skeleton connections (pairs of landmark indices in the 21-point model)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm bridge
]


class OverlayUI:
    """Renders the real-time visual overlay on the camera feed."""

    def __init__(self, config: dict, bins: list[BinRegion]):
        self._show_bins = config.get("show_bin_boundaries", True)
        self._show_hands = config.get("show_hand_landmarks", True)
        self._highlight = config.get("highlight_active_bin", True)
        self._alpha = config.get("overlay_alpha", 0.35)
        self._show_fps = config.get("show_fps", True)
        self._bins = bins

        # FPS tracker
        self._frame_times: deque[float] = deque(maxlen=30)

    def render(
        self,
        frame: np.ndarray,
        hands: list,
        events: list[BinEvent],
    ) -> np.ndarray:
        """
        Draw all overlays on a copy of *frame* and return the annotated image.

        Parameters
        ----------
        frame : BGR image from camera
        hands : list of HandDetection from the hand tracker
        events : list of BinEvent from the assignment engine
        """
        now = time.time()
        self._frame_times.append(now)
        display = frame.copy()

        active_bin_ids = {ev.bin_id for ev in events if ev.bin_id is not None}

        if self._show_bins:
            self._draw_bins(display, active_bin_ids)

        if self._show_hands:
            for hand in hands:
                self._draw_hand(display, hand)

        self._draw_status_bar(display, events)

        if self._show_fps:
            self._draw_fps(display)

        return display

    # ── Bin rendering ────────────────────────────────────────

    def _draw_bins(self, img: np.ndarray, active_ids: set[str]) -> None:
        overlay = img.copy()

        for idx, region in enumerate(self._bins):
            color = _BIN_COLORS[idx % len(_BIN_COLORS)]
            is_active = region.bin_id in active_ids
            thickness = 3 if is_active else 2

            # Prefer the segmentation polygon when the detector supplied one;
            # fall back to the bounding box (manual layout / box-only models).
            pts = self._polygon_points(region)
            if pts is not None:
                if self._highlight and is_active:
                    cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(img, [pts], True, color, thickness)
            else:
                x1, y1 = int(region.x_min), int(region.y_min)
                x2, y2 = int(region.x_max), int(region.y_max)
                if self._highlight and is_active:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Label at centroid
            cx = int((region.x_min + region.x_max) / 2)
            cy = int((region.y_min + region.y_max) / 2)
            cv2.putText(img, region.label, (cx - 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.addWeighted(overlay, self._alpha, img, 1 - self._alpha, 0, img)

    @staticmethod
    def _polygon_points(region) -> np.ndarray | None:
        """Return the region's segmentation polygon as an int32 (N,1,2) array
        for cv2 poly drawing, or None when no usable polygon is present."""
        poly = getattr(region, "polygon", None)
        if not poly or len(poly) < 3:
            return None
        return np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)

    # ── Hand rendering ───────────────────────────────────────

    def _draw_hand(self, img: np.ndarray, hand) -> None:
        lms = hand.landmarks
        if not lms:
            return

        # Single fixed color — we only care about where the hand is, not its grip.
        skel_color = (0, 255, 0)
        point_color = (255, 0, 0)

        # Draw bone connections
        for i, j in _HAND_CONNECTIONS:
            if i < len(lms) and j < len(lms):
                p1 = (int(lms[i].x), int(lms[i].y))
                p2 = (int(lms[j].x), int(lms[j].y))
                cv2.line(img, p1, p2, skel_color, 2)

        # Draw keypoints
        for lm in lms:
            cv2.circle(img, (int(lm.x), int(lm.y)), 4, point_color, -1)

        # Draw bounding box with handedness label
        bbox = getattr(hand, "bounding_box", None)
        if bbox:
            bx1, by1, bx2, by2 = [int(v) for v in bbox]
            cv2.rectangle(img, (bx1, by1), (bx2, by2), skel_color, 1)

            label = hand.handedness.upper()

            cv2.putText(img, label, (bx1, by1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, skel_color, 2)

    # ── Status bar ───────────────────────────────────────────

    def _draw_status_bar(self, img: np.ndarray, events: list[BinEvent]) -> None:
        """Semi-transparent bar at the bottom showing bin assignments."""
        h, w = img.shape[:2]
        bar_h = 40
        bar_y = h - bar_h

        overlay = img.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        if not events:
            text = "No hands detected"
        else:
            parts = []
            for ev in events:
                if ev.bin_id is not None:
                    parts.append(f"{ev.handedness} -> {ev.bin_label}")
                else:
                    parts.append(f"{ev.handedness} -> outside")
            text = "  |  ".join(parts)

        cv2.putText(img, text, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── FPS counter ──────────────────────────────────────────

    def _draw_fps(self, img: np.ndarray) -> None:
        if len(self._frame_times) < 2:
            return
        elapsed = self._frame_times[-1] - self._frame_times[0]
        fps = (len(self._frame_times) - 1) / max(elapsed, 1e-6)
        cv2.putText(img, f"FPS: {fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
