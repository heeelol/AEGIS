"""
Overlay UI
==========
OpenCV-based real-time visualization that draws:
  - Bin boundary rectangles (color-coded)
  - Hand landmarks and connections
  - Active-bin highlighting when a hand is inside a bin
  - Status bar showing current bin assignments
"""

from __future__ import annotations

import cv2
import numpy as np

from src.engine.bin_assignment import BinEvent, BinRegion
from src.trackers.base_hand_tracker import HandDetection

# Distinct colors for up to 12 bins (BGR format)
_BIN_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 50),  (50, 200, 150),  (150, 50, 200),
    (200, 200, 200), (128, 0, 0),     (0, 128, 0),
]

# Hand skeleton connections (pairs of landmark indices)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]


class OverlayUI:
    """Renders the visual overlay on each frame."""

    def __init__(self, config: dict, bins: list[BinRegion]):
        self._show_bins = config.get("show_bin_boundaries", True)
        self._show_hands = config.get("show_hand_landmarks", True)
        self._highlight = config.get("highlight_active_bin", True)
        self._alpha = config.get("overlay_alpha", 0.35)
        self._bins = bins

    def render(
        self,
        frame: np.ndarray,
        hands: list[HandDetection],
        events: list[BinEvent],
    ) -> np.ndarray:
        """Draw overlays on a copy of *frame* and return the annotated image."""
        display = frame.copy()

        # Collect IDs of bins currently being reached into
        active_bin_ids = {ev.bin_id for ev in events if ev.bin_id is not None}

        if self._show_bins:
            self._draw_bins(display, active_bin_ids)

        if self._show_hands:
            for hand in hands:
                self._draw_hand(display, hand)

        self._draw_status_bar(display, events)

        return display

    # ── Drawing helpers ──────────────────────────────────────

    def _draw_bins(self, img: np.ndarray, active_ids: set[str]) -> None:
        overlay = img.copy()

        for idx, region in enumerate(self._bins):
            color = _BIN_COLORS[idx % len(_BIN_COLORS)]
            x1, y1 = int(region.x_min), int(region.y_min)
            x2, y2 = int(region.x_max), int(region.y_max)

            # Fill active bins with semi-transparent highlight
            if self._highlight and region.bin_id in active_ids:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # Always draw the border
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label
            cx, cy = int(region.centroid[0]), int(region.centroid[1])
            cv2.putText(img, region.label, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Blend the filled overlay
        cv2.addWeighted(overlay, self._alpha, img, 1 - self._alpha, 0, img)

    def _draw_hand(self, img: np.ndarray, hand: HandDetection) -> None:
        lms = hand.landmarks
        if not lms:
            return

        # Draw connections
        for i, j in _HAND_CONNECTIONS:
            if i < len(lms) and j < len(lms):
                p1 = (int(lms[i].x), int(lms[i].y))
                p2 = (int(lms[j].x), int(lms[j].y))
                cv2.line(img, p1, p2, (0, 255, 0), 1)

        # Draw keypoints
        for lm in lms:
            cv2.circle(img, (int(lm.x), int(lm.y)), 3, (0, 0, 255), -1)

    def _draw_status_bar(self, img: np.ndarray, events: list[BinEvent]) -> None:
        """Semi-transparent bar at the top showing bin assignments."""
        h, w = img.shape[:2]
        bar_h = 40
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        if not events:
            text = "No hands detected"
        else:
            parts = []
            for ev in events:
                if ev.bin_id is not None:
                    parts.append(f"{ev.handedness} hand -> {ev.bin_label}")
                else:
                    parts.append(f"{ev.handedness} hand -> outside")
            text = "  |  ".join(parts)

        cv2.putText(img, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
