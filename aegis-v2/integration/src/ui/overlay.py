"""
Overlay UI
==========
OpenCV-based real-time visualization drawn on every camera frame.

Renders:
  - Bin boundary rectangles with color-coded fill (active vs inactive)
  - Hand skeleton (21 landmarks + connections)
  - Grab gesture indicator (red = grabbing, blue = open)
  - FSM state badge + gate progress bar
  - Status bar with current bin assignments
  - FPS counter
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from integration.src.engine.bin_assignment import BinEvent, BinRegion
from integration.src.engine.fsm import FSMState

# Up to 12 distinct bin colors (BGR)
_BIN_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 50),  (50, 200, 150),  (150, 50, 200),
    (200, 200, 200), (128, 0, 0),     (0, 128, 0),
]

# FSM state → display color (BGR) and label
_FSM_DISPLAY = {
    FSMState.IDLE:          ((180, 180, 180), "IDLE"),
    FSMState.GATE_1_SPATIAL:((0, 200, 255),   "GATE 1: SPATIAL"),
    FSMState.GATE_2_INTENT: ((0, 180, 255),   "GATE 2: INTENT"),
    FSMState.GATE_3_VERIFY: ((0, 140, 255),   "GATE 3: VERIFY"),
    FSMState.SUCCESS:       ((0, 220, 0),     "SUCCESS"),
    FSMState.ERROR:         ((0, 0, 255),     "ERROR"),
}

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
        fsm_state: FSMState = FSMState.IDLE,
        fsm_info: dict | None = None,
    ) -> np.ndarray:
        """
        Draw all overlays on a copy of *frame* and return the annotated image.

        Parameters
        ----------
        frame : BGR image from camera
        hands : list of HandDetection from the hand tracker
        events : list of BinEvent from the assignment engine
        fsm_state : current FSM state
        fsm_info : dict from fsm.get_state_info() (for elapsed time display)
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

        self._draw_fsm_badge(display, fsm_state, fsm_info)
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

        is_grabbing = getattr(hand, "is_grabbing", False)
        grab_score = getattr(hand, "grab_score", 0.0)

        # Skeleton color: red if grabbing, green if open
        skel_color = (0, 0, 255) if is_grabbing else (0, 255, 0)
        point_color = (0, 80, 255) if is_grabbing else (255, 0, 0)

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
            if is_grabbing:
                label += " GRAB"
            label += f" ({grab_score:.0%})"

            cv2.putText(img, label, (bx1, by1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, skel_color, 2)

    # ── FSM badge ────────────────────────────────────────────

    def _draw_fsm_badge(
        self,
        img: np.ndarray,
        state: FSMState,
        info: dict | None,
    ) -> None:
        """Draw a state badge in the top-right corner with gate progress."""
        h, w = img.shape[:2]
        color, label = _FSM_DISPLAY.get(state, ((180, 180, 180), state.value))

        # Badge background
        badge_w, badge_h = 260, 60
        bx = w - badge_w - 10
        by = 10
        overlay = img.copy()
        cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (bx, by), (bx + badge_w, by + badge_h), color, 2)

        # State label
        cv2.putText(img, label, (bx + 10, by + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Elapsed time & bin ID
        if info:
            elapsed = info.get("elapsed_time", 0)
            bin_id = info.get("bin_id", "—")
            cv2.putText(img, f"Bin: {bin_id}  {elapsed:.1f}s",
                        (bx + 10, by + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Gate progress dots
        gate_states = [FSMState.GATE_1_SPATIAL, FSMState.GATE_2_INTENT, FSMState.GATE_3_VERIFY]
        gate_labels = ["S", "I", "V"]
        current_idx = gate_states.index(state) if state in gate_states else -1

        for i, (gs, gl) in enumerate(zip(gate_states, gate_labels)):
            cx = bx + badge_w - 70 + i * 22
            cy = by + 18
            if state == FSMState.SUCCESS:
                dot_color = (0, 220, 0)
            elif i < current_idx:
                dot_color = (0, 220, 0)      # passed
            elif i == current_idx:
                dot_color = (0, 200, 255)     # current
            else:
                dot_color = (100, 100, 100)   # pending
            cv2.circle(img, (cx, cy), 8, dot_color, -1)
            cv2.putText(img, gl, (cx - 4, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

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
                grab_marker = ""
                if ev.bin_id is not None:
                    parts.append(f"{ev.handedness} -> {ev.bin_label}{grab_marker}")
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
