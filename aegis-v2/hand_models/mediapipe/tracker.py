"""
MediaPipe Hand Tracker (Tasks API)
===================================
Uses Google AI Edge / MediaPipe Tasks Hand Landmarker for real-time
hand detection, landmark extraction, and grab/release gesture estimation.

This is the primary tested backend. Others are experimental.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from hand_models.common import BaseHandTracker, HandDetection, HandLandmark, TrackerRegistry

logger = logging.getLogger("hand_models.mediapipe")

# MediaPipe Tasks API aliases
BaseOptions = mp.tasks.BaseOptions
MPHandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 21 MediaPipe landmarks
_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def _landmark_confidence(lm) -> float:
    """Per-landmark confidence from a MediaPipe landmark.

    The Tasks-API hand model exposes a ``visibility`` field but leaves it
    unset (``None``) — ``getattr(lm, "visibility", 1.0)`` returns that ``None``
    rather than the default, which then breaks float comparisons downstream.
    Treat an absent or ``None`` visibility as fully confident.
    """
    visibility = getattr(lm, "visibility", None)
    return 1.0 if visibility is None else float(visibility)


class MediaPipeTracker(BaseHandTracker):
    """Hand tracker using MediaPipe Tasks Hand Landmarker (VIDEO mode)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._model_path = config.get("model_path", None)
        self._grab_threshold = float(config.get("grab_threshold_ratio", 1.15))
        self._grab_confirm = max(int(config.get("grab_confirm_frames", 3)), 1)
        self._release_confirm = max(int(config.get("release_confirm_frames", 2)), 1)
        self._landmarker = None
        self._grab_history: dict[str, deque[bool]] = {}
        self._stable_grab: dict[str, bool] = {}
        self._last_ts_ms = 0

    @property
    def name(self) -> str:
        return "MediaPipe Tasks Hand Landmarker"

    def load_model(self) -> None:
        model_path = self._model_path
        if model_path is None:
            # Try to find hand_landmarker.task in common locations
            candidates = [
                Path(__file__).parent / "hand_landmarker.task",
                Path(__file__).parents[2] / "models" / "hand_landmarker.task",
            ]
            for p in candidates:
                if p.exists():
                    model_path = str(p)
                    break

        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(
                "hand_landmarker.task not found. Download from:\n"
                "  https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
                f"Searched: {[str(c) for c in candidates]}"
            )

        logger.info("Loading MediaPipe Hand Landmarker: %s", model_path)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._confidence,
            min_hand_presence_confidence=self._confidence,
            min_tracking_confidence=self._confidence,
        )
        self._landmarker = MPHandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        if self._landmarker is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect_for_video(mp_image, self._next_ts())

        detections: list[HandDetection] = []

        hand_lm_list = getattr(results, "hand_landmarks", None) or []
        world_lm_list = getattr(results, "hand_world_landmarks", None) or []
        handed_list = getattr(results, "handedness", None) or []

        for idx, (hand_lms, handedness_cats) in enumerate(zip(hand_lm_list, handed_list)):
            landmarks = []
            for lm_idx, lm in enumerate(hand_lms):
                landmarks.append(HandLandmark(
                    name=_LANDMARK_NAMES[lm_idx],
                    x=lm.x * w,
                    y=lm.y * h,
                    z=lm.z,
                    confidence=_landmark_confidence(lm),
                ))

            # Handedness
            cat = handedness_cats[0] if handedness_cats else None
            handedness = (
                getattr(cat, "category_name", None)
                or getattr(cat, "display_name", None)
                or "unknown"
            ).lower()

            # Bounding box
            xs = [l.x for l in landmarks]
            ys = [l.y for l in landmarks]
            bbox = (min(xs), min(ys), max(xs), max(ys))

            # Grab detection
            grab_lms = (
                list(world_lm_list[idx])
                if idx < len(world_lm_list) and world_lm_list[idx]
                else hand_lms
            )
            raw_grab, grab_score = self._estimate_grab(grab_lms)
            is_grabbing = self._debounce_grab(handedness, raw_grab)

            detections.append(HandDetection(
                hand_id=idx,
                handedness=handedness,
                landmarks=landmarks,
                bounding_box=bbox,
                is_grabbing=is_grabbing,
                grab_score=grab_score,
            ))

        return detections

    def release(self) -> None:
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception:
                pass
            logger.info("MediaPipe Hand Landmarker released")

    # --- Internal helpers ---

    def _next_ts(self) -> int:
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    @staticmethod
    def _dist3d(a, b) -> float:
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

    def _estimate_grab(self, landmarks) -> tuple[bool, float]:
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm_size = max(self._dist3d(wrist, middle_mcp), 1e-6)

        palm_ids = [0, 5, 9, 13, 17]
        cx = sum(landmarks[i].x for i in palm_ids) / len(palm_ids)
        cy = sum(landmarks[i].y for i in palm_ids) / len(palm_ids)
        cz = sum(landmarks[i].z for i in palm_ids) / len(palm_ids)

        class _Pt:
            def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z

        palm_center = _Pt(cx, cy, cz)
        tip_ids = [4, 8, 12, 16, 20]
        dists = [self._dist3d(landmarks[i], palm_center) / palm_size for i in tip_ids]
        curled = sum(d < self._grab_threshold for d in dists)
        return curled >= 4, curled / len(tip_ids)

    def _debounce_grab(self, key: str, raw: bool) -> bool:
        hist = self._grab_history.get(key)
        if hist is None:
            hist = deque(maxlen=max(self._grab_confirm, self._release_confirm))
            self._grab_history[key] = hist
        hist.append(raw)
        stable = self._stable_grab.get(key, False)
        if len(hist) >= self._grab_confirm and all(list(hist)[-self._grab_confirm:]):
            stable = True
        elif len(hist) >= self._release_confirm and not any(list(hist)[-self._release_confirm:]):
            stable = False
        self._stable_grab[key] = stable
        return stable


# Auto-register
TrackerRegistry.register("mediapipe", MediaPipeTracker)
