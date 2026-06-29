"""
MediaPipe Hands — default hand-tracking backend.
=================================================
Uses the MediaPipe **Tasks** API (HandLandmarker). The legacy
`mp.solutions.hands` API was removed in MediaPipe 0.10.x, so this backend loads
a `hand_landmarker.task` model asset instead.

Model asset (downloaded once to models/pretrained/hand_landmarker.task):
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Swap this backend out by changing `hand_tracker.backend` in settings.yaml to any
other registered name.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .base_hand_tracker import BaseHandTracker, HandDetection, HandLandmark
from .registry import TrackerRegistry

logger = logging.getLogger("aegis.trackers.mediapipe")

# 21 MediaPipe hand landmarks, in the canonical index order (same for Tasks API).
_MP_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Default model-asset location, resolved relative to the repo root
# (src/trackers/mediapipe_tracker.py -> parents[2] == aegis-core).
_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "pretrained" / "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class MediaPipeHandTracker(BaseHandTracker):
    """Hand tracker using Google MediaPipe HandLandmarker (Tasks API)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._landmarker = None
        self._model_path = Path(config.get("model_asset_path", _DEFAULT_MODEL))

    def load_model(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Hand landmark model not found: {self._model_path}\n"
                f"Download it once with:\n  curl -L -o \"{self._model_path}\" {_MODEL_URL}"
            )

        logger.info("Initializing MediaPipe HandLandmarker (max_hands=%d)", self._max_hands)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._confidence,
            min_hand_presence_confidence=self._confidence,
            min_tracking_confidence=self._confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        # Keep a handle to mp for building Image objects in detect()
        self._mp = mp

    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        if self._landmarker is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        detections: list[HandDetection] = []
        if not result.hand_landmarks:
            return detections

        for idx, hand_lms in enumerate(result.hand_landmarks):
            # Handedness (Tasks API returns a parallel list of Category lists)
            handedness = "unknown"
            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                handedness = result.handedness[idx][0].category_name.lower()

            landmarks = []
            for lm_idx, lm in enumerate(hand_lms):
                name = _MP_LANDMARK_NAMES[lm_idx] if lm_idx < len(_MP_LANDMARK_NAMES) else f"lm_{lm_idx}"
                landmarks.append(HandLandmark(
                    name=name,
                    x=lm.x * w,
                    y=lm.y * h,
                    z=getattr(lm, "z", 0.0),
                    confidence=getattr(lm, "visibility", 1.0) or 1.0,
                ))

            xs = [l.x for l in landmarks]
            ys = [l.y for l in landmarks]
            bbox = (min(xs), min(ys), max(xs), max(ys))

            detections.append(HandDetection(
                hand_id=idx,
                handedness=handedness,
                landmarks=landmarks,
                bounding_box=bbox,
            ))

        return detections

    def release(self) -> None:
        if self._landmarker:
            self._landmarker.close()
            logger.info("MediaPipe HandLandmarker released")


# ── Auto-register on import ─────────────────────────────────
TrackerRegistry.register("mediapipe", MediaPipeHandTracker)
