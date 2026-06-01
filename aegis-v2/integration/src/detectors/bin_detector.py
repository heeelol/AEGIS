"""
Bin Detector Adapter
=====================
Wraps the cv-models bin boundary detector for use in the integration pipeline.
Handles both segmentation-polygon and bounding-box fallback modes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("aegis.detectors")

# Allow importing from cv-models
_CV_MODELS_PATH = str(Path(__file__).resolve().parents[3] / "cv-models" / "scripts")
if _CV_MODELS_PATH not in sys.path:
    sys.path.insert(0, _CV_MODELS_PATH)


class BinDetector:
    """
    Detects bins from a frame and produces geofence coordinates.
    Uses the trained YOLOv8-seg model from cv-models/.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.6):
        self._model_path = model_path
        self._conf = conf_threshold
        self._model = None
        self.detected_bins: Dict[str, Dict] = {}

    def _load(self) -> None:
        from ultralytics import YOLO
        logger.info("Loading bin detector: %s", self._model_path)
        self._model = YOLO(self._model_path)

    def detect_bins(self, frame: np.ndarray) -> Dict[str, Dict]:
        """
        Detect bins and return geofence dict:
          {"bin_0_0": {"x_min": .., "x_max": .., "y_min": .., "y_max": .., "confidence": ..}, ...}

        Also stores polygon data when segmentation masks are available.
        """
        if self._model is None:
            self._load()

        results = self._model.predict(frame, conf=self._conf, verbose=False)
        if not results or len(results) == 0 or len(results[0].boxes) == 0:
            logger.warning("No bins detected")
            return {}

        det = results[0]
        boxes = det.boxes.xyxy.cpu().numpy()
        confs = det.boxes.conf.cpu().numpy()

        # Check for segmentation masks
        has_masks = det.masks is not None
        mask_polys = list(det.masks.xy) if has_masks else [None] * len(boxes)

        # Sort spatially (row-major)
        sorted_indices = self._sort_spatially(boxes)
        geofences: Dict[str, Dict] = {}

        for rank, idx in enumerate(sorted_indices):
            x1, y1, x2, y2 = boxes[idx]
            row, col = rank // 4, rank % 4
            bin_id = f"bin_{row}_{col}"
            margin = 3

            entry = {
                "x_min": max(0, int(x1) - margin),
                "x_max": int(x2) + margin,
                "y_min": max(0, int(y1) - margin),
                "y_max": int(y2) + margin,
                "confidence": float(confs[idx]),
            }

            # Attach polygon if available (from segmentation)
            if mask_polys[idx] is not None:
                entry["polygon"] = np.array(mask_polys[idx], dtype=np.float32).tolist()

            geofences[bin_id] = entry

        self.detected_bins = geofences
        logger.info("Detected %d bin(s)", len(geofences))
        return geofences

    def _sort_spatially(self, boxes: np.ndarray) -> list[int]:
        """Sort box indices by row then column."""
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        # Sort by y first, then x
        indices = list(range(len(boxes)))
        indices.sort(key=lambda i: (centers_y[i], centers_x[i]))
        return indices

    def visualize(self, frame: np.ndarray, geofences: Optional[Dict] = None) -> np.ndarray:
        gf = geofences or self.detected_bins
        vis = frame.copy()
        for bid, coords in gf.items():
            # Draw the segmentation polygon when available; otherwise the box.
            poly = coords.get("polygon")
            if poly and len(poly) >= 3:
                pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            else:
                cv2.rectangle(
                    vis,
                    (coords["x_min"], coords["y_min"]),
                    (coords["x_max"], coords["y_max"]),
                    (0, 255, 0), 2,
                )
            cv2.putText(vis, f"{bid} ({coords['confidence']:.2f})",
                        (coords["x_min"], coords["y_min"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis


class DynamicGeofenceManager:
    """Wraps BinDetector with temporal smoothing for coordinate stability."""

    def __init__(self, model_path: str = "yolov8n.pt", smoothing_window: int = 5):
        self.detector = BinDetector(model_path)
        self._window = smoothing_window
        self._history: list[Dict] = []

    def update(self, frame: np.ndarray) -> Dict[str, Dict]:
        gf = self.detector.detect_bins(frame)
        if not gf:
            return self._history[-1] if self._history else {}
        self._history.append(gf)
        if len(self._history) > self._window:
            self._history.pop(0)
        return self._smooth()

    def _smooth(self) -> Dict[str, Dict]:
        if not self._history:
            return {}
        latest = self._history[-1]
        smoothed = {}
        for bid in latest:
            vals = {k: [] for k in ("x_min", "x_max", "y_min", "y_max")}
            for h in self._history:
                if bid in h:
                    for k in vals:
                        vals[k].append(h[bid][k])
            smoothed[bid] = {k: int(np.median(v)) for k, v in vals.items() if v}
            smoothed[bid]["confidence"] = latest[bid].get("confidence", 0.0)
        return smoothed
