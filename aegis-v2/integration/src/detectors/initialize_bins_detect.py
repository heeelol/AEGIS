"""Axis-aligned DETECT bin detector for the aegis-v2 pipeline.

Drop-in alternative to ``initialize_bins_obb`` for a standard YOLOv8 **detect**
model (e.g. the aegis-core FSV-series ``best.pt``). It returns the SAME shape the
OBB detector does — ``[{id, corners, center, area, conf}]`` — so GridSession
calibration and bin assignment consume it unchanged. An axis-aligned box is just
a rectangle, so its 4 corners drop straight into the rotated-box machinery.

Selected via ``bin_detector.task: "detect"`` in settings.yaml. Reuses the OBB
module's phantom/duplicate filter and nearest-center rule so behaviour matches.
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import initialize_bins_obb as _obb  # reuse _filter_detections + assign_point

logger = logging.getLogger("aegis.detectors.initialize_bins_detect")

# parents[0]=detectors [1]=src [2]=integration [3]=aegis-v2
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_PATH = _REPO_ROOT / "cv-models" / "models" / "weights" / "fsv3_detect.pt"

# Re-export the nearest-center attribution rule unchanged.
assign_point = _obb.assign_point


def load_model(model_path=None):
    """Load a YOLOv8 detect model. Returns the model, or ``None`` on failure (soft)."""
    p = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    if not p.exists():
        logger.warning("Detect model weights not found: %s", p)
        return None
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover - environment dependent
        logger.warning("ultralytics not installed (%s) — detect model unavailable", e)
        return None
    model = YOLO(str(p))
    logger.info("Loaded DETECT model: %s (classes: %s)", p.name, getattr(model, "names", {}))
    return model


def detect_bins(image, model, expected_count=None,
                overlap_thresh=0.35, min_area_frac=0.25):
    """Detect bins with a DETECT model; return ``[{id, corners, center, area, conf}]``.

    Reads ``result.boxes.xyxy`` (axis-aligned) and expresses each as a 4-corner
    box so the downstream rotated-box code (grid calibration, point-in-polygon
    attribution) works identically. Returns ``[]`` when the model is ``None`` or
    finds nothing.
    """
    if model is None:
        return []

    results = model.predict(source=image, conf=0.25, imgsz=640, verbose=False, device="cpu")
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    import numpy as np

    xyxy = boxes.xyxy.cpu().numpy()                 # (N, 4) [x1, y1, x2, y2]
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None

    raw = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        area = float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
        conf = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        raw.append({"corners": corners, "center": (cx, cy), "area": area, "conf": conf})

    kept = _obb._filter_detections(raw, expected_count, overlap_thresh, min_area_frac)
    for j, b in enumerate(kept):
        b["id"] = j
    if len(kept) != len(raw):
        logger.info("Detections: %d raw -> %d kept (removed %d likely phantom/duplicate)",
                    len(raw), len(kept), len(raw) - len(kept))
    return kept
