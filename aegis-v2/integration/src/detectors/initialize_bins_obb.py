"""Native OBB bin detector for the aegis-v2 pipeline.

Adapted from aegis-core's ``scripts/inference/initialize_bins_obb.py`` — the
detection core (RAW rotated boxes, no overlap trimming; nearest-center rule at
attribution time) is kept verbatim. Two things change so the module is
self-contained inside aegis-v2 and usable as a library by the pipeline:

  * ``load_model`` resolves the bundled weights at
    ``cv-models/models/weights/project_9_yolov8_obb_1.pt`` when no path is given,
    and fails *soft* (returns ``None`` + warns) when the weights or ultralytics
    are missing, so the pipeline can still boot.
  * cv2 / numpy / ultralytics are imported lazily inside the functions that need
    them, so ``import initialize_bins_obb`` is cheap and never hard-fails in a
    headless / dependency-light environment.

The pipeline calls ``load_model(path)`` once at startup and ``detect_bins(frame,
model, expected_count)`` per snapshot.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("aegis.detectors.initialize_bins_obb")

MODEL_NAME = "project_9_yolov8_obb_1.pt"

# aegis-v2 repo root: .../aegis-v2/integration/src/detectors/initialize_bins_obb.py
#   parents[0]=detectors  [1]=src  [2]=integration  [3]=aegis-v2
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_PATH = _REPO_ROOT / "cv-models" / "models" / "weights" / MODEL_NAME


# ─────────────────────────── Model loading ───────────────────────────

def load_model(model_path=None):
    """Load the native OBB YOLO model. Returns the model, or ``None`` on failure.

    ``model_path`` may be an absolute/relative path string; when ``None`` the
    bundled weights under ``cv-models/models/weights/`` are used. Missing weights
    or a missing ultralytics install are logged as warnings and yield ``None`` so
    the caller (the pipeline) keeps running without bin detection.
    """
    p = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    if not p.exists():
        logger.warning("OBB model weights not found: %s", p)
        return None
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover - environment dependent
        logger.warning("ultralytics not installed (%s) — OBB model unavailable", e)
        return None
    model = YOLO(str(p))
    logger.info("Loaded OBB model: %s (classes: %s)", p.name, getattr(model, "names", {}))
    return model


# ─────────────── Detection (OBB -> rotated boxes + centers) ───────────────

def _quad_iou(a, b):
    """IoU of two convex quadrilaterals (rotated boxes)."""
    import cv2
    import numpy as np

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    if inter <= 0:
        return 0.0
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def _filter_detections(raw, expected_count, overlap_thresh, min_area_frac):
    """Remove phantom / duplicate detections using priors we already have.

    1) drop detections far smaller than the typical bin (seam phantoms),
    2) suppress overlapping duplicates (keep the higher-confidence one),
    3) if the bin COUNT is known, keep only the strongest N.
    """
    import numpy as np

    if not raw:
        return []

    med_area = float(np.median([b["area"] for b in raw]))
    dets = [b for b in raw if b["area"] >= min_area_frac * med_area]

    dets.sort(key=lambda b: b["conf"], reverse=True)
    kept = []
    for b in dets:
        if all(_quad_iou(b["corners"], k["corners"]) < overlap_thresh for k in kept):
            kept.append(b)

    if expected_count is not None and len(kept) > expected_count:
        kept = kept[:expected_count]   # already sorted by confidence
    return kept


def detect_bins(image, model, expected_count=None,
                overlap_thresh=0.35, min_area_frac=0.25):
    """Detect bins and return ``[{id, corners, center, area, conf}]`` from an OBB model.

    Reads ``result.obb`` directly — each detection is already a rotated 4-corner
    box. ``corners`` = (4, 2) int array; ``center`` = (cx, cy). No boundary
    trimming is applied; overlaps are resolved at attribution time by the
    nearest-center rule. Returns ``[]`` when the model is ``None`` or finds
    nothing.
    """
    if model is None:
        return []

    results = model.predict(source=image, conf=0.5, imgsz=640, verbose=False, device="cpu")
    if not results:
        return []

    obb = results[0].obb
    if obb is None:
        return []

    corners_all = obb.xyxyxyxy.cpu().numpy()      # (N, 4, 2) rotated-box corners (pixels)
    if len(corners_all) == 0:
        return []
    confs = obb.conf.cpu().numpy() if obb.conf is not None else None

    import cv2
    import numpy as np

    raw = []
    for i in range(len(corners_all)):
        corners = corners_all[i].astype(np.int32)            # (4, 2)
        cx = float(corners[:, 0].mean())
        cy = float(corners[:, 1].mean())
        area = float(cv2.contourArea(corners.astype(np.float32)))
        conf = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        raw.append({"corners": corners, "center": (cx, cy), "area": area, "conf": conf})

    kept = _filter_detections(raw, expected_count, overlap_thresh, min_area_frac)
    for j, b in enumerate(kept):
        b["id"] = j
    if len(kept) != len(raw):
        logger.info("Detections: %d raw -> %d kept (removed %d likely phantom/duplicate)",
                    len(raw), len(kept), len(raw) - len(kept))
    return kept


# ───────────────────────── nearest-center rule ─────────────────────────

def assign_point(x, y, bins):
    """Assign point (x, y) to a bin (nearest center among boxes that contain it).

    Returns the bin id, or ``None`` if the point is outside every box. Boxes are
    oriented (rotated), so containment is a point-in-rotated-rect test.
    """
    import cv2

    hits = [
        b for b in bins
        if cv2.pointPolygonTest(b["corners"], (float(x), float(y)), False) >= 0
    ]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]["id"]
    best = min(hits, key=lambda b: (x - b["center"][0]) ** 2 + (y - b["center"][1]) ** 2)
    return best["id"]
