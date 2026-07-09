"""Shared helpers for the interactive bin-initializer scripts.

These functions were byte-identical across initialize_bins_nearest.py,
initialize_bins_obb.py and initialize_bins_obb_cut_overlap.py. They are the
backend-agnostic pieces: rotated-box IoU + phantom/duplicate filtering, the
Fix-1 nearest-center attribution (single point + whole-image ownership), camera
enumeration, and the click-probe callback. Each script still owns its own
detect_bins / draw_overlay / save_bins / mode functions (those differ per model).
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────── Detection filtering (rotated boxes) ───────────────────────

def _quad_iou(a, b):
    """IoU of two convex quadrilaterals (rotated boxes)."""
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


# ─────────────────────────── Fix 1: nearest-center rule ────────────────────────────

def assign_point(x, y, bins):
    """Assign point (x, y) to a bin.

    THIS IS FIX 1. If the point is inside several bins' boxes, the bin whose center
    is closest wins. Returns the bin id, or None if the point is outside every box.
    Boxes are oriented (rotated), so containment is a point-in-rotated-rect test.
    """
    hits = [
        b for b in bins
        if cv2.pointPolygonTest(b["corners"], (float(x), float(y)), False) >= 0
    ]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]["id"]

    # Overlap -> nearest center wins
    best = min(hits, key=lambda b: (x - b["center"][0]) ** 2 + (y - b["center"][1]) ** 2)
    return best["id"]


def compute_ownership(shape, bins):
    """For every pixel, which bin owns it (nearest-center among boxes that contain it).

    Returns an int array (h, w): the owning bin id, or -1 if the pixel is in no box.
    This is just assign_point() applied to the whole image, vectorised for drawing.
    """
    h, w = shape[:2]
    owner = np.full((h, w), -1, dtype=np.int32)
    best = np.full((h, w), np.inf, dtype=np.float32)
    ys, xs = np.mgrid[0:h, 0:w]

    for b in bins:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [b["corners"]], 1)        # pixels inside the rotated box
        inside = mask.astype(bool)
        cx, cy = b["center"]
        dist = (xs - cx) ** 2 + (ys - cy) ** 2
        upd = inside & (dist < best)
        owner[upd] = b["id"]
        best[upd] = dist[upd]
    return owner


# ──────────────────────────────── Camera + click probe ─────────────────────────────

def list_available_cameras():
    """List available cameras (same helper style as initialize_bins.py)."""
    logger.info("Scanning for available cameras...")
    cams = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cams.append({"index": idx, "resolution": f"{w}x{h}"})
            cap.release()
    for c in cams:
        logger.info(f"  Camera {c['index']}: {c['resolution']}")
    return cams


def _click_state():
    return {"probe": None}


def _mouse_cb(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["probe"] = (x, y)
