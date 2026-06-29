"""Initialize bins from a native OBB model — RAW boxes (no overlap cutting/shrinking).

Same as initialize_bins_obb.py, but each bin is drawn and saved as its FULL detected
rotated box. Overlapping bins are left overlapping in the drawing; the overlap is only
resolved at attribution time by the nearest-center rule (assign_point). The territory
shading still shows the nearest-center split, but the box OUTLINES are not trimmed.

Use this when you want the original detected boxes untouched. Use initialize_bins_obb.py
if you want each boundary shrunk so the boxes don't overlap.

NOTE: it detects whatever class the OBB model was trained on. Point MODEL_NAME at an
OBB model trained on BINS to use this as a bin initializer.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "project_9_yolov8_obb_1.pt"   # OBB model under models/custom/

# Distinct BGR colors for up to a handful of bins
BIN_COLORS = [
    (0, 200, 255), (0, 255, 0), (255, 100, 0), (255, 0, 200),
    (0, 165, 255), (200, 200, 0), (128, 0, 255), (0, 255, 128),
]


# ─────────────────────── Detection (OBB -> rotated boxes + centers) ───────────────────────

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


def detect_bins(image, model, expected_count=None,
                overlap_thresh=0.35, min_area_frac=0.25):
    """Detect bins and return [{id, corners, center}] from a native OBB model.

    Reads result.obb directly - each detection is already a rotated 4-corner box.
    corners = (4, 2) int array; center = (cx, cy). NO boundary trimming is applied.
    """
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
        logger.info(f"Detections: {len(raw)} raw -> {len(kept)} kept "
                    f"(removed {len(raw) - len(kept)} likely phantom/duplicate)")
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


# ──────────────────────────────── Visualization ────────────────────────────────────

def draw_overlay(image, bins, probe=None):
    """Draw the ownership shading + RAW rotated boxes, centers, labels."""
    display = image.copy()

    if bins:
        owner = compute_ownership(image.shape, bins)
        shade = np.zeros_like(image)
        for b in bins:
            shade[owner == b["id"]] = BIN_COLORS[b["id"] % len(BIN_COLORS)]
        # Blend the territory shading lightly over the image
        display = cv2.addWeighted(display, 0.75, shade, 0.25, 0)

    for b in bins:
        color = BIN_COLORS[b["id"] % len(BIN_COLORS)]
        cv2.polylines(display, [b["corners"]], True, color, 2)   # full rotated box (not trimmed)
        cx, cy = int(b["center"][0]), int(b["center"][1])
        cv2.drawMarker(display, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)
        lx = int(b["corners"][:, 0].min())
        ly = int(b["corners"][:, 1].min())
        cv2.putText(display, f"Bin {b['id']}", (lx, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if probe is not None:
        px, py = int(probe[0]), int(probe[1])
        assigned = assign_point(probe[0], probe[1], bins)
        label = f"({px},{py}) -> " + (f"Bin {assigned}" if assigned is not None else "none")
        cv2.circle(display, (px, py), 7, (0, 0, 255), -1)
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return display


# ──────────────────────────────── Save / load ──────────────────────────────────────

def save_bins(bins, out_path, frame=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"id": b["id"], "corners": b["corners"].tolist(),
             "center": list(b["center"]), "conf": float(b.get("conf", 0.0))}
            for b in bins]
    payload = {"bins": data, "rule": "nearest_center", "source": "obb"}
    if frame is not None:
        payload["frame_h"], payload["frame_w"] = int(frame.shape[0]), int(frame.shape[1])
        snap_path = out_path.parent / "snapshot.jpg"
        cv2.imwrite(str(snap_path), frame)
        logger.info(f"✓ Saved snapshot image -> {snap_path}")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"✓ Saved raw bin boxes + centers -> {out_path}")


# ──────────────────────────────────── Modes ────────────────────────────────────────

def _click_state():
    return {"probe": None}


def _mouse_cb(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["probe"] = (x, y)


def initialize_image(image_path, model, save_path, expected_count=None):
    """Single image: detect bins, show ownership map, click to test attribution."""
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return
    image = cv2.rotate(image, cv2.ROTATE_180)

    bins = detect_bins(image, model, expected_count)
    if not bins:
        logger.warning("No bins detected.")
        return
    logger.info(f"✓ Detected {len(bins)} bins")
    save_bins(bins, save_path, image)

    win = "Bins (OBB raw) - click to test, 'q' to quit"
    cv2.namedWindow(win)
    state = _click_state()
    cv2.setMouseCallback(win, _mouse_cb, state)

    logger.info("Click anywhere to see which bin that point belongs to. 'q' to quit.")
    while True:
        cv2.imshow(win, draw_overlay(image, bins, state["probe"]))
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


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


def initialize_webcam(model, save_path, camera_idx=0, expected_count=None):
    """Webcam: re-detect bins periodically, show ownership map, click to test."""
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_idx}!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "Bins (OBB raw) - click to test, 's' to save, 'q' to quit"
    cv2.namedWindow(win)
    state = _click_state()
    cv2.setMouseCallback(win, _mouse_cb, state)

    logger.info("Click to test a point. 's' = save current bins, 'q' = quit.")
    bins, frame_count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame_count += 1
        if frame_count % 5 == 0 or not bins:
            detected = detect_bins(frame, model, expected_count)
            if detected:
                bins = detected

        cv2.imshow(win, draw_overlay(frame, bins, state["probe"]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and bins:
            save_bins(bins, save_path, frame)

    cap.release()
    cv2.destroyAllWindows()


def initialize_batch(folder_path, model, out_dir, expected_count=None):
    """Batch: save an ownership-map image for every photo in a folder."""
    folder = Path(folder_path)
    images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.JPG")) + sorted(folder.glob("*.png"))
    if not images:
        logger.error(f"No images found in {folder}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.rotate(image, cv2.ROTATE_180)
        bins = detect_bins(image, model, expected_count)
        overlay = draw_overlay(image, bins)
        out = out_dir / f"{img_path.stem}_obb_raw.jpg"
        cv2.imwrite(str(out), overlay)
        logger.info(f"  {img_path.name}: {len(bins)} bins -> {out.name}")
    logger.info("✓ Batch complete")


# ───────────────────────────────────── main ────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "models" / "custom" / MODEL_NAME
    save_path = base_dir / "runs" / "bins_obb_raw" / "bins.json"

    if not model_path.exists():
        logger.error(f"OBB model not found: {model_path}")
        logger.info("Train one with scripts/training/train_project_9_obb.py, "
                    "or set MODEL_NAME to an existing OBB model.")
        return

    logger.info("Loading trained OBB model...")
    model = YOLO(str(model_path))
    names = getattr(model, "names", {})
    logger.info(f"✓ Model loaded — classes: {names}\n")

    # Known bin count is the strongest guard against hallucinated extra detections.
    ec_in = input("How many bins/objects are in view? (press Enter to auto-detect): ").strip()
    expected_count = int(ec_in) if ec_in.isdigit() else None

    print("\nInitialization modes:")
    print("  1 = Single image")
    print("  2 = Continuous (webcam)")
    print("  3 = Batch (folder)")
    mode = input("\nSelect mode (1-3): ").strip()

    if mode == "1":
        test_dir = base_dir / "models" / "data" / "Project 9_1.yolov8-obb" / "train" / "images"
        imgs = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))
        if imgs:
            initialize_image(str(imgs[0]), model, save_path, expected_count)
        else:
            logger.error(f"No images in {test_dir}")

    elif mode == "2":
        cams = list_available_cameras()
        if cams:
            choice = input("\nSelect camera (default 0): ").strip()
            camera_idx = int(choice) if choice.isdigit() else 0
            initialize_webcam(model, save_path, camera_idx=camera_idx, expected_count=expected_count)
        else:
            logger.error("No cameras available!")

    elif mode == "3":
        folder = input("Enter folder path: ").strip()
        initialize_batch(folder, model, save_path.parent, expected_count)

    else:
        logger.error("Invalid mode!")


if __name__ == "__main__":
    main()
