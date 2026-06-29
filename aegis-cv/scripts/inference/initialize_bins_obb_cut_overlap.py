"""Initialize bins from a native OBB model, with nearest-center overlap resolution (Fix 1).

This is the OBB counterpart of initialize_bins_nearest.py. Instead of running a
segmentation model and fitting a rotated rectangle to each mask, it reads the
ORIENTED boxes straight from an OBB model's `result.obb` (each detection is already
a rotated 4-corner box).

Everything else matches the _nearest script:
  * assign_point() - the nearest-center rule (Fix 1) for overlapping boxes,
  * an "ownership map" so you can SEE each bin's territory,
  * click anywhere to test which bin a point is assigned to,
  * phantom/duplicate filtering + an optional known bin count.

NOTE: it detects whatever class the OBB model was trained on. The current default
model (project_9_2_obb.pt) has class "Camera-Face-Operator" - point MODEL_NAME at an
OBB model trained on BINS to use this as a bin initializer.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from bin_init_common import (
    _filter_detections, assign_point, list_available_cameras, _click_state, _mouse_cb,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "project_9_yolov8_obb_1_rotate.pt"   # OBB model under models/custom/

# Distinct BGR colors for up to a handful of bins
BIN_COLORS = [
    (0, 200, 255), (0, 255, 0), (255, 100, 0), (255, 0, 200),
    (0, 165, 255), (200, 200, 0), (128, 0, 255), (0, 255, 128),
]


# ─────────────────────── Detection (OBB -> rotated boxes + centers) ───────────────────────


def detect_bins(image, model, expected_count=None,
                overlap_thresh=0.35, min_area_frac=0.25):
    """Detect bins and return [{id, corners, center}] from a native OBB model.

    Reads result.obb directly - each detection is already a rotated 4-corner box, so
    no minAreaRect fitting is needed. corners = (4, 2) int array; center = (cx, cy).

    _filter_detections removes phantom/duplicate boxes. Pass expected_count = the known
    number of bins for the strongest guard.
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
    attach_boundaries(image.shape, kept)   # trim overlaps -> non-overlapping boundaries
    if len(kept) != len(raw):
        logger.info(f"Detections: {len(raw)} raw -> {len(kept)} kept "
                    f"(removed {len(raw) - len(kept)} likely phantom/duplicate)")
    return kept


# ─────────────────────────── Fix 1: nearest-center rule ────────────────────────────


def attach_boundaries(shape, bins, step=0.96, max_iter=30):
    """Give each bin a non-overlapping ROTATED RECTANGLE boundary as b['boundary'].

    Overlapping boxes are shrunk minimally (center + angle kept fixed) until they no
    longer overlap, so each boundary stays a clean 4-corner rectangle — not an
    irregular polygon. Non-overlapping bins keep their full box.
    """
    if not bins:
        return bins

    # Represent each box as a mutable rotated-rect [(cx,cy),(w,h),angle].
    rects = []
    for b in bins:
        (cx, cy), (w, h), a = cv2.minAreaRect(b["corners"].astype(np.float32))
        rects.append([cx, cy, w, h, a])

    def as_cv(r):
        return ((r[0], r[1]), (r[2], r[3]), r[4])

    for _ in range(max_iter):
        overlapping = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                ret, region = cv2.rotatedRectangleIntersection(as_cv(rects[i]), as_cv(rects[j]))
                if ret != 0 and region is not None and cv2.contourArea(region) > 1.0:
                    overlapping = True
                    for k in (i, j):
                        rects[k][2] *= step   # shrink width
                        rects[k][3] *= step   # shrink height
        if not overlapping:
            break

    for b, r in zip(bins, rects):
        b["boundary"] = cv2.boxPoints(as_cv(r)).astype(np.int32)
    return bins


def _bnd(b):
    """The non-overlapping boundary if present, else the raw box corners (as int array)."""
    return np.asarray(b.get("boundary", b["corners"]), dtype=np.int32)


# ──────────────────────────────── Visualization ────────────────────────────────────

def draw_overlay(image, bins, probe=None):
    """Draw the ownership map, boxes, centers, labels. `probe` = (x, y) point to test."""
    display = image.copy()

    if bins:
        shade = np.zeros_like(image)
        for b in bins:
            cv2.fillPoly(shade, [_bnd(b)], BIN_COLORS[b["id"] % len(BIN_COLORS)])
        # Blend the (non-overlapping) territory shading lightly over the image
        display = cv2.addWeighted(display, 0.75, shade, 0.25, 0)

    for b in bins:
        color = BIN_COLORS[b["id"] % len(BIN_COLORS)]
        bnd = _bnd(b)
        cv2.polylines(display, [bnd], True, color, 2)   # trimmed, non-overlapping boundary
        cx, cy = int(b["center"][0]), int(b["center"][1])
        cv2.drawMarker(display, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)
        lx = int(bnd[:, 0].min())
        ly = int(bnd[:, 1].min())
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

def save_bins(bins, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"id": b["id"], "corners": b["corners"].tolist(),
             "boundary": _bnd(b).tolist(),   # non-overlapping boundary
             "center": list(b["center"])} for b in bins]
    with open(out_path, "w") as f:
        json.dump({"bins": data, "rule": "nearest_center", "source": "obb"}, f, indent=2)
    logger.info(f"✓ Saved bin boundaries + centers -> {out_path}")


# ──────────────────────────────────── Modes ────────────────────────────────────────


def initialize_image(image_path, model, save_path, expected_count=None):
    """Single image: detect bins, show ownership map, click to test attribution."""
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return

    bins = detect_bins(image, model, expected_count)
    if not bins:
        logger.warning("No bins detected.")
        return
    logger.info(f"✓ Detected {len(bins)} bins")
    save_bins(bins, save_path)

    win = "Bins (OBB) - click to test, 'q' to quit"
    cv2.namedWindow(win)
    state = _click_state()
    cv2.setMouseCallback(win, _mouse_cb, state)

    logger.info("Click anywhere to see which bin that point belongs to. 'q' to quit.")
    while True:
        cv2.imshow(win, draw_overlay(image, bins, state["probe"]))
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


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

    win = "Bins (OBB) - click to test, 's' to save, 'q' to quit"
    cv2.namedWindow(win)
    state = _click_state()
    cv2.setMouseCallback(win, _mouse_cb, state)

    logger.info("Click to test a point. 's' = save current bins, 'q' = quit.")
    bins, frame_count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
            save_bins(bins, save_path)

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
        bins = detect_bins(image, model, expected_count)
        overlay = draw_overlay(image, bins)
        out = out_dir / f"{img_path.stem}_obb.jpg"
        cv2.imwrite(str(out), overlay)
        logger.info(f"  {img_path.name}: {len(bins)} bins -> {out.name}")
    logger.info("✓ Batch complete")


# ───────────────────────────────────── main ────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "models" / "custom" / MODEL_NAME
    save_path = base_dir / "runs" / "bins_obb" / "bins.json"

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
