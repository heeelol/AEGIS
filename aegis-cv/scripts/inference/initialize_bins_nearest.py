"""Initialize bins as BOUNDING BOXES and resolve overlaps with the nearest-center rule (Fix 1).

Bins next to each other can have overlapping bounding boxes. This script fixes that:
when a point falls inside more than one bin's box, it is assigned to the bin whose
CENTER is closest. The result is that every point on the workstation belongs to exactly
ONE bin - the overlap stops being ambiguous.

It mirrors initialize_bins.py (single image / webcam / batch), but:
  * stores each bin as an ORIENTED (rotated) bounding box + center, fitted to the bin's
    mask with cv2.minAreaRect - so the box hugs a slanted bin (no corner gaps, minimal
    overlap) while staying a simple, stable 4-corner box (not a noisy polygon),
  * provides assign_point() - the actual Fix 1 runtime rule,
  * draws an "ownership map" so you can SEE each bin's territory and how overlaps
    are split down the middle,
  * lets you CLICK anywhere to test which bin that point is assigned to.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from bin_init_common import (
    _filter_detections, assign_point, compute_ownership, list_available_cameras, _click_state, _mouse_cb,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Distinct BGR colors for up to a handful of bins
BIN_COLORS = [
    (0, 200, 255), (0, 255, 0), (255, 100, 0), (255, 0, 200),
    (0, 165, 255), (200, 200, 0), (128, 0, 255), (0, 255, 128),
]


# ─────────────────────── Detection (bins -> boxes + centers) ───────────────────────


def detect_bins(image, model, expected_count=None,
                overlap_thresh=0.35, min_area_frac=0.25):
    """Detect bins and return [{id, corners, center}] using ORIENTED boxes.

    Each bin's segmentation mask is wrapped in the smallest ROTATED rectangle
    (cv2.minAreaRect) so the box hugs a slanted bin - no corner gaps, minimal overlap.
    corners = (4, 2) int array of the rotated box; center = (cx, cy).

    The model sometimes hallucinates an extra bin when bins touch or sit in an
    unfamiliar position. _filter_detections removes those. Pass expected_count = the
    known number of bins for the strongest guard.
    """
    results = model.predict(source=image, conf=0.5, imgsz=640, verbose=False, device="cpu")
    if not results:
        return []

    result = results[0]
    if result.masks is None or len(result.masks) == 0:
        return []

    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else None

    raw = []
    for i, poly in enumerate(result.masks.xy):  # mask outline in original-image pixel coords
        poly = np.asarray(poly, dtype=np.float32)
        if len(poly) < 3:
            continue
        rect = cv2.minAreaRect(poly)                     # ((cx, cy), (w, h), angle)
        corners = cv2.boxPoints(rect).astype(np.int32)   # (4, 2) rotated-box corners
        (cx, cy), (bw, bh), _ = rect
        conf = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        raw.append({"corners": corners, "center": (float(cx), float(cy)),
                    "area": float(bw * bh), "conf": conf})

    kept = _filter_detections(raw, expected_count, overlap_thresh, min_area_frac)
    for j, b in enumerate(kept):
        b["id"] = j
    if len(kept) != len(raw):
        logger.info(f"Detections: {len(raw)} raw -> {len(kept)} kept "
                    f"(removed {len(raw) - len(kept)} likely phantom/duplicate)")
    return kept


# ─────────────────────────── Fix 1: nearest-center rule ────────────────────────────


# ──────────────────────────────── Visualization ────────────────────────────────────

def draw_overlay(image, bins, probe=None):
    """Draw the ownership map, boxes, centers, labels. `probe` = (x, y) point to test."""
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
        cv2.polylines(display, [b["corners"]], True, color, 2)   # rotated box outline
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

def save_bins(bins, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"id": b["id"], "corners": b["corners"].tolist(), "center": list(b["center"])} for b in bins]
    with open(out_path, "w") as f:
        json.dump({"bins": data, "rule": "nearest_center"}, f, indent=2)
    logger.info(f"✓ Saved bin boxes + centers -> {out_path}")


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

    win = "Bins (nearest-center) - click to test, 'q' to quit"
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

    win = "Bins (nearest-center) - click to test, 's' to save, 'q' to quit"
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
        out = out_dir / f"{img_path.stem}_nearest.jpg"
        cv2.imwrite(str(out), overlay)
        logger.info(f"  {img_path.name}: {len(bins)} bins -> {out.name}")
    logger.info("✓ Batch complete")


# ───────────────────────────────────── main ────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "models" / "custom" / "bin_segmentation_training_5_medium.pt"
    save_path = base_dir / "runs" / "bins_nearest" / "bins.json"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    logger.info("Loading trained segmentation model...")
    model = YOLO(str(model_path))
    logger.info("✓ Model loaded\n")

    # Known bin count is the strongest guard against hallucinated extra bins.
    ec_in = input("How many bins are in view? (press Enter to auto-detect): ").strip()
    expected_count = int(ec_in) if ec_in.isdigit() else None

    print("\nInitialization modes:")
    print("  1 = Single image")
    print("  2 = Continuous (webcam)")
    print("  3 = Batch (folder)")
    mode = input("\nSelect mode (1-3): ").strip()

    if mode == "1":
        test_dir = base_dir / "models" / "data" / "CDE3301.yolov8-seg" / "images" / "test"
        imgs = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.JPG"))
        if imgs:
            initialize_image(str(imgs[0]), model, save_path, expected_count)
        else:
            logger.error(f"No test images in {test_dir}")

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
