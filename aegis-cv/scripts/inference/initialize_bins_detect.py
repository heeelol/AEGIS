"""Test a per-bin DETECT model (B0, B1, ...) on images or a live camera.

Modelled on initialize_bins.py, adapted for a YOLOv8 *detect* model: that
script reads segmentation masks (result.masks); these bin models produce
bounding boxes only (result.boxes), so bin info comes from the boxes directly.

Modes (same UX as initialize_bins.py):
    1 = Single image
    2 = Continuous (webcam)   <- point this at the fixed mounted camera
    3 = Batch (folder)

Usage:
    python scripts/inference/initialize_bins_detect.py --dataset B1
"""

import argparse
import logging
from pathlib import Path

import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Overfit / small models are not calibrated, so keep the threshold lenient.
CONF_THRESHOLD = 0.25


def extract_bin_info(result):
    """Extract bin boxes from a YOLO detect result. Returns a list or None."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    bins = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])
        conf = float(boxes.conf[i])
        bins.append({
            "id": i,
            "center": {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2},
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "width": x2 - x1, "height": y2 - y1},
            "conf": conf,
        })
    return bins


def visualize_bins(image, bins):
    """Draw detected bins (box + center + label + confidence) on a copy."""
    display = image.copy()
    for bin_info in bins:
        b = bin_info["bbox"]
        cv2.rectangle(display, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 2)
        cx, cy = int(bin_info["center"]["x"]), int(bin_info["center"]["y"])
        cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
        label = f"Bin {bin_info['id']} {bin_info['conf']:.2f}"
        cv2.putText(display, label, (b["x1"], b["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return display


def initialize_bins_image(image_path, model):
    """One-time detection from a single image. Returns (bins, image) or None."""
    logger.info("Processing: %s", Path(image_path).name)
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("Failed to load image: %s", image_path)
        return None

    results = model.predict(source=str(image_path), conf=CONF_THRESHOLD,
                            imgsz=640, verbose=False, device="cpu")
    if not results:
        return None

    bins = extract_bin_info(results[0])
    if bins:
        logger.info("✓ Detected %d bins", len(bins))
    else:
        logger.warning("⚠ No bins detected")
    return bins, image


def list_available_cameras():
    """List the first few available cameras."""
    logger.info("Scanning for available cameras...")
    available = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append({"index": idx, "resolution": f"{w}x{h}"})
            cap.release()
    if available:
        logger.info("Found %d camera(s):", len(available))
        for cam in available:
            logger.info("  Camera %s: %s", cam["index"], cam["resolution"])
    else:
        logger.warning("No cameras found!")
    return available


def initialize_continuous_webcam(model, camera_idx=1):
    """Continuous detection on a live camera feed (the fixed mounted camera)."""
    logger.info("Opening camera %s...", camera_idx)
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.warning("Camera %s not available with DirectShow, trying default...", camera_idx)
        cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        logger.error("Failed to open camera %s!", camera_idx)
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    logger.info("✓ Webcam opened")
    logger.info("Controls: 'q'=quit, 's'=save, 'i'=initialize (store coordinates)\n")

    frame_count = 0
    current_bins = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % 4 == 0:  # inference every N frames for speed
            results = model.predict(source=frame, conf=CONF_THRESHOLD,
                                    imgsz=640, verbose=False, device="cpu")
            if results:
                current_bins = extract_bin_info(results[0])

        if current_bins:
            display = visualize_bins(frame, current_bins)
            num_bins = len(current_bins)
        else:
            display = frame.copy()
            num_bins = 0

        cv2.putText(display, f"Bins detected: {num_bins} | Frame: {frame_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press 'i' to initialize/save coordinates",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("Bin Detection - CONTINUOUS", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"bin_detect_{frame_count}.jpg"
            cv2.imwrite(filename, display)
            logger.info("✓ Saved: %s", filename)
        elif key == ord("i"):
            if current_bins:
                logger.info("\n✓ INITIALIZED! Stored %d bin coordinates", len(current_bins))
                for bin_info in current_bins:
                    logger.info("  Bin %s: center=%s, conf=%.2f",
                                bin_info["id"], bin_info["center"], bin_info["conf"])
            else:
                logger.warning("⚠ No bins detected yet!")

    cap.release()
    cv2.destroyAllWindows()


def initialize_batch_folder(folder_path, model):
    """Run detection over every image in a folder, saving *_det.jpg overlays."""
    folder = Path(folder_path)
    images = (sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.JPG"))
              + sorted(folder.glob("*.png")))
    if not images:
        logger.error("No images found in %s", folder)
        return

    logger.info("Found %d images\n", len(images))
    processed = 0
    for img_path in images:
        result = initialize_bins_image(str(img_path), model)
        if result and result[0]:
            bins, image = result
            display = visualize_bins(image, bins)
            cv2.imwrite(str(img_path.parent / f"{img_path.stem}_det.jpg"), display)
            processed += 1

    logger.info("\n✓ Batch complete! Processed %d/%d images with bins",
                processed, len(images))


def default_single_image(dataset):
    """Pick a default test image from the normalized <dataset>.yolo splits."""
    yolo = BASE_DIR / "models" / "data" / f"{dataset}.yolo" / "images"
    for split in ("test", "val", "train"):
        split_dir = yolo / split
        imgs = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png")) \
            if split_dir.exists() else []
        if imgs:
            return imgs[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Test a per-bin detect model.")
    parser.add_argument("--dataset", default="B0", help="Dataset name, e.g. B0, B1")
    args = parser.parse_args()
    dataset = args.dataset

    model_path = BASE_DIR / "runs" / f"{dataset}_detector" / "weights" / "best.pt"
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        logger.error("Train it first: python scripts/training/train_bin_yolo.py --dataset %s", dataset)
        return

    logger.info("Loading %s detect model...", dataset)
    model = YOLO(str(model_path))
    logger.info("✓ Model loaded\n")

    print(f"\n{dataset} detection test modes:")
    print("  1 = Single image")
    print("  2 = Continuous (webcam)")
    print("  3 = Batch (folder)")
    mode = input("\nSelect mode (1-3): ").strip()

    if mode == "1":
        test_image = default_single_image(dataset)
        if not test_image:
            logger.error("No default test image found for %s", dataset)
            return
        result = initialize_bins_image(str(test_image), model)
        if result and result[0]:
            bins, image = result
            cv2.imshow(f"{dataset} Bin Detection", visualize_bins(image, bins))
            logger.info("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif mode == "2":
        cameras = list_available_cameras()
        if cameras:
            for cam in cameras:
                print(f"  {cam['index']} = Camera {cam['index']} ({cam['resolution']})")
            choice = input("\nSelect camera (default 1): ").strip()
            camera_idx = int(choice) if choice.isdigit() else 1
            initialize_continuous_webcam(model, camera_idx=camera_idx)
        else:
            logger.error("No cameras available!")

    elif mode == "3":
        initialize_batch_folder(input("Enter folder path: ").strip(), model)

    else:
        logger.error("Invalid mode!")


if __name__ == "__main__":
    main()
