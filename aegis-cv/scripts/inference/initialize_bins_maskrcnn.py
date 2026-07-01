"""Initialize bin coordinates using the trained Mask R-CNN model (one-time or continuous).

Mask R-CNN counterpart of initialize_bins.py. The torchvision model is loaded from a
.pth state dict (not the Ultralytics .pt format) and run directly with torch. Input
images are scaled to [0,1] to match T.ToTensor() used in training - feeding [0,255]
produces 0 detections.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Detection confidence threshold (Mask R-CNN score)
CONF_THRESHOLD = 0.5


def load_maskrcnn(model_path, device):
    """Load the trained Mask R-CNN R-50-FPN with a binary (bg + bin) box head."""
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Match the head used during training: 2 classes (background + bin)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, 2)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, 2 * 4)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def predict_maskrcnn(model, image_bgr, device, conf=CONF_THRESHOLD):
    """Run Mask R-CNN on a BGR image and return the raw prediction dict (CPU tensors)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Scale to [0,1] to match T.ToTensor() used in training. torchvision's Mask R-CNN
    # normalizes internally with ImageNet mean/std assuming [0,1] input.
    image_tensor = (torch.from_numpy(image_rgb).float() / 255.0).permute(2, 0, 1).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    # Keep only confident detections
    scores = prediction["scores"].cpu().numpy()
    keep = scores >= conf

    return {
        "boxes": prediction["boxes"].cpu().numpy()[keep],
        "scores": scores[keep],
        "masks": prediction["masks"].cpu().numpy()[keep],  # (N, 1, H, W) in [0,1]
    }


def extract_bin_info(image, prediction):
    """Extract bin information from a Mask R-CNN prediction dict."""

    masks = prediction["masks"]
    boxes = prediction["boxes"]

    if masks is None or len(masks) == 0:
        return None

    img_h, img_w = image.shape[:2]

    bins = []
    for i in range(len(masks)):
        # Mask R-CNN returns full-resolution soft masks (N, 1, H, W); no scaling needed.
        mask = masks[i, 0]
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Bounding box from the detector
        x1, y1, x2, y2 = map(int, boxes[i])
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        # Largest contour as the bin polygon (already in image space)
        largest_contour = max(contours, key=cv2.contourArea)
        polygon = largest_contour.reshape(-1, 2).astype(int).tolist()

        bin_info = {
            "id": i,
            "center": {"x": center_x, "y": center_y},
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": width, "height": height},
            "polygon": polygon,
            "mask": mask_uint8,
            "score": float(prediction["scores"][i]),
            "area": cv2.contourArea(largest_contour),
        }

        bins.append(bin_info)

    return bins if bins else None


def visualize_bins(image, bins):
    """Draw bins on image."""

    display = image.copy()
    for i, bin_info in enumerate(bins):
        x1, y1, x2, y2 = bin_info["bbox"]["x1"], bin_info["bbox"]["y1"], bin_info["bbox"]["x2"], bin_info["bbox"]["y2"]

        # Draw bounding box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw polygon (exact boundary)
        polygon = np.array(bin_info["polygon"], dtype=np.int32)
        cv2.polylines(display, [polygon], True, (255, 0, 0), 2)

        # Draw center
        cx, cy = int(bin_info["center"]["x"]), int(bin_info["center"]["y"])
        cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

        # Label
        label = f"Bin {i} ({bin_info['score']:.2f})"
        cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return display


def initialize_bins_image(image_path, model, device):
    """One-time initialization from single image."""

    logger.info(f"Processing: {Path(image_path).name}")
    image = cv2.imread(str(image_path))

    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None

    prediction = predict_maskrcnn(model, image, device)
    bins = extract_bin_info(image, prediction)

    if bins:
        logger.info(f"✓ Detected {len(bins)} bins")
    else:
        logger.warning("⚠ No bins detected")

    return bins, image


def list_available_cameras():
    """List all available cameras."""
    logger.info("Scanning for available cameras...")
    available_cameras = []

    for idx in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available_cameras.append({
                "index": idx,
                "resolution": f"{width}x{height}"
            })
            cap.release()

    if available_cameras:
        logger.info(f"Found {len(available_cameras)} camera(s):")
        for cam in available_cameras:
            logger.info(f"  Camera {cam['index']}: {cam['resolution']}")
        return available_cameras
    else:
        logger.warning("No cameras found!")
        return []


def initialize_continuous_webcam(model, device, camera_idx=1):
    """Continuous initialization mode - webcam feed."""

    logger.info(f"Opening camera {camera_idx}...")
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)

    if not cap.isOpened():
        logger.warning(f"Camera {camera_idx} not available with DirectShow, trying default backend...")
        cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_idx}!")
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

        # Run inference every N frames for speed (Mask R-CNN is heavier than YOLO)
        if frame_count % 4 == 0:
            prediction = predict_maskrcnn(model, frame, device)
            current_bins = extract_bin_info(frame, prediction)

        # Visualize
        if current_bins:
            display = visualize_bins(frame, current_bins)
            num_bins = len(current_bins)
        else:
            display = frame.copy()
            num_bins = 0

        # Stats
        cv2.putText(display, f"Bins detected: {num_bins} | Frame: {frame_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press 'i' to initialize/save coordinates",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("Bin Initialization (Mask R-CNN) - CONTINUOUS", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"bin_init_{frame_count}.jpg"
            cv2.imwrite(filename, display)
            logger.info(f"✓ Saved: {filename}")
        elif key == ord('i'):
            if current_bins:
                logger.info(f"\n✓ INITIALIZED! Stored {len(current_bins)} bin coordinates")
                for i, bin_info in enumerate(current_bins):
                    logger.info(f"  Bin {i}: center={bin_info['center']}, area={bin_info['area']:.0f}px")
            else:
                logger.warning("⚠ No bins detected yet!")

    cap.release()
    cv2.destroyAllWindows()


def initialize_batch_folder(folder_path, model, device):
    """Process all images in a folder."""

    folder = Path(folder_path)
    images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.JPG")) + sorted(folder.glob("*.png"))

    if not images:
        logger.error(f"No images found in {folder}")
        return

    logger.info(f"Found {len(images)} images\n")

    all_bins = {}
    for img_path in images:
        result = initialize_bins_image(str(img_path), model, device)
        if result:
            bins, image = result
            if not bins:
                continue
            all_bins[img_path.name] = bins

            # Save visualization
            display = visualize_bins(image, bins)
            viz_path = img_path.parent / f"{img_path.stem}_init.jpg"
            cv2.imwrite(str(viz_path), display)

    logger.info(f"\n✓ Batch processing complete!")
    logger.info(f"Processed {len(all_bins)} images with bins")


def main():
    """Test Mask R-CNN segmentation model."""

    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "models" / "custom" / "bin_segmentation_training_5_maskrcnn.pth"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info("Loading trained Mask R-CNN model...")
    logger.info(f"Device: {device}")
    model = load_maskrcnn(model_path, device)
    logger.info("✓ Model loaded\n")

    # Mode selection
    print("\nInitialization modes:")
    print("  1 = Single image")
    print("  2 = Continuous (webcam)")
    print("  3 = Batch (folder)")
    mode = input("\nSelect mode (1-3): ").strip()

    if mode == "1":
        # Single image
        test_image = base_dir / "models" / "data" / "CDE3301.yolov8-seg" / "images" / "test"
        test_images = list(test_image.glob("*.jpg")) + list(test_image.glob("*.JPG"))

        if test_images:
            logger.info("")
            result = initialize_bins_image(str(test_images[0]), model, device)
            if result:
                bins, image = result
                if bins:
                    display = visualize_bins(image, bins)
                    cv2.imshow("Bin Initialization (Mask R-CNN)", display)
                    logger.info("\nPress any key to close...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    elif mode == "2":
        # Continuous - list cameras and let user select
        logger.info("")
        cameras = list_available_cameras()

        if cameras:
            logger.info("")
            for cam in cameras:
                print(f"  {cam['index']} = Camera {cam['index']} ({cam['resolution']})")

            camera_choice = input("\nSelect camera (default is typically 1 for Logitech USB): ").strip()
            camera_idx = int(camera_choice) if camera_choice.isdigit() else 1

            logger.info("")
            initialize_continuous_webcam(model, device, camera_idx=camera_idx)
        else:
            logger.error("No cameras available!")

    elif mode == "3":
        # Batch
        folder = input("Enter folder path: ").strip()
        logger.info("")
        initialize_batch_folder(folder, model, device)

    else:
        logger.error("Invalid mode!")


if __name__ == "__main__":
    main()
