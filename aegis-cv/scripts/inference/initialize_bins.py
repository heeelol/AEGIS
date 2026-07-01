"""Initialize bin coordinates using trained segmentation model (one-time or continuous)."""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_bin_info(image, result):
    """Extract bin information from segmentation result."""
    
    if not result.masks:
        return None
    
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    
    # Get scale factors (masks are in model output space, need to scale to image space)
    img_h, img_w = image.shape[:2]
    mask_h, mask_w = masks[0].shape
    scale_x = img_w / mask_w
    scale_y = img_h / mask_h
    
    bins = []
    for i, mask in enumerate(masks):
        # Get mask contour
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Get bounding box
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Get polygon points and scale to image dimensions
        largest_contour = max(contours, key=cv2.contourArea)
        polygon_mask_space = largest_contour.reshape(-1, 2)
        polygon = (polygon_mask_space * np.array([scale_x, scale_y])).astype(int).tolist()
        
        bin_info = {
            "id": i,
            "center": {"x": center_x, "y": center_y},
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": width, "height": height},
            "polygon": polygon,
            "mask": mask_uint8,
            "area": cv2.contourArea(largest_contour)
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
        cv2.putText(display, f"Bin {i}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return display


def initialize_bins_image(image_path, model):
    """One-time initialization from single image."""
    
    logger.info(f"Processing: {Path(image_path).name}")
    image = cv2.imread(str(image_path))
    
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    # Inference
    results = model.predict(
        source=str(image_path),
        conf=0.5,
        imgsz=640,
        verbose=False,
        device='cpu'
    )
    
    if not results:
        return None
    
    result = results[0]
    bins = extract_bin_info(image, result)
    
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


def initialize_continuous_webcam(model, camera_idx=1):
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
        
        # Run inference every N frames for speed
        if frame_count % 4 == 0:
            results = model.predict(
                source=frame,
                conf=0.5,
                imgsz=640,
                verbose=False,
                device='cpu'
            )
            
            if results and results[0].masks:
                current_bins = extract_bin_info(frame, results[0])
        
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
        
        cv2.imshow("Bin Initialization - CONTINUOUS", display)
        
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


def initialize_batch_folder(folder_path, model):
    """Process all images in a folder."""
    
    folder = Path(folder_path)
    images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.JPG")) + sorted(folder.glob("*.png"))
    
    if not images:
        logger.error(f"No images found in {folder}")
        return
    
    logger.info(f"Found {len(images)} images\n")
    
    all_bins = {}
    for img_path in images:
        result = initialize_bins_image(str(img_path), model)
        if result:
            bins, image = result
            all_bins[img_path.name] = bins
            
            # Save visualization
            display = visualize_bins(image, bins)
            viz_path = img_path.parent / f"{img_path.stem}_init.jpg"
            cv2.imwrite(str(viz_path), display)
    
    logger.info(f"\n✓ Batch processing complete!")
    logger.info(f"Processed {len(all_bins)} images with bins")


def main():
    """Test segmentation model."""
    
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "models" / "custom" / "bin_segmentation_training_5_medium.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Load model
    logger.info("Loading trained segmentation model...")
    model = YOLO(str(model_path))
    logger.info("✓ Model loaded\n")
    
    # Mode selection
    print("\nInitialization modes:")
    print("  1 = Single image")
    print("  2 = Continuous (webcam)")
    print("  3 = Batch (folder)")
    mode = input("\nSelect mode (1-3): ").strip()
    
    if mode == "1":
        # Single image
        test_image = base_dir / "models" / "data" / "Project 9.yolov8-seg" / "images" / "test"
        test_images = list(test_image.glob("*.jpg")) + list(test_image.glob("*.JPG"))
        
        if test_images:
            logger.info("")
            result = initialize_bins_image(str(test_images[0]), model)
            if result:
                bins, image = result
                display = visualize_bins(image, bins)
                cv2.imshow("Bin Initialization", display)
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
            initialize_continuous_webcam(model, camera_idx=camera_idx)
        else:
            logger.error("No cameras available!")
    
    elif mode == "3":
        # Batch
        folder = input("Enter folder path: ").strip()
        logger.info("")
        initialize_batch_folder(folder, model)
    
    else:
        logger.error("Invalid mode!")


if __name__ == "__main__":
    main()

