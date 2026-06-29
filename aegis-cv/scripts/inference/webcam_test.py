"""Simple webcam test for bin boundary detector."""

import logging
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run webcam test."""
    # Configuration
    INFERENCE_SIZE = 224  # Ultra-aggressive for CPU
    FRAME_SKIP = 4  # Process every 4th frame only
    CONF_THRESHOLD = 0.5
    
    # Find model in models/custom/
    model_path = Path(__file__).parent.parent.parent / "models" / "custom" / "bin_detector.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.info("Checking runs directory as fallback...")
        runs_dir = Path(__file__).parent.parent.parent / "runs" / "obb"
        detector_runs = sorted(runs_dir.glob("**/bin_detector*"))
        detector_runs = [d for d in detector_runs if d.is_dir()]
        if not detector_runs:
            logger.error("No trained model found!")
            return
        model_path = detector_runs[-1] / "weights" / "best.pt"
    
    logger.info(f"Model: {model_path.name}\n")
    
    # Load model
    logger.info("Loading model...")
    model = YOLO(str(model_path))
    
    # Force CPU (RTX 5060 sm_120 not compatible with PyTorch CUDA kernels)
    device = "cpu"
    logger.info(f"Using device: {device} (RTX 5060 sm_120 not supported by CUDA)\n")
    model.to(device)
    
    logger.info(f"✓ Ready! (Inference size: {INFERENCE_SIZE}x{INFERENCE_SIZE})\n")
    
    # Open webcam
    logger.info("Opening webcam (DirectShow)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        logger.warning("DirectShow failed, trying default backend...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open webcam!")
            return
    
    # Optimize webcam buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Test first frame
    ret, test_frame = cap.read()
    if not ret:
        logger.error("Webcam opened but cannot read frames!")
        cap.release()
        return
    
    logger.info(f"✓ Webcam opened ({test_frame.shape[1]}x{test_frame.shape[0]})\n")
    logger.info("Controls: 'q'=quit, 's'=save, '+'/'-'=adjust speed\n")
    
    frame_count = 0
    skip_count = 0
    fps_start = time.time()
    fps_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        skip_count += 1
        
        # Skip frames for performance
        if skip_count < FRAME_SKIP:
            continue
        skip_count = 0
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=INFERENCE_SIZE,
            verbose=False,
            device=device,
            half=False
        )
        
        # Annotate
        result = results[0] if results else None
        display = result.plot() if result else frame
        
        # Stats
        detections = len(result.boxes) if result and result.boxes else 0
        
        # FPS counter
        fps_frames += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = fps_frames / elapsed
            fps_start = time.time()
            fps_frames = 0
        else:
            fps = 0
        
        # Draw info
        info_text = f"Frame: {frame_count} | Detections: {detections} | FPS: {fps:.1f}"
        cv2.putText(
            display,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        # Display
        cv2.imshow("Bin Detector", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"frame_{frame_count}.jpg"
            cv2.imwrite(filename, display)
            logger.info(f"✓ Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Done!")


if __name__ == "__main__":
    main()
