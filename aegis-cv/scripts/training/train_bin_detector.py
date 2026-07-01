"""
Training script for fine-tuning YOLOv8 on bin detection.
Prepares dataset in YOLO format and trains model to detect bin edges/bounding boxes.
"""

import argparse
import logging
from pathlib import Path
import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def prepare_dataset_yaml(dataset_path: str, output_yaml: str = "bin_dataset.yaml") -> None:
    """
    Create dataset YAML for YOLO training.
    Expected directory structure:
        dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
    
    Args:
        dataset_path: Path to dataset directory
        output_yaml: Output YAML filename
    """
    dataset_config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,  # 1 class: bin
        'names': {0: 'bin'}
    }
    
    with open(output_yaml, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    logger.info(f"Created dataset YAML: {output_yaml}")
    print(f"""
Dataset YAML created at: {output_yaml}

Expected directory structure:
{Path(dataset_path).absolute()}
├── images/
│   ├── train/          (annotated training images)
│   ├── val/            (validation images)
│   └── test/           (test images)
└── labels/
    ├── train/          (YOLO format .txt labels)
    ├── val/
    └── test/

To prepare your dataset:
1. Collect images of your bin array
2. Annotate with bounding boxes using LabelImg or Roboflow
3. Export in YOLO format (one .txt file per image)
4. Organize into train/val/test split
    """)


def train_bin_detector(
    dataset_yaml: str,
    model_size: str = "n",  # nano, small, medium, large
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: int = 0,
    patience: int = 20,
) -> YOLO:
    """
    Fine-tune YOLOv8 on bin detection dataset.
    
    Args:
        dataset_yaml: Path to dataset configuration YAML
        model_size: YOLOv8 model size (n/s/m/l/x)
        epochs: Number of training epochs
        batch_size: Training batch size
        imgsz: Input image size
        device: GPU device ID
        patience: Early stopping patience
        
    Returns:
        Trained YOLO model
    """
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Starting YOLOv8 bin detection training...")
    logger.info(f"  Model: YOLOv8{model_size}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Image size: {imgsz}")
    
    # Load base model
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Train on bin dataset
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project="runs/bin_detection",
        name="yolov8_bins",
        patience=patience,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
        amp=True,  # Automatic Mixed Precision
        mosaic=1.0,  # Data augmentation
        flipud=0.5,  # Flip up-down
        fliplr=0.5,  # Flip left-right
        degrees=10,  # Rotation
        hsv_h=0.015,  # HSV hue
        hsv_s=0.7,  # HSV saturation
        hsv_v=0.4,  # HSV value
    )
    
    logger.info(f"✓ Training complete!")
    logger.info(f"  Best model: {results.save_dir}/weights/best.pt")
    logger.info(f"  Results: {results.save_dir}")
    
    return model


def evaluate_model(model_path: str, dataset_yaml: str) -> None:
    """
    Evaluate trained model on test set.
    
    Args:
        model_path: Path to trained model
        dataset_yaml: Dataset YAML path
    """
    logger.info("Evaluating trained model...")
    
    model = YOLO(model_path)
    metrics = model.val(data=dataset_yaml)
    
    logger.info(f"mAP50: {metrics.box.map50:.3f}")
    logger.info(f"mAP50-95: {metrics.box.map:.3f}")


def test_real_time(model_path: str, camera_id: int = 0) -> None:
    """
    Test real-time bin detection on live camera feed.
    
    Args:
        model_path: Path to trained model
        camera_id: Camera device ID (0 for default)
    """
    import cv2
    
    logger.info("Starting real-time bin detection test...")
    logger.info("Press 'q' to quit, 's' to save frame")
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_id}")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Visualize
        annotated = results[0].plot()
        
        # Display stats
        n_detections = len(results[0].boxes)
        cv2.putText(
            annotated,
            f"Detections: {n_detections}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        
        cv2.imshow("Bin Detection", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"test_detection_{frame_count}.jpg", annotated)
            logger.info(f"Saved frame: test_detection_{frame_count}.jpg")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate YOLOv8 for bin detection"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Prepare dataset command
    prep_parser = subparsers.add_parser('prepare', help='Prepare dataset YAML')
    prep_parser.add_argument('--dataset-path', required=True, help='Path to dataset directory')
    prep_parser.add_argument('--output-yaml', default='bin_dataset.yaml', help='Output YAML filename')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train bin detector')
    train_parser.add_argument('--dataset-yaml', required=True, help='Path to dataset YAML')
    train_parser.add_argument('--model-size', default='n', choices=['n', 's', 'm', 'l', 'x'])
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch-size', type=int, default=16)
    train_parser.add_argument('--imgsz', type=int, default=640)
    train_parser.add_argument('--device', type=int, default=0)
    train_parser.add_argument('--patience', type=int, default=20)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--model', required=True, help='Path to trained model')
    eval_parser.add_argument('--dataset-yaml', required=True, help='Path to dataset YAML')
    
    # Test real-time command
    test_parser = subparsers.add_parser('test', help='Real-time detection test')
    test_parser.add_argument('--model', required=True, help='Path to trained model')
    test_parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_dataset_yaml(args.dataset_path, args.output_yaml)
    
    elif args.command == 'train':
        train_bin_detector(
            dataset_yaml=args.dataset_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            patience=args.patience,
        )
    
    elif args.command == 'eval':
        evaluate_model(args.model, args.dataset_yaml)
    
    elif args.command == 'test':
        test_real_time(args.model, args.camera)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
