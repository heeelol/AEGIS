"""
Fine-tuning Script for Custom Hand Gesture Recognition
Trains YOLOv8-Pose model on custom "Grab" gesture dataset.
"""

import argparse
import logging
from pathlib import Path

import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def train_custom_gesture(
    data_yaml: str,
    model_name: str = "yolov8n-pose",
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: int = 0,
    project: str = "models/custom",
) -> None:
    """
    Fine-tune YOLOv8-Pose on custom gesture dataset.
    
    Args:
        data_yaml: Path to dataset YAML (Roboflow format)
        model_name: Base YOLOv8 model (yolov8n-pose, yolov8m-pose, etc.)
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Training batch size
        device: GPU device ID
        project: Output project directory
    """
    logging.basicConfig(level=logging.INFO)
    
    # Load base model
    logger.info(f"Loading base model: {model_name}")
    model = YOLO(f"{model_name}.pt")
    
    # Train on custom dataset
    logger.info(f"Starting fine-tuning on custom gesture dataset")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name="grab_pose",
        patience=20,  # Early stopping
        save=True,
        save_period=5,
        verbose=True,
    )
    
    logger.info(f"Training complete. Best model saved to: {results.save_dir}")
    
    # Validate model
    logger.info("Running validation...")
    val_results = model.val()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for custom gestures")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file (Roboflow format)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-pose",
        help="Base model variant"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID"
    )
    
    args = parser.parse_args()
    
    train_custom_gesture(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
