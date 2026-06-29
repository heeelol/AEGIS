"""
TensorRT INT8 Quantization Script
Converts YOLOv8 PyTorch models to optimized TensorRT engines for edge deployment.
"""

import argparse
import logging
from pathlib import Path

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def quantize_to_tensorrt(
    model_path: str,
    int8: bool = True,
    dynamic_batch: bool = False,
    output_dir: str = "models/custom",
    calibration_data: str = None,
) -> None:
    """
    Convert YOLOv8 model to TensorRT INT8 engine.
    
    Args:
        model_path: Path to trained YOLOv8 model (.pt file)
        int8: Enable INT8 quantization
        dynamic_batch: Allow dynamic batch sizes
        output_dir: Directory for output .engine file
        calibration_data: Optional path to calibration dataset for INT8
    """
    logging.basicConfig(level=logging.INFO)
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load YOLOv8 model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Export to TensorRT
    logger.info("Converting to TensorRT INT8 engine...")
    logger.info(f"  INT8 Quantization: {int8}")
    logger.info(f"  Target: NVIDIA GPU (TensorRT)")
    
    try:
        # Export with TensorRT backend
        engine_path = model.export(
            format="engine",
            imgsz=640,
            half=False,  # Use FP32 for stability
            int8=int8,
            dynamic=dynamic_batch,
            simplify=True,
            optimize=True,
            workspace=4,  # GB - increase for better optimization
        )
        
        logger.info(f"✓ TensorRT engine exported successfully")
        logger.info(f"  Engine path: {engine_path}")
        logger.info(f"  Size: {Path(engine_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        # Benchmark inference speed
        logger.info("Benchmarking inference speed...")
        model_engine = YOLO(str(engine_path))
        results = model_engine.benchmark(imgsz=640, half=False)
        
        logger.info("✓ Quantization complete!")
        return engine_path
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise


def validate_engine(engine_path: str, test_images: str = None) -> None:
    """Validate TensorRT engine inference."""
    logger.info(f"Validating TensorRT engine: {engine_path}")
    
    model = YOLO(engine_path)
    
    if test_images:
        results = model.predict(test_images, conf=0.5)
        logger.info(f"✓ Engine validation successful ({len(results)} images processed)")
    else:
        logger.info("✓ Engine loaded successfully (no test images provided)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 models to TensorRT INT8 engines"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLOv8 model (.pt file)"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        default=True,
        help="Enable INT8 quantization"
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Allow dynamic batch sizes"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/custom",
        help="Output directory for .engine file"
    )
    parser.add_argument(
        "--validate",
        type=str,
        help="Path to test images for validation"
    )
    
    args = parser.parse_args()
    
    engine_path = quantize_to_tensorrt(
        model_path=args.model,
        int8=args.int8,
        dynamic_batch=args.dynamic_batch,
        output_dir=args.output_dir,
    )
    
    if args.validate:
        validate_engine(engine_path, args.validate)


if __name__ == "__main__":
    main()
