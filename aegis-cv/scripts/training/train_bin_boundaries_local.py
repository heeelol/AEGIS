"""Local GPU-accelerated training for bin boundary detection.

Uses NVIDIA CUDA for fast training on your Acer Nitro V 16 laptop.
"""

import logging
from pathlib import Path

import torch
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def verify_gpu():
    """Verify NVIDIA GPU is available for training."""
    logger.info("=" * 60)
    logger.info("GPU VERIFICATION")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available. Please install PyTorch with CUDA support:")
        logger.error("pip uninstall torch torchvision torchaudio -y")
        logger.error(
            "pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
        raise RuntimeError("CUDA not available")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    logger.info(f"✓ GPU Name: {gpu_name}")
    logger.info(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    logger.info(f"✓ PyTorch Version: {torch.__version__}")
    logger.info("=" * 60)


def train_bin_detector():
    """Train YOLOv8 model for bin boundary detection locally."""
    logger.info("\n" + "=" * 60)
    logger.info("LOCAL BIN BOUNDARY DETECTOR TRAINING")
    logger.info("=" * 60 + "\n")

    # Note: GPU training not available (RTX 5060 sm_120 not supported by PyTorch 2.5.1+cu121)
    # Using CPU training instead - slower but reliable
    logger.info("⚠️  GPU training skipped (sm_120 not supported)")
    logger.info("Training will use CPU instead - this may take 30-60 minutes")
    logger.info("")

    # Dataset path
    dataset_dir = Path(__file__).parent.parent / "models" / "data" / "CDE3301.yolov8-obb"
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        logger.error(f"Dataset not found: {data_yaml}")
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    logger.info(f"Dataset path: {dataset_dir}")
    logger.info(f"Data config: {data_yaml}")

    # Load pretrained YOLOv8 model (nano for fast training on laptop)
    logger.info("\nLoading YOLOv8 OBB model (nano)...")
    model = YOLO("yolov8n-obb.pt")  # OBB = Oriented Bounding Box

    logger.info("Starting training with CPU (GPU incompatible with sm_120 compute capability)...\n")

    # Training with optimized settings for laptop CPU
    # Note: RTX 5060 sm_120 is not supported by PyTorch 2.5.1+cu121
    # CPU training will be slower but will work reliably
    results = model.train(
        data=str(data_yaml),  # Path to dataset.yaml
        epochs=50,  # Number of training epochs
        imgsz=640,  # Image size
        batch=8,  # Reduced batch size for CPU (was 16 for GPU)
        workers=0,  # CPU workers for data loading (disable for CPU training)
        device='cpu',  # Use CPU instead of incompatible GPU
        amp=False,  # Disable Mixed Precision on CPU
        patience=10,  # Early stopping patience
        save=True,  # Save checkpoints
        name="bin_detector",  # Experiment name
        project="runs",  # Project directory
        exist_ok=False,  # Create new directory each run
        verbose=True,
        seed=42,  # Reproducible results
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {results.save_dir}")
    logger.info(f"Best model: {results.save_dir}/weights/best.pt")

    return results


def test_model(model_path: str = "runs/bin_detector/weights/best.pt"):
    """Test the trained model on a sample image."""
    logger.info("\nTesting trained model...")

    if not Path(model_path).exists():
        logger.warning(f"Model not found: {model_path}")
        return

    model = YOLO(model_path)
    logger.info(f"Loaded model: {model_path}")

    # Test on validation set
    val_results = model.val()
    logger.info(f"Validation mAP: {val_results.box.map50}")


if __name__ == "__main__":
    try:
        # Verify GPU is available
        verify_gpu()

        # Run training
        results = train_bin_detector()

        # Test model
        test_model()

    except KeyboardInterrupt:
        logger.info("\n⚠ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise
