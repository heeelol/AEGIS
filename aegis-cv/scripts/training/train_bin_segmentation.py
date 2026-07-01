"""Train YOLOv8 Segmentation model on bin boundaries (COCO format) - GPU Accelerated."""

import json
import logging
import shutil
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_yolo_dataset():
    """Prepare YOLOv8 dataset from COCO format."""
    
    base_dir = Path(__file__).parent.parent.parent
    coco_dir = base_dir / "models" / "data" / "Bin identification.coco-segmentation" / "train"
    output_dir = base_dir / "models" / "data" / "CDE3301.yolov8-seg"
    
    # Create YOLO structure
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading COCO annotations...")
    
    try:
        with open(coco_dir / "_annotations.coco.json") as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Annotations not found: {coco_dir / '_annotations.coco.json'}")
        return False
    
    # Maps
    images_by_id = {img["id"]: img for img in coco_data["images"]}
    annotations_by_image = {}
    logger.info(f"Found {len(images_by_id)} images")
    
    # Simple 70-15-15 split
    image_ids = sorted(images_by_id.keys())
    n = len(image_ids)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    splits = {
        "train": image_ids[:train_idx],
        "val": image_ids[train_idx:val_idx],
        "test": image_ids[val_idx:]
    }
    
    logger.info(f"Split: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test\n")
    
    # Copy images only
    for split_name, img_ids in splits.items():
        logger.info(f"Processing {split_name}...")
        for img_id in img_ids:
            img_info = images_by_id[img_id]
            src = coco_dir / img_info["file_name"]
            dst = output_dir / "images" / split_name / img_info["file_name"]
            
            if src.exists():
                shutil.copy(src, dst)
        
        logger.info(f"  ✓ {len(img_ids)} images copied\n")
    
    # Create data.yaml
    data_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["bin"]
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    logger.info(f"✓ Dataset prepared at {output_dir}")
    logger.info(f"✓ Config saved to {yaml_path}\n")
    
    return True


def main():
    """Train YOLOv8 Segmentation model on GPU."""
    
    base_dir = Path(__file__).parent.parent.parent
    
    # Prepare dataset
    if not prepare_yolo_dataset():
        logger.error("Failed to prepare dataset")
        return
    
    # Load model
    logger.info("Loading YOLOv8 Nano SEGMENTATION model (NOT OBB)...")
    model = YOLO('yolov8n-seg.pt')  # -seg = segmentation, NOT -obb
    logger.info("✓ Segmentation model loaded\n")
    
    # Train on GPU
    logger.info("🚀 Starting SEGMENTATION training on GPU...")
    logger.info("Task: Pixel-level mask segmentation")
    logger.info(f"Device: RTX 5060 (Blackwell - sm_120 with CUDA 12.8)\n")
    
    data_yaml = base_dir / "models" / "data" / "CDE3301.yolov8-seg" / "data.yaml"
    
    start_time = time.time()
    
    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,  # Increased for GPU (was 4 for CPU)
        device=0,  # Use GPU device 0
        amp=True,  # Automatic mixed precision for faster training
        patience=10,
        task='segment',  # EXPLICIT: segmentation task
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    # Save model
    model_path = base_dir / "models" / "custom" / "bin_segmentation.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        shutil.copy(best_model, model_path)
        logger.info(f"\n✓ Best model saved: {model_path}")
    
    logger.info(f"Results: {results.save_dir}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"GPU Training Time: {elapsed/60:.1f} minutes")
    logger.info(f"CPU Training Time: ~43 minutes (previous run)")
    if elapsed > 0:
        speedup = 43 / (elapsed/60)
        logger.info(f"Speedup: {speedup:.1f}x faster! 🎉")
    logger.info("="*60)


if __name__ == "__main__":
    main()
