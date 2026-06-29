"""Train YOLOv8 Segmentation model on Training_9 dataset (COCO format) - GPU Accelerated."""

import json
import logging
import shutil
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def print_epoch_progress(trainer):
    """Print epoch progress using trainer metrics."""
    if hasattr(trainer, 'epoch') and hasattr(trainer, 'epochs'):
        epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        progress = int((epoch / total_epochs) * 20)
        bar = "█" * progress + "░" * (20 - progress)
        
        metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
        loss_box = metrics.get('loss/box', 0)
        loss_seg = metrics.get('loss/seg', 0)
        loss_cls = metrics.get('loss/cls', 0)
        
        logger.info(f"Epoch {epoch}/{total_epochs} [{bar}] {(epoch/total_epochs)*100:.0f}%")
        logger.info(f"  Loss - Box: {loss_box:.4f} | Seg: {loss_seg:.4f} | Cls: {loss_cls:.4f}")


def prepare_yolo_dataset():
    """Prepare YOLOv8 dataset from Training_9 COCO format."""
    
    base_dir = Path(__file__).parent.parent.parent
    coco_dir = base_dir / "models" / "data" / "Project 9.coco-segmentation" / "train"
    output_dir = base_dir / "models" / "data" / "project_9.yolov8-seg"
    
    # Create YOLO structure
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading COCO annotations from Training_9...")
    
    try:
        with open(coco_dir / "_annotations.coco.json") as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Annotations not found: {coco_dir / '_annotations.coco.json'}")
        return False
    
    # Maps
    images_by_id = {img["id"]: img for img in coco_data["images"]}
    categories_by_id = {cat["id"]: cat for cat in coco_data.get("categories", [])}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    logger.info(f"Found {len(images_by_id)} images")
    logger.info(f"Found {len(categories_by_id)} categories: {[cat['name'] for cat in categories_by_id.values()]}")
    
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
    
    # Convert COCO to YOLO format with segmentation masks
    for split_name, img_ids in splits.items():
        logger.info(f"Processing {split_name}...")
        images_with_labels = 0
        
        for img_id in img_ids:
            img_info = images_by_id[img_id]
            
            # Clean filename - remove "PNG image " prefix if present
            original_filename = img_info["file_name"]
            if original_filename.startswith("PNG image "):
                clean_filename = original_filename[9:]  # Remove "PNG image " (9 characters)
            else:
                clean_filename = original_filename
            
            src = coco_dir / original_filename
            dst_img = output_dir / "images" / split_name / clean_filename
            
            if src.exists():
                shutil.copy(src, dst_img)
                
                # Convert annotations to YOLO format
                image_height = img_info["height"]
                image_width = img_info["width"]
                
                yolo_annotations = []
                for ann in annotations_by_image.get(img_id, []):
                    category_id = ann["category_id"]
                    # Remap category ID: COCO uses 1-based IDs, YOLO uses 0-based
                    yolo_category_id = category_id - 1 if category_id > 0 else 0
                    
                    # Handle segmentation (polygon coordinates)
                    if "segmentation" in ann and ann["segmentation"]:
                        for segmentation in ann["segmentation"]:
                            # segmentation is a list of coordinates [x1, y1, x2, y2, ...]
                            normalized_coords = []
                            for i in range(0, len(segmentation), 2):
                                x = segmentation[i] / image_width
                                y = segmentation[i+1] / image_height
                                # Clamp to [0, 1]
                                x = max(0, min(1, x))
                                y = max(0, min(1, y))
                                normalized_coords.extend([f"{x:.6f}", f"{y:.6f}"])
                            
                            if normalized_coords:
                                yolo_annotations.append(f"{yolo_category_id} " + " ".join(normalized_coords))
                
                # Write YOLO format labels
                if yolo_annotations:
                    # Replace extension with .txt while keeping the rest of the filename intact
                    label_filename = clean_filename.rsplit(".", 1)[0] + ".txt"
                    label_file = output_dir / "labels" / split_name / label_filename
                    with open(label_file, 'w') as f:
                        f.write("\n".join(yolo_annotations))
                    images_with_labels += 1
        
        logger.info(f"  ✓ {len(img_ids)} images processed, {images_with_labels} with labels\n")
    
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
    """Train YOLOv8 Segmentation model on Training_9 dataset with GPU."""
    
    base_dir = Path(__file__).parent.parent.parent
    
    # Prepare dataset
    if not prepare_yolo_dataset():
        logger.error("Failed to prepare dataset")
        return
    
    # Load model
    logger.info("Loading YOLOv8 Nano SEGMENTATION model...")
    model = YOLO('yolov8n-seg.pt')  # Nano segmentation model
    logger.info("✓ Segmentation model loaded\n")
    
    # Train on GPU
    logger.info("🚀 Starting SEGMENTATION training on GPU for Training_9...")
    logger.info("Task: Pixel-level mask segmentation")
    logger.info(f"Device: RTX 5060 (Blackwell with CUDA 12.8)\n")
    
    data_yaml = base_dir / "models" / "data" / "project_9.yolov8-seg" / "data.yaml"
    
    start_time = time.time()
    
    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # Use GPU device 0
        amp=True,  # Automatic mixed precision for faster training
        patience=10,
        task='segment',  # Segmentation task
        verbose=True,
        project=str(base_dir / "runs" / "segment" / "project_9")  # Organize results
    )
    
    elapsed = time.time() - start_time
    
    # Save model
    model_path = base_dir / "models" / "custom" / "bin_segmentation_project_9.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        shutil.copy(best_model, model_path)
        logger.info(f"\n✓ Best model saved: {model_path}")
    
    logger.info(f"Results: {results.save_dir}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING_9 MODEL TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"GPU Training Time: {elapsed/60:.1f} minutes")
    logger.info(f"Dataset: Training_9 (COCO-Segmentation format)")
    logger.info(f"Model Saved: {model_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
