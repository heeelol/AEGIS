"""Train YOLOv8 Segmentation on merged COCO datasets - GPU Accelerated."""

import json
import logging
import shutil
import time
from pathlib import Path
from collections import defaultdict

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def merge_datasets_to_yolo():
    """Merge multiple COCO datasets and convert to YOLO format."""
    
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "models" / "data"
    output_dir = data_dir / "CDE3301.yolov8-seg-multi"
    
    # Dataset paths
    datasets = [
        data_dir / "dataset_01_roboflow" / "train",
        data_dir / "dataset_02_aegis" / "train",
        data_dir / "dataset_03_bin_identification" / "train",
    ]
    
    # Filter existing
    datasets = [d for d in datasets if d.parent.exists()]
    
    if not datasets:
        logger.error("No datasets found")
        return None
    
    logger.info(f"Merging {len(datasets)} COCO datasets...")
    for i, ds in enumerate(datasets, 1):
        logger.info(f"  {i}. {ds.parent.name}")
    logger.info("")
    
    # Create YOLO structure
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Load and merge datasets
    all_images = []
    all_annotations = []
    image_id_offset = 0
    
    for dataset_idx, dataset_path in enumerate(datasets, 1):
        logger.info(f"Loading dataset {dataset_idx}...")
        
        coco_json = dataset_path / "_annotations.coco.json"
        if not coco_json.exists():
            logger.warning(f"  Not found: {coco_json}")
            continue
        
        with open(coco_json) as f:
            coco_data = json.load(f)
        
        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        logger.info(f"  {len(images)} images, {len(annotations)} annotations")
        
        # Offset IDs
        for img in images:
            img["id"] += image_id_offset
            img["dataset_idx"] = dataset_idx
            img["original_path"] = dataset_path
        
        for ann in annotations:
            ann["id"] += image_id_offset * 100000
            ann["image_id"] += image_id_offset
        
        all_images.extend(images)
        all_annotations.extend(annotations)
        image_id_offset = max([img["id"] for img in all_images]) // 100000 + 1
    
    logger.info(f"\nMerged: {len(all_images)} images, {len(all_annotations)} annotations")
    
    # Split data
    image_ids = sorted([img["id"] for img in all_images])
    n = len(image_ids)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    splits = {
        "train": set(image_ids[:train_idx]),
        "val": set(image_ids[train_idx:val_idx]),
        "test": set(image_ids[val_idx:])
    }
    
    logger.info(f"Split: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test\n")
    
    # Build lookup
    images_by_id = {img["id"]: img for img in all_images}
    annotations_by_image = defaultdict(list)
    for ann in all_annotations:
        annotations_by_image[ann["image_id"]].append(ann)
    
    # Process each split
    for split_name, img_ids in splits.items():
        logger.info(f"Processing {split_name}...")
        
        img_count = 0
        
        for img_id in img_ids:
            img_info = images_by_id[img_id]
            src_img = Path(img_info["original_path"]) / img_info["file_name"]
            
            if not src_img.exists():
                continue
            
            # Copy image
            dst_img = output_dir / "images" / split_name / img_info["file_name"]
            shutil.copy(src_img, dst_img)
            img_count += 1
            
            # Convert to YOLO format
            img_width = img_info["width"]
            img_height = img_info["height"]
            
            label_lines = []
            for ann in annotations_by_image[img_id]:
                if "segmentation" not in ann or not ann["segmentation"]:
                    continue
                
                seg = ann["segmentation"]
                
                # Skip RLE format
                if isinstance(seg, dict):
                    continue
                
                # Get polygon
                polygon = seg[0] if isinstance(seg, list) and len(seg) > 0 else seg
                
                # Skip invalid
                if not isinstance(polygon, list) or len(polygon) < 6:
                    continue
                
                # Normalize to [0, 1]
                normalized = []
                try:
                    for i in range(0, len(polygon), 2):
                        x = max(0, min(1, polygon[i] / img_width))
                        y = max(0, min(1, polygon[i + 1] / img_height))
                        normalized.append(f"{x:.6f} {y:.6f}")
                except (IndexError, TypeError, ZeroDivisionError):
                    continue
                
                if normalized:
                    label_lines.append(f"0 {' '.join(normalized)}")
            
            # Write label file
            label_file = output_dir / "labels" / split_name / (src_img.stem + ".txt")
            with open(label_file, 'w') as f:
                f.write("\n".join(label_lines))
        
        logger.info(f"  ✓ {img_count} images copied\n")
    
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
    
    logger.info(f"✓ Dataset prepared: {output_dir}")
    logger.info(f"✓ Config saved: {yaml_path}\n")
    
    return str(yaml_path)


def main():
    """Main training pipeline."""
    
    base_dir = Path(__file__).parent.parent.parent
    
    # Prepare merged dataset
    data_yaml = merge_datasets_to_yolo()
    if not data_yaml:
        logger.error("Failed to prepare dataset")
        return
    
    # Load model
    logger.info("Loading YOLOv8 Nano SEGMENTATION model...")
    model = YOLO("yolov8n-seg.pt")
    
    # Train
    logger.info("Starting training on GPU (RTX 5060)...\n")
    start_time = time.time()
    
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        amp=True,
        patience=10,
        save=True,
        project=str(base_dir / "runs" / "segment"),
        name="train-multi"
    )
    
    elapsed = (time.time() - start_time) / 3600
    
    # Save best model
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        dest = base_dir / "models" / "custom" / "bin_segmentation_multi.pt"
        shutil.copy(best_model, dest)
        logger.info(f"\n✓ Best model saved: {dest}")
    
    logger.info(f"Results: {results.save_dir}")
    logger.info(f"Training completed in {elapsed:.2f} hours")


if __name__ == "__main__":
    main()
