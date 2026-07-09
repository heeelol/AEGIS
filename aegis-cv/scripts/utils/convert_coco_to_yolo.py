"""Convert COCO segmentation format to YOLOv8 format."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def coco_to_yolo_seg(coco_dir, output_dir, val_split=0.15, test_split=0.15):
    """Convert COCO format to YOLOv8 segmentation format."""
    
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)
    
    # Create output structure
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading COCO annotations from {coco_dir / 'train' / '_annotations.coco.json'}")
    
    # Load COCO annotations
    with open(coco_dir / "train" / "_annotations.coco.json") as f:
        coco_data = json.load(f)
    
    # Build image and annotation maps
    images_by_id = {img["id"]: img for img in coco_data["images"]}
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    logger.info(f"Found {len(images_by_id)} images with {len(coco_data['annotations'])} annotations\n")
    
    # Split dataset
    image_ids = list(images_by_id.keys())
    train_ids, temp_ids = train_test_split(image_ids, test_size=val_split+test_split, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=test_split/(val_split+test_split), random_state=42)
    
    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }
    
    logger.info(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test\n")
    
    # Process each split
    for split_name, image_ids_list in splits.items():
        logger.info(f"Processing {split_name}...")
        
        for img_id in image_ids_list:
            img_info = images_by_id[img_id]
            img_filename = img_info["file_name"]
            img_path = coco_dir / "train" / img_filename
            
            if not img_path.exists():
                logger.warning(f"  Image not found: {img_path}")
                continue
            
            # Copy image
            output_img = output_dir / "images" / split_name / img_filename
            import shutil
            shutil.copy(img_path, output_img)
            
            # Get image dimensions
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # Get annotations for this image
            anns = annotations_by_image.get(img_id, [])
            
            if not anns:
                logger.warning(f"  No annotations for {img_filename}")
                continue
            
            # Convert to YOLO format
            yolo_label = ""
            for ann in anns:
                segmentation = ann.get("segmentation", [])
                if not segmentation or not segmentation[0]:
                    continue
                
                # Use first segmentation (polygon)
                polygon = segmentation[0]
                
                # Normalize coordinates (0-1)
                norm_polygon = []
                for i in range(0, len(polygon), 2):
                    x = polygon[i] / w
                    y = polygon[i+1] / h
                    norm_polygon.append(f"{x} {y}")
                
                # Class 0 = bin
                class_id = 0
                yolo_line = f"{class_id} " + " ".join(norm_polygon)
                yolo_label += yolo_line + "\n"
            
            # Save label
            label_path = output_dir / "labels" / split_name / f"{Path(img_filename).stem}.txt"
            with open(label_path, "w") as f:
                f.write(yolo_label)
            
            logger.info(f"  ✓ {img_filename}")
    
    logger.info(f"\n✓ Conversion complete! Output: {output_dir}")


if __name__ == "__main__":
    coco_dir = Path("models/data/Bin identification.coco-segmentation")
    output_dir = Path("models/data/CDE3301.yolov8-seg")
    
    coco_to_yolo_seg(str(coco_dir), str(output_dir))
