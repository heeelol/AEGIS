"""Train a YOLOv8 segmentation model on a bin dataset (COCO format), GPU-accelerated.

Consolidates the former per-dataset scripts — train_training_1..5, the Training_5
size variants (_small/_medium/_large), and project_9 — into one config-driven
trainer. Every dataset shared the *same* COCO->YOLO conversion and training recipe;
only the dataset, base weights, batch size, and output names differed. Those are
the rows of CONFIGS below; the recipe (epochs/imgsz/device/patience/split) is fixed.

Usage:
    python scripts/training/train_seg.py --list                  # show known configs
    python scripts/training/train_seg.py --config training_5
    python scripts/training/train_seg.py --config training_5_large
    python scripts/training/train_seg.py --config project_9 --print-config   # resolve only
"""

import argparse
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# aegis-cv/  (this file is at aegis-cv/scripts/training/train_seg.py)
BASE_DIR = Path(__file__).parent.parent.parent

# Fixed recipe — identical in all nine original scripts.
EPOCHS = 50
IMGSZ = 640
DEVICE = 0          # GPU device 0
PATIENCE = 10
TASK = "segment"


@dataclass(frozen=True)
class SegConfig:
    """The only values that differed between the old per-dataset train scripts."""
    name: str          # run label; -> runs/segment/<name>
    coco_name: str     # dir under models/data/ holding the COCO export (.../train/_annotations.coco.json)
    yolo_name: str     # dir under models/data/ to write the converted YOLO dataset
    weights: str       # base checkpoint (selects model size)
    model_out: str     # filename under models/custom/ for the best weights
    batch: int = 16


# small/medium/large reuse the Training_5 data, varying only the base model size
# (and batch=8 for large, to fit VRAM).
CONFIGS = {
    "training_1":        SegConfig("training_1",        "Training_1.coco-segmentation", "Training_1.yolov8-seg", "yolov8n-seg.pt", "bin_segmentation_training_1.pt"),
    "training_2":        SegConfig("training_2",        "Training_2.coco-segmentation", "Training_2.yolov8-seg", "yolov8n-seg.pt", "bin_segmentation_training_2.pt"),
    "training_3":        SegConfig("training_3",        "Training_3.coco-segmentation", "Training_3.yolov8-seg", "yolov8n-seg.pt", "bin_segmentation_training_3.pt"),
    "training_4":        SegConfig("training_4",        "Training_4.coco-segmentation", "Training_4.yolov8-seg", "yolov8n-seg.pt", "bin_segmentation_training_4.pt"),
    "training_5":        SegConfig("training_5",        "Training_5.coco-segmentation", "Training_5.yolov8-seg", "yolov8n-seg.pt", "bin_segmentation_training_5.pt"),
    "training_5_small":  SegConfig("training_5_small",  "Training_5.coco-segmentation", "Training_5.yolov8-seg", "yolov8s-seg.pt", "bin_segmentation_training_5_small.pt"),
    "training_5_medium": SegConfig("training_5_medium", "Training_5.coco-segmentation", "Training_5.yolov8-seg", "yolov8m-seg.pt", "bin_segmentation_training_5_medium.pt"),
    "training_5_large":  SegConfig("training_5_large",  "Training_5.coco-segmentation", "Training_5.yolov8-seg", "yolov8l-seg.pt", "bin_segmentation_training_5_large.pt", batch=8),
    "project_9":         SegConfig("project_9",         "Project 9.coco-segmentation",  "project_9.yolov8-seg",  "yolov8n-seg.pt", "bin_segmentation_project_9.pt"),
}


def prepare_yolo_dataset(cfg: SegConfig) -> bool:
    """Convert the COCO segmentation export to YOLOv8-seg format and write data.yaml."""
    coco_dir = BASE_DIR / "models" / "data" / cfg.coco_name / "train"
    output_dir = BASE_DIR / "models" / "data" / cfg.yolo_name

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading COCO annotations from {cfg.coco_name}...")
    try:
        with open(coco_dir / "_annotations.coco.json") as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Annotations not found: {coco_dir / '_annotations.coco.json'}")
        return False

    images_by_id = {img["id"]: img for img in coco_data["images"]}
    categories_by_id = {cat["id"]: cat for cat in coco_data.get("categories", [])}

    annotations_by_image: dict = {}
    for ann in coco_data.get("annotations", []):
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    logger.info(f"Found {len(images_by_id)} images")
    logger.info(f"Found {len(categories_by_id)} categories: {[cat['name'] for cat in categories_by_id.values()]}")

    # Deterministic 70/15/15 split by sorted image id.
    image_ids = sorted(images_by_id.keys())
    n = len(image_ids)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    splits = {
        "train": image_ids[:train_idx],
        "val": image_ids[train_idx:val_idx],
        "test": image_ids[val_idx:],
    }
    logger.info(f"Split: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test\n")

    for split_name, img_ids in splits.items():
        logger.info(f"Processing {split_name}...")
        images_with_labels = 0
        for img_id in img_ids:
            img_info = images_by_id[img_id]

            # Clean filename - remove "PNG image " prefix if present (9 chars).
            original_filename = img_info["file_name"]
            if original_filename.startswith("PNG image "):
                clean_filename = original_filename[9:]
            else:
                clean_filename = original_filename

            src = coco_dir / original_filename
            dst_img = output_dir / "images" / split_name / clean_filename
            if not src.exists():
                continue
            shutil.copy(src, dst_img)

            image_height = img_info["height"]
            image_width = img_info["width"]

            yolo_annotations = []
            for ann in annotations_by_image.get(img_id, []):
                category_id = ann["category_id"]
                # COCO is 1-based, YOLO is 0-based.
                yolo_category_id = category_id - 1 if category_id > 0 else 0
                if "segmentation" in ann and ann["segmentation"]:
                    for segmentation in ann["segmentation"]:
                        normalized_coords = []
                        for i in range(0, len(segmentation), 2):
                            x = max(0, min(1, segmentation[i] / image_width))
                            y = max(0, min(1, segmentation[i + 1] / image_height))
                            normalized_coords.extend([f"{x:.6f}", f"{y:.6f}"])
                        if normalized_coords:
                            yolo_annotations.append(f"{yolo_category_id} " + " ".join(normalized_coords))

            if yolo_annotations:
                label_filename = clean_filename.rsplit(".", 1)[0] + ".txt"
                label_file = output_dir / "labels" / split_name / label_filename
                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_annotations))
                images_with_labels += 1

        logger.info(f"  ✓ {len(img_ids)} images processed, {images_with_labels} with labels\n")

    data_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["bin"],
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    logger.info(f"✓ Dataset prepared at {output_dir}")
    logger.info(f"✓ Config saved to {yaml_path}\n")
    return True


def train(cfg: SegConfig) -> None:
    """Prepare the dataset, train YOLOv8-seg, and copy the best weights to models/custom/."""
    from ultralytics import YOLO  # lazy: keeps --list/--print-config cheap

    if not prepare_yolo_dataset(cfg):
        logger.error("Failed to prepare dataset")
        return

    logger.info(f"Loading segmentation base model {cfg.weights}...")
    model = YOLO(cfg.weights)
    logger.info("✓ Segmentation model loaded\n")

    logger.info(f"🚀 Starting SEGMENTATION training on GPU for {cfg.name}...")
    logger.info("Task: Pixel-level mask segmentation")

    data_yaml = BASE_DIR / "models" / "data" / cfg.yolo_name / "data.yaml"
    start_time = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=cfg.batch,
        device=DEVICE,
        amp=True,
        patience=PATIENCE,
        task=TASK,
        verbose=True,
        project=str(BASE_DIR / "runs" / "segment" / cfg.name),
    )
    elapsed = time.time() - start_time

    model_path = BASE_DIR / "models" / "custom" / cfg.model_out
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        shutil.copy(best_model, model_path)
        logger.info(f"\n✓ Best model saved: {model_path}")
    logger.info(f"Results: {results.save_dir}")

    logger.info("\n" + "=" * 60)
    logger.info(f"{cfg.name.upper()} MODEL TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"GPU Training Time: {elapsed/60:.1f} minutes")
    logger.info(f"Dataset: {cfg.coco_name} (COCO-Segmentation format)")
    logger.info(f"Model Saved: {model_path}")
    logger.info("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=sorted(CONFIGS), help="which dataset/model config to train")
    ap.add_argument("--list", action="store_true", help="list known configs and exit")
    ap.add_argument("--print-config", action="store_true", help="print the resolved config and exit (no training)")
    args = ap.parse_args()

    if args.list:
        for name, cfg in CONFIGS.items():
            print(f"{name:20s} weights={cfg.weights:16s} batch={cfg.batch:<3d} -> {cfg.model_out}")
        return

    if not args.config:
        ap.error("--config is required (or use --list)")

    cfg = CONFIGS[args.config]
    if args.print_config:
        print(json.dumps(cfg.__dict__, indent=2))
        return

    train(cfg)


if __name__ == "__main__":
    main()
