"""Normalize a per-bin Roboflow dataset into YOLO detect format.

Handles BOTH export formats Roboflow produces:
  * COCO     (``<src>/<split>/_annotations.coco.json`` + images)
  * YOLOv8   (``<src>/<split>/images`` + ``<src>/<split>/labels`` + data.yaml)

Everything is collapsed to a single class (``0 = bin``) and written to:

    models/data/<DATASET>.yolo/
    ├── images/{train,val[,test]}
    ├── labels/{train,val[,test]}
    └── data_<DATASET>.yaml

Splits are auto-detected (Roboflow ``valid`` -> YOLO ``val``). If no validation
split exists, val is mirrored from train as a SMOKE-TEST fallback (with a warning).

The source folder is resolved from ``--source`` if given, else by trying
``<DATASET>.coco``, ``<DATASET>.yolov8``, then ``<DATASET>``. Use ``--source``
when the folder name has spaces or a suffix that differs from the dataset key,
e.g. key ``FSV1`` <- folder ``Final Setup V1.yolov8``.

Usage:
    python scripts/training/prepare_bin_dataset.py --dataset B1
    python scripts/training/prepare_bin_dataset.py --dataset FSV1 --source "Final Setup V1.yolov8"
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "models" / "data"

# Source split folder name -> YOLO split name
SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


# --------------------------------------------------------------------------- #
# Source resolution / format detection
# --------------------------------------------------------------------------- #
def resolve_source(dataset, source):
    """Find the source dataset folder under models/data."""
    if source:
        src = DATA_DIR / source
        if not src.exists():
            raise FileNotFoundError(f"--source folder not found: {src}")
        return src
    for suffix in (".coco", ".yolov8", ""):
        candidate = DATA_DIR / f"{dataset}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No source for '{dataset}' (tried {dataset}.coco / {dataset}.yolov8 / {dataset}). "
        f"Pass --source explicitly."
    )


def detect_format(src):
    """Return 'coco' or 'yolo' based on what the source folder contains."""
    if list(src.glob("*/_annotations.coco.json")):
        return "coco"
    # YOLO: a split with a labels/ dir, or a data.yaml at the root.
    if list(src.glob("*/labels")) or (src / "data.yaml").exists():
        return "yolo"
    raise ValueError(f"Could not determine dataset format for: {src}")


# --------------------------------------------------------------------------- #
# COCO -> YOLO
# --------------------------------------------------------------------------- #
def coco_box_to_yolo(bbox, img_w, img_h):
    """COCO [x, y, w, h] (top-left px) -> YOLO (cx, cy, w, h) normalized & clamped."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return tuple(min(max(v, 0.0), 1.0) for v in (cx, cy, w / img_w, h / img_h))


def convert_coco_split(coco_json, src_images_dir, dst_images_dir, dst_labels_dir):
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    n_images = n_boxes = 0
    for img_id, img in images.items():
        src_img = src_images_dir / img["file_name"]
        if not src_img.exists():
            logger.warning("Image in COCO json missing on disk: %s", src_img)
            continue
        shutil.copy2(src_img, dst_images_dir / img["file_name"])

        lines = []
        for ann in anns_by_image.get(img_id, []):
            cx, cy, nw, nh = coco_box_to_yolo(ann["bbox"], img["width"], img["height"])
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_boxes += 1
        (dst_labels_dir / f"{Path(img['file_name']).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        n_images += 1
    return n_images, n_boxes


def discover_coco_splits(src):
    present = {}
    for coco_name, yolo_name in SPLIT_MAP.items():
        coco_json = src / coco_name / "_annotations.coco.json"
        if coco_json.exists() and yolo_name not in present:
            present[yolo_name] = (coco_json, src / coco_name)
    return present


# --------------------------------------------------------------------------- #
# YOLO -> YOLO (copy + force single class)
# --------------------------------------------------------------------------- #
def copy_yolo_split(src_split, dst_images_dir, dst_labels_dir):
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    src_images = src_split / "images"
    src_labels = src_split / "labels"

    n_images = n_boxes = 0
    for img in sorted(src_images.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        shutil.copy2(img, dst_images_dir / img.name)

        label_in = src_labels / f"{img.stem}.txt"
        lines = []
        if label_in.exists():
            for line in label_in.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    # Force class 0 (single-class bin detector).
                    lines.append("0 " + " ".join(parts[1:5]))
                    n_boxes += 1
        (dst_labels_dir / f"{img.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        n_images += 1
    return n_images, n_boxes


def discover_yolo_splits(src):
    present = {}
    for split_name, yolo_name in SPLIT_MAP.items():
        split_dir = src / split_name
        if (split_dir / "images").exists() and yolo_name not in present:
            present[yolo_name] = split_dir
    return present


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def prepare(dataset: str, source: str = None) -> Path:
    src = resolve_source(dataset, source)
    fmt = detect_format(src)
    out_dir = DATA_DIR / f"{dataset}.yolo"

    logger.info("Source: %s (%s format)", src, fmt)
    if out_dir.exists():
        logger.info("Removing previous output: %s", out_dir)
        shutil.rmtree(out_dir)

    counts = {}
    if fmt == "coco":
        present = discover_coco_splits(src)
        if "train" not in present:
            raise FileNotFoundError(f"No train split with annotations in {src}")
        for yolo_split, (coco_json, src_dir) in present.items():
            n, b = convert_coco_split(
                coco_json, src_dir,
                out_dir / "images" / yolo_split, out_dir / "labels" / yolo_split,
            )
            counts[yolo_split] = n
            logger.info("%s: %d images, %d boxes", yolo_split, n, b)
    else:  # yolo
        present = discover_yolo_splits(src)
        if "train" not in present:
            raise FileNotFoundError(f"No train/images split in {src}")
        for yolo_split, split_dir in present.items():
            n, b = copy_yolo_split(
                split_dir,
                out_dir / "images" / yolo_split, out_dir / "labels" / yolo_split,
            )
            counts[yolo_split] = n
            logger.info("%s: %d images, %d boxes", yolo_split, n, b)

    # Fallback: no real validation split -> mirror train (smoke test only).
    val_is_real = "val" in counts
    if not val_is_real:
        shutil.copytree(out_dir / "images" / "train", out_dir / "images" / "val")
        shutil.copytree(out_dir / "labels" / "train", out_dir / "labels" / "val")
        counts["val"] = counts["train"]
        logger.warning("No val split found -> mirrored train as val (SMOKE TEST only)")

    config = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "bin"},
    }
    if "test" in counts:
        config["test"] = "images/test"

    data_yaml = out_dir / f"data_{dataset}.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("\n%s", "=" * 60)
    logger.info("%s -> YOLO ready", dataset)
    logger.info("Dataset: %s", out_dir)
    logger.info("Data config: %s", data_yaml)
    logger.info("Splits: %s | real val: %s", counts, val_is_real)
    logger.info("%s", "=" * 60)
    if counts["train"] <= 5:
        logger.warning("Only %d training image(s) -- smoke test; expect overfit.", counts["train"])
    return data_yaml


def main():
    parser = argparse.ArgumentParser(description="Normalize a per-bin dataset to YOLO format.")
    parser.add_argument("--dataset", required=True, help="Clean dataset key, e.g. B0, B1, FSV1")
    parser.add_argument("--source", default=None,
                        help="Source folder under models/data (for spaces/odd suffixes)")
    args = parser.parse_args()
    prepare(args.dataset, args.source)


if __name__ == "__main__":
    main()
