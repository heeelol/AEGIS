"""Merge several per-bin datasets into one combined YOLO dataset.

Combines the train splits of all inputs, and pulls val/test from inputs that
have *real* validation/test splits (train-only datasets contribute train only,
so their train-as-val fallback never pollutes the combined val). Output:

    models/data/<OUT>.yolo/{images,labels}/{train,val[,test]} + data_<OUT>.yaml

Filenames are prefixed per source (``<KEY>__``) so datasets that share image
names don't collide. Reuses the format detection / conversion from
prepare_bin_dataset.py, so each input may be COCO or YOLOv8.

Usage:
    python scripts/training/combine_bin_datasets.py --out FS12 \
        --inputs "FSV1=Final Setup V1.yolov8" "FSV2=Final Setup v2.coco"

Then train/test like any other dataset key:
    python scripts/training/train_bin_yolo.py --dataset FS12 --device 0
    python scripts/inference/initialize_bins_detect.py --dataset FS12
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import yaml

import prepare_bin_dataset as prep  # same directory

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = prep.DATA_DIR


def add_coco_split(coco_json, src_images, dst_img, dst_lbl, prefix):
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    coco = json.loads(Path(coco_json).read_text())
    images = {im["id"]: im for im in coco["images"]}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    n = 0
    for img_id, im in images.items():
        src = src_images / im["file_name"]
        if not src.exists():
            continue
        out_name = f"{prefix}{im['file_name']}"
        shutil.copy2(src, dst_img / out_name)
        lines = []
        for a in anns.get(img_id, []):
            cx, cy, w, h = prep.coco_box_to_yolo(a["bbox"], im["width"], im["height"])
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (dst_lbl / f"{Path(out_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        n += 1
    return n


def add_yolo_split(src_split, dst_img, dst_lbl, prefix):
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    src_images = src_split / "images"
    src_labels = src_split / "labels"

    n = 0
    for img in sorted(src_images.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        out_name = f"{prefix}{img.name}"
        shutil.copy2(img, dst_img / out_name)
        lines = []
        label_in = src_labels / f"{img.stem}.txt"
        if label_in.exists():
            for line in label_in.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    lines.append("0 " + " ".join(parts[1:5]))
        (dst_lbl / f"{prefix}{img.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        n += 1
    return n


def combine(out_key, inputs):
    out_dir = DATA_DIR / f"{out_key}.yolo"
    if out_dir.exists():
        logger.info("Removing previous output: %s", out_dir)
        shutil.rmtree(out_dir)

    counts = {"train": 0, "val": 0, "test": 0}
    for key, source in inputs:
        src = prep.resolve_source(key, source)
        fmt = prep.detect_format(src)
        prefix = f"{key}__"
        logger.info("Adding %s (%s, %s)", key, src.name, fmt)

        if fmt == "coco":
            splits = prep.discover_coco_splits(src)  # yolo_split -> (json, dir)
            for yolo_split, (coco_json, src_dir) in splits.items():
                added = add_coco_split(
                    coco_json, src_dir,
                    out_dir / "images" / yolo_split, out_dir / "labels" / yolo_split,
                    prefix,
                )
                counts[yolo_split] = counts.get(yolo_split, 0) + added
                logger.info("  %s += %d", yolo_split, added)
        else:
            splits = prep.discover_yolo_splits(src)  # yolo_split -> dir
            for yolo_split, split_dir in splits.items():
                added = add_yolo_split(
                    split_dir,
                    out_dir / "images" / yolo_split, out_dir / "labels" / yolo_split,
                    prefix,
                )
                counts[yolo_split] = counts.get(yolo_split, 0) + added
                logger.info("  %s += %d", yolo_split, added)

    # Fallback: no real val from any input -> mirror combined train.
    val_is_real = counts.get("val", 0) > 0
    if not val_is_real:
        shutil.copytree(out_dir / "images" / "train", out_dir / "images" / "val")
        shutil.copytree(out_dir / "labels" / "train", out_dir / "labels" / "val")
        counts["val"] = counts["train"]
        logger.warning("No real val from any input -> mirrored train (SMOKE TEST only)")

    config = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "bin"},
    }
    if counts.get("test", 0) > 0:
        config["test"] = "images/test"

    data_yaml = out_dir / f"data_{out_key}.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("\n%s", "=" * 60)
    logger.info("Combined dataset '%s' ready", out_key)
    logger.info("Dataset: %s", out_dir)
    logger.info("Splits: %s | real val: %s", counts, val_is_real)
    logger.info("%s", "=" * 60)
    return data_yaml


def parse_input(token):
    """'KEY=SOURCE' -> (KEY, SOURCE); 'KEY' -> (KEY, None)."""
    if "=" in token:
        key, source = token.split("=", 1)
        return key.strip(), source.strip()
    return token.strip(), None


def main():
    parser = argparse.ArgumentParser(description="Combine per-bin datasets into one.")
    parser.add_argument("--out", required=True, help="Output dataset key, e.g. FS12")
    parser.add_argument("--inputs", required=True, nargs="+",
                        help="Inputs as KEY or KEY=SOURCE_FOLDER")
    args = parser.parse_args()
    combine(args.out, [parse_input(t) for t in args.inputs])


if __name__ == "__main__":
    main()
