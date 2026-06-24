"""Train a per-bin YOLOv8 DETECT model (B0, B1, ...).

Trains ``yolov8n.pt`` (axis-aligned boxes, NOT OBB) on a converted
``<DATASET>.yolo`` dataset and writes a predictable output path:

    runs/<DATASET>_detector/weights/best.pt

so the tester can always find it. Auto-runs the converter if the YOLO
dataset is missing.

Usage:
    python scripts/training/train_bin_yolo.py --dataset B1
    python scripts/training/train_bin_yolo.py --dataset B1 --device 0   # GPU
    python scripts/training/train_bin_yolo.py --dataset B0 --epochs 50  # tiny smoke test

Notes
-----
* device defaults to 'cpu' for reliability. The python_clinic env has
  torch 2.11.0+cu128 with working CUDA, so '--device 0' trains on the GPU
  (much faster once a dataset is real-sized).
* Output path overwrites each run (exist_ok) so the tester is stable.
* patience defaults to 100 (= epochs), i.e. early-stopping effectively OFF.
  On tiny fixed-camera datasets, early stopping halts training too soon and
  leaves the model under-confident (FSV2 stopped at 41 epochs and missed bins
  at the 0.25 threshold until retrained for the full 100). Overfitting to a
  fixed camera view is acceptable here. Lower --patience to re-enable it.
"""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

import prepare_bin_dataset as prep  # same directory

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RUN_PROJECT = BASE_DIR / "runs"


def data_yaml_for(dataset: str) -> Path:
    return BASE_DIR / "models" / "data" / f"{dataset}.yolo" / f"data_{dataset}.yaml"


def train(dataset, epochs, batch, imgsz, device, model_weights, source=None, patience=100):
    data_yaml = data_yaml_for(dataset)
    if not data_yaml.exists():
        logger.info("YOLO dataset missing; preparing %s first...", dataset)
        prep.prepare(dataset, source)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Preparation did not produce {data_yaml}")

    run_name = f"{dataset}_detector"
    logger.info("Loading %s (detect)...", model_weights)
    model = YOLO(model_weights)

    logger.info("Training %s on device=%s ...", dataset, device)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        amp=(device != "cpu"),  # AMP is a no-op on CPU
        patience=patience,
        project=str(RUN_PROJECT),
        name=run_name,
        exist_ok=True,
        seed=42,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("\n%s", "=" * 60)
    logger.info("TRAINING COMPLETE (%s)", dataset)
    logger.info("Best model: %s", best)
    logger.info("Test it: python scripts/inference/initialize_bins_detect.py --dataset %s", dataset)
    logger.info("%s", "=" * 60)

    try:
        metrics = model.val(device=device)
        logger.info("mAP50: %.3f | mAP50-95: %.3f", metrics.box.map50, metrics.box.map)
    except Exception as e:  # best-effort
        logger.warning("Validation step skipped: %s", e)

    return best


def main():
    parser = argparse.ArgumentParser(description="Train a per-bin YOLOv8 detect model.")
    parser.add_argument("--dataset", default="B0", help="Dataset key, e.g. B0, B1, FSV1")
    parser.add_argument("--source", default=None,
                        help="Source folder under models/data (for spaces/odd suffixes)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="'cpu' or GPU id like '0'")
    parser.add_argument("--weights", default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--patience", type=int, default=100,
                        help="Early-stopping patience (default 100 = effectively off for "
                             "small fixed-camera datasets); lower it to re-enable early stop")
    args = parser.parse_args()

    train(args.dataset, args.epochs, args.batch, args.imgsz, args.device,
          args.weights, args.source, args.patience)


if __name__ == "__main__":
    main()
