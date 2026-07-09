"""Train a YOLOv8 OBB (oriented bounding box) model on the Project 9_1 dataset.

The dataset (models/data/Project 9_1.yolov8-obb) is already in YOLO-OBB format,
but two things need handling before training:
  1. All images sit in train/ — valid/ and test/ are empty. A validation split is
     created by moving ~15% of train into valid/ (one-time; skipped if valid exists).
  2. The Roboflow data.yaml uses relative paths and the folder name has a space, which
     can confuse the trainer. We write a data_local.yaml with an absolute `path:`.

Change MODEL below for a different size (yolov8n/s/m/l-obb.pt). With only ~106
images and one class, a smaller model (s or n) often generalises better — watch the
val metrics for overfitting.
"""

import logging
import re
import shutil
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Training config (edit here) ─────────────────────────────
DATASET_NAME = "Project 9.yolov8-obb (1)"   # folder under models/data/ to train on
MODEL = "yolov8m-obb.pt"     # yolov8n-obb.pt | yolov8s-obb.pt | yolov8m-obb.pt | yolov8l-obb.pt
EPOCHS = 50
BATCH = 16
IMGSZ = 640
PATIENCE = 20
VAL_FRACTION = 0.15
WORKERS = 2                  # DataLoader workers. Low on Windows: each is a process
                             # that duplicates RAM; 8 (default) can exhaust system memory.
AMP = False                  # Mixed precision OFF — AMP is a common trigger of CUDA
                             # "illegal memory access" on new GPUs (Blackwell) + OBB.
MOSAIC = 0.0                 # Mosaic aug OFF — mosaic can create degenerate rotated
                             # boxes that crash the OBB loss. Raise to 1.0 to re-enable.
DEGREES = 180                # Random rotation aug (±deg). Rotates images AND OBB labels,
                             # so the model learns tilted boxes from upright annotations.
                             # Lower (90/45) if edge bins get clipped out of frame.
RESUME = False               # True → continue the most recent run from its last.pt

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def print_epoch_progress(trainer):
    """Log a progress bar + losses (+ mAP50) at the end of each epoch."""
    epoch = getattr(trainer, "epoch", 0) + 1
    total = getattr(trainer, "epochs", EPOCHS)
    filled = int(epoch / total * 20)
    bar = "█" * filled + "░" * (20 - filled)

    parts = []
    try:
        losses = trainer.label_loss_items(getattr(trainer, "tloss", None), prefix="train")
        parts = [f"{k.split('/')[-1]}: {float(v):.4f}" for k, v in losses.items()]
    except Exception:
        pass

    # mAP becomes available after validation at fit-epoch end.
    metrics = getattr(trainer, "metrics", {}) or {}
    map50 = metrics.get("metrics/mAP50(B)")
    map_str = f" | mAP50: {map50:.4f}" if isinstance(map50, (int, float)) else ""

    logger.info(f"Epoch {epoch}/{total} [{bar}] {epoch/total*100:.0f}%  "
                + " | ".join(parts) + map_str)


def ensure_val_split(dataset_dir: Path, val_frac: float = VAL_FRACTION) -> None:
    """Move ~val_frac of train images+labels into valid/ if valid/ is empty."""
    train_img = dataset_dir / "train" / "images"
    train_lbl = dataset_dir / "train" / "labels"
    val_img = dataset_dir / "valid" / "images"
    val_lbl = dataset_dir / "valid" / "labels"
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    existing = [p for p in val_img.glob("*") if p.suffix.lower() in _IMG_EXTS]
    if existing:
        logger.info(f"Validation split already present ({len(existing)} images) — leaving as is")
        return

    images = sorted(p for p in train_img.glob("*") if p.suffix.lower() in _IMG_EXTS)
    if not images:
        logger.error(f"No training images found in {train_img}")
        return

    # Deterministic stride selection (no randomness → reproducible).
    step = max(2, round(1.0 / val_frac))   # ~15% → every 6th image
    val_images = images[::step]

    moved = 0
    for img in val_images:
        lbl = train_lbl / (img.stem + ".txt")
        shutil.move(str(img), str(val_img / img.name))
        if lbl.exists():
            shutil.move(str(lbl), str(val_lbl / lbl.name))
        moved += 1

    remaining = len(images) - moved
    logger.info(f"Created validation split: moved {moved} images to valid/, {remaining} remain in train/")


def _slug(name: str) -> str:
    """Filesystem-safe slug, e.g. 'Project 9.yolov8-obb (1)' -> 'project_9_yolov8_obb_1'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def _latest_last_pt(run_root: Path) -> Path | None:
    """Find the newest <run_root>/*/weights/last.pt for resuming."""
    cands = sorted(run_root.glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def write_local_yaml(dataset_dir: Path) -> Path:
    """Write data_local.yaml with an absolute path root (handles spaces / relative paths)."""
    orig_yaml = dataset_dir / "data.yaml"
    names = {0: "object"}
    if orig_yaml.exists():
        with open(orig_yaml) as f:
            orig = yaml.safe_load(f) or {}
        names = orig.get("names", names)

    data = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
    }
    out = dataset_dir / "data_local.yaml"
    with open(out, "w") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)
    logger.info(f"✓ Wrote dataset config: {out}")
    logger.info(f"  Classes: {names}")
    return out


def main():
    base_dir = Path(__file__).parent.parent.parent
    dataset_dir = base_dir / "models" / "data" / DATASET_NAME
    slug = _slug(DATASET_NAME)
    run_project = base_dir / "runs" / "obb" / slug
    name_suffix = "_rotate" if DEGREES > 0 else ""   # tag rotation-augmented models
    out_model = base_dir / "models" / "custom" / f"{slug}{name_suffix}.pt"

    logger.info("=" * 60)
    logger.info(f"YOLOv8 OBB TRAINING — {DATASET_NAME}")
    logger.info("=" * 60 + "\n")

    if not dataset_dir.exists():
        logger.error(f"Dataset not found: {dataset_dir}")
        return

    # 1. Prepare splits + local config
    ensure_val_split(dataset_dir)
    data_yaml = write_local_yaml(dataset_dir)

    start = time.time()

    # 2. Train — either resume the last run, or start fresh.
    if RESUME:
        last = _latest_last_pt(run_project)
        if last is None:
            logger.error("RESUME=True but no previous checkpoint found — run once without RESUME first.")
            return
        logger.info(f"\nResuming from checkpoint: {last}")
        model = YOLO(str(last))
        model.add_callback("on_fit_epoch_end", print_epoch_progress)
        results = model.train(resume=True)   # continues with the run's saved settings
    else:
        logger.info(f"\nLoading OBB model: {MODEL}")
        model = YOLO(MODEL)
        model.add_callback("on_fit_epoch_end", print_epoch_progress)
        logger.info("✓ Model loaded\n")

        logger.info("🚀 Starting OBB training...")
        logger.info(f"  Epochs: {EPOCHS} | Batch: {BATCH} | imgsz: {IMGSZ} | "
                    f"workers: {WORKERS} | patience: {PATIENCE}\n")
        results = model.train(
            data=str(data_yaml),
            task="obb",
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=0,            # GPU 0; set to "cpu" if no CUDA
            amp=AMP,
            mosaic=MOSAIC,
            degrees=DEGREES,     # rotation aug → teaches tilted boxes
            patience=PATIENCE,
            workers=WORKERS,     # low worker count → avoids Windows RAM blow-up
            cache=False,         # do NOT cache images in RAM
            verbose=True,
            project=str(run_project),
        )

    elapsed = time.time() - start

    # 4. Save best weights to models/custom/
    model_path = out_model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, model_path)
        logger.info(f"\n✓ Best model saved: {model_path}")

    logger.info("\n" + "=" * 60)
    logger.info("OBB TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Training time : {elapsed/60:.1f} minutes")
    logger.info(f"Base model    : {MODEL}")
    logger.info(f"Results dir   : {results.save_dir}")
    logger.info(f"Model saved   : {model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
