# Per-Bin YOLO Detector — Train & Test

YOLOv8 **detect** models (axis-aligned bounding boxes, single class `bin`), one per
bin dataset (`B0`, `B1`, `FSV1`, …). The scripts are **dataset-generic**: pass a
`--dataset` key and the same three scripts handle any bin, in either Roboflow export
format (COCO or YOLOv8).

## ⚠️ Use the right Python env

`ultralytics` / `torch` live **only** in the `python_clinic` conda env, which is under
**anaconda3** — but your shell's active conda is **miniconda3**, so
`conda activate python_clinic` fails with `EnvironmentNameNotFound`.

**Just call the interpreter by full path (no activation needed):**

```powershell
cd C:\Users\yapor\OneDrive\Desktop\CDE3301\TEnterns\aegis-core
$py = "C:\Users\yapor\anaconda3\envs\python_clinic\python.exe"
& $py <script> --dataset B1          # use  & $py  in place of  python
```

Paste commands as separate lines (Enter after each), not one block — pasted blocks
can collide with live program output.

## The three scripts

| Step | Script | What it does |
|------|--------|--------------|
| Prepare | `scripts/training/prepare_bin_dataset.py` | Normalizes a Roboflow dataset → `<KEY>.yolo`, single class `bin`. Auto-detects **COCO or YOLOv8** format and train/valid/test splits (`valid`→`val`). If no val split, mirrors train as val (smoke test) and warns. |
| Train | `scripts/training/train_bin_yolo.py` | Trains `yolov8n.pt` → `runs/<KEY>_detector/weights/best.pt`. Auto-prepares first if needed. |
| Test | `scripts/inference/initialize_bins_detect.py` | Interactive tester (detect-adapted from `initialize_bins.py`). Loads the dataset's `best.pt`. |

### Dataset keys vs source folders

`--dataset` is a **clean key** used for output paths and run names (`B0`, `B1`, `FSV1`).
The prep step finds the source folder under `models/data/` by trying `<KEY>.coco`,
`<KEY>.yolov8`, then `<KEY>`. When the folder name has spaces or a mismatched suffix,
pass `--source` explicitly:

| Key | Source folder | Format |
|-----|---------------|--------|
| `B0` | `B0.coco` | COCO (auto-found) |
| `B1` | `B1.coco` | COCO (auto-found) |
| `FSV1` | `Final Setup V1.yolov8` | YOLOv8 (needs `--source`) |

## Run training

```powershell
# COCO dataset, auto-found by key, GPU
& $py scripts/training/train_bin_yolo.py --dataset B1 --device 0

# YOLOv8 dataset with a spaced folder name -> pass --source
& $py scripts/training/train_bin_yolo.py --dataset FSV1 --source "Final Setup V1.yolov8" --device 0

# CPU (reliable; fine for tiny / smoke datasets)
& $py scripts/training/train_bin_yolo.py --dataset B0 --device cpu --epochs 50
```

Output model: `runs/<KEY>_detector/weights/best.pt` (overwrites each run).
Flags: `--epochs` (100), `--batch` (16), `--device` (`cpu` or GPU id `0`), `--weights` (`yolov8n.pt`), `--patience` (100).

**`--patience` defaults to 100 (= epochs), i.e. early-stopping effectively OFF.** On tiny
fixed-camera datasets a real val split otherwise triggers early stopping too soon, leaving
the model under-confident (FSV2 stopped at epoch 41 and missed bins at conf 0.25 until
retrained for the full 100). Overfitting to a fixed view is fine here. Lower `--patience`
to re-enable early stopping once datasets grow larger.

You can also prepare without training: `& $py scripts/training/prepare_bin_dataset.py --dataset FSV1 --source "Final Setup V1.yolov8"`.

## Run testing

```powershell
& $py scripts/inference/initialize_bins_detect.py --dataset FSV1
```

Prompts for a mode:

- **`1` Single image** — runs on a normalized test/val/train image; opens a window. Press any key to close.
- **`2` Continuous (webcam)** — lists cameras, you pick one; live detection. **Use this for the fixed mounted camera.**
  Keys: `q` quit · `s` save frame · `i` print/store bin coordinates.
- **`3` Batch (folder)** — type a folder path; writes `*_det.jpg` overlays next to each image.

## GPU note

The `python_clinic` env has `torch 2.11.0+cu128` with **working CUDA on the RTX 5060
(sm_120)** — a verified GPU matmul passes. The old "sm_120 unsupported" comment in
`train_bin_boundaries_local.py` is obsolete here, so prefer `--device 0` for real training.

## Validation splits matter

Datasets exported **train-only** (B0, FSV1) get `val = copy of train`, so their mAP is
**optimistic/meaningless**. Datasets with a real `valid` split (B1) get trustworthy
metrics. For a dataset you care about, export real train/valid/test splits from Roboflow.

## Adding a new bin / setup (B2, FSV2, …)

1. Export from Roboflow as **COCO** or **YOLOv8** to `models/data/<FOLDER>/`
   (prefer real train/valid/test splits).
2. `& $py scripts/training/train_bin_yolo.py --dataset <KEY> [--source "<FOLDER>"] --device 0`
3. `& $py scripts/inference/initialize_bins_detect.py --dataset <KEY>`

## Related (pre-existing) scripts

- `scripts/training/train_bin_detector.py` — fuller, GPU-oriented detect trainer with augmentation.
- `scripts/inference/initialize_bins.py` — the original **segmentation** tester these were adapted from.
