"""
Bottom-bin visibility guide
============================
Generates an alignment overlay for the outer-limit / bin-visibility experiment:
finds the bottom-row bins in a baseline (fully-visible, camera-at-trained-
position) reference photo, then draws horizontal lines across each bin at
20/40/60/80/100% of its height so you can physically occlude down to a known
fraction, photograph it, and drop that photo into ``Dev_test_data`` for
``inspect_bin_detections.py`` to test.

Lines are measured from the TOP of each bin's bounding box down. Each line is
labeled with both directions — "40% down / 60% up" — so it works whichever
edge you're physically occluding from (rack lip cutting off the near/bottom
edge, or the far/top edge receding out of frame — whichever your rig does).

Run (from aegis-v2/):
    python tools/bin_visibility_guide.py
    python tools/bin_visibility_guide.py --image tools/Dev_test_data/65cm.jpg
    python tools/bin_visibility_guide.py --levels 10,25,50,75,100

Output: ``<image>_visibility_guide.jpg`` next to the source image, plus a
printed pixel-coordinate table (exact y for each % line, per bin) for cases
where measuring against the photo on-screen isn't precise enough.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("aegis.tools.bin_visibility_guide")

_ROOT = Path(__file__).resolve().parent.parent  # aegis-v2/
sys.path.insert(0, str(_ROOT.parent))  # repo root
sys.path.insert(0, str(_ROOT))          # aegis-v2/

_DEFAULT_CONFIG = _ROOT / "integration" / "config" / "settings.yaml"
_DEFAULT_DIR = _ROOT / "tools" / "Dev_test_data"

_LEVEL_COLORS = {
    20: (0, 0, 255),      # red
    40: (0, 128, 255),    # orange
    60: (0, 255, 255),    # yellow
    80: (0, 255, 128),    # light green
    100: (0, 255, 0),     # green
}
_FALLBACK_COLOR = (255, 0, 255)  # magenta, for levels outside the palette above
_BBOX_COLOR = (255, 255, 255)


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_model_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = _ROOT / raw
    return p


def bbox_of(corners) -> tuple:
    pts = np.asarray(corners, dtype=np.float32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return float(x1), float(y1), float(x2), float(y2)


def draw_guide(image, bottom_bins: list, levels: list) -> np.ndarray:
    vis = image.copy()
    for bin_idx, d in enumerate(bottom_bins):
        x1, y1, x2, y2 = bbox_of(d["corners"])
        height = y2 - y1
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), _BBOX_COLOR, 1)

        for level in levels:
            y = y1 + (level / 100.0) * height
            color = _LEVEL_COLORS.get(level, _FALLBACK_COLOR)
            cv2.line(vis, (int(x1), int(y)), (int(x2), int(y)), color, 2)
            label = f"{level}% down / {100 - level}% up"
            label_y = int(y) - 6 if level < 100 else int(y) - 6
            cv2.putText(vis, label, (int(x1) + 4, max(12, label_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        cv2.putText(vis, f"bin {bin_idx}", (int(x1) + 4, max(12, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BBOX_COLOR, 1, cv2.LINE_AA)
    return vis


def print_table(bottom_bins: list, levels: list) -> None:
    for bin_idx, d in enumerate(bottom_bins):
        x1, y1, x2, y2 = bbox_of(d["corners"])
        height = y2 - y1
        logger.info("bin %d: x=[%.0f, %.0f]  y_top=%.0f  y_bottom=%.0f  height=%.0fpx",
                     bin_idx, x1, x2, y1, y2, height)
        for level in levels:
            y = y1 + (level / 100.0) * height
            logger.info("    %3d%% down (%3d%% up) -> y=%.0f", level, 100 - level, y)


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw bottom-bin visibility-percentage guide lines")
    ap.add_argument("--image", default=None,
                     help="baseline reference photo (default: first image in Dev_test_data)")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG), help="path to settings.yaml")
    ap.add_argument("--model", default=None, help="override bin_detector.model_path")
    ap.add_argument("--task", default=None, choices=["detect", "obb"], help="override bin_detector.task")
    ap.add_argument("--levels", default="20,40,60,80,100", help="comma-separated visibility percentages")
    ap.add_argument("--out", default=None, help="output path (default: <image>_visibility_guide.jpg)")
    args = ap.parse_args()

    levels = sorted(int(x) for x in args.levels.split(","))

    if args.image:
        image_path = Path(args.image)
    else:
        candidates = sorted(p for p in _DEFAULT_DIR.iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "_annotated" not in p.stem
                             and "_guide" not in p.stem) if _DEFAULT_DIR.exists() else []
        if not candidates:
            logger.error("No --image given and no images found in %s", _DEFAULT_DIR)
            return
        image_path = candidates[0]
        logger.info("No --image given, using %s", image_path.name)

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("Could not read image: %s", image_path)
        return

    config = load_config(Path(args.config))
    det_cfg = config.get("bin_detector", {})
    task = (args.task or det_cfg.get("task", "obb")).lower()
    if task == "detect":
        from integration.src.detectors import initialize_bins_detect as detector
    else:
        from integration.src.detectors import initialize_bins_obb as detector
    from integration.src.detectors import grid_allocator as ga

    model_path = resolve_model_path(args.model or det_cfg.get("model_path"))
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        return
    model = detector.load_model(str(model_path))
    if model is None:
        logger.error("Model failed to load — nothing to guide from.")
        return

    dets = detector.detect_bins(image, model, expected_count=9)
    if not dets:
        logger.error("No bins detected in %s — need a clean baseline shot to build the guide.", image_path.name)
        return

    rows = ga.split_rows_by_y(dets, num_rows=2)
    bottom_bins = sorted(rows[1], key=lambda d: d["center"][0])
    if not bottom_bins:
        logger.error("Could not identify a bottom row (found %d detection(s), 0 in bottom row).", len(dets))
        return
    logger.info("Found %d bottom-row bin(s)", len(bottom_bins))

    vis = draw_guide(image, bottom_bins, levels)
    out_path = Path(args.out) if args.out else image_path.with_name(f"{image_path.stem}_visibility_guide.jpg")
    cv2.imwrite(str(out_path), vis)
    logger.info("Saved guide: %s", out_path)

    print_table(bottom_bins, levels)


if __name__ == "__main__":
    main()
