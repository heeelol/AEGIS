"""
Bottom-bin visibility guide — live overlay
============================================
Live-camera version of ``bin_visibility_guide.py``. Opens the camera (fixed
at the trained position), detects the bottom-row bins ONCE on a clean warm-up
frame to lock in each bin's 20/40/60/80/100%-height guide lines, then keeps
those lines burned into the live preview while you physically occlude the
bin and align it to a line. Press the matching number key to save a CLEAN
(no-overlay) capture straight into ``Dev_test_data`` for
``inspect_bin_detections.py`` to test.

Reads camera + model settings from the SAME settings.yaml the real pipeline
uses. Since the camera doesn't move during this experiment, the guide lines
are computed once and reused every frame — press 'd' to recompute them if
you bump the camera or want a fresh baseline.

Run (from aegis-v2/):
    python tools/bin_visibility_guide_live.py
    python tools/bin_visibility_guide_live.py --levels 10,25,50,75,100

Controls:
    1..5  = save a clean capture labeled with that level (e.g. '3' -> 60pct)
    s     = save a clean capture with no label (outer_capture_NN.jpg)
    d     = redetect bottom bins / recompute guide lines from the current frame
    q     = quit
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
logger = logging.getLogger("aegis.tools.bin_visibility_guide_live")

_ROOT = Path(__file__).resolve().parent.parent  # aegis-v2/
sys.path.insert(0, str(_ROOT.parent))  # repo root
sys.path.insert(0, str(_ROOT))          # aegis-v2/

_DEFAULT_CONFIG = _ROOT / "integration" / "config" / "settings.yaml"
_DEFAULT_OUT_DIR = _ROOT / "tools" / "Dev_test_data"

_LEVEL_COLORS = {
    20: (0, 0, 255),      # red
    40: (0, 128, 255),    # orange
    60: (0, 255, 255),    # yellow
    80: (0, 255, 128),    # light green
    100: (0, 255, 0),     # green
}
_FALLBACK_COLOR = (255, 0, 255)
_BBOX_COLOR = (255, 255, 255)


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_model_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = _ROOT / raw
    return p


def open_camera(source, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    logger.info("Opening camera: %s", source)

    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.warning("DirectShow could not open camera %s; trying default backend...", source)
            cap.release()
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {source}")

    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)  # re-assert after resolution
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera negotiated: %dx%d @ %.1f fps", actual_w, actual_h, cap.get(cv2.CAP_PROP_FPS))
    return cap


def grab_warm_frame(cap: cv2.VideoCapture, warmup: int, rotate_180: bool):
    frame = None
    for _ in range(max(1, warmup)):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
    if frame is not None and rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def bbox_of(corners) -> tuple:
    pts = np.asarray(corners, dtype=np.float32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return float(x1), float(y1), float(x2), float(y2)


def compute_guide_lines(detector, model, ga, frame, levels: list):
    """Detect the bottom row once and return [(x1, x2, {level: y}), ...] per bin."""
    dets = detector.detect_bins(frame, model, expected_count=9)
    if not dets:
        logger.warning("No bins detected on this frame — can't compute guide lines")
        return None
    rows = ga.split_rows_by_y(dets, num_rows=2)
    bottom_bins = sorted(rows[1], key=lambda d: d["center"][0])
    if not bottom_bins:
        logger.warning("No bottom-row bins found (got %d detection(s) total)", len(dets))
        return None

    lines = []
    for d in bottom_bins:
        x1, y1, x2, y2 = bbox_of(d["corners"])
        height = y2 - y1
        level_ys = {level: y1 + (level / 100.0) * height for level in levels}
        lines.append((x1, x2, level_ys))
    logger.info("Locked guide lines for %d bottom bin(s)", len(lines))
    return lines


def draw_guide_overlay(frame, lines, levels: list):
    vis = frame.copy()
    for bin_idx, (x1, x2, level_ys) in enumerate(lines):
        for level in levels:
            y = level_ys[level]
            color = _LEVEL_COLORS.get(level, _FALLBACK_COLOR)
            cv2.line(vis, (int(x1), int(y)), (int(x2), int(y)), color, 2)
            cv2.putText(vis, f"{level}%", (int(x1) + 4, max(12, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(vis, f"bin {bin_idx}", (int(x1) + 4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BBOX_COLOR, 1, cv2.LINE_AA)
    return vis


def next_free_path(out_dir: Path, stem: str) -> Path:
    p = out_dir / f"{stem}.jpg"
    n = 2
    while p.exists():
        p = out_dir / f"{stem}_{n}.jpg"
        n += 1
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Live camera overlay for the bottom-bin visibility experiment")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG), help="path to settings.yaml")
    ap.add_argument("--model", default=None, help="override bin_detector.model_path")
    ap.add_argument("--task", default=None, choices=["detect", "obb"], help="override bin_detector.task")
    ap.add_argument("--camera", default=None, help="override camera.source")
    ap.add_argument("--levels", default="20,40,60,80,100", help="comma-separated visibility percentages")
    ap.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR), help="folder to save captures into")
    args = ap.parse_args()

    levels = sorted(int(x) for x in args.levels.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config))
    cam_cfg = config.get("camera", {})
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

    source = args.camera if args.camera is not None else cam_cfg.get("source", 0)
    cap = open_camera(source, cam_cfg.get("width", 1280), cam_cfg.get("height", 720), cam_cfg.get("fps", 30))
    rotate_180 = bool(cam_cfg.get("rotate_180", False))
    warmup = cam_cfg.get("warmup_frames", 30)

    logger.info("Warming up for baseline detection...")
    baseline = grab_warm_frame(cap, warmup, rotate_180)
    lines = compute_guide_lines(detector, model, ga, baseline, levels) if baseline is not None else None

    key_to_level = {ord(str(i + 1)): lvl for i, lvl in enumerate(levels[:9])}
    save_count = 0

    key_hint = "  ".join(f"{chr(k)}={v}%" for k, v in key_to_level.items())
    logger.info("Controls: %s  |  s = save unlabeled  |  d = redetect  |  q = quit", key_hint)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            if rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            display = draw_guide_overlay(frame, lines, levels) if lines else frame.copy()
            if lines is None:
                cv2.putText(display, "No guide lines (press 'd' to detect)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, f"{key_hint}   s=save  d=redetect  q=quit", (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imshow("Bin visibility guide (live)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                lines = compute_guide_lines(detector, model, ga, frame, levels)
            elif key == ord("s"):
                out_path = next_free_path(out_dir, "outer_capture")
                cv2.imwrite(str(out_path), frame)
                save_count += 1
                logger.info("Saved: %s (%d total)", out_path, save_count)
            elif key in key_to_level:
                level = key_to_level[key]
                out_path = next_free_path(out_dir, f"outer_{level}pct")
                cv2.imwrite(str(out_path), frame)
                save_count += 1
                logger.info("Saved: %s (%d total)", out_path, save_count)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Done — %d capture(s) saved to %s", save_count, out_dir)


if __name__ == "__main__":
    main()
