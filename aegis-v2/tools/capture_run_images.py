"""
Phase 2 run image capture
===========================
Live camera feed with bin detection boxes + confidence drawn on top (same
production ``detect_bins()`` path the pipeline uses), for walking through the
Phase 2 test sheet run by run. Press 's' to snap a clean (no-overlay) photo
into the current run's folder; switch runs with 'n'/'p' or by typing a number
with 'r'. Once a run's folder has ~20 photos, move to the next run and repeat
— no restart needed between runs.

Photos go to ``Dev_test_data/run_<N>/img_<NN>.jpg`` (zero-padded, sorts
cleanly). Restarting the script resumes numbering from whatever's already in
the folder, so nothing gets overwritten.

Run (from aegis-v2/):
    python tools/capture_run_images.py --run 5
    python tools/capture_run_images.py --run 5 --expected-count 9   # (default)

Controls:
    s = save a clean capture into the current run's folder
    n = next run (run number + 1)
    p = previous run (run number - 1)
    r = type a specific run number
    q = quit
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
logger = logging.getLogger("aegis.tools.capture_run_images")

_ROOT = Path(__file__).resolve().parent.parent  # aegis-v2/
sys.path.insert(0, str(_ROOT.parent))  # repo root
sys.path.insert(0, str(_ROOT))          # aegis-v2/

_DEFAULT_CONFIG = _ROOT / "integration" / "config" / "settings.yaml"
_DEFAULT_OUT_ROOT = _ROOT / "tools" / "Dev_test_data"

_BOX_COLOR = (0, 255, 0)
_LABEL_COLOR = (0, 255, 0)


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


def draw_detections(image, dets: list) -> np.ndarray:
    vis = image.copy()
    for d in dets:
        pts = np.asarray(d["corners"], dtype=np.int32).reshape(-1, 1, 2)
        conf = float(d.get("conf", 0.0))
        cv2.polylines(vis, [pts], True, _BOX_COLOR, 2)
        cx, cy = int(d["center"][0]), int(d["center"][1])
        cv2.putText(vis, f"#{d.get('id', '?')} {conf:.2f}", (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _LABEL_COLOR, 2)
    return vis


def run_dir(out_root: Path, run: int) -> Path:
    return out_root / f"run_{run:02d}"


def existing_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len(list(folder.glob("img_*.jpg")))


def next_free_path(folder: Path) -> Path:
    n = existing_count(folder) + 1
    p = folder / f"img_{n:02d}.jpg"
    while p.exists():  # in case files were deleted out of sequence
        n += 1
        p = folder / f"img_{n:02d}.jpg"
    return p


def prompt_run_number(current: int) -> int:
    try:
        raw = input(f"\nRun number (currently {current}): ").strip()
    except (EOFError, KeyboardInterrupt):
        return current
    if not raw.isdigit():
        logger.warning("Not a number, keeping run %d", current)
        return current
    return int(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live capture tool for the Phase 2 test sheet, organized by run number")
    ap.add_argument("--run", type=int, default=1, help="starting run number")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG), help="path to settings.yaml")
    ap.add_argument("--model", default=None, help="override bin_detector.model_path")
    ap.add_argument("--task", default=None, choices=["detect", "obb"], help="override bin_detector.task")
    ap.add_argument("--camera", default=None, help="override camera.source")
    ap.add_argument("--expected-count", type=int, default=9,
                     help="known bin count for filtering (matches the pipeline's calibration call). "
                          "Pass 0 to disable.")
    ap.add_argument("--out-root", default=str(_DEFAULT_OUT_ROOT), help="parent folder for run_<N> subfolders")
    args = ap.parse_args()

    config = load_config(Path(args.config))
    cam_cfg = config.get("camera", {})
    det_cfg = config.get("bin_detector", {})

    task = (args.task or det_cfg.get("task", "obb")).lower()
    if task == "detect":
        from integration.src.detectors import initialize_bins_detect as detector
    else:
        from integration.src.detectors import initialize_bins_obb as detector

    model_path = resolve_model_path(args.model or det_cfg.get("model_path"))
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        return
    model = detector.load_model(str(model_path))
    if model is None:
        logger.error("Model failed to load — nothing to capture against.")
        return

    expected_count = args.expected_count if args.expected_count > 0 else None

    def detect(image):
        return detector.detect_bins(image, model, expected_count=expected_count)

    source = args.camera if args.camera is not None else cam_cfg.get("source", 0)
    cap = open_camera(source, cam_cfg.get("width", 1280), cam_cfg.get("height", 720), cam_cfg.get("fps", 30))
    rotate_180 = bool(cam_cfg.get("rotate_180", False))

    out_root = Path(args.out_root)
    current_run = args.run
    logger.info("Controls: s=save  n=next run  p=prev run  r=type run number  q=quit")

    try:
        while True:
            folder = run_dir(out_root, current_run)
            ret, frame = cap.read()
            if not ret:
                continue
            if rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            dets = detect(frame)
            display = draw_detections(frame, dets)
            confs = sorted((float(d["conf"]) for d in dets), reverse=True)
            summary = f"{len(dets)} detection(s)" + (f" avg={sum(confs) / len(confs):.2f}" if confs else "")
            saved = existing_count(folder)
            cv2.putText(display, f"Run {current_run}  |  {saved} saved  |  {summary}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "s=save  n=next  p=prev  r=set run  q=quit",
                        (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imshow("Phase 2 run capture (live)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                current_run += 1
                logger.info("-> run %d (%d already saved)", current_run, existing_count(run_dir(out_root, current_run)))
            elif key == ord("p"):
                current_run = max(1, current_run - 1)
                logger.info("-> run %d (%d already saved)", current_run, existing_count(run_dir(out_root, current_run)))
            elif key == ord("r"):
                current_run = prompt_run_number(current_run)
                logger.info("-> run %d (%d already saved)", current_run, existing_count(run_dir(out_root, current_run)))
            elif key == ord("s"):
                folder.mkdir(parents=True, exist_ok=True)
                out_path = next_free_path(folder)
                cv2.imwrite(str(out_path), frame)
                logger.info("Saved: %s (%d/20 in run %d)", out_path, existing_count(folder), current_run)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Done — last run was %d", current_run)


if __name__ == "__main__":
    main()
