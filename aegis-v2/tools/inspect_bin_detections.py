"""
Bin detection inspector
========================
Runs the bin-boundary model on a folder of still images (default:
``tools/Dev_test_data/``) through the SAME ``detect_bins()`` path the live
pipeline uses (phantom/duplicate suppression + keep-top-N-by-confidence when
the bin count is known) and draws what survives, with its confidence score.
Useful for judging a newly trained/retrained model (e.g. FSV3) against known
reference shots before/after wiring it into the live pipeline.

Reads model path + task from the SAME settings.yaml the real pipeline uses,
so it inspects whichever model is currently configured. Override with
--model / --task for side-by-side comparisons. Pass --raw to see every
detection BEFORE filtering (useful for diagnosing why something got dropped).

Run (from aegis-v2/):
    python tools/inspect_bin_detections.py
    python tools/inspect_bin_detections.py --dir tools/Dev_test_data --expected-count 9
    python tools/inspect_bin_detections.py --model cv-models/models/weights/bin_detector.pt --task obb
    python tools/inspect_bin_detections.py --raw   # skip filtering, show every raw detection
    python tools/inspect_bin_detections.py --live  # live camera feed instead of a folder of stills

Annotated images are written next to the source as ``<name>_annotated.jpg``.
In --live mode, press 's' to save the current (clean, no-overlay) frame into
--dir for later batch inspection, 'q' to quit.
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
logger = logging.getLogger("aegis.tools.inspect_bin_detections")

_ROOT = Path(__file__).resolve().parent.parent  # aegis-v2/
sys.path.insert(0, str(_ROOT.parent))  # repo root
sys.path.insert(0, str(_ROOT))          # aegis-v2/

_DEFAULT_CONFIG = _ROOT / "integration" / "config" / "settings.yaml"
_DEFAULT_DIR = _ROOT / "tools" / "Dev_test_data"

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


def draw_detections(image, dets: list) -> np.ndarray:
    """Draw ``[{id, corners, conf, ...}]`` (filtered or raw) on the image."""
    vis = image.copy()
    for d in dets:
        pts = np.asarray(d["corners"], dtype=np.int32).reshape(-1, 1, 2)
        conf = float(d.get("conf", 0.0))
        cv2.polylines(vis, [pts], True, _BOX_COLOR, 2)
        cx, cy = int(d["center"][0]), int(d["center"][1])
        cv2.putText(vis, f"#{d.get('id', '?')} {conf:.2f}", (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _LABEL_COLOR, 2)
    return vis


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


def next_free_path(out_dir: Path, stem: str) -> Path:
    p = out_dir / f"{stem}.jpg"
    n = 2
    while p.exists():
        p = out_dir / f"{stem}_{n}.jpg"
        n += 1
    return p


def run_live(detect, cam_cfg: dict, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    source = cam_cfg.get("source", 0)
    cap = open_camera(source, cam_cfg.get("width", 1280), cam_cfg.get("height", 720), cam_cfg.get("fps", 30))
    rotate_180 = bool(cam_cfg.get("rotate_180", False))
    save_count = 0

    logger.info("Live mode — 's' = save clean frame to %s, 'q' = quit", save_dir)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            if rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            dets = detect(frame)
            display = draw_detections(frame, dets)
            confs = sorted((float(d["conf"]) for d in dets), reverse=True)
            summary = f"{len(dets)} detection(s)" + (f" avg={sum(confs) / len(confs):.2f}" if confs else "")
            cv2.putText(display, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "s=save  q=quit", (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imshow("Bin detection inspector (live)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                out_path = next_free_path(save_dir, "live_capture")
                cv2.imwrite(str(out_path), frame)
                save_count += 1
                logger.info("Saved: %s (%d total)", out_path, save_count)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Done — %d capture(s) saved to %s", save_count, save_dir)


def inspect_image(detect, image_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("Could not read image: %s", image_path)
        return

    dets = detect(image)
    vis = draw_detections(image, dets)

    out_path = image_path.with_name(f"{image_path.stem}_annotated.jpg")
    cv2.imwrite(str(out_path), vis)

    if dets:
        confs = sorted((float(d["conf"]) for d in dets), reverse=True)
        conf_list = ", ".join(f"{c:.2f}" for c in confs)
        logger.info("%s: %d detection(s) -> [%s] (avg %.2f) -> %s",
                     image_path.name, len(confs), conf_list, sum(confs) / len(confs), out_path.name)
    else:
        logger.warning("%s: 0 detections -> %s", image_path.name, out_path.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw bin detections + confidence on a folder of test images")
    ap.add_argument("--dir", default=str(_DEFAULT_DIR), help="folder of images to inspect")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG), help="path to settings.yaml")
    ap.add_argument("--model", default=None, help="override bin_detector.model_path")
    ap.add_argument("--task", default=None, choices=["detect", "obb"], help="override bin_detector.task")
    ap.add_argument("--expected-count", type=int, default=9,
                    help="known bin count, used to drop the weakest extras (matches the pipeline's "
                         "calibration call). Pass 0 to disable this step.")
    ap.add_argument("--raw", action="store_true",
                    help="skip phantom/duplicate/top-N filtering entirely — show every raw detection")
    ap.add_argument("--live", action="store_true", help="live camera feed instead of a folder of stills")
    ap.add_argument("--camera", default=None, help="(--live) override camera.source")
    args = ap.parse_args()

    config = load_config(Path(args.config))
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

    logger.info("Loading %s model: %s", task, model_path)
    model = detector.load_model(str(model_path))
    if model is None:
        logger.error("Model failed to load — nothing to inspect.")
        return

    expected_count = None if (args.raw or args.expected_count <= 0) else args.expected_count
    filter_kwargs = {"overlap_thresh": 1.0, "min_area_frac": 0.0} if args.raw else {}
    if args.raw:
        logger.info("--raw: filtering disabled, showing every detection")

    def detect(image):
        return detector.detect_bins(image, model, expected_count=expected_count, **filter_kwargs)

    if args.live:
        cam_cfg = dict(config.get("camera", {}))
        if args.camera is not None:
            cam_cfg["source"] = args.camera
        run_live(detect, cam_cfg, Path(args.dir))
        return

    image_dir = Path(args.dir)
    skip_markers = ("_annotated", "_guide")
    images = sorted(p for p in image_dir.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                     and not any(m in p.stem for m in skip_markers))
    if not images:
        logger.error("No images found in %s", image_dir)
        return

    logger.info("Inspecting %d image(s) in %s", len(images), image_dir)
    for image_path in images:
        inspect_image(detect, image_path)


if __name__ == "__main__":
    main()
