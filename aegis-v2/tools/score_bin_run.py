"""
Phase 2 bin-detection run scorer
==================================
Scores one row of the Phase 2 "Bin detection (camera fixed, contents and
colour vary)" test sheet from a folder of individually-captured photos (or,
if you have one, a recorded video clip):

  1. Load the sample frames — either every image in ``--dir``, or frames
     extracted from ``--video`` at ~1 fps.
  2. Detect bins in each sampled frame with the SAME production
     ``detect_bins()`` path the pipeline uses.
  3. Match each frame's detections to the 9 canonical grid slots (via
     ``grid_calibrator.match_to_grid``, exactly like the pipeline's kit-init
     step — no ``expected_count`` cap, so a genuinely-missing bin shows up as
     missing rather than getting silently backfilled).
  4. Aggregate presence-per-slot across the sample -> overall accuracy%,
     Pass/Fail against the 95% criterion, and which specific slot(s) failed.

The 9-slot grid is calibrated ONCE from a clean reference image (default:
``Dev_test_data/65cm.jpg``, the empty-blue-bin baseline) since the camera
stays fixed for all of Phase 2 — content/colour changes don't move the bins.

Run (from aegis-v2/):
    python tools/score_bin_run.py --dir tools/Dev_test_data/B1_Empty --run B1 --fill Empty \\
        --setting "TE parts - black, small"
    python tools/score_bin_run.py --video recordings/B2_full.mp4 --run B2 --fill Full \\
        --group "Bin contents" --setting "Metal shelf brackets"

Prints a row matching the test-sheet columns, plus a per-slot failure
breakdown, and writes annotated evidence frames (green=present bin,
red=missing) to ``Dev_test_data/<run>_<fill>/`` (or straight into ``--dir``
when scoring a photo folder, so the annotated copies sit next to the
originals).
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
logger = logging.getLogger("aegis.tools.score_bin_run")

_ROOT = Path(__file__).resolve().parent.parent  # aegis-v2/
sys.path.insert(0, str(_ROOT.parent))                                  # repo root
sys.path.insert(0, str(_ROOT))                                          # aegis-v2/
sys.path.insert(0, str(_ROOT / "integration" / "src" / "detectors"))    # grid_calibrator's bare `import grid_allocator`

_DEFAULT_CONFIG = _ROOT / "integration" / "config" / "settings.yaml"
_DEFAULT_CALIBRATION_IMAGE = _ROOT / "tools" / "Dev_test_data" / "65cm.jpg"
_DEFAULT_OUT_ROOT = _ROOT / "tools" / "Dev_test_data"


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_model_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = _ROOT / raw
    return p


def extract_frames(video_path: Path, target_fps: float, rotate_180: bool) -> list:
    """Returns [(name, frame), ...]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(source_fps / target_fps))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            f = cv2.rotate(frame, cv2.ROTATE_180) if rotate_180 else frame
            frames.append((f"frame_{len(frames) + 1:02d}", f))
        idx += 1
    cap.release()
    logger.info("Extracted %d frame(s) from %s (source ~%.1f fps, step=%d)",
                len(frames), video_path.name, source_fps, step)
    return frames


def load_images_from_dir(dir_path: Path, rotate_180: bool) -> list:
    """Returns [(name, frame), ...] for every photo in the folder (non-recursive)."""
    skip_markers = ("_annotated", "_guide", "_scored")
    paths = sorted(p for p in dir_path.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                    and not any(m in p.stem for m in skip_markers))
    frames = []
    for p in paths:
        f = cv2.imread(str(p))
        if f is None:
            logger.warning("Could not read image: %s — skipping", p)
            continue
        frames.append((p.stem, cv2.rotate(f, cv2.ROTATE_180) if rotate_180 else f))
    logger.info("Loaded %d image(s) from %s", len(frames), dir_path)
    return frames


def build_calibration(image_path: Path, model, detector, gc):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read calibration image: {image_path}")
    dets = detector.detect_bins(image, model, expected_count=9)
    calibration = gc.calibrate_grid(dets)  # raises ValueError if not a clean 6+3
    logger.info("Calibrated 9 slots from %s", image_path.name)
    return calibration


def draw_occupancy(frame, occupancy: dict, calibration: dict):
    vis = frame.copy()
    for sid, info in occupancy.items():
        idx = info["index"]
        if info["present"]:
            pts = np.asarray(info["corners"], dtype=np.int32).reshape(-1, 1, 2)
            cx, cy = int(info["center"][0]), int(info["center"][1])
            color = (0, 255, 0)
            cv2.polylines(vis, [pts], True, color, 2)
            cv2.putText(vis, f"{idx} {info.get('confidence', 0):.2f}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cal = calibration.get(sid)
            if cal is None:
                continue
            pts = np.asarray(cal["corners"], dtype=np.int32).reshape(-1, 1, 2)
            cx, cy = int(cal["center"][0]), int(cal["center"][1])
            color = (0, 0, 255)
            cv2.polylines(vis, [pts], True, color, 2)
            cv2.putText(vis, f"{idx} MISSING", (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def score(frames: list, model, detector, calibration: dict, gc) -> list:
    """Returns one occupancy dict (slot_id -> info) per frame."""
    results = []
    for frame in frames:
        dets = detector.detect_bins(frame, model)  # no expected_count: mirrors pipeline._init_kit
        occ = gc.match_to_grid(dets, calibration)
        results.append(occ)
    return results


def summarize(occupancies: list, calibration: dict) -> dict:
    n = len(occupancies)
    slot_ids = sorted(calibration.keys(), key=lambda s: calibration[s]["index"])
    per_slot = {}
    total_present = 0
    for sid in slot_ids:
        present_count = sum(1 for occ in occupancies if occ[sid]["present"])
        confs = [occ[sid]["confidence"] for occ in occupancies if occ[sid]["present"] and "confidence" in occ[sid]]
        per_slot[sid] = {
            "index": calibration[sid]["index"],
            "present_count": present_count,
            "total": n,
            "rate": present_count / n if n else 0.0,
            "avg_conf": (sum(confs) / len(confs)) if confs else None,
        }
        total_present += present_count

    accuracy = (total_present / (9 * n) * 100.0) if n else 0.0
    return {"per_slot": per_slot, "accuracy": accuracy, "n_frames": n}


def format_failure_notes(summary: dict) -> str:
    problems = [s for s in summary["per_slot"].values() if s["present_count"] < s["total"]]
    if not problems:
        return ""
    problems.sort(key=lambda s: s["present_count"])
    parts = []
    for s in problems:
        miss = s["total"] - s["present_count"]
        parts.append(f"slot {s['index']} missing {miss}/{s['total']} frames")
    return "; ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a Phase 2 bin-detection test run from photos or a recorded clip")
    ap.add_argument("--dir", default=None, help="folder of individually-captured photos to score")
    ap.add_argument("--video", default=None, help="path to a recorded clip (alternative to --dir)")
    ap.add_argument("--run", default="", help="run label, e.g. B1 (for the printed row)")
    ap.add_argument("--group", default="Bin contents", help="Group column")
    ap.add_argument("--setting", default="", help="Exact setting column, e.g. 'TE parts - black, small'")
    ap.add_argument("--fill", default="", choices=["", "Empty", "Partial", "Full"], help="Fill level column")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG), help="path to settings.yaml")
    ap.add_argument("--model", default=None, help="override bin_detector.model_path")
    ap.add_argument("--task", default=None, choices=["detect", "obb"], help="override bin_detector.task")
    ap.add_argument("--calibration-image", default=str(_DEFAULT_CALIBRATION_IMAGE),
                     help="clean reference photo used to lock the 9 slot positions once")
    ap.add_argument("--fps", type=float, default=1.0, help="(--video only) frame extraction rate")
    ap.add_argument("--sample-frames", type=int, default=20, help="how many frames/photos to score")
    ap.add_argument("--skip-start", type=int, default=2,
                     help="(--video only) frames to discard from the start (settling time)")
    ap.add_argument("--pass-threshold", type=float, default=95.0, help="pass criterion, %% bins detected")
    ap.add_argument("--rotate-180", action="store_true", default=None,
                     help="rotate frames 180°; default is camera.rotate_180 for --video, off for --dir "
                          "(photos from this repo's live tools are already rotated)")
    ap.add_argument("--out-dir", default=None,
                     help="evidence output folder (default: Dev_test_data/<run>_<fill> for --video, "
                          "or straight into --dir)")
    args = ap.parse_args()

    if bool(args.dir) == bool(args.video):
        logger.error("Pass exactly one of --dir or --video")
        return

    config = load_config(Path(args.config))
    det_cfg = config.get("bin_detector", {})
    cam_cfg = config.get("camera", {})

    task = (args.task or det_cfg.get("task", "obb")).lower()
    if task == "detect":
        from integration.src.detectors import initialize_bins_detect as detector
    else:
        from integration.src.detectors import initialize_bins_obb as detector
    import grid_calibrator as gc  # bare import, see sys.path setup above

    model_path = resolve_model_path(args.model or det_cfg.get("model_path"))
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        return
    model = detector.load_model(str(model_path))
    if model is None:
        logger.error("Model failed to load — nothing to score.")
        return

    try:
        calibration = build_calibration(Path(args.calibration_image), model, detector, gc)
    except ValueError as e:
        logger.error("Calibration failed: %s", e)
        return

    if args.video:
        rotate_180 = args.rotate_180 if args.rotate_180 is not None else bool(cam_cfg.get("rotate_180", False))
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error("Video not found: %s", video_path)
            return
        all_named = extract_frames(video_path, args.fps, rotate_180)
        sample = all_named[args.skip_start: args.skip_start + args.sample_frames]
        run_tag = f"{args.run or 'run'}_{args.fill or 'unlabeled'}"
        out_dir = Path(args.out_dir) if args.out_dir else _DEFAULT_OUT_ROOT / run_tag
    else:
        rotate_180 = bool(args.rotate_180)  # default off — photo folders are already correctly oriented
        image_dir = Path(args.dir)
        if not image_dir.is_dir():
            logger.error("Folder not found: %s", image_dir)
            return
        all_named = load_images_from_dir(image_dir, rotate_180)
        sample = all_named[: args.sample_frames]
        out_dir = Path(args.out_dir) if args.out_dir else image_dir

    if len(sample) < args.sample_frames:
        logger.warning("Only %d frame(s) available (wanted %d) — scoring what's there",
                        len(sample), args.sample_frames)
    if not sample:
        logger.error("No frames to score.")
        return

    names = [n for n, _ in sample]
    frames = [f for _, f in sample]
    occupancies = score(frames, model, detector, calibration, gc)
    summary = summarize(occupancies, calibration)
    passed = summary["accuracy"] >= args.pass_threshold
    failure_notes = format_failure_notes(summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame, occ in zip(names, frames, occupancies):
        vis = draw_occupancy(frame, occ, calibration)
        cv2.imwrite(str(out_dir / f"{name}_scored.jpg"), vis)

    print()
    print("Run\tGroup\tExact setting\tFill level\tSample size\tBins detected accurately (%)\tPass/Fail\tFailure notes\tEvidence ref")
    print(f"{args.run}\t{args.group}\t{args.setting}\t{args.fill}\t{summary['n_frames']}\t"
          f"{summary['accuracy']:.0f}%\t{'Pass' if passed else 'Fail'}\t{failure_notes}\t{out_dir}")
    print()
    logger.info("Per-slot breakdown:")
    for sid, s in sorted(summary["per_slot"].items(), key=lambda kv: kv[1]["index"]):
        conf_str = f", avg conf {s['avg_conf']:.2f}" if s["avg_conf"] is not None else ""
        logger.info("  slot %d: %d/%d frames%s", s["index"], s["present_count"], s["total"], conf_str)


if __name__ == "__main__":
    main()
