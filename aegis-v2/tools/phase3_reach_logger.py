"""
Phase 3 hand-reach event logger (target-gated)
=================================================
Live capture tool for the "Phase 3 — Hand pose" test sheet: reuses the SAME
hand tracker (mediapipe, via ``TrackerRegistry``), bin-assignment engine
(``BinAssignmentEngine``, ``finger_vote`` method per settings.yaml), and
overlay renderer (``OverlayUI``) the real pipeline uses.

Unlike a passive logger, this one is TARGET-GATED: only the current step's
intended bin is registered with the assignment engine (``set_bin_map_from_geofences``
with a single entry). A hand passing through the bottom row on its way to a
top bin therefore can't register a false hit on the bottom bin — that bin
isn't in the engine's map at all, so it just returns no match, not a wrong
match. This replaces the earlier passive-log-everything design, which logged
every transit-through bin as if it were an intended reach.

You give the run's full intended target sequence up front (``--bins``/
``--repeats`` or ``--targets``, same spec language as ``score_phase3_run.py``).
Reach into the highlighted target as many times as you intend (watch the
on-screen "samples confirmed" counter), then press SPACE to lock in that
target's tally and gate onto the next one. Because gating is manual rather
than auto-advance-at-N, a target where detection keeps missing shows up
honestly as fewer confirmed samples than you attempted — that's the signal.

Since each invocation is scoped to ONE run's target sequence (each sheet row
has different bins/repeats/hand), there's no cross-run 'n'/'p' browsing here
— just re-run with the next row's ``--run``/`--bins`/`--repeats`.

Run (from aegis-v2/):
    python tools/phase3_reach_logger.py --run 1 --bins 1-9 --repeats 10 --hand right

Controls:
    SPACE = lock in the current target's tally, gate onto the next target
    s     = save a snapshot of the current frame (with overlay) as evidence
    c     = recalibrate the 9-slot grid from the current frame
    q     = quit (flushes reach_log.csv + target_summary.csv)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("aegis.tools.phase3_reach_logger")

_ROOT = Path(__file__).resolve().parent.parent  # aegis-v2/
sys.path.insert(0, str(_ROOT.parent))  # repo root
sys.path.insert(0, str(_ROOT))          # aegis-v2/
sys.path.insert(0, str(_ROOT / "integration" / "src" / "detectors"))  # grid_calibrator's bare imports

_DEFAULT_CONFIG = _ROOT / "integration" / "config" / "settings.yaml"
_DEFAULT_OUT_ROOT = _ROOT / "tools" / "Dev_test_data"
_LOG_COLUMNS = ["index", "timestamp", "hand_id", "handedness", "target_index", "target_bin_id", "confidence"]
_SUMMARY_COLUMNS = ["target_index", "bin_id", "repeats_intended", "samples_confirmed"]


def slot_to_bin_id(n: int) -> str:
    """1..6 -> top row bin_0_0..bin_0_5, 7..9 -> bottom row bin_1_0..bin_1_2."""
    if 1 <= n <= 6:
        return f"bin_0_{n - 1}"
    if 7 <= n <= 9:
        return f"bin_1_{n - 7}"
    raise ValueError(f"Slot number must be 1..9, got {n}")


def parse_targets_spec(spec: str) -> list:
    """'1x10,2x10,9' -> [(bin_0_0,10)]*... flattened to one (bin_id, repeats) per block."""
    blocks = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "x" in token:
            slot_str, repeat_str = token.split("x", 1)
            slot, repeat = int(slot_str), int(repeat_str)
        else:
            slot, repeat = int(token), 1
        blocks.append((slot_to_bin_id(slot), repeat))
    return blocks


def parse_bins_range(bins_range: str, repeats: int) -> list:
    start_str, end_str = bins_range.split("-", 1)
    start, end = int(start_str), int(end_str)
    return [(slot_to_bin_id(slot), repeats) for slot in range(start, end + 1)]


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
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("Camera negotiated: %dx%d @ %.1f fps",
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                cap.get(cv2.CAP_PROP_FPS))
    return cap


def grab_warm_frame(cap, warmup: int, rotate_180: bool):
    frame = None
    for _ in range(max(1, warmup)):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
    if frame is not None and rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def calibrate_geofences(frame, bin_model, bin_detector, gc):
    """Detect the 9 bins and return a geofence dict via GridSession, like the pipeline does."""
    from integration.src.detectors.grid_session import GridSession
    dets = bin_detector.detect_bins(frame, bin_model, expected_count=9)
    session = GridSession()
    session.calibrate(dets)  # raises ValueError if not a clean 6+3
    return session.to_geofences()


_DISABLED_COLOR = (90, 90, 90)
_ACTIVE_COLOR = (0, 255, 255)


def _bin_corners(c: dict):
    poly = c.get("polygon")
    if poly:
        return np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2), None
    return None, (int(c["x_min"]), int(c["y_min"]), int(c["x_max"]), int(c["y_max"]))


def draw_gated_overlay(frame, all_geofences: dict, target_bin_id, hands: list, overlay_ui) -> np.ndarray:
    """Every non-target bin is drawn dim/grey and labeled 'disabled' (it's not
    even registered with the assignment engine, so it truly can't fire); the
    current target is drawn bold with a 'REACH HERE' label. Hand landmarks are
    drawn via OverlayUI's own (private but stable) hand renderer, reused as-is
    rather than reimplemented."""
    display = frame.copy()

    for bid, c in all_geofences.items():
        if bid == target_bin_id:
            continue
        pts, box = _bin_corners(c)
        cx, cy = int((c["x_min"] + c["x_max"]) / 2), int((c["y_min"] + c["y_max"]) / 2)
        if pts is not None:
            cv2.polylines(display, [pts], True, _DISABLED_COLOR, 1)
        else:
            cv2.rectangle(display, box[:2], box[2:], _DISABLED_COLOR, 1)
        cv2.putText(display, f"{bid} (disabled)", (cx - 45, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _DISABLED_COLOR, 1)

    if target_bin_id is not None and target_bin_id in all_geofences:
        c = all_geofences[target_bin_id]
        pts, box = _bin_corners(c)
        cx, cy = int((c["x_min"] + c["x_max"]) / 2), int((c["y_min"] + c["y_max"]) / 2)
        if pts is not None:
            cv2.polylines(display, [pts], True, _ACTIVE_COLOR, 3)
        else:
            cv2.rectangle(display, box[:2], box[2:], _ACTIVE_COLOR, 3)
        cv2.putText(display, "REACH HERE", (cx - 55, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, _ACTIVE_COLOR, 2)
        cv2.putText(display, target_bin_id, (cx - 30, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _ACTIVE_COLOR, 2)

    for hand in hands:
        overlay_ui._draw_hand(display, hand)

    return display


def append_row(csv_path: Path, row: dict, columns: list) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class HandDebouncer:
    """Per-hand debounced bin state: only reports a change once it holds for N frames."""

    def __init__(self, debounce_frames: int):
        self._debounce = debounce_frames
        self._confirmed: dict = {}   # hand_id -> bin_id (or None)
        self._candidate: dict = {}   # hand_id -> (bin_id, count)

    def reset(self) -> None:
        self._confirmed.clear()
        self._candidate.clear()

    def update(self, events: list) -> list:
        """Returns newly-confirmed (hand_id, handedness, bin_id, confidence) transitions this frame."""
        confirmed_now = []
        for ev in events:
            cand_bin, cand_count = self._candidate.get(ev.hand_id, (object(), 0))
            if ev.bin_id == cand_bin:
                cand_count += 1
            else:
                cand_bin, cand_count = ev.bin_id, 1
            self._candidate[ev.hand_id] = (cand_bin, cand_count)

            if cand_count >= self._debounce and self._confirmed.get(ev.hand_id, object()) != cand_bin:
                self._confirmed[ev.hand_id] = cand_bin
                confirmed_now.append((ev.hand_id, ev.handedness, cand_bin, ev.confidence))
        return confirmed_now


def main() -> None:
    ap = argparse.ArgumentParser(description="Target-gated live hand-reach logger for the Phase 3 test sheet")
    ap.add_argument("--run", type=int, required=True, help="run number (output folder phase3_run_<N>)")
    ap.add_argument("--targets", default=None, help="explicit sequence, e.g. '1x10,2x10,9x5'")
    ap.add_argument("--bins", default=None, help="slot range shorthand, e.g. '1-9' or '7-9'")
    ap.add_argument("--repeats", type=int, default=10, help="(--bins only) reaches to aim for per target")
    ap.add_argument("--hand", default="any", choices=["any", "left", "right"],
                     help="only count confirmations from this hand (default: any)")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG), help="path to settings.yaml")
    ap.add_argument("--camera", default=None, help="override camera.source")
    ap.add_argument("--bin-model", default=None, help="override bin_detector.model_path")
    ap.add_argument("--bin-task", default=None, choices=["detect", "obb"], help="override bin_detector.task")
    ap.add_argument("--debounce", type=int, default=3, help="frames a bin assignment must hold before logging")
    ap.add_argument("--out-root", default=str(_DEFAULT_OUT_ROOT), help="parent folder for phase3_run_<N> subfolders")
    args = ap.parse_args()

    if bool(args.targets) == bool(args.bins):
        logger.error("Pass exactly one of --targets or --bins")
        return
    target_blocks = parse_targets_spec(args.targets) if args.targets else parse_bins_range(args.bins, args.repeats)
    if not target_blocks:
        logger.error("Empty target sequence.")
        return

    config = load_config(Path(args.config))
    cam_cfg = config.get("camera", {})
    det_cfg = config.get("bin_detector", {})
    ht_cfg = config.get("hand_tracker", {})
    assign_cfg = config.get("bin_assignment", {})
    ui_cfg = config.get("ui", {})

    task = (args.bin_task or det_cfg.get("task", "obb")).lower()
    if task == "detect":
        from integration.src.detectors import initialize_bins_detect as bin_detector
    else:
        from integration.src.detectors import initialize_bins_obb as bin_detector
    import grid_calibrator as gc  # bare import, see sys.path setup above

    bin_model_path = resolve_model_path(args.bin_model or det_cfg.get("model_path"))
    if not bin_model_path.exists():
        logger.error("Bin detector model not found: %s", bin_model_path)
        return
    bin_model = bin_detector.load_model(str(bin_model_path))
    if bin_model is None:
        logger.error("Bin detector failed to load.")
        return

    from hand_models.common import TrackerRegistry
    import hand_models.mediapipe.tracker  # noqa: F401  (registers "mediapipe")
    logger.info("Loading hand tracker: %s", ht_cfg.get("backend", "mediapipe"))
    hand_tracker = TrackerRegistry.create(ht_cfg.get("backend", "mediapipe"), ht_cfg)

    from integration.src.engine import BinAssignmentEngine, BinRegion
    from integration.src.ui.overlay import OverlayUI
    assignment = BinAssignmentEngine(assign_cfg)

    source = args.camera if args.camera is not None else cam_cfg.get("source", 0)
    cap = open_camera(source, cam_cfg.get("width", 1280), cam_cfg.get("height", 720), cam_cfg.get("fps", 30))
    rotate_180 = bool(cam_cfg.get("rotate_180", False))
    warmup = cam_cfg.get("warmup_frames", 30)

    logger.info("Warming up for baseline calibration...")
    baseline = grab_warm_frame(cap, warmup, rotate_180)
    if baseline is None:
        logger.error("Failed to capture a baseline frame.")
        cap.release()
        return
    try:
        all_geofences = calibrate_geofences(baseline, bin_model, bin_detector, gc)
    except ValueError as e:
        logger.error("Calibration failed: %s", e)
        cap.release()
        return
    logger.info("Calibrated %d bin slot(s)", len(all_geofences))
    all_bins = [BinRegion(bin_id=bid, label=bid, x_min=c["x_min"], x_max=c["x_max"],
                           y_min=c["y_min"], y_max=c["y_max"], confidence=c.get("confidence", 0.0),
                           polygon=c.get("polygon")) for bid, c in all_geofences.items()]
    overlay = OverlayUI(ui_cfg, all_bins)

    def gate_to(bin_id: str) -> None:
        """Only this ONE bin is registered with the assignment engine — a hand
        elsewhere in frame (e.g. transiting past the bottom row en route to a
        top bin) simply gets no match instead of a false hit on the wrong bin."""
        if bin_id not in all_geofences:
            raise KeyError(f"{bin_id} was not part of the calibrated grid: {sorted(all_geofences)}")
        assignment.set_bin_map_from_geofences({bin_id: all_geofences[bin_id]})

    folder = Path(args.out_root) / f"phase3_run_{args.run:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "reach_log.csv"
    summary_path = folder / "target_summary.csv"

    debouncer = HandDebouncer(args.debounce)
    target_idx = 0
    samples_confirmed = 0
    log_index = 0
    gate_to(target_blocks[target_idx][0])
    debouncer.reset()

    logger.info("Target 1/%d: %s (aim for %d)", len(target_blocks), target_blocks[0][0], target_blocks[0][1])
    logger.info("Controls: SPACE=lock in & next target  s=save snapshot  c=recalibrate  q=quit")

    def finalize_target() -> None:
        nonlocal samples_confirmed
        bin_id, repeats_intended = target_blocks[target_idx]
        append_row(summary_path, {"target_index": target_idx + 1, "bin_id": bin_id,
                                   "repeats_intended": repeats_intended, "samples_confirmed": samples_confirmed},
                   _SUMMARY_COLUMNS)
        logger.info("Target %d/%d (%s) locked in: %d/%d confirmed",
                    target_idx + 1, len(target_blocks), bin_id, samples_confirmed, repeats_intended)

    try:
        while True:
            done = target_idx >= len(target_blocks)
            ret, frame = cap.read()
            if not ret:
                continue
            if rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            if not done:
                hands = hand_tracker.detect(frame)
                events = assignment.assign(hands, frame.shape, None)
                target_bin_id = target_blocks[target_idx][0]
                for hand_id, handedness, bin_id, confidence in debouncer.update(events):
                    if bin_id != target_bin_id:
                        continue  # an "exit" (bin_id None) confirmation — not a sample
                    if args.hand != "any" and handedness != args.hand:
                        continue
                    samples_confirmed += 1
                    log_index += 1
                    append_row(log_path, {"index": log_index, "timestamp": f"{time.time():.3f}",
                                           "hand_id": hand_id, "handedness": handedness,
                                           "target_index": target_idx + 1, "target_bin_id": bin_id,
                                           "confidence": f"{confidence:.3f}"}, _LOG_COLUMNS)
                    logger.info("Target %d/%d (%s): sample %d confirmed (%s hand, conf %.2f)",
                                target_idx + 1, len(target_blocks), target_bin_id, samples_confirmed,
                                handedness, confidence)
                display = draw_gated_overlay(frame, all_geofences, target_bin_id, hands, overlay)
            else:
                display = draw_gated_overlay(frame, all_geofences, None, [], overlay)

            if done:
                instruction = f"ALL {len(target_blocks)} TARGETS DONE for run {args.run}"
            else:
                bin_id, repeats = target_blocks[target_idx]
                instruction = (f"Step {target_idx + 1}/{len(target_blocks)}: reach into {bin_id}  "
                                f"({samples_confirmed}/{repeats} confirmed)")
            cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(display, instruction, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(display, "SPACE=lock in & next target  s=save  c=recalibrate  q=quit",
                        (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imshow("Phase 3 reach logger (live, target-gated)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if not done:
                    finalize_target()
                break
            elif key == ord("c"):
                try:
                    all_geofences = calibrate_geofences(frame, bin_model, bin_detector, gc)
                    all_bins = [BinRegion(bin_id=bid, label=bid, x_min=c["x_min"], x_max=c["x_max"],
                                           y_min=c["y_min"], y_max=c["y_max"], confidence=c.get("confidence", 0.0),
                                           polygon=c.get("polygon")) for bid, c in all_geofences.items()]
                    overlay = OverlayUI(ui_cfg, all_bins)
                    if not done:
                        gate_to(target_blocks[target_idx][0])
                    logger.info("Recalibrated %d bin slot(s)", len(all_geofences))
                except ValueError as e:
                    logger.error("Recalibration failed: %s", e)
            elif key == ord("s"):
                n_snaps = len(list(folder.glob("snapshot_*.jpg")))
                out_path = folder / f"snapshot_{n_snaps + 1:02d}.jpg"
                cv2.imwrite(str(out_path), display)
                logger.info("Saved snapshot: %s", out_path)
            elif key == 32 and not done:  # SPACE
                finalize_target()
                target_idx += 1
                samples_confirmed = 0
                debouncer.reset()
                if target_idx < len(target_blocks):
                    gate_to(target_blocks[target_idx][0])
                    logger.info("Target %d/%d: %s (aim for %d)", target_idx + 1, len(target_blocks),
                                target_blocks[target_idx][0], target_blocks[target_idx][1])
                else:
                    logger.info("All targets done for run %d — press 'q' to quit.", args.run)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Done — %s / %s", summary_path, log_path)


if __name__ == "__main__":
    main()
