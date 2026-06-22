"""
Snapshot OBB driver
===================
Two modes over the aegis-core OBB handoff (``bins.json`` + ``snapshot.jpg``):

* ``--calibrate`` (once per workstation, all 9 bins): define the grid via
  ``grid_calibrator.calibrate_grid`` and write ``grid_calibration.json`` (+ overlay).
* default match mode (per kitting list): load the calibration and match the snapshot's
  detections to the nearest slot via ``grid_calibrator.match_to_grid``, writing
  ``occupancy.json`` and a green(present)/grey(absent) overlay.

The older band-based ``index_bins`` (via ``grid_allocator``) is kept as a library
fallback but is no longer used by ``main``.

Run (from aegis-v2/integration) as a PATH SCRIPT — this bypasses the detectors
package __init__, which eagerly imports cv2 / ultralytics:
    # calibrate the workstation (all 9 bins present)
    python src/detectors/snapshot_obb.py --bins ../../aegis-core/runs/bins_obb_raw/bins.json --calibrate
    # then, per kitting list
    python src/detectors/snapshot_obb.py --bins ../../aegis-core/runs/bins_obb_raw/bins.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Allow both ``python -m src.detectors.snapshot_obb`` and direct path import in tests.
sys.path.insert(0, os.path.dirname(__file__))
import grid_allocator as ga  # noqa: E402
import grid_calibrator as gc  # noqa: E402

logger = logging.getLogger("aegis.detectors.snapshot_obb")


def index_bins(payload: dict) -> dict:
    """Pure transform: raw handoff payload -> indexed grid dict.

    ``payload`` is the parsed ``bins.json`` ({"frame_w", "bins": [...]}). Returns
    {"bins": {bin_i: {...}}, "frame_w", "frame_h", "source", "rule"}.
    """
    frame_w = int(payload.get("frame_w") or 1280)
    detections = payload.get("bins", [])
    grid = ga.allocate_grid(detections, frame_w)
    return {
        "bins": grid,
        "frame_w": frame_w,
        "frame_h": int(payload.get("frame_h") or 0),
        "source": "obb",
        "rule": "rotate180_then_band_split",
    }


def build_calibration(payload: dict) -> dict:
    """Pure: parsed bins.json (all 9 bins) -> calibration payload."""
    slots = gc.calibrate_grid(payload.get("bins", []))
    return {
        "source": "obb",
        "frame_w": int(payload.get("frame_w") or 1280),
        "frame_h": int(payload.get("frame_h") or 0),
        "slots": slots,
    }


def build_occupancy(payload: dict, calibration_payload: dict) -> dict:
    """Pure: parsed bins.json + calibration payload -> per-slot occupancy."""
    occ = gc.match_to_grid(payload.get("bins", []), calibration_payload["slots"])
    return {
        "source": "obb",
        "rule": "calibrated_nearest_slot",
        "frame_w": int(payload.get("frame_w") or 1280),
        "frame_h": int(payload.get("frame_h") or 0),
        "slots": occ,
    }


def _draw_overlay(image, grid: dict):
    import cv2
    import numpy as np
    vis = image.copy()
    for info in grid.values():
        idx = info["index"]
        if info["detected"]:
            pts = np.array(info["corners"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cx, cy = int(info["center"][0]), int(info["center"][1])
            cv2.putText(vis, str(idx), (cx - 10, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # (undetected cells are left unmarked on the snapshot)
    return vis


def _draw_calibration(image, cal_payload: dict):
    import cv2
    import numpy as np
    vis = image.copy()
    for info in cal_payload["slots"].values():
        pts = np.array(info["corners"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (255, 200, 0), 2)
        cx, cy = int(info["center"][0]), int(info["center"][1])
        cv2.putText(vis, str(info["index"]), (cx - 10, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
    return vis


def _draw_occupancy(image, occ_payload: dict, cal_payload: dict):
    import cv2
    import numpy as np
    vis = image.copy()
    for sid, info in occ_payload["slots"].items():
        idx = info["index"]
        if info["present"]:
            pts = np.array(info["corners"], dtype=np.int32).reshape((-1, 1, 2))
            cx, cy = int(info["center"][0]), int(info["center"][1])
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)            # present = green
            cv2.putText(vis, str(idx), (cx - 10, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cal = cal_payload["slots"].get(sid)                        # absent = grey calib box
            if cal is None:
                continue
            pts = np.array(cal["corners"], dtype=np.int32).reshape((-1, 1, 2))
            cx, cy = int(cal["center"][0]), int(cal["center"][1])
            cv2.polylines(vis, [pts], True, (130, 130, 130), 2)
            cv2.putText(vis, str(idx), (cx - 10, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (130, 130, 130), 2)
    return vis


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    here = os.path.dirname(__file__)
    default_bins = os.path.abspath(os.path.join(
        here, "..", "..", "..", "..", "aegis-core", "runs", "bins_obb_raw", "bins.json"))

    ap = argparse.ArgumentParser(description="Calibrate the bin grid, or match a snapshot to it")
    ap.add_argument("--bins", default=default_bins, help="path to aegis-core bins.json")
    ap.add_argument("--calibrate", action="store_true",
                    help="define the grid from an all-9-bins snapshot (writes grid_calibration.json)")
    ap.add_argument("--calibration", default=None,
                    help="path to grid_calibration.json (default: sibling of --bins)")
    ap.add_argument("--out", default=None, help="output json path")
    ap.add_argument("--no-show", action="store_true", help="don't open the overlay window")
    args = ap.parse_args()

    if not os.path.exists(args.bins):
        logger.error("bins.json not found: %s — run the aegis-core OBB script first.", args.bins)
        return
    try:
        with open(args.bins) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Could not read %s: %s", args.bins, e)
        return

    bins_dir = os.path.dirname(args.bins)
    cal_path = args.calibration or os.path.join(bins_dir, "grid_calibration.json")

    def _load_image():
        snap = os.path.join(bins_dir, "snapshot.jpg")
        if not os.path.exists(snap):
            logger.warning("No snapshot.jpg next to bins.json — skipping overlay.")
            return None
        import cv2
        image = cv2.imread(snap)
        if image is None:
            logger.error("Could not read snapshot image: %s — skipping overlay.", snap)
        return image

    if args.calibrate:
        try:
            cal_payload = build_calibration(payload)
        except ValueError as e:
            logger.error("Calibration failed: %s", e)
            return
        os.makedirs(os.path.dirname(os.path.abspath(cal_path)), exist_ok=True)
        with open(cal_path, "w") as f:
            json.dump(cal_payload, f, indent=2)
        logger.info("✓ Wrote calibration %s (9 slots)", cal_path)
        image = _load_image()
        if image is not None:
            import cv2
            vis = _draw_calibration(image, cal_payload)
            vis_path = os.path.join(bins_dir, "grid_calibration_overlay.jpg")
            cv2.imwrite(vis_path, vis)
            logger.info("✓ Wrote overlay %s", vis_path)
            if not args.no_show:
                cv2.imshow("Calibrated grid (1-9) - 'q' to close", vis)
                while cv2.waitKey(20) & 0xFF != ord("q"):
                    pass
                cv2.destroyAllWindows()
        return

    # match mode
    if not os.path.exists(cal_path):
        logger.error("No calibration at %s - calibrate this workstation first "
                     "(run with --calibrate on an all-9-bins snapshot).", cal_path)
        return
    try:
        with open(cal_path) as f:
            cal_payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Could not read calibration %s: %s — recalibrate.", cal_path, e)
        return
    if "slots" not in cal_payload:
        logger.error("Calibration %s has no 'slots' — recalibrate this workstation.", cal_path)
        return
    cal_w, cal_h = cal_payload.get("frame_w"), cal_payload.get("frame_h")
    new_w, new_h = payload.get("frame_w"), payload.get("frame_h")
    if (cal_w and new_w and cal_w != new_w) or (cal_h and new_h and cal_h != new_h):
        logger.warning("Snapshot frame (%sx%s) differs from calibration (%sx%s); "
                       "matching may be off — recalibrate at the current resolution.",
                       new_w, new_h, cal_w, cal_h)
    try:
        occ_payload = build_occupancy(payload, cal_payload)
    except (KeyError, ZeroDivisionError, TypeError, ValueError) as e:
        logger.error("Calibration %s is corrupt/incomplete (%s) — recalibrate.", cal_path, e)
        return
    out_path = args.out or os.path.join(bins_dir, "occupancy.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(occ_payload, f, indent=2)
    present = sum(1 for v in occ_payload["slots"].values() if v["present"])
    logger.info("✓ Wrote %s (%d/9 slots occupied)", out_path, present)
    image = _load_image()
    if image is not None:
        import cv2
        vis = _draw_occupancy(image, occ_payload, cal_payload)
        vis_path = os.path.join(bins_dir, "occupancy_overlay.jpg")
        cv2.imwrite(vis_path, vis)
        logger.info("✓ Wrote overlay %s", vis_path)
        if not args.no_show:
            cv2.imshow("Bin occupancy (green=present, grey=absent) - 'q' to close", vis)
            while cv2.waitKey(20) & 0xFF != ord("q"):
                pass
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
