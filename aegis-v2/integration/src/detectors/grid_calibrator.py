"""
Grid Calibrator
===============
Pure functions for the "calibrate from a full snapshot, then match" workflow.
No camera / model / cv2 — just geometry, so it unit-tests in milliseconds.

* ``calibrate_grid`` turns an all-9-bins snapshot into the 9 fixed slots (the grid),
  indexed 1..6 (top) / 7..9 (bottom), storing each slot's real centre + box.
* ``match_to_grid`` matches a later snapshot's detections to the nearest calibrated
  slot (same row, one-to-one, within a distance cutoff) -> per-slot occupancy.

A detection is ``{"corners": [[x,y]*4], "center": [cx, cy], "conf": float}``.
"""
from __future__ import annotations

import logging
import math

import grid_allocator as ga  # reuse the proven row split

logger = logging.getLogger("aegis.detectors.grid_calibrator")


def _cx(det) -> float:
    return float(det["center"][0])


def _cy(det) -> float:
    return float(det["center"][1])


def _median(values: list) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def calibrate_grid(detections: list, layout: list[int] | None = None) -> dict:
    """Full-grid snapshot -> the fixed slots (the grid).

    ``layout`` is the per-row bin count (default ``[6, 3]`` = 6 top + 3 bottom;
    ``[6]`` = a single top row of 6). Splits into ``len(layout)`` rows by y-gap,
    validates each row's bin count, orders each row left->right, and assigns
    indices 1..N (top row first). Every row renders against a ``max(layout)``-slot
    track, so a row with fewer bins spans wider (the 3 bottom bins span 2 slots
    each). Stores each slot's centre, box, and the row's median centre-to-centre
    spacing (the match cutoff). Raises ValueError if the snapshot doesn't match.
    """
    layout = list(layout) if layout else [6, 3]
    num_rows = len(layout)
    track = max(layout) if layout else 1

    if not all(math.isfinite(_cx(d)) and math.isfinite(_cy(d)) for d in detections):
        raise ValueError(
            "Calibration snapshot contains non-finite detection centres; "
            "retake the snapshot."
        )
    rows = ga.split_rows_by_y(detections, num_rows=num_rows)
    got = [len(r) for r in rows]
    if got != layout:
        want = " + ".join(str(n) for n in layout)
        raise ValueError(
            f"Calibration needs rows of {want} bins (total {sum(layout)}); got "
            f"{' / '.join(str(n) for n in got)}. Retake the snapshot with all "
            f"{sum(layout)} bins clearly separated into {num_rows} row(s)."
        )

    slots: dict = {}
    index = 1
    for row_idx, (count, row_dets) in enumerate(zip(layout, rows)):
        layer = ga._layer_name(row_idx, num_rows)
        span = track // count if count and track % count == 0 else 1
        ordered = sorted(row_dets, key=_cx)
        centers = [_cx(d) for d in ordered]
        diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        row_spacing = _median(diffs) if diffs else 0.0
        if count >= 2 and row_spacing <= 0:
            raise ValueError(
                f"Degenerate {layer}-row spacing ({row_spacing}); bins overlap in x. "
                f"Retake the calibration snapshot with all bins clearly separated."
            )
        for d in ordered:
            slots[f"slot_{index}"] = {
                "index": index,
                "row": row_idx,
                "layer": layer,
                "span": span,
                "center": [_cx(d), _cy(d)],
                "corners": d["corners"],
                "row_spacing": row_spacing,
            }
            index += 1
    logger.info("Calibrated %d slot(s) across %d row(s) (layout=%s)",
                len(slots), num_rows, layout)
    return slots


def match_to_grid(detections: list, calibration: dict) -> dict:
    """Match a kitting-list snapshot's detections to calibrated slots.

    Each detection is assigned a row (nearer of the two calibrated row mean-y's),
    then matched greedily to the nearest *same-row* calibrated slot centre, one-to-one,
    provided the distance is within ``0.5 * row_spacing``. Matched slots get the live
    box and ``present=True``; unmatched slots stay ``present=False`` ("bin not there").
    Detections matching no slot within cutoff are dropped with a warning.

    ``calibration`` is the slots dict from ``calibrate_grid`` (or the ``"slots"`` value
    loaded from grid_calibration.json).
    """
    # Skeleton: every calibrated slot, present=False to start.
    result: dict = {}
    for sid, info in calibration.items():
        result[sid] = {
            "index": info["index"],
            "row": info["row"],
            "layer": info["layer"],
            "span": info["span"],
            "present": False,
        }
    if not detections:
        return result

    # Drop detections with non-finite centres (raw OBB output can emit NaN/inf).
    finite = [d for d in detections if math.isfinite(_cx(d)) and math.isfinite(_cy(d))]
    if len(finite) != len(detections):
        logger.warning("Dropped %d detection(s) with non-finite centre",
                       len(detections) - len(finite))
    detections = finite
    if not detections:
        return result

    # Calibrated row mean-y, to assign each detection to its nearest row. Built
    # from the rows actually present in the calibration (1, 2, or more), so a
    # single-row layout (e.g. [6]) doesn't divide by an empty row.
    row_cys: dict = {}
    for info in calibration.values():
        row_cys.setdefault(info["row"], []).append(info["center"][1])
    row_mean = {r: sum(v) / len(v) for r, v in row_cys.items()}

    # Same-row (distance, det_idx, slot_id) pairs within the match cutoff (0.5 * row spacing).
    pairs = []
    for di, d in enumerate(detections):
        cx, cy = _cx(d), _cy(d)
        row = min(row_mean, key=lambda r: abs(cy - row_mean[r]))
        for sid, info in calibration.items():
            if info["row"] != row:
                continue
            sx, sy = info["center"]
            dist = math.hypot(cx - sx, cy - sy)
            cutoff = 0.5 * float(info["row_spacing"])
            if dist <= cutoff:
                pairs.append((dist, di, sid))

    # Greedy nearest, one-to-one.
    pairs.sort(key=lambda t: t[0])
    used_dets: set = set()
    used_slots: set = set()
    for dist, di, sid in pairs:
        if di in used_dets or sid in used_slots:
            continue
        d = detections[di]
        result[sid].update({
            "present": True,
            "center": [_cx(d), _cy(d)],
            "corners": d["corners"],
            "confidence": float(d.get("conf", 0.0)),
        })
        used_dets.add(di)
        used_slots.add(sid)

    dropped = len(detections) - len(used_dets)
    if dropped:
        logger.warning("Dropped %d detection(s) not matched to any slot", dropped)
    return result
