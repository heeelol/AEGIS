"""
Grid Allocator
==============
Pure functions that place raw OBB bin detections onto a FIXED, pre-indexed grid.
No camera / model / cv2 — just geometry, so it unit-tests in milliseconds.

Contract
--------
* The grid is indexed FIRST and is fixed. Indices come from the grid position,
  never from detection order.
* A detection is dropped into the cell it occupies; a cell with no detection stays
  ``detected=False`` (no renumbering of any other index).
* Bin spans are hardcoded by row (top = 1 cell, bottom = 2 cells). Inferring span /
  size from box geometry, and generalising beyond the fixed grid, are FUTURE_TASKS.md.

Output: dict keyed ``bin_{index}`` (index is the canonical 1..N bin id).
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger("aegis.detectors.grid_allocator")

# Fixed rig: top row of 6 single-cell bins, bottom row of 3 double-cell bins.
DEFAULT_LAYOUT = [[1, 1, 1, 1, 1, 1], [2, 2, 2]]

# A vertical gap counts as a row boundary only if it exceeds this fraction of the
# median bin height (keeps a single shared-y row from being force-split).
Y_GAP_FACTOR = 0.5


def _layer_name(row: int, num_rows: int) -> str:
    if num_rows == 2:
        return "top" if row == 0 else "bottom"
    return f"row{row}"


def build_skeleton(layout: list[list[int]]) -> dict:
    """Fixed, pre-indexed grid. Index runs 1..N across rows (top row first)."""
    skeleton: dict = {}
    num_rows = len(layout)
    index = 1
    for row, spans in enumerate(layout):
        row_slots = sum(int(s) for s in spans)
        num_bins = len(spans)
        slot_start = 0
        for col, span in enumerate(spans):
            span = int(span)
            skeleton[f"bin_{index}"] = {
                "index": index,
                "layer": _layer_name(row, num_rows),
                "row": row,
                "col": col,
                "slot_start": slot_start,
                "span": span,
                "row_slots": row_slots,
                "num_bins": num_bins,
                "detected": False,
                "confidence": 0.0,
            }
            slot_start += span
            index += 1
    return skeleton


def _cy(det) -> float:
    return float(det["center"][1])


def _cx(det) -> float:
    return float(det["center"][0])


def _box_height(det) -> float:
    ys = [p[1] for p in det.get("corners", [])]
    return float(max(ys) - min(ys)) if ys else 0.0


def _median(values: list) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def split_rows_by_y(detections: list, num_rows: int) -> list:
    """Group detections into ``num_rows`` rows by splitting at SIGNIFICANT y-gaps.

    A gap counts as a row boundary only if it exceeds ~half the median bin height,
    so detections that share a row (near-equal y) are never force-split. At most
    ``num_rows - 1`` boundaries are taken (the largest qualifying gaps). When fewer
    qualify, later rows stay empty and earlier rows absorb the detections (a single
    detected band lands in row 0). Robust when fewer detections than rows are present.
    """
    rows = [[] for _ in range(num_rows)]
    if not detections:
        return rows
    ordered = sorted(detections, key=_cy)
    if num_rows == 1 or len(ordered) == 1:
        rows[0] = list(ordered)
        return rows

    median_h = _median([_box_height(d) for d in ordered])
    if median_h > 0:
        threshold = Y_GAP_FACTOR * median_h
    else:
        # Heights unknown/degenerate: fall back to a fraction of the total y-span
        # (0 when all detections share a y, so a single row is not force-split).
        threshold = 0.25 * (_cy(ordered[-1]) - _cy(ordered[0]))
    # gaps[i] = vertical distance between ordered[i] and ordered[i+1]
    gaps = [(ordered[i + 1]["center"][1] - ordered[i]["center"][1], i)
            for i in range(len(ordered) - 1)]
    significant = [(g, i) for g, i in gaps if g > threshold]
    # largest qualifying gaps first (sort by gap value only — never by index)
    significant.sort(key=lambda t: t[0], reverse=True)
    boundaries = sorted(i for _, i in significant[:num_rows - 1])

    band = 0
    start = 0
    for b in boundaries:
        rows[band] = ordered[start:b + 1]
        band += 1
        start = b + 1
    rows[band] = ordered[start:]
    return rows


def _assign_row(dets_in_row: list, cells: list, frame_w: float, skeleton: dict) -> None:
    """Place each detection into its frame-band cell (nearest unused on collision).

    ``cells`` is the row's list of (bin_id, info) ordered left->right. The band a
    detection falls in is ``floor(cx / frame_w * num_bins)``; with the rig filling the
    frame this is the bin's physical column. A collision snaps to the nearest free band.
    """
    num_bins = len(cells)
    if num_bins == 0:
        return
    used: set[int] = set()
    for det in sorted(dets_in_row, key=_cx):
        frac = _cx(det) / float(frame_w)
        pref = max(0, min(num_bins - 1, int(math.floor(frac * num_bins))))
        col = pref
        if col in used:
            # Collision: another detection already claimed this band. DELIBERATE under
            # the "rig fills the frame" assumption — if a real bin is missing AND a
            # duplicate lands in one band, the duplicate fills the empty cell. Acceptable
            # for the fixed rig (robust dedup is in FUTURE_TASKS.md). Nearest free band:
            free = [c for c in range(num_bins) if c not in used]
            if not free:
                logger.warning("Extra detection dropped (row full): cx=%.1f", _cx(det))
                continue
            col = min(free, key=lambda c: abs((c + 0.5) / num_bins - frac))
            logger.info("Detection snapped band %d->%d (collision)", pref, col)
        used.add(col)
        bin_id, _info = cells[col]
        skeleton[bin_id].update({
            "corners": det["corners"],
            "center": [float(det["center"][0]), float(det["center"][1])],
            "confidence": float(det.get("conf", 0.0)),
            "detected": True,
        })


def allocate_grid(detections: list, frame_w: int, layout: list = DEFAULT_LAYOUT) -> dict:
    """Place raw OBB detections onto the fixed pre-indexed grid.

    Parameters
    ----------
    detections : list of {"corners": [[x,y]*4], "center": [cx, cy], "conf": float}
        Upright (already rotated) detections, any order.
    frame_w : snapshot width in pixels (band reference; rig fills the frame).
    layout : per-row bin spans; defaults to the fixed 6+3 rig.

    Returns
    -------
    dict keyed ``bin_{index}`` (see module docstring).
    """
    skeleton = build_skeleton(layout)
    num_rows = len(layout)
    if not detections or num_rows == 0:
        return skeleton

    # Drop detections with non-finite centres (real OBB output can emit NaN/inf).
    finite = [d for d in detections
              if math.isfinite(_cx(d)) and math.isfinite(_cy(d))]
    if len(finite) != len(detections):
        logger.warning("Dropped %d detection(s) with non-finite centre",
                       len(detections) - len(finite))
    detections = finite
    if not detections:
        return skeleton

    # (row, col) -> bin_id lookup, cells ordered left->right per row
    cells_by_row: list = [[] for _ in range(num_rows)]
    for bin_id, info in skeleton.items():
        cells_by_row[info["row"]].append((bin_id, info))
    for r in range(num_rows):
        cells_by_row[r].sort(key=lambda t: t[1]["col"])

    rows = split_rows_by_y(detections, num_rows)
    for r in range(num_rows):
        _assign_row(rows[r], cells_by_row[r], frame_w, skeleton)

    detected = sum(1 for v in skeleton.values() if v["detected"])
    per_row = [sum(1 for d in rows[r]) for r in range(num_rows)]
    expected = [len(layout[r]) for r in range(num_rows)]
    if per_row != expected:
        logger.warning("Detected counts %s differ from expected %s", per_row, expected)
    logger.info("Grid allocated: %d/%d cells filled", detected, len(skeleton))
    return skeleton
