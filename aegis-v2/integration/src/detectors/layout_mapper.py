"""
Layout Mapper
=============
Pure function that maps raw bin detections (bounding boxes from ``BinDetector``)
onto a *declared* bin layout. No camera, model, or heavy dependencies — just
geometry, so it can be unit-tested in milliseconds.

Design contract
---------------
* The **declared layout always defines the grid shape.** Every bin in the spec
  appears in the output regardless of what the model found.
* **Detections only fill in real pixel boxes** for the slots they match.
* A declared slot the model did not detect is returned with ``detected=False``
  (the dashboard renders it as a grey placeholder; no hand-matching there).

Layout spec
-----------
A list of rows, each a list of per-bin slot *spans* (left → right). Example for
a rig with a 6-slot top row of single-slot bins and a 6-slot bottom row of
three double-slot bins::

    [[1, 1, 1, 1, 1, 1],
     [2, 2, 2]]

Output
------
A geofence dict keyed by ``bin_{row}_{col}``::

    {
      "bin_1_0": {
        "row": 1, "col": 0, "slot_start": 0, "span": 2, "row_slots": 6,
        "detected": True, "confidence": 0.91,
        "x_min": .., "x_max": .., "y_min": .., "y_max": ..   # only if detected
      },
      ...
    }
"""

from __future__ import annotations

import logging

logger = logging.getLogger("aegis.detectors.layout_mapper")

# A detection box is (x_min, y_min, x_max, y_max[, confidence]).


def _x_center(box) -> float:
    return (box[0] + box[2]) / 2.0


def _y_center(box) -> float:
    return (box[1] + box[3]) / 2.0


def build_skeleton(layout_spec: list[list[int]]) -> dict:
    """Build the full declared grid (every bin, no detection yet)."""
    skeleton: dict = {}
    for row, spans in enumerate(layout_spec):
        row_slots = sum(int(s) for s in spans)
        slot_start = 0
        for col, span in enumerate(spans):
            span = int(span)
            skeleton[f"bin_{row}_{col}"] = {
                "row": row,
                "col": col,
                "slot_start": slot_start,
                "span": span,
                "row_slots": row_slots,
                "detected": False,
                "confidence": 0.0,
            }
            slot_start += span
    return skeleton


def _assign_row(boxes_in_row: list, declared_bins: list, frame_w: int, skeleton: dict) -> None:
    """Greedily snap each detected box to its nearest unused declared slot.

    Matching is by *nearest expected slot centre* (in normalised x), not a naive
    left-to-right zip, so a missing bin in the middle of a row leaves the correct
    slot empty instead of shifting its neighbours.
    """
    used: set[str] = set()
    for box in sorted(boxes_in_row, key=_x_center):
        d = _x_center(box) / float(frame_w)
        best_id = None
        best_dist = None
        for bin_id, info in declared_bins:
            if bin_id in used:
                continue
            expected = (info["slot_start"] + info["span"] / 2.0) / info["row_slots"]
            dist = abs(expected - d)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_id = bin_id
        if best_id is None:
            # More boxes than declared slots in this row — drop the extra.
            logger.warning("Extra detection in row dropped (no free slot): x_center=%.1f", _x_center(box))
            continue
        used.add(best_id)
        conf = float(box[4]) if len(box) > 4 else 0.0
        skeleton[best_id].update({
            "x_min": int(box[0]),
            "y_min": int(box[1]),
            "x_max": int(box[2]),
            "y_max": int(box[3]),
            "confidence": conf,
            "detected": True,
        })


def map_detections_to_layout(
    boxes: list,
    layout_spec: list[list[int]],
    frame_w: int,
    frame_h: int,
) -> dict:
    """Map raw detection boxes onto the declared layout.

    Parameters
    ----------
    boxes : list of (x_min, y_min, x_max, y_max[, confidence])
        Raw detections from the bin model (any order).
    layout_spec : list of rows, each a list of per-bin slot spans.
    frame_w, frame_h : snapshot dimensions in pixels.

    Returns
    -------
    dict : geofence dict keyed by ``bin_{row}_{col}`` (see module docstring).
    """
    skeleton = build_skeleton(layout_spec)
    num_rows = len(layout_spec)
    if not boxes or num_rows == 0:
        return skeleton

    # Group declared bins by row, ordered left → right.
    declared_by_row: dict[int, list] = {r: [] for r in range(num_rows)}
    for bin_id, info in skeleton.items():
        declared_by_row[info["row"]].append((bin_id, info))
    for r in declared_by_row:
        declared_by_row[r].sort(key=lambda t: t[1]["slot_start"])

    # Assign each detection to a row by vertical band (rows are physically
    # separated top → bottom on a fixed rig). Empty rows stay all-placeholder.
    boxes_by_row: dict[int, list] = {r: [] for r in range(num_rows)}
    for box in boxes:
        r = int(_y_center(box) / float(frame_h) * num_rows)
        r = max(0, min(num_rows - 1, r))
        boxes_by_row[r].append(box)

    for r in range(num_rows):
        _assign_row(boxes_by_row[r], declared_by_row[r], frame_w, skeleton)

    detected = sum(1 for v in skeleton.values() if v["detected"])
    logger.info("Layout mapped: %d/%d declared bins detected", detected, len(skeleton))
    return skeleton
