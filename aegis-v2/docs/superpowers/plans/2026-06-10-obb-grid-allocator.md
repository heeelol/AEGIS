# OBB Snapshot → 2-Layer Grid Allocation + 1–9 Indexing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From a startup snapshot, detect the rig's bins with the OBB model and allocate them into a fixed 6×2 grid with permanent indices 1–9, correcting the upside-down camera.

**Architecture:** aegis-core runs the model (rotate frame 180° → OBB detect → write `bins.json` + `snapshot.jpg`). aegis-v2 owns a **pure** allocator (`grid_allocator.allocate_grid`) that splits detections into 2 rows by y-gap and places each into its grid cell by frame-band (`pitch = frame_w / bins_per_row`), plus a thin driver that reads the handoff, writes `bins_indexed.json`, and draws a 1–9 overlay.

**Tech Stack:** Python 3, OpenCV (`cv2`), Ultralytics YOLO (OBB), pytest. No NumPy required in the allocator (pure lists/math); cv2 only in the driver/core.

**Reference spec:** `aegis-v2/docs/superpowers/specs/2026-06-10-obb-grid-allocator-design.md`
**Deferred work:** repo-root `FUTURE_TASKS.md` (do not implement here).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `aegis-core/scripts/inference/initialize_bins_obb.py` (modify) | Use `project_9_yolov8_obb_1.pt`; rotate each frame 180°; save `conf` + `frame_w/h` in `bins.json` and write `snapshot.jpg`. |
| `aegis-v2/integration/src/detectors/grid_allocator.py` (create) | **Pure** allocator: skeleton + index assignment, y-gap row split, band placement → indexed dict. |
| `aegis-v2/integration/tests/test_grid_allocator.py` (create) | Pure unit tests for the allocator. |
| `aegis-v2/integration/src/detectors/snapshot_obb.py` (create) | Driver: read `bins.json` + `snapshot.jpg` → `bins_indexed.json` + overlay image. |

---

## Task 1: aegis-core — OBB weight, 180° rotation, richer handoff

**Files:**
- Modify: `aegis-core/scripts/inference/initialize_bins_obb.py`

This task touches camera/model I/O, so it is verified by a smoke run on a sample image rather than a unit test.

- [ ] **Step 1: Point at the non-rotate OBB weight**

In `initialize_bins_obb.py`, change line 26:

```python
MODEL_NAME = "project_9_yolov8_obb_1.pt"   # OBB model under models/custom/
```

- [ ] **Step 2: Rotate every captured frame 180° (camera is mounted upside-down)**

In `initialize_image()`, immediately after `image = cv2.imread(str(image_path))` and its `None` check, add:

```python
    image = cv2.rotate(image, cv2.ROTATE_180)
```

In `initialize_webcam()`, inside the `while True:` loop, right after the `if not ret: break` line, add:

```python
        frame = cv2.rotate(frame, cv2.ROTATE_180)
```

In `initialize_batch()`, after `image = cv2.imread(str(img_path))` and its `if image is None: continue`, add:

```python
        image = cv2.rotate(image, cv2.ROTATE_180)
```

- [ ] **Step 3: Extend `save_bins()` to carry confidence, frame size, and the snapshot image**

Replace the whole `save_bins` function (lines ~193–199) with:

```python
def save_bins(bins, out_path, frame=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"id": b["id"], "corners": b["corners"].tolist(),
             "center": list(b["center"]), "conf": float(b.get("conf", 0.0))}
            for b in bins]
    payload = {"bins": data, "rule": "nearest_center", "source": "obb"}
    if frame is not None:
        payload["frame_h"], payload["frame_w"] = int(frame.shape[0]), int(frame.shape[1])
        snap_path = out_path.parent / "snapshot.jpg"
        cv2.imwrite(str(snap_path), frame)
        logger.info(f"✓ Saved snapshot image -> {snap_path}")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"✓ Saved raw bin boxes + centers -> {out_path}")
```

- [ ] **Step 4: Pass the frame into the save calls**

In `initialize_image()`, change `save_bins(bins, save_path)` to:

```python
    save_bins(bins, save_path, image)
```

In `initialize_webcam()`, change the save line under the `s` key from `save_bins(bins, save_path)` to:

```python
            save_bins(bins, save_path, frame)
```

- [ ] **Step 5: Smoke-test on a sample training image**

Run (from `aegis-core`, with the project venv active):

```bash
cd aegis-core
python scripts/inference/initialize_bins_obb.py
```

Enter blank for count, choose mode `1` (single image). Press `q` to close the window.
Expected: `runs/bins_obb_raw/bins.json` exists and contains `frame_w`, `frame_h`, and a
`conf` on each bin; `runs/bins_obb_raw/snapshot.jpg` exists. (If the OBB weight is missing
the script logs a clear error — confirm the weight path `models/custom/project_9_yolov8_obb_1.pt`.)

- [ ] **Step 6: Commit**

```bash
git add aegis-core/scripts/inference/initialize_bins_obb.py
git commit -m "feat(core): OBB bin init rotates 180, uses project_9 weight, saves frame+conf+snapshot"
```

---

## Task 2: aegis-v2 — pure allocator skeleton + index assignment

**Files:**
- Create: `aegis-v2/integration/src/detectors/grid_allocator.py`
- Test: `aegis-v2/integration/tests/test_grid_allocator.py`

- [ ] **Step 1: Write the failing test for the pre-indexed skeleton**

Create `aegis-v2/integration/tests/test_grid_allocator.py`:

```python
"""Unit tests for grid_allocator — pure geometry, no camera/model/cv2.

Imported by path so the test doesn't drag in the detectors package __init__
(which pulls cv2 / ultralytics).
"""
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "detectors")),
)

import grid_allocator as ga  # noqa: E402

FRAME_W = 1280


def test_skeleton_is_fully_preindexed():
    sk = ga.build_skeleton(ga.DEFAULT_LAYOUT)
    assert len(sk) == 9
    # top row -> indices 1..6, span 1
    assert sk["bin_1"]["index"] == 1 and sk["bin_1"]["span"] == 1
    assert sk["bin_1"]["layer"] == "top" and sk["bin_1"]["slot_start"] == 0
    assert sk["bin_6"]["index"] == 6 and sk["bin_6"]["slot_start"] == 5
    # bottom row -> indices 7..9, span 2, slot_start 0/2/4
    assert sk["bin_7"]["index"] == 7 and sk["bin_7"]["span"] == 2
    assert sk["bin_7"]["layer"] == "bottom" and sk["bin_7"]["slot_start"] == 0
    assert sk["bin_8"]["slot_start"] == 2 and sk["bin_9"]["slot_start"] == 4
    # every cell starts undetected, row_slots 6 for both rows
    assert all(c["detected"] is False for c in sk.values())
    assert sk["bin_1"]["row_slots"] == 6 and sk["bin_9"]["row_slots"] == 6
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd aegis-v2/integration && python -m pytest tests/test_grid_allocator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grid_allocator'`.

- [ ] **Step 3: Implement the skeleton builder**

Create `aegis-v2/integration/src/detectors/grid_allocator.py`:

```python
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
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd aegis-v2/integration && python -m pytest tests/test_grid_allocator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aegis-v2/integration/src/detectors/grid_allocator.py aegis-v2/integration/tests/test_grid_allocator.py
git commit -m "feat(v2): grid_allocator pre-indexed skeleton (1-9 fixed grid)"
```

---

## Task 3: aegis-v2 — split detections into rows by largest y-gap

**Files:**
- Modify: `aegis-v2/integration/src/detectors/grid_allocator.py`
- Test: `aegis-v2/integration/tests/test_grid_allocator.py`

- [ ] **Step 1: Add a helper to build synthetic detections, then the failing test**

Append to `tests/test_grid_allocator.py`:

```python
def make_det(cx, cy, conf=0.9, half=20):
    """A detection dict with a square box centred on (cx, cy)."""
    return {
        "corners": [[cx - half, cy - half], [cx + half, cy - half],
                    [cx + half, cy + half], [cx - half, cy + half]],
        "center": [float(cx), float(cy)],
        "conf": conf,
    }


def test_split_rows_by_largest_y_gap():
    # 6 top dets near y=200, 3 bottom dets near y=560 (big gap between bands)
    top = [make_det(x, 200) for x in (100, 300, 500, 700, 900, 1100)]
    bottom = [make_det(x, 560) for x in (200, 640, 1080)]
    rows = ga.split_rows_by_y(top + bottom, num_rows=2)
    assert len(rows) == 2
    assert len(rows[0]) == 6 and all(d["center"][1] == 200 for d in rows[0])
    assert len(rows[1]) == 3 and all(d["center"][1] == 560 for d in rows[1])


def test_split_rows_single_row_only():
    top = [make_det(x, 200) for x in (100, 500, 900)]
    rows = ga.split_rows_by_y(top, num_rows=2)
    assert len(rows[0]) == 3 and rows[1] == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd aegis-v2/integration && python -m pytest tests/test_grid_allocator.py::test_split_rows_by_largest_y_gap -v`
Expected: FAIL — `AttributeError: module 'grid_allocator' has no attribute 'split_rows_by_y'`.

- [ ] **Step 3: Implement the y-gap splitter**

Append to `grid_allocator.py`:

```python
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

    threshold = 0.5 * _median([_box_height(d) for d in ordered])
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
```

> **Note (plan correction):** an earlier draft force-split into `num_rows-1` bands
> unconditionally (wrongly splitting a single detected row) and sorted `(gap, index)`
> tuples (a latent tie-break bug). The version above only splits at gaps larger than
> ~half the median bin height and sorts by gap value alone.

- [ ] **Step 4: Run to verify both split tests pass**

Run: `cd aegis-v2/integration && python -m pytest tests/test_grid_allocator.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add aegis-v2/integration/src/detectors/grid_allocator.py aegis-v2/integration/tests/test_grid_allocator.py
git commit -m "feat(v2): grid_allocator y-gap row split"
```

---

## Task 4: aegis-v2 — full `allocate_grid` (band placement + collisions + misses)

**Files:**
- Modify: `aegis-v2/integration/src/detectors/grid_allocator.py`
- Test: `aegis-v2/integration/tests/test_grid_allocator.py`

- [ ] **Step 1: Write failing tests for the full allocation**

Append to `tests/test_grid_allocator.py`:

```python
def top_dets():
    # one bin centred in each of the 6 equal frame bands (band width = 1280/6)
    return [make_det(int((c + 0.5) * FRAME_W / 6), 200) for c in range(6)]


def bottom_dets(cols=(0, 1, 2)):
    # bottom row has 3 bins -> band width = 1280/3
    return [make_det(int((c + 0.5) * FRAME_W / 3), 560) for c in cols]


def test_full_detection_maps_to_1_through_9():
    res = ga.allocate_grid(top_dets() + bottom_dets(), FRAME_W)
    assert all(res[f"bin_{i}"]["detected"] for i in range(1, 10))
    assert res["bin_1"]["corners"] is not None
    assert res["bin_7"]["confidence"] == 0.9


def test_missing_middle_top_bin_leaves_that_index_empty():
    dets = [d for c, d in enumerate(top_dets()) if c != 2]  # drop band 2 -> index 3
    res = ga.allocate_grid(dets + bottom_dets(), FRAME_W)
    assert res["bin_3"]["detected"] is False        # the gap lands on index 3…
    assert res["bin_2"]["detected"] is True          # …neighbours unaffected
    assert res["bin_4"]["detected"] is True
    assert sum(res[f"bin_{i}"]["detected"] for i in range(1, 7)) == 5


def test_missing_middle_bottom_bin():
    res = ga.allocate_grid(top_dets() + bottom_dets(cols=(0, 2)), FRAME_W)
    assert res["bin_8"]["detected"] is False         # middle bottom = index 8
    assert res["bin_7"]["detected"] is True and res["bin_9"]["detected"] is True


def test_out_of_order_x_is_placed_by_position():
    res = ga.allocate_grid(list(reversed(top_dets())) + bottom_dets(), FRAME_W)
    assert all(res[f"bin_{i}"]["detected"] for i in range(1, 7))


def test_empty_detection_returns_full_placeholder_grid():
    res = ga.allocate_grid([], FRAME_W)
    assert len(res) == 9
    assert all(res[f"bin_{i}"]["detected"] is False for i in range(1, 10))


def test_extra_detection_in_row_is_dropped():
    extra = make_det(int(0.5 * FRAME_W / 6) + 3, 200)   # 7th box in 6-band top row
    res = ga.allocate_grid(top_dets() + [extra] + bottom_dets(), FRAME_W)
    assert len(res) == 9
    assert all(res[f"bin_{i}"]["detected"] for i in range(1, 7))
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd aegis-v2/integration && python -m pytest tests/test_grid_allocator.py -v`
Expected: FAIL — `AttributeError: module 'grid_allocator' has no attribute 'allocate_grid'`.

- [ ] **Step 3: Implement `allocate_grid`**

Append to `grid_allocator.py`:

```python
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
            # nearest unused band by distance to band centre
            free = [c for c in range(num_bins) if c not in used]
            if not free:
                logger.warning("Extra detection dropped (row full): cx=%.1f", _cx(det))
                continue
            col = min(free, key=lambda c: abs((c + 0.5) / num_bins - frac))
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
```

- [ ] **Step 4: Run the whole test file to verify all pass**

Run: `cd aegis-v2/integration && python -m pytest tests/test_grid_allocator.py -v`
Expected: PASS (all tests, ~10).

- [ ] **Step 5: Commit**

```bash
git add aegis-v2/integration/src/detectors/grid_allocator.py aegis-v2/integration/tests/test_grid_allocator.py
git commit -m "feat(v2): grid_allocator.allocate_grid band placement with gap/collision handling"
```

---

## Task 5: aegis-v2 — snapshot driver (`bins.json` → `bins_indexed.json` + overlay)

**Files:**
- Create: `aegis-v2/integration/src/detectors/snapshot_obb.py`
- Test: `aegis-v2/integration/tests/test_grid_allocator.py` (add a pure-transform test)

- [ ] **Step 1: Write the failing test for the pure transform `index_bins`**

Append to `tests/test_grid_allocator.py`:

```python
def test_index_bins_pure_transform():
    import snapshot_obb as so
    payload = {
        "frame_w": FRAME_W, "frame_h": 720,
        "bins": [
            {"id": 0, "corners": [[10, 190], [30, 190], [30, 210], [10, 210]],
             "center": [int(0.5 * FRAME_W / 6), 200], "conf": 0.8},
            {"id": 1, "corners": [[10, 550], [30, 550], [30, 570], [10, 570]],
             "center": [int(0.5 * FRAME_W / 3), 560], "conf": 0.7},
        ],
    }
    out = so.index_bins(payload)
    assert out["bins"]["bin_1"]["detected"] is True       # top-left small bin
    assert out["bins"]["bin_7"]["detected"] is True       # bottom-left big bin
    assert out["bins"]["bin_2"]["detected"] is False      # nothing there
    assert out["rule"] == "rotate180_then_band_split"
```

Note: `snapshot_obb` imports `grid_allocator`; both live in `src/detectors`, which is on
`sys.path` from the test header, so the import resolves.

- [ ] **Step 2: Run to verify it fails**

Run: `cd aegis-v2/integration && python -m pytest tests/test_grid_allocator.py::test_index_bins_pure_transform -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'snapshot_obb'`.

- [ ] **Step 3: Implement the driver**

Create `aegis-v2/integration/src/detectors/snapshot_obb.py`:

```python
"""
Snapshot OBB driver
===================
Reads the aegis-core OBB handoff (``bins.json`` + ``snapshot.jpg``), allocates the
detections onto the fixed 1-9 grid via ``grid_allocator``, writes ``bins_indexed.json``,
and draws an overlay with the indices for visual verification.

Run (from aegis-v2/integration) as a PATH SCRIPT — this bypasses the detectors
package __init__, which eagerly imports cv2 / ultralytics:
    python src/detectors/snapshot_obb.py \
        --bins ../../aegis-core/runs/bins_obb_raw/bins.json
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

logger = logging.getLogger("aegis.detectors.snapshot_obb")


def index_bins(payload: dict) -> dict:
    """Pure transform: raw handoff payload -> indexed grid dict.

    ``payload`` is the parsed ``bins.json`` ({"frame_w", "bins": [...]}). Returns
    {"bins": {bin_i: {...}}, "frame_w", "source", "rule"}.
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    here = os.path.dirname(__file__)
    default_bins = os.path.abspath(os.path.join(
        here, "..", "..", "..", "..", "aegis-core", "runs", "bins_obb_raw", "bins.json"))

    ap = argparse.ArgumentParser(description="Allocate OBB snapshot bins to the 1-9 grid")
    ap.add_argument("--bins", default=default_bins, help="path to aegis-core bins.json")
    ap.add_argument("--out", default=None, help="output bins_indexed.json path")
    ap.add_argument("--no-show", action="store_true", help="don't open the overlay window")
    args = ap.parse_args()

    if not os.path.exists(args.bins):
        logger.error("bins.json not found: %s — run the aegis-core OBB script first.", args.bins)
        return
    with open(args.bins) as f:
        payload = json.load(f)

    out = index_bins(payload)
    out_path = args.out or os.path.join(os.path.dirname(args.bins), "bins_indexed.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    filled = sum(1 for v in out["bins"].values() if v["detected"])
    logger.info("✓ Wrote %s (%d/9 cells filled)", out_path, filled)

    # Overlay on the sibling snapshot.jpg, if present.
    snap = os.path.join(os.path.dirname(args.bins), "snapshot.jpg")
    if os.path.exists(snap):
        import cv2
        image = cv2.imread(snap)
        vis = _draw_overlay(image, out["bins"])
        vis_path = os.path.join(os.path.dirname(args.bins), "bins_indexed_overlay.jpg")
        cv2.imwrite(vis_path, vis)
        logger.info("✓ Wrote overlay %s", vis_path)
        if not args.no_show:
            cv2.imshow("Indexed bins (1-9) — 'q' to close", vis)
            while cv2.waitKey(20) & 0xFF != ord("q"):
                pass
            cv2.destroyAllWindows()
    else:
        logger.warning("No snapshot.jpg next to bins.json — skipping overlay.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the transform test passes**

Run: `cd aegis-v2/integration && python -m pytest tests/test_grid_allocator.py -v`
Expected: PASS (all tests including `test_index_bins_pure_transform`).

- [ ] **Step 5: Commit**

```bash
git add aegis-v2/integration/src/detectors/snapshot_obb.py aegis-v2/integration/tests/test_grid_allocator.py
git commit -m "feat(v2): snapshot_obb driver -> bins_indexed.json + 1-9 overlay"
```

---

## Task 6: End-to-end verification

**Files:** none (manual run).

- [ ] **Step 1: Generate the handoff in aegis-core**

```bash
cd aegis-core
python scripts/inference/initialize_bins_obb.py
```
Mode 2 (webcam) with the rig in view; press `s` to save, `q` to quit.
Expected: `aegis-core/runs/bins_obb_raw/bins.json` (+ `frame_w`, `conf`) and `snapshot.jpg`.

- [ ] **Step 2: Allocate + visualise in aegis-v2**

```bash
cd aegis-v2/integration
python src/detectors/snapshot_obb.py
```
Expected: `bins_indexed.json` written; an overlay window shows **1–6** across the top
small bins (left→right) and **7–9** across the bottom big bins (left→right). Any
undetected bin simply has no number — its index is reserved, not reused.

- [ ] **Step 3: Confirm the full test suite is green**

```bash
cd aegis-v2/integration && python -m pytest tests/test_grid_allocator.py -v
```
Expected: all PASS.

---

## Self-Review Notes

- **Spec §3 (orientation + handoff):** Task 1 (rotation, weight, conf/frame_w/snapshot.jpg).
- **Spec §4 (algorithm):** Task 2 (skeleton/index), Task 3 (y-gap rows), Task 4 (band placement, collisions, misses).
- **Spec §5 (output contract):** `build_skeleton` fields + `index_bins` rule string (Task 2, Task 5).
- **Spec §7 (testing):** Tasks 2–5 unit tests + Task 6 e2e.
- **Deferred (FUTURE_TASKS.md):** generalized inference, size/span detection, dashboard/ESP wiring — intentionally not in this plan.
