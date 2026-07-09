# Workstation Grid Calibration + Nearest-Slot Matching — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define the bin grid from a one-time "all 9 bins" calibration snapshot per workstation, then per kitting list match each detected bin to its nearest calibrated slot to report which slots have a bin and which don't.

**Architecture:** A new **pure** module `grid_calibrator.py` provides `calibrate_grid(detections)` (all-9 snapshot → 9 indexed slots with real centers/boxes) and `match_to_grid(detections, calibration)` (nearest-center, same-row, one-to-one matching → per-slot occupancy). The driver `snapshot_obb.py` gains a `--calibrate` mode (writes `grid_calibration.json`) and a default match mode (loads it, writes occupancy + overlay). The existing band-based `grid_allocator.allocate_grid` is left untouched as a fallback; `grid_calibrator` reuses its `split_rows_by_y`.

**Tech Stack:** Python 3, pure stdlib for the allocator logic (no cv2/numpy in `grid_calibrator`); cv2 only inside driver functions; pytest.

**Reference spec:** `aegis-v2/docs/superpowers/specs/2026-06-11-grid-calibration-matching-design.md`
**Branch:** `FableDoesGridAllocator`. **Deferred work:** repo-root `FUTURE_TASKS.md`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `aegis-v2/integration/src/detectors/grid_calibrator.py` (create) | Pure: `calibrate_grid` + `match_to_grid`. |
| `aegis-v2/integration/tests/test_grid_calibrator.py` (create) | Pure unit tests. |
| `aegis-v2/integration/src/detectors/snapshot_obb.py` (modify) | Add pure `build_calibration`/`build_occupancy` wrappers + `--calibrate`/match `main()` + overlays. |

All commands run from: `C:\Users\chenx\Documents\TEnterns\aegis-v2\integration`. Stage only the files each task names; never `git add -A` (the repo has unrelated staged deletions). Do not push.

---

## Task 1: `grid_calibrator.calibrate_grid` (pure)

**Files:**
- Create: `src/detectors/grid_calibrator.py`
- Test: `tests/test_grid_calibrator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grid_calibrator.py`:

```python
"""Unit tests for grid_calibrator — pure geometry, no camera/model/cv2.

Imported by path so the test doesn't drag in the detectors package __init__
(which pulls cv2 / ultralytics).
"""
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "detectors")),
)

import pytest  # noqa: E402
import grid_calibrator as gc  # noqa: E402

FRAME_W = 1280


def make_det(cx, cy, conf=0.9, half=20):
    return {
        "corners": [[cx - half, cy - half], [cx + half, cy - half],
                    [cx + half, cy + half], [cx - half, cy + half]],
        "center": [float(cx), float(cy)],
        "conf": conf,
    }


def top_dets():
    # 6 bins across the top band (y=200), evenly spaced
    return [make_det(int((c + 0.5) * FRAME_W / 6), 200) for c in range(6)]


def bottom_dets():
    # 3 big bins across the bottom band (y=560)
    return [make_det(int((c + 0.5) * FRAME_W / 3), 560) for c in range(3)]


def full():
    return top_dets() + bottom_dets()


def test_calibrate_builds_9_indexed_slots():
    cal = gc.calibrate_grid(full())
    assert len(cal) == 9
    assert cal["slot_1"]["index"] == 1 and cal["slot_1"]["row"] == 0
    assert cal["slot_1"]["layer"] == "top" and cal["slot_1"]["span"] == 1
    assert cal["slot_6"]["index"] == 6 and cal["slot_6"]["row"] == 0
    assert cal["slot_7"]["index"] == 7 and cal["slot_7"]["row"] == 1
    assert cal["slot_7"]["layer"] == "bottom" and cal["slot_7"]["span"] == 2
    assert cal["slot_9"]["index"] == 9
    # each slot keeps its real centre + box + a positive per-row spacing
    assert cal["slot_1"]["center"][1] == 200 and cal["slot_7"]["center"][1] == 560
    assert "corners" in cal["slot_1"]
    assert cal["slot_1"]["row_spacing"] > 0 and cal["slot_7"]["row_spacing"] > 0


def test_calibrate_orders_left_to_right():
    cal = gc.calibrate_grid(list(reversed(top_dets())) + bottom_dets())
    xs = [cal[f"slot_{i}"]["center"][0] for i in range(1, 7)]
    assert xs == sorted(xs)  # slot_1..6 increase left->right regardless of input order


def test_calibrate_rejects_wrong_counts():
    with pytest.raises(ValueError):
        gc.calibrate_grid(top_dets()[:5] + bottom_dets())   # 5 top
    with pytest.raises(ValueError):
        gc.calibrate_grid(top_dets() + bottom_dets()[:2])   # 2 bottom
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_grid_calibrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grid_calibrator'`.

- [ ] **Step 3: Implement `calibrate_grid`**

Create `src/detectors/grid_calibrator.py`:

```python
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


def calibrate_grid(detections: list) -> dict:
    """All-9-bins snapshot -> the 9 fixed slots (the grid).

    Splits into 2 rows by y-gap, validates exactly 6 top + 3 bottom, orders each
    row left->right, and assigns indices 1..6 (top) / 7..9 (bottom). Stores each
    slot's centre, box, and the row's median centre-to-centre spacing (used as the
    match cutoff later). Raises ValueError if the snapshot isn't a clean 6+3.
    """
    rows = ga.split_rows_by_y(detections, num_rows=2)
    top, bottom = rows[0], rows[1]
    if len(top) != 6 or len(bottom) != 3:
        raise ValueError(
            f"Calibration needs 6 top + 3 bottom bins; got {len(top)} top / "
            f"{len(bottom)} bottom. Retake the calibration snapshot with all 9 bins."
        )

    slots: dict = {}
    index = 1
    for row_idx, layer, span, row_dets in (
        (0, "top", 1, top),
        (1, "bottom", 2, bottom),
    ):
        ordered = sorted(row_dets, key=_cx)
        centers = [_cx(d) for d in ordered]
        diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        row_spacing = _median(diffs)
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
    return slots
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_grid_calibrator.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/detectors/grid_calibrator.py tests/test_grid_calibrator.py
git commit -m "feat(v2): grid_calibrator.calibrate_grid — define grid from all-9 snapshot"
```

---

## Task 2: `grid_calibrator.match_to_grid` (pure)

**Files:**
- Modify: `src/detectors/grid_calibrator.py`
- Test: `tests/test_grid_calibrator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_grid_calibrator.py`:

```python
def present_indices(occ):
    return {info["index"] for info in occ.values() if info["present"]}


def test_match_all_present():
    cal = gc.calibrate_grid(full())
    occ = gc.match_to_grid(full(), cal)
    assert present_indices(occ) == set(range(1, 10))
    assert occ["slot_1"]["present"] is True
    assert occ["slot_1"]["confidence"] == 0.9
    assert "corners" in occ["slot_1"]


def test_match_missing_middle_top_bin():
    cal = gc.calibrate_grid(full())
    dets = [d for c, d in enumerate(top_dets()) if c != 2] + bottom_dets()  # drop top index 3
    occ = gc.match_to_grid(dets, cal)
    assert occ["slot_3"]["present"] is False
    assert occ["slot_2"]["present"] is True and occ["slot_4"]["present"] is True
    assert present_indices(occ) == {1, 2, 4, 5, 6, 7, 8, 9}


def test_match_tolerates_small_jitter():
    cal = gc.calibrate_grid(full())
    jittered = [make_det(d["center"][0] + 8, d["center"][1] - 6) for d in full()]
    occ = gc.match_to_grid(jittered, cal)
    assert present_indices(occ) == set(range(1, 10))


def test_match_drops_stray_far_detection():
    cal = gc.calibrate_grid(full())
    stray = make_det(30, 30)  # top-left corner, far from every slot
    occ = gc.match_to_grid(full() + [stray], cal)
    assert present_indices(occ) == set(range(1, 10))  # 9 reals matched, stray dropped


def test_match_uses_nearest_center_not_index_order():
    cal = gc.calibrate_grid(full())
    # one bin sitting on slot_2's calibrated centre -> must match slot_2, not slot_1
    s2 = cal["slot_2"]["center"]
    occ = gc.match_to_grid([make_det(s2[0], s2[1])], cal)
    assert occ["slot_2"]["present"] is True
    assert occ["slot_1"]["present"] is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_grid_calibrator.py -v`
Expected: FAIL — `AttributeError: module 'grid_calibrator' has no attribute 'match_to_grid'`.

- [ ] **Step 3: Implement `match_to_grid`**

Append to `src/detectors/grid_calibrator.py`:

```python
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

    # Calibrated row mean-y, to assign each detection a row.
    row_cys: dict = {0: [], 1: []}
    for info in calibration.values():
        row_cys[info["row"]].append(info["center"][1])
    row_mean = {r: (sum(v) / len(v) if v else 0.0) for r, v in row_cys.items()}

    # All (distance, det_idx, slot_id, cutoff) candidate pairs within the same row.
    pairs = []
    for di, d in enumerate(detections):
        cx, cy = _cx(d), _cy(d)
        row = 0 if abs(cy - row_mean[0]) <= abs(cy - row_mean[1]) else 1
        for sid, info in calibration.items():
            if info["row"] != row:
                continue
            sx, sy = info["center"]
            dist = math.hypot(cx - sx, cy - sy)
            cutoff = 0.5 * float(info.get("row_spacing", 0.0))
            pairs.append((dist, di, sid, cutoff))

    pairs.sort(key=lambda t: t[0])
    used_dets: set = set()
    used_slots: set = set()
    for dist, di, sid, cutoff in pairs:
        if di in used_dets or sid in used_slots:
            continue
        if cutoff > 0 and dist > cutoff:
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_grid_calibrator.py -v`
Expected: PASS (8 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/detectors/grid_calibrator.py tests/test_grid_calibrator.py
git commit -m "feat(v2): grid_calibrator.match_to_grid — nearest-slot occupancy"
```

---

## Task 3: driver — `--calibrate` mode + match mode

**Files:**
- Modify: `src/detectors/snapshot_obb.py`
- Test: `tests/test_grid_calibrator.py`

This task adds two **pure** wrappers (`build_calibration`, `build_occupancy`) that are unit-tested, plus the cv2/file `main()` that uses them.

- [ ] **Step 1: Write the failing tests for the pure wrappers**

Append to `tests/test_grid_calibrator.py`:

```python
def test_build_calibration_and_occupancy_wrappers():
    import snapshot_obb as so
    payload = {"frame_w": FRAME_W, "frame_h": 720, "bins": full()}
    cal_payload = so.build_calibration(payload)
    assert cal_payload["frame_w"] == FRAME_W
    assert cal_payload["source"] == "obb"
    assert len(cal_payload["slots"]) == 9

    # a later snapshot missing the middle top bin
    later = {"frame_w": FRAME_W, "frame_h": 720,
             "bins": [d for c, d in enumerate(top_dets()) if c != 2] + bottom_dets()}
    occ_payload = so.build_occupancy(later, cal_payload)
    assert occ_payload["rule"] == "calibrated_nearest_slot"
    assert occ_payload["slots"]["slot_3"]["present"] is False
    assert occ_payload["slots"]["slot_2"]["present"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_grid_calibrator.py::test_build_calibration_and_occupancy_wrappers -v`
Expected: FAIL — `AttributeError: module 'snapshot_obb' has no attribute 'build_calibration'`.

- [ ] **Step 3: Add the wrappers and rewrite `main()`**

In `src/detectors/snapshot_obb.py`, add `import grid_calibrator as gc` next to the existing `import grid_allocator as ga` (keep both). Add these two pure functions (next to `index_bins`):

```python
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
```

Then replace the entire existing `main()` function with the version below (the old band-based `index_bins` and `_draw_overlay` stay in the file, unused by `main` now but kept as a fallback):

```python
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
            return None, None
        import cv2
        image = cv2.imread(snap)
        if image is None:
            logger.error("Could not read snapshot image: %s — skipping overlay.", snap)
        return image, snap

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
        image, _ = _load_image()
        if image is not None:
            import cv2
            vis = _draw_calibration(image, cal_payload)
            vis_path = os.path.join(bins_dir, "grid_calibration_overlay.jpg")
            cv2.imwrite(vis_path, vis)
            logger.info("✓ Wrote overlay %s", vis_path)
            if not args.no_show:
                cv2.imshow("Calibrated grid (1-9) — 'q' to close", vis)
                while cv2.waitKey(20) & 0xFF != ord("q"):
                    pass
                cv2.destroyAllWindows()
        return

    # match mode
    if not os.path.exists(cal_path):
        logger.error("No calibration at %s — calibrate this workstation first "
                     "(run with --calibrate on an all-9-bins snapshot).", cal_path)
        return
    with open(cal_path) as f:
        cal_payload = json.load(f)
    occ_payload = build_occupancy(payload, cal_payload)
    out_path = args.out or os.path.join(bins_dir, "occupancy.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(occ_payload, f, indent=2)
    present = sum(1 for v in occ_payload["slots"].values() if v["present"])
    logger.info("✓ Wrote %s (%d/9 slots occupied)", out_path, present)
    image, _ = _load_image()
    if image is not None:
        import cv2
        vis = _draw_occupancy(image, occ_payload, cal_payload)
        vis_path = os.path.join(bins_dir, "occupancy_overlay.jpg")
        cv2.imwrite(vis_path, vis)
        logger.info("✓ Wrote overlay %s", vis_path)
        if not args.no_show:
            cv2.imshow("Bin occupancy (green=present, grey=absent) — 'q' to close", vis)
            while cv2.waitKey(20) & 0xFF != ord("q"):
                pass
            cv2.destroyAllWindows()
```

- [ ] **Step 4: Run the wrapper test + full suite**

Run: `python -m pytest tests/test_grid_calibrator.py tests/test_grid_allocator.py -v`
Expected: PASS (grid_calibrator 9 tests incl. the wrapper test; grid_allocator 12 still green).

- [ ] **Step 5: Verify the module still parses and cv2 stays inside functions**

Run: `python -c "import ast; ast.parse(open('src/detectors/snapshot_obb.py').read())"`
Expected: exit 0. (cv2/numpy must remain imported only inside functions.)

- [ ] **Step 6: Commit**

```bash
git add src/detectors/snapshot_obb.py tests/test_grid_calibrator.py
git commit -m "feat(v2): snapshot_obb --calibrate + match modes (grid_calibration.json, occupancy overlay)"
```

---

## Task 4: end-to-end verification (camera-free)

**Files:** none (smoke run).

- [ ] **Step 1: Synthesize an all-9 calibration snapshot and calibrate**

```bash
python -c "import json,os; W=1280; os.makedirs('runs/_cal',exist_ok=True); mk=lambda i,cx,cy:{'id':i,'corners':[[cx-20,cy-20],[cx+20,cy-20],[cx+20,cy+20],[cx-20,cy+20]],'center':[cx,cy],'conf':0.9}; b=[mk(c,int((c+0.5)*W/6),200) for c in range(6)]+[mk(6+c,int((c+0.5)*W/3),560) for c in range(3)]; json.dump({'frame_w':W,'frame_h':720,'bins':b}, open('runs/_cal/bins.json','w'))"
python src/detectors/snapshot_obb.py --bins runs/_cal/bins.json --calibrate --no-show
```

Expected: `✓ Wrote calibration runs/_cal/grid_calibration.json (9 slots)`.

- [ ] **Step 2: Synthesize a kitting-list snapshot missing one bin and match**

```bash
python -c "import json; W=1280; mk=lambda i,cx,cy:{'id':i,'corners':[[cx-20,cy-20],[cx+20,cy-20],[cx+20,cy+20],[cx-20,cy+20]],'center':[cx,cy],'conf':0.9}; b=[mk(c,int((c+0.5)*W/6),200) for c in range(6) if c!=2]+[mk(6+c,int((c+0.5)*W/3),560) for c in range(3)]; json.dump({'frame_w':W,'frame_h':720,'bins':b}, open('runs/_cal/bins.json','w'))"
python src/detectors/snapshot_obb.py --bins runs/_cal/bins.json --no-show
python -c "import json; o=json.load(open('runs/_cal/occupancy.json'))['slots']; print({k:v['present'] for k,v in o.items()})"
```

Expected: `✓ Wrote runs/_cal/occupancy.json (8/9 slots occupied)`, and the printed dict shows `slot_3` → `False`, all others `True`.

- [ ] **Step 3: Clean up the smoke artifacts (do not commit them)**

```bash
rm -rf runs/_cal
```

- [ ] **Step 4: Confirm the whole suite is green**

```bash
python -m pytest tests/ -v
```
Expected: all PASS (grid_calibrator + grid_allocator).

---

## Self-Review Notes
- **Spec §4 (calibrate):** Task 1 (`calibrate_grid`, 6+3 validation, indexing, row_spacing).
- **Spec §5 (match):** Task 2 (`match_to_grid`, same-row, greedy nearest, ½-spacing cutoff, present flags).
- **Spec §2/§3 (workflow + driver):** Task 3 (`--calibrate` writes `grid_calibration.json`; match writes `occupancy.json` + overlay; "calibrate first" error).
- **Spec §6 (edges):** bad-count ValueError (Task 1 test), stray-drop + missing-bin (Task 2 tests), missing-calibration error (Task 3 main).
- **Spec §8 (testing):** Tasks 1–2 unit tests + Task 4 camera-free e2e.
- **Deferred (FUTURE_TASKS.md):** mixed rows, image-based occupancy, ESP/dashboard wiring — not in this plan.

## Live hardware step (user, not in this plan)
Once on the rig: `initialize_bins_obb.py` (all 9 bins) → `snapshot_obb.py --calibrate` to set the workstation grid; then per kitting list, `initialize_bins_obb.py` → `snapshot_obb.py` to get the occupancy overlay.
