# Workstation Grid Calibration + Nearest-Slot Matching — Design

**Date:** 2026-06-11
**Status:** Approved (design); implementation pending
**Branch:** `FableDoesGridAllocator`
**Scope:** aegis-v2 — replace the assumption-based band allocation with a calibration that
defines the grid from a real "all bins present" snapshot, then matches each kitting-list
snapshot against it.

---

## 1. Problem & Goal

The current allocator assumes the rig fills the frame and divides it into equal columns to
place bins. That assumption is fragile across camera framings and can't robustly tell
"is the bin there or not."

Instead, **let the bins define the grid**. The fixed rig is still **6 top slots (small/
medium, 1-cell, indices 1–6) + 3 bottom slots (big, 2-cell, indices 7–9)**, small/medium
top-only and big bottom-only, with **fixed indices 1–9**. We:

1. **Calibrate once per workstation** from a snapshot with **all 9 bins present** — the 9
   detected bin boundaries *become* the 9 grid slots.
2. **Per kitting list**, take a snapshot and **match** each detected bin to the **nearest
   calibrated slot**, yielding per-slot occupancy ("bin present / not there").

### Goals
- Robust "is the bin there or not" without assuming even spacing or a frame-filling rig.
- **Zero labeling / no fiducials / no corner-clicking** — calibration is just one good snapshot.
- Adapts to a new workstation by re-calibrating (no retraining).

### Decisions captured during brainstorming
- Grid is the **fixed 6+3** with **fixed indices 1–9** (index = canonical bin id; reverted
  from the earlier mixed-row idea). Size is hardcoded by row (top 1-cell, bottom 2-cell).
- **Calibration scope = per workstation** (once); **matching = per kitting list**.
- The OBB model is reused as-is (`project_9_yolov8_obb_1.pt`); **no new training**, no HBB.
- Matching: **nearest calibrated slot centre, one-to-one, same row**, with a distance
  cutoff (~half the row's slot spacing) to drop stray detections.

---

## 2. Workflow

```
[once per workstation]                         [per kitting list]
all-9-bins snapshot                            kitting-list snapshot
   │ OBB detect (6 top + 3 bottom)                │ OBB detect (any subset)
   ▼                                              ▼
calibrate_grid ──► grid_calibration.json ──►  match_to_grid ──► occupancy
 (9 slots: index,                              (each detection → nearest
  row, center, box)                             calibrated slot; rest = absent)
```

The aegis-core OBB step (rotate 180°, detect, save `bins.json` + `snapshot.jpg`) is
unchanged and feeds both modes.

---

## 3. Architecture

A new pure module **`grid_calibrator.py`** (no cv2/model) sits beside the existing
`grid_allocator.py`:

- `calibrate_grid(detections)` → the 9-slot calibration (pure geometry; unit-testable).
- `match_to_grid(detections, calibration)` → per-slot occupancy (pure geometry).

The existing `grid_allocator.allocate_grid` (band-based) is **kept but no longer the
primary path** — it stays as a fallback / for comparison. The driver
(`snapshot_obb.py`) gains a `--calibrate` mode that writes `grid_calibration.json`, and a
default match mode that loads it. Keeping calibration vs matching in one small pure module
isolates the risky geometry for fast unit tests.

A detection is `{"corners": [[x,y]*4], "center": [cx, cy], "conf": float}` (from the
aegis-core handoff), already upright.

---

## 4. Calibration (`calibrate_grid`)

Input: detections from the **all-9** snapshot.

1. **Split into 2 rows** by largest y-gap (reuse `grid_allocator.split_rows_by_y`).
2. **Validate**: top row has exactly 6, bottom exactly 3. Otherwise raise/return an error
   so the operator re-takes the calibration snapshot (a bad calibration poisons everything).
3. **Order & index**: sort each row by `cx` ascending; top → indices **1–6**, bottom →
   **7–9**.
4. **Store per slot**: `index`, `row` (0 top / 1 bottom), `layer` ("top"/"bottom"),
   `span` (1 top / 2 bottom), `center` `[cx, cy]`, `corners`, plus the row's **slot
   spacing** (median centre-to-centre distance) used later as the match cutoff.

### `grid_calibration.json`
```json
{
  "source": "obb", "frame_w": 1280, "frame_h": 720,
  "slots": {
    "slot_1": { "index": 1, "row": 0, "layer": "top", "span": 1,
                "center": [cx, cy], "corners": [[x,y],...],
                "row_spacing": 213.0 },
    "...": {},
    "slot_9": { "index": 9, "row": 1, "layer": "bottom", "span": 2, "...": {} }
  }
}
```

---

## 5. Matching (`match_to_grid`)

Input: detections from a **kitting-list** snapshot + the loaded calibration.

1. **Assign each detection a row** by comparing its `cy` to the calibration's top-row mean
   `cy` and bottom-row mean `cy`; pick the nearer row.
2. **Greedy nearest-centre, one-to-one**: consider all (detection, same-row slot) pairs;
   repeatedly take the smallest centre-to-centre distance, assign that detection to that
   slot, remove both. Skip a pair if the distance exceeds the **cutoff = 0.5 × that row's
   `row_spacing`** (drops a stray/spurious detection rather than snapping it to a far slot).
3. **Result per slot**: matched → `present: true` with the live `corners`/`center`/`conf`;
   unmatched calibration slots → `present: false` ("bin not there"). Detections that match
   nothing are dropped with a warning.

### Match output (per kitting list) — keyed by the fixed slot index
```json
{
  "source": "obb", "rule": "calibrated_nearest_slot",
  "slots": {
    "slot_1": { "index": 1, "layer": "top", "span": 1, "present": true,
                "center": [cx, cy], "corners": [[x,y],...], "confidence": 0.91 },
    "slot_4": { "index": 4, "layer": "top", "span": 1, "present": false }
  }
}
```
`index` is the canonical bin id (fixed by calibration) the ESP load cells / hand events key
off. The full 9-slot grid is always present; `present:false` = bin not there.

---

## 6. Edge Handling
- **Calibration ≠ 6+3 detected** → error, ask operator to re-take (do not write a partial
  calibration).
- **Match: detection beyond cutoff from every same-row slot** → spurious, dropped (warn).
- **Two detections nearest the same slot** → greedy gives it to the closer one; the other
  takes its next-nearest slot within cutoff, else is dropped.
- **No calibration file present at match time** → driver errors with "calibrate this
  workstation first."

---

## 7. Files

| File | Change |
|------|--------|
| `aegis-v2/integration/src/detectors/grid_calibrator.py` | **new** — pure `calibrate_grid` + `match_to_grid` |
| `aegis-v2/integration/tests/test_grid_calibrator.py` | **new** — pure unit tests |
| `aegis-v2/integration/src/detectors/snapshot_obb.py` | add `--calibrate` mode (writes `grid_calibration.json`) and default match mode (loads it, writes occupancy + overlay) |
| `aegis-v2/integration/src/detectors/grid_allocator.py` | unchanged (kept as fallback) |

---

## 8. Testing

`test_grid_calibrator.py` (pure, no camera/model):
- **Calibration:** full 6+3 → slots 1–9 with correct row/index/centers; rejects 5-top or
  4-bottom inputs; left→right ordering within a row.
- **Matching:** all 9 present → all `present:true` at the right slots; a missing middle bin
  → that slot `present:false`, neighbours unaffected; a shifted/jittered detection still
  matches its slot; a stray detection far from all slots is dropped; a bin that moved closer
  to a neighbouring slot matches the nearer slot (proves nearest-centre, not index order).

Manual e2e: `snapshot_obb.py --calibrate` on an all-9 snapshot → `grid_calibration.json`;
then `snapshot_obb.py` on a kitting-list snapshot → occupancy overlay (present slots
boxed/numbered, absent slots greyed).

---

## 9. Out of Scope (see repo-root `FUTURE_TASKS.md`)
- Mixed-size rows / fully adaptable multi-workstation layouts (variable slot counts).
- Image-based occupancy (confirming "empty" against bare-shelf pixels) to catch OBB
  false-negatives — current matching trusts the detector.
- ESP load-cell weight → slot-index linkage and dashboard wiring.
