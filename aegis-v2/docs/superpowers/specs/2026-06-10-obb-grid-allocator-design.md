# OBB Snapshot → 2-Layer Grid Allocation + 1–9 Indexing — Design

**Date:** 2026-06-10
**Status:** Approved (design); implementation pending
**Scope:** Use the trained OBB model `project_9_yolov8_obb_1.pt` to snapshot the bin
rig and allocate detected bins into a fixed 2-layer grid with permanent indices 1–9.

---

## 1. Problem & Goal

**The grid is indexed first; the snapshot fills it.** We define a **fixed 6×2 grid**
(6 columns, 2 rows = 12 cells) with permanent, hardcoded indices. When the snapshot is
taken, each detected bin is dropped into the grid cell it occupies. The index belongs to
the **grid position**, never to detection order.

The fixed grid for this rig:

- **Top row:** 6 cells, **1 cell per bin** (small/medium bins) → indices **1–6** (left → right).
- **Bottom row:** 6 cells, **2 cells per bin** (large bins) → 3 bins, indices **7–9** (left → right).
- 9 bins max; the bottom row is the same total width as the top (6 cells).

The camera is mounted **upside-down** (180°): without correction the image shows the
bottom (large-bin) layer at the top and left/right mirrored.

On a startup snapshot we: detect the real bins with the OBB model, correct the
orientation, drop each detected bin into its fixed grid cell, and stamp it with that
cell's **permanent index 1–9**. That index is the canonical bin ID used downstream (ESP
load cells, hand-in-bin events, dashboard).

**Missing bins leave a gap — indices never renumber.** If a grid cell has no detected
bin, that index stays empty (`detected:false`); the surrounding indices are unaffected.

**Bin-size detection is NOT implemented yet.** We do not currently distinguish small/
medium from large bins from the image — the 1-cell-vs-2-cell distinction is **hardcoded
by row** (top row = 1 cell, bottom row = 2 cells). A robust size-differentiation system
is planned for the future (see §8).

### Decisions captured during brainstorming
- **Model running** lives in **aegis-core** (`initialize_bins_obb.py`). **Allocation +
  indexing** lives in **aegis-v2** (new pure module).
- **Weight file:** `project_9_yolov8_obb_1.pt` (not the `_rotate` variant).
- **Orientation fix:** **rotate the captured frame 180°** in aegis-core before
  detection — *not* per-axis remapping in the allocator. After rotation everything is
  physically upright.
- **Allocation algorithm:** **geometry-driven relative-spacing (Method B)** is the chosen
  direction. For the *current* fixed rig (rig fills frame, known cell count per row) it
  reduces to a clean band assignment (`pitch = frame_w / K`). The generalized inference —
  variable rows/cells, pitch inferred from detections, multi-workstation — is deferred to
  `FUTURE_TASKS.md`. **Principle: make it work entirely first, then make it adaptable.**
- **Index 1–9 is the canonical bin ID**, not just a display label, and is **fixed to the
  grid position** — assigned by where a bin sits in the grid, never by detection order.
- **Output:** `bins_indexed.json` + an overlay image with indices drawn.
- **Core → v2 handoff:** a **JSON file** (`bins.json` written by core, read by v2).
- **On missing bins:** the full 9-bin grid always renders; an undetected bin is
  returned with `detected:false` (grey placeholder, no hand-matching there) **without
  renumbering any other index**.
- **Bin size is hardcoded by row** (top = 1 cell, bottom = 2 cells); image-based size
  differentiation is deferred to future work.

---

## 2. Architecture

```
[aegis-core]                                  [aegis-v2 / integration]
camera frame ─► rotate 180° ─► OBB model ─► bins.json ─► grid_allocator ─► bins_indexed.json
               (cv2.ROTATE_180)  detect      (raw boxes   (upright split    + overlay (1–9)
                                              + centers)    + 1–9 indexing)
```

The risky geometry (clustering into layers, assigning indices, handling misses) is
isolated in a **pure function** (`grid_allocator.allocate_grid`) with no camera/model
dependency, so it is unit-testable in milliseconds with synthetic coordinates.

---

## 3. Orientation Fix (aegis-core)

In `scripts/inference/initialize_bins_obb.py`:

1. `MODEL_NAME = "project_9_yolov8_obb_1.pt"` (drop `_rotate`).
2. Immediately after every frame read (webcam mode, single-image mode, batch mode),
   apply `frame = cv2.rotate(frame, cv2.ROTATE_180)`.
3. Extend the save step so the handoff carries everything v2 needs:
   - `bins.json` gains `conf` per bin and top-level `frame_w` / `frame_h`.
   - the rotated snapshot frame is also written as a sibling `snapshot.jpg` (v2 draws the
     verification overlay on it).

After this, detection, the saved `bins.json`, and the overlay are all in upright space.
The `bins.json` schema becomes:

```json
{ "frame_w": 1280, "frame_h": 720, "source": "obb", "rule": "nearest_center",
  "bins": [ { "id": 0, "corners": [[x,y],...4], "center": [cx,cy], "conf": 0.91 }, ... ] }
```

---

## 4. The Allocation Algorithm (`allocate_grid`, aegis-v2)

Input: the list of detected bins (`{corners, center, confidence}`) — already upright —
plus the snapshot's `frame_w` (used to locate a bin's column within the fixed grid).

The fixed grid is built **first**, with every cell pre-indexed, then detections are
snapped into it. Indices come from the grid, not the detections.

1. **Build the fixed pre-indexed grid** from the declared layout `[[1,1,1,1,1,1],[2,2,2]]`:
   top cells → indices 1–6 at `slot_start` 0..5 (`span 1`); bottom cells → indices 7–9 at
   `slot_start` 0,2,4 (`span 2`); `row_slots = 6` for both. All start `detected:false`.
2. **Split detections into 2 layers by y-center.** Sort by `cy`; split at the **largest
   *significant* gap** between consecutive `cy` values — a gap only counts as a row
   boundary if it exceeds ~half the median bin height, so a single detected layer is not
   force-split. Smaller-`cy` cluster = **top** row; larger-`cy` cluster = **bottom** row.
   Warn if the split is not 6 (top) / 3 (bottom).
3. **Place each detection into its cell by relative spacing** (Method B), within its row.
   The row has a fixed number of cells `K` (top `K=6`, bottom `K=3`) and — per the
   **"rig fills the frame"** assumption — the cell pitch is `frame_w / K`. Each detection
   maps to the cell its centre falls in: `cell = clamp(floor(cx / frame_w * K), 0, K-1)`,
   snapping to the **nearest unused** cell on collision. This is position-based, not
   order-based: a missing bin in the middle leaves *its* cell empty instead of shifting
   neighbours into the wrong index, and (because the frame is filled) an end gap is caught
   too.
4. **Stamp the matched cell** with the detection's `corners`/`center`/`confidence` and
   `detected:true`. Unmatched cells keep `detected:false`. Extra detections beyond a row's
   capacity are dropped with a warning.

> **Why this is Method B but looks simple here.** The chosen direction is geometry-driven
> relative-spacing inference (Method B). Under the *current* fixed-rig assumptions —
> the rig fills the frame and each row has a known cell count `K` — the pitch is just
> `frame_w / K`, so B reduces to a clean band assignment. The *generalized* B (infer the
> number of rows/cells and the pitch from the detections themselves, with no fixed `K`)
> is deferred to `FUTURE_TASKS.md` §A. This keeps the current version simple and working
> while leaving the scalable algorithm as a direct extension of the same idea.
>
> **Spans are hardcoded by row** for now (top = 1 cell, bottom = 2 cells); inferring span
> from box width (size differentiation) is `FUTURE_TASKS.md` §B.

### Edge handling
- Fewer than 2 distinct y-bands (e.g. only one row detected) → the detected row is
  filled; the other row stays all `detected:false`.
- Empty detection → all 9 cells stay placeholders (the grid still renders).

---

## 5. Data Contract

### Declared layout (constant for this rig)
```python
LAYOUT = [[1, 1, 1, 1, 1, 1],   # top: 6 bins × 1 slot
          [2, 2, 2]]            # bottom: 3 bins × 2 slots
```

### `bins_indexed.json` (allocator output), keyed by `bin_{index}`
```json
{
  "bins": {
    "bin_1": { "index": 1, "layer": "top",    "slot_start": 0, "span": 1,
               "row_slots": 6, "center": [cx, cy], "corners": [[x,y],...],
               "confidence": 0.91, "detected": true },
    "bin_7": { "index": 7, "layer": "bottom", "slot_start": 0, "span": 2,
               "row_slots": 6, "center": [cx, cy], "corners": [[x,y],...],
               "confidence": 0.88, "detected": true }
  },
  "source": "obb",
  "rule": "rotate180_then_band_split"
}
```

`slot_start`: top layer 0..5; bottom layer 0,2,4. `row_slots`: 6 for both. Each cell also
carries `row`, `col`, and `num_bins` (bins in that row) for completeness. This matches
the slot-proportional contract the existing dashboard `layout_mapper`/`state.py` already
consume, so wiring into the dashboard later is a drop-in.

---

## 6. Files

| File | Change |
|------|--------|
| `aegis-core/scripts/inference/initialize_bins_obb.py` | weight → `project_9_yolov8_obb_1.pt`; 180° rotate after each frame read |
| `aegis-v2/integration/src/detectors/grid_allocator.py` | **new** — pure allocator (`allocate_grid`) |
| `aegis-v2/integration/src/detectors/snapshot_obb.py` | **new** — driver: read `bins.json` + snapshot → `bins_indexed.json` + overlay |
| `aegis-v2/integration/tests/test_grid_allocator.py` | **new** — pure unit tests |

---

## 7. How to Test

### Unit tests (no camera/model, milliseconds)
```
cd aegis-v2/integration
pytest tests/test_grid_allocator.py
```
Cases: full 6+3 detection (correct 1–9 mapping); largest-gap layer split; x-ascending
ordering within a layer; missing middle bin leaves a `detected:false` gap without
shifting indices; single-layer-only detection; empty detection; extra/spurious detection
dropped.

### End-to-end (real OBB model + camera)
1. **aegis-core** — run the OBB initializer (rotates 180°, uses `project_9_yolov8_obb_1.pt`):
   ```
   cd aegis-core
   python scripts/inference/initialize_bins_obb.py
   ```
   Mode 2 (webcam); press `s` to save → `aegis-core/runs/bins_obb_raw/bins.json` (+ rotated snapshot).
2. **aegis-v2** — run the snapshot driver on that output (run as a path script, not
   `-m`, so the detectors package `__init__` — which imports cv2/ultralytics — is skipped):
   ```
   cd aegis-v2/integration
   python src/detectors/snapshot_obb.py
   ```
   → writes `bins_indexed.json` and shows an overlay with **1–9** drawn. Verify: index
   **1** = top-left small bin, **6** = top-right, **7–9** = the three large bins left→right.

---

## 8. Out of Scope (later iterations)

**Guiding principle: make this work entirely first, then make it adaptable.** All deferred
work is tracked in the repo-root **`FUTURE_TASKS.md`** (always consulted when working in
this repo). In summary:

- **(A) Adaptability & scalability** — generalized geometry-driven inference (variable rows
  and cell counts, pitch inferred from detections) for multi-workstation use, instead of
  the fixed 6×2 grid.
- **(B) Robust bin-size differentiation** — detect 1-cell vs 2-cell span from box geometry
  instead of hardcoding it by row.
- **(C) Downstream integration** — wire `bins_indexed.json` into the live dashboard/pipeline;
  link ESP load-cell weights to bin indices 1–9; hand-in-bin attribution; distinguish
  "undetected-but-expected" grey from "not-in-job" grey.

See `FUTURE_TASKS.md` for the full, maintained list.
