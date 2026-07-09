# OBB Two-Snapshot Grid Flow in the aegis-v2 Pipeline

**Date:** 2026-06-15
**Status:** Approved, implementing

## Goal

Replace the seg-model bin detection in `integration/src/pipeline.py` with the native
OBB detector (`initialize_bins_obb.py` + `project_9_yolov8_obb_1.pt`), driven by a
two-snapshot operator flow, and render the result on the existing FastAPI dashboard,
which is already preprogrammed for a **6-bin top row + 3-bin bottom row** grid.

When the operator runs `python -m integration.src.pipeline`, the pipeline must start
the camera, hand tracker, and dashboard; key `1` locks the workstation grid; key `2`
initialises the kit; and the dashboard's 6+3 grid fills in present bins and greys out
missing ones — end to end, on a real run.

## Two-snapshot flow

1. **Key `1` — Workstation grid calibration.** Snapshot → OBB detect (expecting 9
   bins) → `grid_calibrator.calibrate_grid` → the 9 fixed slots (1–6 top, 7–9 bottom).
   In-memory only; press `1` again to recalibrate (resets any kit). Requires all 9
   bins in view (calibration raises `ValueError` otherwise; the pipeline logs it and
   keeps running).
2. **Key `2` — Kit initialisation.** Snapshot → OBB detect → `match_to_grid` against
   the calibrated grid → per-slot occupancy. Slots with no matched detection are
   **missing bins** (`present=False`). Pressing `2` before `1` logs a warning and is a
   no-op. Snapshot **2 references snapshot 1** for the missing-bin boxes.
3. **Key `q`** quits (unchanged).

## Components

### New: `integration/src/detectors/initialize_bins_obb.py`
Copied from aegis-core. Only the model base-path resolution changes so the model loads
from `cv-models/models/weights/project_9_yolov8_obb_1.pt` inside aegis-v2 (the repo
becomes self-contained). Its `detect_bins(image, model, expected_count)` — phantom /
duplicate filtering, optional expected-count cap — is what the pipeline calls.

### New: `integration/src/detectors/grid_session.py`
Pure-geometry state holder (no camera/model), unit-testable like `grid_calibrator`:

- `calibrate(detections)` → wraps `calibrate_grid`; stores the grid; clears occupancy.
- `init_kit(detections)` → `match_to_grid`; stores occupancy; raises `RuntimeError` if
  uncalibrated.
- `calibrated` (bool), `calibration`, `occupancy`, `missing_slots` (list of indices).
- `to_geofences(present_only=False)` → converts slots to the existing
  `bin_{row}_{col}` geofence dict the rest of the pipeline already consumes.

**Slot → dashboard-bin mapping** (matches the preprogrammed 6+3 grid):

| Grid slot index | Row | bin id | span | slot_start | row_slots |
|---|---|---|---|---|---|
| 1–6 (top) | 0 | `bin_0_0`…`bin_0_5` | 1 | 0…5 | 6 |
| 7–9 (bottom) | 1 | `bin_1_0`…`bin_1_2` | 2 | 0,2,4 | 6 |

Each geofence carries `x_min/x_max/y_min/y_max` (bbox of the OBB corners),
`polygon` (the 4 corners), `confidence`, `span`, `slot_start`, `row_slots`, and
`detected`. For **present** bins the box comes from the snapshot-2 live detection; for
**missing** bins it comes from the snapshot-1 calibrated slot, with `detected=False`.

### Changed: `integration/src/pipeline.py`
- `_detect_bins()` (startup seg snapshot) is removed; bins are not locked at INIT.
- `_load_obb_model()` loads the OBB model once at startup.
- The main loop handles keys `1` and `2`, calling `GridSession` and then a shared
  `_apply_bins(geofences)` that pushes to `PipelineState.update_bins`, resets the
  assignment engine bin map (present bins only), rebuilds the overlay, and — after a
  kit init — applies the work order.
- A `camera.rotate_180` config flag rotates every frame to match the rig.
- The manual-layout fallback (`_build_manual_geofences`) stays; it never used the seg
  model.

### Changed: `integration/src/detectors/__init__.py`
Stops importing `BinDetector` (deleted). This also drops the eager cv2/ultralytics
import that fired on any `detectors` import.

### Changed: `integration/config/settings.yaml`
- `bin_detector.model_path` → `./cv-models/models/weights/project_9_yolov8_obb_1.pt`.
- Add `camera.rotate_180: true`.
- `work_order.targets` updated to a 6+3 shape to match the rig.

## UI integration

The dashboard needs **no code changes**. `PipelineState.update_bins` already reads
`span`, `slot_start`, `row_slots`, and `detected`; `get_layout()` already renders
slot-proportional rows; `_calculate_bin_status` already greys `detected=False` bins and
flags wrong-bin entries. `GridSession.to_geofences()` produces exactly that contract for
all 9 slots, so calibrating (key 1) shows the locked grid and kit init (key 2) fills
present/grey-missing automatically.

The OpenCV overlay reflects whatever geofences are currently applied (calibrated grid
after `1`, present bins after `2`).

## Removals (superseded seg-model lineage)

- `integration/src/detectors/bin_detector.py`, `binDetector.py`
- `integration/src/detectors/layout_mapper.py`, `tests/test_layout_mapper.py`
- `cv-models/scripts/` (`train_bin_segmentation.py`, `inference_bin_detector.py`,
  `coco_to_yolov8_seg.py`)

Kept: `cv-models/models/weights/best.pt` (the trained seg model, per request), and
`grid_allocator.py` (live import of `grid_calibrator`; named target of the
FUTURE_TASKS Method-B work).

## Testing

- New `tests/test_grid_session.py` (pure geometry): calibrate→init_kit happy path; kit
  with a missing bin; init before calibrate raises; recalibration clears the kit;
  `to_geofences` bbox/polygon/slot-mapping correctness for both top (span 1) and bottom
  (span 2) rows; `present_only` filtering.
- Existing grid tests stay green.
- Pipeline verified by compile + dry import + dashboard boot (camera hardware can't run
  headless). A live-run checklist is provided for the rig.

## Acceptance

`python -m integration.src.pipeline` runs; `1` locks the 6+3 grid (dashboard shows 9
slots); `2` populates the dashboard with present bins (white/active) and missing bins
(grey); `q` quits cleanly. Full pytest suite passes.
