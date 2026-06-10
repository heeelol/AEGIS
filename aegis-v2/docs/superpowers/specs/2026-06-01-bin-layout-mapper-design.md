# Bin Layout Mapper & Slot-Proportional Dashboard — Design

**Date:** 2026-06-01
**Status:** Implemented (dashboard preview); pipeline/camera wiring pending
**Scope:** aegis-v2 — integrate a trained bin model with the operator dashboard for a fixed, non-uniform bin rig.

---

## 1. Problem & Goal

The physical rig is **fixed and non-uniform**:

- **Top layer:** 6 slots → 6 small/medium bins, **1 slot each**.
- **Bottom layer:** 6 slots → 3 large bins, **2 slots each**.
- 9 bins total.

The operator dashboard must **represent** this layout faithfully (slot-proportional: bottom bins drawn twice as wide, aligned under the top pairs), while a trained CV model detects the real bins from a startup snapshot. The dashboard uses *pseudo positions* (a logical slot grid), not raw pixels — pixels are only needed internally for hand-to-bin matching.

### Decisions captured during brainstorming
- **UI representation:** slot-proportional (Option A), not uniform boxes.
- **Layout source:** **hybrid** — config declares the grid; detection fills in real pixel boxes.
- **On detection mismatch:** **use only what's detected** — the declared grid always shows; undetected slots render as grey placeholders (no hand-matching there).

---

## 2. Architecture

A new **pure module** sits between the detector and the rest of the system:

```
Snapshot ─► BinDetector.detect_raw() ─► LayoutMapper ─► geofence dict ─► Engine · Overlay · Dashboard
            (raw boxes)                 (boxes + layout spec → slots)
```

Chosen over extending `BinDetector` in place (Approach 1) or reusing `manual_layout` (Approach 3) because the risky geometry — clustering boxes into rows, assigning slots, handling misses — is isolated as a pure function that is trivially unit-testable with synthetic coordinates (no camera/model).

---

## 3. Data Contract

### Layout spec (config)
A list of rows, each a list of per-bin slot **spans** (left → right):

```yaml
layout:
  rows:
    - [1, 1, 1, 1, 1, 1]   # top: 6 bins × 1 slot   → 6 slots
    - [2, 2, 2]            # bottom: 3 bins × 2 slots → 6 slots
```

### Enriched geofence entry (mapper output)
```python
"bin_1_0": {
    "x_min","x_max","y_min","y_max",  # real pixels — only when detected
    "confidence",
    "row": 1, "col": 0,               # col = bin index within row
    "slot_start": 0, "span": 2,       # position in the row's slot track
    "row_slots": 6,                    # total slots in this row
    "detected": true                   # false → grey placeholder
}
```

**IDs:** `bin_{row}_{col}` (col = bin index within row), preserving compatibility with FSM events, work order, and `_parse_bin_id`.

---

## 4. The Mapping Algorithm (`map_detections_to_layout`)

1. **Build the declared skeleton** from the layout spec — every bin present regardless of detection (guarantees the grid shape).
2. **Assign detections to rows by vertical band:** `row = clamp(int(y_center / frame_h * num_rows), 0, num_rows-1)`. Robust for a fixed rig with clearly separated rows; allows empty rows (single-row / no detection cases).
3. **Assign within a row by nearest expected slot centre** (normalised x): each detected box snaps to the nearest *unused* declared slot. This is the key robustness property — a missing bin in the middle of a row leaves the correct slot empty instead of shifting its neighbours. Naive left-to-right zip would cascade-mislabel.
4. **Fill the contract:** matched bins get real pixel boxes + `detected=True`; unmatched declared bins get `detected=False`. Extra detections (more boxes than slots) are dropped with a warning.

---

## 5. UI Changes

- **`state.py`** — `BinStatus` gains `span`, `slot_start`, `row_slots`, `detected` (defaulted, backward-compatible). `update_bins()` copies them. `get_layout()` emits per-bin slot placement (`{id, slot_start, span, detected}` and `row_slots` per layer), with a **uniform fallback** (`slot_start = position`, `span = 1`, `row_slots = bin count`) when no explicit layout is present, so the legacy layout still renders. `_calculate_bin_status()` returns `"grey"` for undetected bins.
- **`app.js`** — `renderBins()` lays out each row as a slot track: `grid-template-columns: repeat(row_slots, 1fr)` and per bin `grid-column: slot_start+1 / span N`. Undetected bins reuse the existing grey rendering.
- **`dev_mock.py`** — drives the 6+3 layout through the mapper (omitting one bottom bin to demo the grey placeholder).

Untouched: FSM, bin-assignment engine, OpenCV overlay, dashboard backend (`dashboard.py`).

---

## 6. Testing

`integration/tests/test_layout_mapper.py` (pure, no camera/model) — 6 cases:
full detection, missing middle bin (no neighbour shift), out-of-order x, single row only, empty detection, extra/spurious detection. **All passing.**

Manual check: `scripts/dev_mock.py` → `http://localhost:8080` renders the slot-proportional grid with a grey placeholder, live.

---

## 7. Files

| File | Change |
|------|--------|
| `integration/src/detectors/layout_mapper.py` | **new** — pure mapper |
| `integration/tests/test_layout_mapper.py` | **new** — 6 unit tests |
| `integration/src/ui/state.py` | slot fields; `get_layout()` reshape; grey-when-undetected |
| `integration/src/ui/static/app.js` | slot-track rendering |
| `scripts/dev_mock.py` | 6+3 layout via mapper |

---

## 8. Remaining Work (out of scope for this iteration)

- **Pipeline/camera wiring:** `pipeline._detect_bins()` → `detector.detect_raw()` → `map_detections_to_layout()`; add `detect_raw()` to `BinDetector`; add `layout:` to `settings.yaml`. Requires the trained `best.pt`.
- **Optional:** distinguish "undetected-but-expected" grey from "not-in-job" grey (currently both render identically).
- **Optional:** x-position calibration if bins do not span the full frame width (mapper currently assumes bins are laid roughly across the frame).
