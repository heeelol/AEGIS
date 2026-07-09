# Load-Cell-Driven Pick Count (bin_0_5) — Design

**Date:** 2026-06-17
**Status:** Approved (design); pending implementation plan
**Component:** `aegis-v2/integration/src/pipeline.py` (+ config: `inventory.yaml`, `settings.yaml`)

---

## Problem

The full load-cell data path already exists and runs every frame, but the last
link is missing:

- `LoadCellReader` streams `{bin_id: grams}` from an ESP32 over serial. ✅
- `InventoryTracker` converts a bin's (negative) weight into items taken,
  `round(-weight / unit_g)`. ✅ — **but it is only used in the standalone tester,
  never in the live pipeline.**
- `PipelineState` already carries a raw `weight` per bin and merges the
  load-cell layout into the dashboard. ✅
- `Pipeline._init_loadcells()` already builds the reader and polls
  `get_weights()` every 30 frames. ✅

So weights reach the dashboard as raw grams, but **nothing turns those grams
into a pick count**. `pick_count` (the dashboard's "current") only moves via the
manual +/- buttons (`adjust_pick_count` / `set_pick_count`). There is no
automatic pick counting today from any source.

## Goal

Close the loop for **bin_0_5**: its load-cell weight should automatically drive
its pick count in the dashboard, so removing items updates "current / target"
(e.g. `2 / 5`) with no manual input. Built generically (keyed by bins mapped in
`inventory.yaml`) so any future mapped bin works the same way; `bin_0_5` is
simply the only mapped bin today.

## Non-goals

- No change to `PipelineState` or the dashboard — `weight` and `current`
  already flow through to the UI.
- No CV/hand-based pick counting (separate concern).
- No multi-bin rollout beyond what `inventory.yaml` declares.
- No firmware/driver work — host-side CP210x driver install is an operator step
  (see Hardware note), not code.

---

## Approach

The single new responsibility is *weight → pick count*. Of three candidate
homes:

- **A — in the pipeline loop (chosen).** `Pipeline` owns an `InventoryTracker`;
  on each load-cell poll it computes `items_taken` and calls
  `state.set_pick_count(bin_id, n)` for mapped bins. Keeps `PipelineState`
  dumb and `InventoryTracker` reusable; one obvious data path.
- B — inside `PipelineState.update_loadcells`. Couples the state container to
  inventory logic.
- C — compute in the dashboard read path (`get_bins`). Recomputes every HTTP
  request and mixes concerns.

## Components & changes

1. **`config/inventory.yaml`** — add the item and bin mapping:
   ```yaml
   items:
     amp: { unit_g: 3.8 }
   bins:
     bin_0_5: amp
   ```
   (Existing `bolt_m6` / `bin_0_0` entries stay as they are.)

2. **`config/settings.yaml`** — under `sensing.loadcells`:
   ```yaml
   enabled: true
   port: "COM5"      # CP210x bridge (driver installed 2026-06-17)
   bin_remap:
     bin_0_0: bin_0_5   # firmware labels its single cell bin_0_0; we want bin_0_5
   ```

3. **`integration/src/sensing/loadcell.py`** — apply `bin_remap` when caching a
   reading: after `_parse_line`, rename any key present in the remap before
   storing. So `get_weights()` and `get_layout()` already speak `bin_0_5`
   downstream; no other component knows about the firmware's key. Empty/absent
   remap → identity (unchanged behavior).

4. **`integration/src/pipeline.py`**:
   - `_init_loadcells()` also constructs an `InventoryTracker` (loads
     `config/inventory.yaml`), stored as `self._inventory`. If the file has no
     usable mappings, the tracker is still built (it just drives no bins).
   - New `_apply_loadcell_counts()`: when the cell `is_connected()`, for each
     bin the tracker knows about, call `state.set_pick_count(bin_id, items_taken[bin_id])`.
     Called once at init and right after each `update_loadcells()` poll in
     `_main_loop` (every 30 frames).

No `state.py` or `dashboard.py` changes.

## Data flow

```
ESP32 (AMP @ 3.8 g/item)  streams {"bins":{"bin_0_0":-7.6}}
  → LoadCellReader bin_remap   bin_0_0 → bin_0_5
  → LoadCellReader.get_weights()            {"bin_0_5": -7.6}
  → InventoryTracker.items_taken()          {"bin_0_5": 2}
  → state.set_pick_count("bin_0_5", 2)
  → dashboard shows  2 / 5  + live grams
```

## Authority & safety

- **Authoritative:** for mapped bins, the load-cell count overwrites the pick
  count each poll. `bin_0_5` is owned by its cell.
- **Connected-guard:** only overwrite while `LoadCellReader.is_connected()` is
  true (data seen within `stale_after`). On disconnect / stale read, stop
  overwriting — the last value stays and manual +/- remains usable; the count is
  never clobbered to 0 by a dropped link.
- **Untouched bins:** any bin absent from `inventory.yaml` is never written by
  this path — today's manual behavior is fully preserved.

## Testing

- **Unit** (`integration/tests/test_loadcell_count.py`, pure — no serial, no
  cv2, follows `test_finger_vote` style with a fake reader/tracker or injected
  weights):
  - mapped bin with negative weight → `set_pick_count` called with the rounded
    count (e.g. `-7.6 g @ 3.8 → 2`).
  - clamp: small positive/zero weight → count `0`, never negative.
  - connected-guard: when `is_connected()` is false, no `set_pick_count` call
    (existing count preserved).
  - unmapped bin present in weights → never written.
  - **remap** (`integration/tests/` for loadcell): a line keyed `bin_0_0` with
    `bin_remap: {bin_0_0: bin_0_5}` → `get_weights()` returns it under
    `bin_0_5`; `get_layout()` reflects the remapped id; identity when remap empty.
- **Live** (after the CP210x driver is installed and a `COMx` exists): run
  `python -m sensing.loadcell --port COMx` to confirm `bin_0_5` streams, then
  run the pipeline and watch `bin_0_5` track `n / 5` as AMPs are removed.

## Hardware note (resolved 2026-06-17)

The ESP32 firmware is flashed and transmitting. The host CP2102 bridge had
**Code 28 (driver not installed)**; the **Silicon Labs CP210x VCP driver** was
installed via `pnputil /add-driver silabser.inf /install` and the device now
enumerates as **COM5** (status OK). A live read confirmed it streams
`{"bins":{"bin_0_0":<grams>}}` at ~−3.8 g idle — hence the `bin_0_0 → bin_0_5`
remap. The −3.8 g was traced to **one AMP missing from "full"**; it was
reincluded, which also fixed the per-unit weight to **3.8 g** (one AMP =
3.8 g). Resolved — not tare drift.
