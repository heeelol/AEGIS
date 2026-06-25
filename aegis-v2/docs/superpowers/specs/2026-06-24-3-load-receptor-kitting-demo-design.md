# 3-Load-Receptor Kitting Demo — Design

**Branch:** `feat/load-receptors-demo`  ·  **Date:** 2026-06-24  ·  **Scope:** full-stack (config, firmware, backend, UI)

## Goal
Demo a kitting flow on **3 load receptors**: 2 BOM source bins + 1 kitting box. Operator picks
from the 2 BOM bins and places items into the box. The **per-bin counter increments only when an
item is placed into the box** (box-cell-driven), not when removed from a bin. Overpick is shown in
the status bar. Kit completes when both bins hit target AND the box total weight matches.

## Physical model (4 load receptors: 3 source bins + box)
| Receptor (load cell id) | Cell | Role | Item | unit_g | Target |
|---|---|---|---|---|---|
| `bin_0_4` | 1 kg | BOM source bin | part_p04 | 3.6 | 3 |
| `bin_0_5` | 5 kg | BOM source bin | part_p05 | 3.1 | 3 |
| `bin_1_2` | 10 kg | BOM source bin | part_p12 | 67.1 | 3 |
| `kit_box` | 5 kg (off-camera) | Kitting box (destination) | — | — | expected 221.4 g |

The other 6 bins have no load cell and `target = 0` → rendered grey "not in BOM".
Expected box total = 3·3.6 + 3·3.1 + 3·67.1 = **221.4 g**. Match tolerance = ±1.5 g.

**Near-equal weights:** bin_0_4 (3.6 g) and bin_0_5 (3.1 g) are too close to tell apart
from the box weight alone, but each bin's **own** load cell bounds its placed-count by what
was removed from it, so final counts stay correct (only a single in-flight item could be
momentarily mis-credited between those two).

## Counting logic (placement-driven)
**Revised 2026-06-25 — per-bin counting (was box-decomposition):** the original
design decomposed the box weight into per-item counts. That failed in practice: a single
5 kg box cell can't resolve a 3.6 g (or 16.6 g) item added on top of a heavy one, so small
items were swallowed by box noise and only the 67 g bin counted. Robust replacement:

1. **Count each BOM bin from its OWN cell** (range-matched: 1 kg cell for 3.6 g parts, etc.):
   `count = round(-bin_weight / unit_g)`. Reliable for small and large items alike.
2. **EMA-smooth** the weights and apply **hysteresis** (a deadband around each .5 boundary)
   so sensor jitter never flickers the count.
3. **The box weight is a verification cross-check only** (`box_verified` = box ≈ expected,
   within a generous `box_tolerance_g`). It is displayed but never drives or gates counts.

## FSM
`INIT` (bins full) → `PICKING` → per-bin `COMPLETE` when count == target → `OVERPICK` when
count > target (status bar: "RETURN N ITEMS TO BIN X") → `KIT_COMPLETE` when **all BOM bins ==
target, no overpick** (box shown as ✓verified cross-check, not a gate) → Complete-kit
(`POST /api/kit/complete`) re-tares all receptors, resets counts, returns to INIT.
Counts refresh every ~3 frames (was 30) for a responsive counter.

## Changes
- **config/settings.yaml**: `work_order.targets` → only bin_1_0 & bin_0_0 nonzero; `loadcells.bin_remap`
  for the 3 firmware ids; add `kit_box` settings (expected total, tolerance).
- **config/inventory.yaml**: `part_a` (63.7), `part_b` (3.6); `bin_1_0: part_a`, `bin_0_0: part_b`; box mapping.
- **firmware/tripleCell.ino**: emit all 3 receptors (`bin_1_0`, `bin_0_0`, `kit_box`). User flashes + calibrates.
- **sensing/inventory.py + pipeline.py**: placement correlation (box-rise + bin-drop → placed-count++).
- **ui/state.py**: kitting-box entity, placed-counts, FSM status, completion check.
- **ui/dashboard.py**: expose box in `/api/bins` or new `/api/kit`; `POST /api/kit/complete`.
- **ui/static**: render kitting box, keep BOM greying, status-bar overpick + completion.

## Testing
- Unit: placement correlation (box-rise attributes to correct bin; bin-drop without box-rise = no count;
  overpick; completion gate on box total). Extend `tests/test_loadcell_count.py`.
- Manual: live with ESP32 (3 cells) + dashboard at `http://localhost:8080`.
