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
1. The **box cell is the trigger**: when box weight rises by ≈ one item's `unit_g`, a placement occurred.
2. **Attribute** the placement to the BOM bin whose weight dropped by ≈ its own `unit_g` in the same
   window (distinct weights 63.7 vs 3.6 make this unambiguous; box delta is a cross-check).
3. Increment that bin's **placed-count** (the `current` shown as `current/target`).
   Removing from a bin without placing does nothing.

## FSM
`INIT` (box tared empty, bins full) → `PICKING` → per-bin `COMPLETE` when placed == target →
`OVERPICK` when placed > target (status bar: "RETURN N ITEMS TO BIN X") → `KIT_COMPLETE` when both
BOM bins == target AND |box_total − 201.9| ≤ tol → Complete-kit (`POST /api/kit/complete`) tares box,
resets counts, returns to INIT.

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
