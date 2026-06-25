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

## Counting logic — box-verified (conservation of mass)
**Revised 2026-06-25 (v3).** A bin's count rises only when the box **verifies** the item is in
it: the weight that left the bin must arrive in the box (`qty = matched box increase / unit_g`).
History: v1 decomposed the *absolute* box weight (small items lost in noise → only the 67 g bin
counted); v2 counted purely per-bin (lost the box verification). v3 keeps both, robustly, by
tracking the box's **change from a committed baseline**:

1. `removed[bin] = round(-bin_weight / unit)` from each bin's own cell (EMA-smoothed, hysteresis).
2. `unaccounted = box_weight − Σ placed·unit` (box weight not yet credited).
3. **Credit** a bin (largest unit first, so the right item matches the box step) when an item
   left it (`placed < removed`) **and** `unaccounted ≥ its unit`. **Un-credit** when the item
   leaves the box (box drops) or returns to the bin.

Because it's a *delta from the committed baseline*, a +3.6 g step is resolvable even with 200 g
already in the box — the v1 failure mode is gone (verified by `test_small_item_counts_on_top_of_heavy_box`).

## FSM
`INIT` (bins full, box empty) → `PICKING` → per-bin `COMPLETE` when verified count == target →
`OVERPICK` when count > target (bin shows "↩ RETURN N"; status bar too) → `KIT_COMPLETE` when
**all BOM bins == target, no overpick** → Complete-kit (`POST /api/kit/complete`) re-tares all
receptors, resets counts, returns to INIT. Counts refresh every ~3 frames (was 30).

UI: the kitting-box panel is **removed** (box is sensing-only); each bin shows a small **"↩ RETURN N"**
badge above its counter on overpick.

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
