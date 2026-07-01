# Sequential Single-Bin Kitting FSM — Design

**Branch:** `feat/load-receptors-demo` · **Date:** 2026-06-25 · Supersedes the counting model in
the 2026-06-24 spec.

## Why
Weight-based multi-bin attribution is ambiguous: (1) item weight varies (±2 g on a 50 g part), so a
48 g placement + a 2 g part from another bin gets mis-attributed; (2) two different items of the same
weight from different bins can't be told apart on overpick. **Fix: only one bin is active at a time**,
so the box's weight change can only come from that one item type.

## Decisions (locked 2026-06-25)
- **No fixed order.** The operator may start any not-done bin; the first **clear ~½-unit drop**
  activates it and **soft-locks** all other not-done bins (trained behaviour).
- **Forward-only.** A completed (green) bin stays locked; touching it is a fault.
- **All 4 red faults enabled** (now load-cell-detectable): `pick-from-wrong-bin`,
  `return-to-wrong-bin`, `remove-from-kit`, `overpack-kit`.
- **Faults auto-clear** when the weights return to a valid state (no acknowledge button).

## Counting — baseline banking
- `baseline_box` = box weight banked from completed bins (0 after tare).
- Active bin count: `placed = round((box − baseline_box) / unit_active)` (hysteresis-debounced).
- On bin completion (`placed == removed == target`): `baseline_box = box` (bank), bin → DONE,
  active → None. Completed counters are frozen (never re-evaluated) → noise-proof.
- `removed[bin]` from each bin's own cell. `holding = removed[active] − placed[active]` (in hand).

## FSM
```
IDLE (active=None; every not-done bin AVAILABLE)
  └ a not-done bin's raw drop ≥ activation_frac → PICKING(bin_i)  [others not-done → LOCKED]
PICKING(bin_i):
  box rises by unit_i          → placed++
  removed > target             → "↩ RETURN N" (overpick; decrements on return; NOT a fault)
  placed==removed==target      → bin_i DONE, bank baseline → IDLE
  ── red faults → FAULT (auto-clear when corrected) ──
  placed > target              → overpack-kit
  holding==0 AND box drops below committed → remove-from-kit
  a non-active bin drops        → pick-from-wrong-bin
  a non-active bin rises (or a DONE bin changes) → return-to-wrong-bin
last bin DONE → KIT_COMPLETE → Complete kit (POST /api/kit/complete) → re-tare → IDLE
```
remove-from-kit is only tested when `holding == 0` (empty hands) — per the operator assumption that
placing and removing never happen simultaneously.

## UI
Per-bin display state (client-derived from `kit.active`, `kit.done`, counts):

| State | Look | Status bar |
|-------|------|------------|
| AVAILABLE | white `0/target`, "ready" | `PICK FROM ANY BIN` |
| ACTIVE | bright, pulsing, big `placed/target`; "↩ RETURN N" if overpicked | `PICK N FROM BIN X` / `RETURN N TO BIN X` |
| LOCKED | dimmed + 🔒 | — |
| DONE | green, frozen `target/target` | — |
| not-in-BOM | grey | — |

Full-screen **red overlay** on any fault (`OVER-PACKED — REMOVE N`, `WRONG BIN — RETURN ITEM TO BIN X`,
`ITEM REMOVED FROM KIT`). Box is sensing-only (no panel).

## Backend surface
`kit` payload adds `active` (bin id|null) and `done` (list); keeps `placed/removed/targets/overpick/
alert/box_grams`. `/api/bins` keeps `current`(=placed)/`removed`/`total`.

## Tests (placement)
activation on first clear drop · only-active-bin counts · variance tolerated · bank-and-advance ·
overpick badge & return · forward-only lock · all 4 faults raise/auto-clear · complete.
