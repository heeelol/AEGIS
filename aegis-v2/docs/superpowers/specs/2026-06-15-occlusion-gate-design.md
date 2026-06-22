# Occlusion Gate — reject extrapolated fingertips that fall into the wrong (top) bin

**Date:** 2026-06-15
**Status:** Approved for implementation

> **Correction (2026-06-17):** Decision #2's anchor priority was **inverted**.
> The original "wrist primary, MCP knuckles fallback" misfired on a genuine reach
> *over* the shelf lip into a top bin: the wrist trails back down over the bottom
> band while the fingers (and knuckles) are genuinely up in the top bin, so the
> gate wrongly reassigned the pick to the bottom bin. The knuckles — not the wrist
> — report which bin the hand is in. The anchor is now **MCP-knuckle centroid
> primary, wrist fallback** (`_occlusion_anchor`). The physical rule in the next
> section still holds; only *which* proximal landmark encodes it changed.

## Problem

MediaPipe Hand Landmarker always returns all 21 landmarks once a hand is
detected. When the operator reaches into a **bottom-layer** bin, the fingers
disappear under the top-shelf lip, but the model still *extrapolates* the
occluded fingertips from the visible part of the hand. The inferred `index_tip`
can land inside a **top** bin's region.

`BinAssignmentEngine.assign()` triggers on a single 2D keypoint inside a flat
bounding box (`_assign_pip` in `integration/src/engine/bin_assignment.py`). A 2D
point-in-box test cannot distinguish "fingertip is really in the top bin" from
"fingertip was hallucinated into the top bin." The result is a **false top-bin
hit** during a genuine bottom-bin pick.

This is the mirror image of the existing `OcclusionHold` problem (which keeps a
*bottom* bin lit while the hand is hidden). The two compose: the gate corrects
the *direction* of the false hit, then the hold keeps the corrected bottom bin
lit. Order: `assign()` (now gated) → `OcclusionHold.apply()`.

## The physical rule it encodes

You cannot have your knuckles/wrist below the top-shelf lip while your fingertip
is genuinely inside a top bin — the arm would have to pass through the shelf. So
when the fingertip lands in a *top* bin but the hand's *proximal anchor* sits in
the *bottom* band, it is an extrapolation artifact → reassign to the bottom bin
beneath.

## Constraints discovered

- MediaPipe per-landmark `visibility`/`presence` for **hands** is not reliably
  populated (`tracker.py` reads `lm.visibility`, usually a constant), so occluded
  fingertips cannot be filtered by confidence. Geometry is the only usable signal.
- The wrist is only **sometimes/partially** visible during a bottom reach
  (confirmed with the user), so the gate cannot depend on the wrist alone.
- The rig is fixed: top rows of single-cell bins, one bottom row of double-cell
  bins. Bin ids are `bin_{row}_{col}`, row 0 = top, bottom = `max(row)`. The grid
  is calibrated live (key 1) or built from `manual_layout`; geofences carry pixel
  extents `x_min/x_max/y_min/y_max`.

## Decisions (from brainstorming)

1. **Home:** integrated into `BinAssignmentEngine`, **not** a separate
   post-processor. The gate needs the wrist/knuckle landmarks, and `assign()`
   already holds the full `HandDetection`. A separate component would only see
   `BinEvent`s (no landmarks), forcing a re-match.
2. **Anchor:** the most-proximal *reliably-available* landmark. **Wrist** if
   finite and in-frame; **else the centroid of the MCP knuckles** (index, middle,
   ring, pinky `_mcp`). Knuckles are the last-but-one to be occluded and far more
   reliable than the fingertip.
3. **Occlusion line:** the **top rim of the bottom bin** (`y_min`), not a midpoint
   between bands. The rim is a real physical feature — exactly where the bin gets
   cut off by the shelf and where fingers vanish under the lip — and it fires
   conservatively (lower line → won't suppress a genuine top reach whose hand
   dips a little). Use the **per-column** rim of the bottom bin beneath the
   anchor; fall back to a **global** line only when no bottom bin is beneath.
4. **On-fire behavior:** **reassign** to the bottom bin beneath (user's choice),
   `method="occlusion_gate"`. Suppress only when there is clearly a bottom reach
   but no bottom bin to assign to.
5. **Conservative by design:** the gate fires *only* when the assigned bin is
   top-row **and** the anchor is at/below the bottom rim. A genuine top reach
   (knuckles up at the top bin) never trips it.

## Architecture

```
hands → BinAssignmentEngine.assign(hands, frame_shape)  → OcclusionHold.apply() → PipelineState
            (stateless geometry + occlusion gate)            (cross-frame latch)
```

The gate is **stateless pure geometry**, computed inside the existing per-hand
loop in `assign()`. No new component, no cross-frame state.

### `set_bin_map` / `set_bin_map_from_geofences` — precompute grid structure

Both bin-map setters recompute, from the current `_bins`:

- `_bottom_row: int` = `max(row over bins)` (parsed from `bin_{row}_{col}`).
- `_top_rows: set[int]` = all rows above `_bottom_row`.
- `_bottom_bins: list[BinRegion]` in the bottom row, sorted by `x_min` (the
  "beneath" lookup table).
- `_global_occ_y: float` = `min(b.y_min for b in _bottom_bins)` — fallback line.

If there is only one row (degenerate / single-layer manual layout), the gate is
inert (`_top_rows` empty).

### `assign(hands, frame_shape=None)` — signature change

`frame_shape` (the pipeline passes `frame.shape`) supplies `(h, w)` for the
in-frame check on the wrist. Optional and defaulted so existing callers/tests
keep working; when `None`, the in-frame check is skipped and the anchor falls to
the MCP centroid if the wrist is non-finite.

### Per-hand gate — `_apply_occlusion_gate(hand, event, frame_shape)`

Called after the per-hand `BinEvent` is computed, before it is appended. Returns
the (possibly rewritten) event.

1. If the gate is disabled, or `event.bin_id is None`, or the event's bin is
   **not** in `_top_rows` → return `event` unchanged.
2. Compute the **anchor** `(ax, ay)`:
   - `wrist = hand.get_landmark("wrist")`; use it if finite and (when
     `frame_shape` given) `0 <= wrist.x <= w and 0 <= wrist.y <= h`.
   - else centroid of the finite MCP knuckles
     (`index_mcp, middle_mcp, ring_mcp, pinky_mcp`).
   - if no usable anchor → return `event` unchanged (cannot judge).
3. Find the bottom bin `B` whose `[x_min, x_max]` contains `ax` (the **anchor**
   x only — *not* the fingertip x). In this rig the top cells span the same total
   width as the bottom cells, so a fingertip in a top bin is always x-aligned with
   some bottom bin; using the fingertip x here would make the global-suppress
   branch (step 5) unreachable. The anchor x is where the arm actually is, so an
   angled reach whose anchor lands beyond/between the bottom bins correctly yields
   "no `B`".
4. **Per-column rim (primary):** if `B` exists and `ay >= B.y_min` → reassign:
   return a `BinEvent` for `B` with `method="occlusion_gate"`, `confidence`
   from `B`, preserving `hand_id`, `handedness`, `hand_point`, `hand_area`.
5. **Global fallback:** if `B` does not exist (anchor x under no bottom bin) and
   `ay >= _global_occ_y` → clearly a bottom reach with no target →
   **suppress** (`bin_id=None`, `method="occlusion_gate"`), logged.
6. Otherwise (anchor above the rim) → genuine top reach → return `event`
   unchanged.

### Direction safety

`y` increases downward (image coords); after `rotate_180` the top row has the
smaller `y`. The condition `ay >= y_min` is derived from the grid's own bottom
band, so it self-adapts to the calibrated layout rather than hardcoding a sign.

## Pipeline wiring

`integration/src/pipeline.py`, `_main_loop`, one changed line:

```python
events = self._assignment.assign(hands, frame.shape)   # ← pass frame shape
events = self._hold.apply(events, hands)
```

No other wiring changes — `set_bin_map_from_geofences` (already called from
`_apply_bins`, `_maybe_apply_manual_layout`, `_create_engines`) now also
precomputes the grid structure.

## Config

New optional block, gate **on by default**:

```yaml
bin_assignment:
  method: "point_in_polygon"
  hand_keypoint: "index_tip"
  overlap_threshold: 0.3
  occlusion_gate:
    enabled: true        # off → assign() behaves exactly as before
```

`BinAssignmentEngine.__init__` reads `config.get("occlusion_gate", {})`.

## UI

A reassigned event lights the bottom bin exactly like a normal pick, then
`OcclusionHold` keeps it lit when the hand vanishes under the lip. The
`method="occlusion_gate"` field is available if a distinct overlay shade is
wanted later (YAGNI for now).

## Testing

Pure-geometry unit tests in `integration/tests/test_occlusion_gate.py` (mirroring
`test_occlusion_hold.py`), driving `BinAssignmentEngine` with a hand-built bin map
and synthetic `HandDetection`s — no camera/model:

- **False top hit, knuckles low** (wrist hidden, MCP centroid below bottom rim,
  `index_tip` in a top bin) → reassigned to the bottom bin beneath.
- **Genuine top reach** (knuckles up near the top bin, `index_tip` in top bin) →
  event untouched.
- **Wrist available and in-frame** → wrist used as anchor (assert it wins over
  the MCP centroid).
- **Wrist off-frame / non-finite** → falls back to MCP centroid.
- **Angled reach** (anchor below global line, anchor x under no bottom bin) →
  suppressed (`bin_id=None`).
- **Anchor above rim, fingertip in top bin** (true top pick at a steep angle) →
  untouched.
- **`enabled: false`** → passthrough (identical to pre-gate behavior).
- **Single-row layout** → gate inert, no reassignment.
- **`assign(hands)` with no `frame_shape`** → still works (in-frame check skipped).

## Out of scope

- Depth/stereo or second-camera disambiguation (future hardware option).
- Distinct dashboard/overlay styling for gated events.
- Per-finger occlusion detection or trajectory-based entry events.
- Changing `nearest_centroid` / `area_overlap` methods — the gate guards the
  default `point_in_polygon` path; it applies wherever a top-row event is
  produced, but tests focus on the default method.
