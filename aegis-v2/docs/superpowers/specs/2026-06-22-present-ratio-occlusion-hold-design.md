# present_ratio-Driven Occlusion Hold

**Date:** 2026-06-22
**Status:** Approved (design), implementing
**Builds on:** [2026-06-22-foreground-occlusion-gate-design.md](2026-06-22-foreground-occlusion-gate-design.md)

## Problem

`OcclusionHold` exists (keeps a bottom bin lit while a hand is hidden under the
shelf) but is **not wired into the pipeline**, and its release rule is broken by
extrapolation. It releases the moment a hand's handedness is "seen again" outside
the held bin — but while the fingers are under the shelf, MediaPipe keeps
emitting a **phantom** extrapolated hand up in a top bin. The hold treats that
phantom as "seen again" and releases too early, so the bottom bin flickers off
mid-pick.

## Goal

Visual continuity **only**: keep the correct bottom bin highlighted in the
overlay/dashboard throughout an occluded pick. No effect on load-cell / inventory
counts. The held event carries `method="occlusion_hold"`.

## Key idea

Use the foreground signal to tell a *phantom* hand from a *real* re-emergence.
A hand's fingertip is **occluded** this frame when the foreground ratio at its
claimed point (`event.hand_point`) is below a floor. Once occluded, we trust the
**last assignment taken while the hand was still visible** and freeze it.

### Two distinct thresholds

| Threshold | Value | Job |
|-----------|-------|-----|
| `present_ratio` (gate) | 0.20 | "Don't *trust* this top-bin hit." Generous, to aggressively catch phantoms and reassign them down. |
| `occlusion_ratio` (hold) | 0.05 | "The fingertip is *genuinely gone*, freeze the last good bin." Strict, so a weakly-visible real hand (ratio ~0.15) is **not** falsely frozen. |

A non-zero floor (0.05), not exactly 0, is used so sensor noise (a stray
foreground pixel) cannot keep the ratio from ever reaching the trigger.

## Revision (2026-06-22): bin-keyed, handedness-independent

The first cut keyed holds by handedness. In testing, MediaPipe's left/right label
**flips** on an occluded/extrapolated hand, so a single hand could register as a
phantom "right" that never released — a stuck `0.0` box for a hand that wasn't
there. Fix: key the hold by **bin**, ignore handedness entirely, and **drop
occluded fingertip events** (the unreliable extrapolated tips) so they never light
a bin or leave a stray marker.

A bin is **held** while (1) it has been *visibly picked* — a genuine,
non-occluded in-bin assignment armed it — and (2) it is still *occupied* (real
foreground in the bin region). Each frame:

1. **Arm** every eligible bin that has a genuine (non-occluded) in-bin event.
2. **Release** any armed bin no longer in `occupied_bins` (forearm left).
3. **Drop** events whose `hand_id` is occluded (fingertip foreground below
   `occlusion_ratio`) — unreliable extrapolations.
4. **Emit** a synthetic `occlusion_hold` event for each armed bin no live event
   still covers.

`apply(events, hands, occluded_ids, occupied_bins)`: `occluded_ids` are the
`hand_id`s occluded this frame; `occupied_bins` the bottom bins with foreground.
Handedness is no longer used as a key. The tuning overlay also excludes
`occlusion_hold` events from its patch boxes, so the held bin shows via the normal
bin highlight, not a `0.0` fingertip marker.

### Pass-through / emergence guard (2026-06-22)

When a hand reaches up to (or above) the rack, the forearm enters from the bottom
of the frame and transits the bottom bin beneath it — keeping its region
"occupied", which falsely held the lower bin (two bins active, or a hold lingering
while the hand hovers above the rack before reaching a top bin).

Fix: `BinAssignmentEngine.bottom_bins_with_hand_above(events, occluded_ids)`
returns the bottom bins whose column holds a **genuinely-visible (non-occluded)
fingertip above the bin's top rim** — i.e. the hand has emerged above the rack
(in a top bin *or* just hovering above it). The pipeline subtracts these from
`occupied_bins`, so such a bin is treated as not-occupied → released and not held.
Occluded fingertips (the extrapolated tip of a real under-shelf reach) are
excluded via `occluded_ids`, so genuine bottom picks keep holding. This
generalizes the earlier top-pick-only guard: the discriminator is *fingertip
visible & above* (emergence) vs *fingertip occluded* (under-shelf reach).

---

### Original design (superseded by the revision above)

## State machine (per handedness)

A hand is **genuinely present** when its handedness is tracked this frame AND it
is **not** occluded (fingertip ratio ≥ `occlusion_ratio`).

**Occupancy signal (the lock fix):** `occupied_bins` is the set of eligible bins
that still contain real foreground (a forearm reaching in), measured by
`region_ratio` over the bin box against `occupancy_ratio`. This is the
authoritative "is the hand still there" signal — without it, a removed hand that
never re-emerges (absent, or a stale phantom) would lock the bin lit forever.

Each frame:

1. **Genuinely present** hand:
   - In an eligible (bottom-row) bin → record it as `last_trusted`.
   - Visible outside the eligible bins → clear `last_trusted`.
   - Either way → **release** any existing hold (real re-emergence).
2. **Held bin no longer occupied** (region foreground below `occupancy_ratio`) →
   **release** (the forearm left; the hand was removed).
3. **Not genuinely present** hand (absent, or present-but-occluded phantom) whose
   `last_trusted` bin is **still occupied** → **latch** (start holding). A bin
   that is not occupied is never latched.
4. **Output:**
   - Drop the live event of any held-and-occluded handedness (ignore the drifting
     phantom; the frozen bin is authoritative).
   - Emit a synthetic `occlusion_hold` event for each held bin not already active.

This unifies all cases: a phantom up top (present + occluded) keeps holding **only
while the forearm is still in the bin**; a genuinely visible hand releases; and a
removed hand releases as soon as its bin region goes background. When no
foreground model is active, `occupied_bins` is `None` and the occupancy checks are
skipped (re-emergence-only release, the prior behavior).

### Thresholds

| Knob | Default | Meaning |
|------|---------|---------|
| `occlusion_ratio` | 0.05 | fingertip foreground floor → "this hand is occluded" |
| `occupancy_ratio` | 0.05 | bottom-bin region foreground floor → "a forearm is still in this bin" |

## Components

### `OcclusionHold` (existing — extended, stays pure/testable)
- `apply(events, hands, occluded=None)` gains `occluded: set[str]` — the set of
  handednesses occluded this frame. Defaults to empty (then behavior degrades to
  the prior absence-only latch, still valid).
- Tracks `last_trusted[handedness]` and `holds[handedness]`.
- `set_eligible_bins` prunes both `holds` and `last_trusted` for bins that are no
  longer eligible.

### `BinAssignmentEngine` (existing)
- Add `bottom_bin_ids() -> set[str]` so the pipeline can set the hold's eligible
  bins without re-deriving grid geometry.

### `pipeline.py` (existing — wires it in)
- `_create_engines`: build `OcclusionHold` from config; read `occlusion_ratio`.
- Where geofences are applied: `hold.set_eligible_bins(engine.bottom_bin_ids())`.
- `_main_loop`, after `assign(...)`: compute the occluded set from
  `presence_fn` at each event's `hand_point`, then
  `events = hold.apply(events, hands, occluded)` before updating state/overlay.
- When no foreground model is active, the occluded set is empty and the hold
  degrades to absence-only latching.

## Configuration (`settings.yaml`, under `bin_assignment`)

```yaml
occlusion_hold:
  enabled: true
  occlusion_ratio: 0.05   # fingertip foreground floor; below = occluded (latch/release)
```

## Testing

- **`OcclusionHold`** (pure, inject `occluded` set):
  - Latch on occlusion: visible-in-bin → then occluded phantom (or suppressed
    `None`) → bottom bin stays lit via `occlusion_hold`.
  - Phantom does not release: repeated occluded frames keep the hold.
  - Real re-emergence releases: a present-and-not-occluded hand ends the hold.
  - No latch without a `last_trusted` eligible bin.
  - Held-and-occluded phantom event is dropped (no double-lit bins on drift).
  - All existing `test_occlusion_hold.py` cases still pass with `occluded` empty.
- **Engine:** `bottom_bin_ids()` returns the bottom-row ids.

## Out of scope

- Count/inventory effects, timeouts, multi-row racks beyond top/bottom.
