# Foreground-Evidence Occlusion Gate

**Date:** 2026-06-22
**Status:** Approved (design), implementing
**Supersedes the decision signal of:** [2026-06-15-occlusion-gate-design.md](2026-06-15-occlusion-gate-design.md)

## Problem

MediaPipe Hand Landmarker always returns all 21 landmarks once a hand is
detected. When the operator reaches into a **bottom-layer** bin, the fingers
disappear under the top-shelf lip, but the model **extrapolates the occluded
landmarks** from the visible part of the hand. The inferred fingertip — and the
inferred knuckles and wrist — can all land inside a **top** bin's region, so the
hand is wrongly counted in the top bin (or "just outside").

The existing occlusion gate (2026-06-15) tries to catch this by checking whether
the MCP knuckle anchor sits below the bottom-bin rim. But the knuckles are
**also extrapolated**, so under a deep reach they read high (above the rim) and
the gate is fooled. **Any rule built on landmark coordinates is checking a guess
against a guess.** It cannot be robust.

## Insight

The camera is fixed and the shelf edge is stable; lighting is stable and the top
bins are mostly static. Under those conditions we have a signal that does **not**
depend on the (possibly fabricated) landmarks: **the actual camera pixels.**

- A genuine reach *over* the lip into a top bin puts a real hand into the top
  bin's image region → new foreground pixels appear there.
- A hand occluded *under* the shelf produces **no** new foreground above the
  shelf, no matter what the landmark model invents.

So the robust discriminator is: *when a hand is assigned to a top bin, is there
really hand-shaped foreground at the claimed fingertip?* If yes → keep it on the
top bin. If no → it's an extrapolation artifact → reassign to the bottom bin in
the same column (or suppress if there is no bottom bin beneath).

## Architecture

Three units, with the image processing kept out of the pure-geometry engine via
dependency injection of a "hand-presence oracle".

### 1. `ForegroundModel` (new — `integration/src/detectors/foreground.py`)
Owns cv2. Wraps an OpenCV MOG2 background subtractor (`detectShadows=True` so
shadows are not mistaken for a hand).
- `update(frame) -> mask`: per-frame binary foreground mask (1 = definite
  foreground, 0 otherwise; MOG2 shadow label 127 is treated as background).
- `patch_ratio(mask, px, py, size=None) -> float`: fraction of foreground in an
  N×N window around an image point.
- `ready` property: `True` only after `warmup_frames` updates, so a cold model
  never rejects a genuine reach.

### 2. `BinAssignmentEngine` (existing — stays cv2/numpy-pure for its logic)
- `assign(hands, frame_shape=None, presence_fn=None)` gains one optional
  parameter: `presence_fn(px, py) -> float`, returning foreground evidence at an
  image point.
- The occlusion gate keeps its existing **structure** (top rows, bottom-bin by
  column via `_bottom_bin_at`, reassign-or-suppress) but swaps the **decision
  signal**:
  - `presence_fn` provided → decide on foreground evidence at the claimed
    fingertip (`event.hand_point`).
  - `presence_fn is None` (gate disabled, model warming up, or unit tests) →
    **fall back to the existing landmark anchor gate**, so prior behavior and all
    existing tests are preserved.

### 3. `pipeline.py` (existing — wires it together)
- `_create_engines`: build a `ForegroundModel` from config when the gate's
  foreground mode is enabled.
- `_main_loop`: each frame, `mask = foreground.update(frame)`; once `ready`,
  build `presence_fn = lambda px, py: foreground.patch_ratio(mask, px, py)` and
  pass it to `assign(hands, frame.shape, presence_fn)`.

## Decision logic (top-bin event, claimed point `(px, py)`)

```
ratio = presence_fn(px, py)
if ratio >= present_ratio:        # real hand up there
    keep the top-bin event
else:                             # extrapolation artifact
    bottom = bottom_bin_at(px)
    if bottom: reassign to bottom
    else:      suppress (bin_id = None)
```

## Configuration (`settings.yaml`, under `bin_assignment.occlusion_gate`)

```yaml
occlusion_gate:
  enabled: true
  foreground:
    present_ratio: 0.10   # min foreground fraction in the fingertip patch to count as a real hand
    patch_size: 41        # px window around the claimed fingertip
    warmup_frames: 30     # frames before the model is trusted (falls back to landmark gate until then)
    history: 500          # MOG2 history
    var_threshold: 16     # MOG2 variance threshold
```

The engine only needs `present_ratio`; the rest configure `ForegroundModel`.

## Error handling / degradation

- Model not `ready` (warmup) or gate disabled → `presence_fn` not passed →
  landmark gate runs (safe, never worse than today).
- Claimed fingertip near the frame edge → patch is clipped; ratio still valid.
- A part held in-hand above the shelf is real foreground → correctly kept on top.
- A hand held perfectly still for many seconds could fade into the background
  model (rare during picking; mitigated by a moderate learning rate). Accepted.

## Testing

- **Engine** (`test_occlusion_gate.py`, pure, inject a fake `presence_fn`):
  - No evidence + knuckles read *high* (would fool the old gate) → reassigned to
    bottom. (Demonstrates the robustness gain.)
  - Strong evidence + knuckles read *low* (old gate would reassign) → kept on
    top.
  - No evidence + no bottom bin beneath → suppressed.
  - `presence_fn=None` → identical to the landmark gate (existing tests).
- **ForegroundModel** (`test_foreground.py`, synthetic frames):
  - `ready` flips only after `warmup_frames`.
  - After learning a static background, a new blob yields high `patch_ratio`
    inside the blob and ~0 outside.

## Out of scope

- Per-bin adaptive thresholds, multi-camera, moving camera.
- Replacing the landmark gate entirely — it stays as the warmup/fallback path.
