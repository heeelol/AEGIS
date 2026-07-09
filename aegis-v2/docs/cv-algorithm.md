# AEGIS v2 — Computer Vision Algorithm

A reference for the full CV pipeline: how the system decides **which bin a hand is
reaching into**, frame by frame, and how it stays correct when the operator's
fingers are hidden under the shelf.

---

## Guiding principle

> **Trust real pixels, not the hand model's guesses.**

MediaPipe always returns all 21 hand landmarks once a hand is detected — including
ones it can't actually see, which it **extrapolates**. When fingers go under the
top shelf, those extrapolated fingertips land in the wrong (upper) bin. So:

- **Landmarks** decide *which* bin a fingertip is in (geometry).
- **Foreground** (background subtraction on a fixed camera) decides whether a hand
  is *really there* — at the fingertip (gate), over the bin region (hold), and
  above the rim (emergence release).

Everything below is built on that split.

---

## Architecture at a glance

```
          ┌─────────────── Initialization (once) ───────────────┐
 Camera → OBB bin detector → grid calibrate/match → geofences ──┐
                                                                 │
          ┌──────────────── Per-frame loop ────────────────┐    │
 frame ─► MediaPipe hands ─► Foreground (MOG2) ─► assign ───┼────┘
                                                  │  bins
                                                  ▼
                                          Occlusion gate
                                                  ▼
                                          Occlusion hold
                                                  ▼
                                     state → dashboard + overlay
```

Orchestrated by `integration/src/pipeline.py` (`Pipeline.run`), which runs two
stages: **initialization** and the **Sense → Analyse → Act loop**.

---

## Stage 1 — Initialization

### 1. Camera (`_open_camera`)
Opens the camera (DirectShow on Windows), forces **MJPG @ 1280×720 / 30 fps**, and
sets buffer size 1 so each read is the freshest frame (no lag backlog). Frames may
be rotated 180° (`camera.rotate_180`).

### 2. Bin detection — *where* the bins are
Two paths, chosen by `bin_detector.manual_layout`:

- **Manual layout** (`[6, 3]`): skip CV; lay an even grid across the frame. The
  headless / dependency-light fallback.
- **OBB two-snapshot flow** (default, `manual_layout: null`): a trained
  **YOLOv8-OBB** (oriented bounding box) model finds the physical bins.
  Operator-driven via two key presses:

  **Press `1` — calibrate** (`_calibrate_grid` → `grid_session.calibrate`):
  1. `initialize_bins_obb.detect_bins` runs YOLO-OBB → raw rotated boxes
     `{corners(4×2), center, area, conf}`. It rejects phantoms: drops boxes far
     smaller than the median, suppresses overlaps by IoU, keeps the strongest
     `expected_count = 9`.
  2. `grid_calibrator.calibrate_grid` splits those 9 into rows by the largest
     vertical gap (`grid_allocator.split_rows_by_y`), validates exactly **6 top +
     3 bottom**, orders each row left→right, and locks them as fixed slots
     `slot_1..9` (each with center, corners, median spacing).

  **Press `2` — init kit** (`_init_kit` → `grid_session.init_kit`):
  - Another OBB snapshot → `grid_calibrator.match_to_grid` greedily matches each
    detection to the nearest *same-row* calibrated slot (within ½ the row
    spacing), one-to-one. Matched slots are `present=True`; unmatched stay
    `present=False` (greyed "bin missing").

### 3. Geofences — the canonical bin map
`grid_session.to_geofences` produces the dict everything downstream consumes,
keyed `bin_{row}_{col}` (top = `bin_0_0..bin_0_5`, bottom = `bin_1_0..bin_1_2`),
each `{x_min, x_max, y_min, y_max, polygon, confidence, detected, span, …}`.
The **assignment engine receives present-only bins** so a hand can't be assigned to
an empty slot; the dashboard/overlay get all of them.

### 4. Models & engines (`_create_engines`)
- **Hand tracker** — MediaPipe Hand Landmarker.
- **`BinAssignmentEngine`** — pure geometry; receives the bin map.
- **`ForegroundModel`** — MOG2 background subtractor (built only when
  `occlusion_gate.foreground` is configured).
- **`OcclusionHold`** — bin-keyed continuity layer; eligible bins set to
  `engine.bottom_bin_ids()`.

---

## Stage 2 — Per-frame loop (`_main_loop`)

### A. Hand detection
`hand_tracker.detect(frame)` → list of `HandDetection`, each with 21 landmarks,
a handedness label, a bounding box, and a debounced grab state. **Key caveat:**
MediaPipe extrapolates occluded landmarks, and its left/right label is unreliable
on occluded hands — the whole design avoids depending on either.

### B. Foreground update
`foreground.update(frame)` → a binary mask (`1` = not the static background = a
real hand/arm). Once warmed up (`warmup_frames`), the pipeline builds
`presence_fn(px, py)` = the foreground fraction in a small patch around a point.

### C. Bin assignment (`engine.assign`, method `finger_vote`)
For each hand, the **index + middle fingertips** each vote for the bin they fall
inside. Agreement (or a single inside tip) wins; a split is broken by deepest
penetration; if neither tip is inside a bin, their centroid is used. Output: one
`BinEvent` per hand `{bin_id, hand_point, method, …}`.

### D. Occlusion **gate** (`_apply_occlusion_gate`)
Corrects the extrapolation bug. **Only** for a hand assigned to a *top* bin, it
checks real foreground at the claimed fingertip via `presence_fn`:
- ratio **≥ `present_ratio`** → a real hand is up there → keep the top assignment.
- ratio **< `present_ratio`** → fingertip extrapolated through the shelf →
  **reassign to the bottom bin in that column**, or suppress if none beneath.

Must measure at the *fingertip* (not the whole hand) — that's what distinguishes a
genuine top reach from an occluded one. Falls back to a landmark heuristic if no
foreground model is ready.

### E. Occlusion **hold** (`hold.apply`, bin-keyed)
Gives visual continuity while fingers are hidden under the shelf. The pipeline
derives three signals from the mask each frame:
- `occluded_ids` — `hand_id`s whose fingertip foreground is **< `occlusion_ratio`**
  (genuinely under the shelf).
- `occupied_bins` — bottom bins whose **region** foreground is **≥
  `occupancy_ratio`** (a forearm is in them)…
- …**minus** `bottom_bins_with_hand_above(events, occluded_ids)` — bins whose
  column has a *visible* fingertip above the rim (the hand emerged above the rack /
  is in a top bin; the forearm is only transiting).

Then, **handedness-independently**:
1. **Arm** an eligible bottom bin when a genuine (non-occluded) in-bin pick lands.
2. **Release** an armed bin no longer in `occupied_bins`.
3. **Drop** occluded-tip events (unreliable extrapolations — kills the stray 0.0
   markers).
4. **Emit** a synthetic `occlusion_hold` event for each armed+occupied bin no live
   event still covers.

Why bin-keyed: MediaPipe's left/right label flips on occluded hands, so keying by
handedness produced stuck phantom holds.

### F. Act
The final event list updates `PipelineState` (web dashboard) and the OpenCV
overlay. Pressing **`m`** swaps to the foreground tuning view
(`overlay.render_foreground_debug`): the mask plus per-fingertip ratios, for
tuning the thresholds by eye (`occlusion_hold` events excluded so no 0.0 boxes).

---

## How the gate and hold work together

| Situation | Fingertip foreground | Bottom-bin region | Result |
|-----------|---------------------|-------------------|--------|
| Visible pick in a bin | high | — | live event lights the bin |
| Reaching **under** the shelf (bottom bin) | ~0 (occluded) | occupied (forearm) | gate reassigns down; hold keeps the bottom bin lit |
| Genuinely in a **top** bin | high | occupied by transiting arm | gate keeps top; hold sees a *visible tip above* → releases the bottom bin |
| **Above the rack**, not in a bin yet | high | occupied by transiting arm | visible tip above → bottom bin released (no lingering hold) |
| Hand removed | — | background | hold releases (bin no longer occupied) |

---

## Thresholds

| Knob (`bin_assignment.*`) | Default | Question it answers |
|---------------------------|---------|---------------------|
| `occlusion_gate.foreground.present_ratio` | 0.20 | Is a *top-bin* fingertip really there? (generous — catch phantoms) |
| `occlusion_hold.occlusion_ratio` | 0.05 | Is this fingertip occluded under the shelf? (strict) |
| `occlusion_hold.occupancy_ratio` | 0.05 | Is a forearm still in this bottom bin? (strict) |
| `bin_assignment.vote_confidence_floor` | 0.50 | Is a voting fingertip reliable enough to count? |

Tune them with the `m` overlay: watch the printed ratios when occluded vs.
weakly-visible, and set each threshold in the gap between the two clusters.

---

## Key data shapes

- **OBB detection** — `{id, corners:(4,2), center:(cx,cy), area, conf}`
- **Calibrated slot** — `slot_{1..9}: {index, row, layer, span, center, corners, row_spacing}`
- **Geofence** — `bin_{row}_{col}: {x_min, x_max, y_min, y_max, polygon, confidence, detected, span, slot_start}`
- **HandDetection** — `{hand_id, handedness, landmarks[21], bounding_box, is_grabbing, grab_score}`
- **BinEvent** — `{hand_id, handedness, bin_id, bin_label, hand_point, hand_area, confidence, method}`
  - `method` ∈ `finger_vote | point_in_polygon | occlusion_gate | occlusion_hold`

---

## File map

| Concern | File |
|---------|------|
| Orchestration / the loop | `integration/src/pipeline.py` |
| OBB bin detection | `integration/src/detectors/initialize_bins_obb.py` |
| Grid calibration + matching | `grid_calibrator.py`, `grid_allocator.py`, `grid_session.py` |
| Hand tracking | `hand_models/mediapipe/tracker.py` |
| Bin assignment + occlusion gate | `integration/src/engine/bin_assignment.py` |
| Occlusion hold | `integration/src/engine/occlusion_hold.py` |
| Foreground model | `integration/src/detectors/foreground.py` |
| Overlay / tuning view | `integration/src/ui/overlay.py` |
| Config (all knobs) | `integration/config/settings.yaml` |
| Design specs | `docs/superpowers/specs/2026-06-*.md` |

---

## Known limitations

- **Hand-off flicker:** a 1–2 frame blip of a bottom bin is possible while the
  fingertip is over the shelf lip (left the bottom box, not yet registered above).
  Transient, not a lock.
- **Two hands stacked in one column** (one in the top bin, one occluded directly
  below): the emergence guard would suppress the lower hold. Rare and visually
  ambiguous.
- **A hand held perfectly still for many seconds** can fade into the MOG2
  background model (foreground decays). Uncommon during active picking.
- Foreground signals assume a **fixed camera + stable scene** — the rig's intended
  configuration.

---

## Design history

The occlusion logic was built iteratively; the rationale for each piece lives in
`docs/superpowers/specs/`:
- `2026-06-15-occlusion-gate-design.md` — original landmark-anchor gate.
- `2026-06-22-foreground-occlusion-gate-design.md` — foreground-evidence gate.
- `2026-06-22-present-ratio-occlusion-hold-design.md` — the hold, its bin-keyed
  rewrite, occupancy release, and the emergence guard.
