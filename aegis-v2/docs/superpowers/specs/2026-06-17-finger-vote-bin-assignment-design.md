# Finger-Vote Bin Assignment — Design

**Date:** 2026-06-17
**Status:** Approved (design); pending implementation plan
**Component:** `aegis-v2/integration/src/engine/bin_assignment.py`

---

## Problem

The bin-assignment criteria treats a single fingertip (`index_tip`) as the
"reaching point" that decides which bin a hand picked from. This is brittle:

- An operator frequently picks with the **middle / ring / pinky** while the
  **index finger is not inside the bin**, so the index-tip test misses the pick
  or attributes it to the wrong bin.
- The current live config actually runs `method: "area_overlap"` (whole-hand
  bbox overlap), so the index-tip intent is only partly realized today, and the
  bbox straddles tight bins easily.

## Goal

Make bin attribution robust to *which* finger does the picking, while resolving
the case where a single hand's fingers span **two adjacent bins** — without a
new tracker, model, or hardware. The decision must be deterministic and compose
with the existing occlusion gate and grab detection.

## Non-goals

- No upstream hand tracker / track-id layer (see `FUTURE_TASKS.md` §D).
- No load-cell weight fusion (deferred; `FUTURE_TASKS.md` §C).
- No change to grab detection or the occlusion gate's own logic.
- No polygon-distance geometry yet (axis-aligned only; noted as future).

---

## Background: why index + middle

Human picking from a bin is a **precision grasp** (Napier, 1956). Within that,
the **tripod / three-jaw chuck** (thumb + index + middle) dominates, and digit
involvement falls off sharply thumb ≈ index > middle > ring > pinky. Ring and
pinky contribute to *power* grasps for larger objects, not precision picks, and
tend to flail — adding false cross-bin votes for little signal.

Decision: vote with **index_tip (landmark 8) + middle_tip (landmark 12)**. Thumb
is excluded — MediaPipe's thumb landmark is the noisiest and first hidden under
the bin lip. Configurable so the set can change later.

References:
- Napier, *The prehensile movements of the human hand* (1956).
- *Finger control in the tripod grasp* — https://pubmed.ncbi.nlm.nih.gov/12632237/
- *A quantitative taxonomy of human hand grasps* — https://pmc.ncbi.nlm.nih.gov/articles/PMC6377750/

## Constraint that shapes the design

Rig geometry is **mixed / row-dependent**: tight top (1-cell) bins let a hand
straddle two adjacent bins; roomy bottom (2-cell) bins comfortably contain a
whole hand. So a tiebreak for split votes is required, but it primarily matters
for the top row.

---

## Design

### Method

Add a new assignment method `finger_vote` to `bin_assignment.py`, alongside the
existing `point_in_polygon` / `nearest_centroid` / `area_overlap`. Selected via
`bin_assignment.method: "finger_vote"`. The existing methods remain unchanged
and the default config is untouched — `finger_vote` is opt-in.

### Pipeline ordering (unchanged contracts)

1. `finger_vote` picks the candidate bin (this design).
2. The **occlusion gate** runs *after*, exactly as today, and may reassign the
   candidate top↔bottom when the proximal anchor (wrist / MCPs) shows the arm is
   reaching under the lip.
3. Grab detection is independent and untouched.

### Algorithm

For each detected hand, gather the configured vote tips (default index_tip,
middle_tip):

1. **Filter** — drop a tip if its coordinate is NaN, off-frame, or its
   confidence/visibility is below `vote_confidence_floor`.
2. **Vote** — for each surviving tip, find the bin whose axis-aligned bounds
   contain it, reusing the same containment test as `_assign_pip`
   (`x_min ≤ px ≤ x_max and y_min ≤ py ≤ y_max`). Each contained tip = one vote.
3. **Aggregate:**

   | Situation | Result |
   |---|---|
   | Both tips vote the same bin | that bin |
   | Tips vote different bins (split) | **deepest-penetration tiebreak** (below) |
   | One tip inside a bin, one outside all | the inside tip's bin |
   | Neither tip inside any bin | fuse: take the **centroid** of the surviving tips, run the single-point containment test; if still outside everything → `bin_id = None` |
   | Only one tip survives filtering | single-point behavior on that tip |
   | No tips survive filtering | fall back to existing behavior (hand center / `area_overlap` path), as today |

### Deepest-penetration tiebreak

When the surviving tips are inside *different* bins, compute each tip's
**interiority** = distance to the nearest edge of its containing bin:

```
interiority = min(px - x_min, x_max - px, py - y_min, y_max - py)
```

The tip with **greater interiority wins** — it is planted deeper inside the real
bin, while the trailing finger merely clips the neighbor's edge (the grasp-
research insight that the working digits go deepest, expressed as geometry).

**Determinism:** on *equal* interiority, the lower bin index wins, so the result
is reproducible.

### Config additions (`bin_assignment`)

```yaml
bin_assignment:
  method: "finger_vote"             # opt-in; default stays area_overlap
  vote_keypoints: [index_tip, middle_tip]   # adjustable digit set
  vote_confidence_floor: 0.5        # drop tips below this confidence/visibility
```

`hand_keypoint` remains for the single-point methods; `vote_keypoints` governs
`finger_vote`.

---

## Edge cases

- **Both tips NaN / off-frame / low-confidence** → fall back to the hand-center /
  `area_overlap` path exactly as the code does today. No regression when the hand
  is poorly tracked.
- **Equal interiority on a split** → lower bin index (deterministic).
- **Single surviving tip** → degrades to old single-point behavior; the change is
  a strict superset of current capability, never worse.
- **Occlusion gate interaction** → voting only selects among *visible* in-frame
  tips; the gate's separate top↔bottom correction still fires on the proximal
  anchor afterward, so a bottom-bin reach whose tips extrapolate into a top bin is
  still corrected.

## Testing

Unit tests mirroring `integration/tests/test_occlusion_gate.py`:

1. Both tips in same bin → that bin.
2. One tip inside, one outside → inside tip's bin (the core pain case).
3. Split across two bins → deeper tip's bin wins.
4. Split with equal interiority → lower bin index wins (determinism).
5. Neither tip inside → centroid fallback (inside → bin; outside → `None`).
6. Single surviving tip (other NaN/off-frame) → single-point behavior.
7. Both tips unusable → falls back to existing hand-center / area_overlap path.
8. Combined `finger_vote` → occlusion_gate ordering: voting picks a top bin, gate
   reassigns to the bottom bin beneath when the anchor is below the rim.

## Files touched

- `integration/src/engine/bin_assignment.py` — new `finger_vote` method, voting +
  tiebreak helpers, config wiring.
- `integration/config/settings.yaml` — documented `vote_keypoints`,
  `vote_confidence_floor` (commented; default method unchanged).
- `integration/tests/test_finger_vote.py` — new test module.
