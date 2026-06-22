# Finger-Vote Bin Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `finger_vote` bin-assignment method that attributes a pick using the index + middle fingertips (with a deepest-penetration tiebreak when they split across two bins), so a pick made with a finger other than the index is still counted to the correct bin.

**Architecture:** Add one new method branch to the existing stateless `BinAssignmentEngine.assign()`. Voting runs first and selects a candidate bin; the existing occlusion gate runs *after*, unchanged, and may still correct a top↔bottom lip-extrapolation. Grab detection is untouched. The feature is opt-in via `bin_assignment.method: "finger_vote"`; the current `area_overlap` default is left in place.

**Tech Stack:** Python 3.8+, pure geometry (no new deps), `pytest` for tests. MediaPipe Hands landmark names (`index_tip` = landmark 8, `middle_tip` = landmark 12) via the `HandDetection`/`HandLandmark` dataclasses in `hand_models/common/base_hand_tracker.py`.

## Global Constraints

- All work lives in `aegis-v2/`. Paths below are relative to that root.
- Tests are pure geometry — no camera, no model, no `cv2`. Mirror `integration/tests/test_occlusion_gate.py` exactly (sys.path insert of `aegis-v2` root, synthetic `HandDetection`s).
- Run tests with: `python -m pytest integration/tests/test_finger_vote.py -v` from the `aegis-v2` directory.
- Coordinate convention: y increases downward, row 0 = top (same as the occlusion-gate tests).
- The default `bin_assignment.method` in `settings.yaml` stays `area_overlap`. Do NOT change the live default.
- New `BinEvent`s from this path set `method="finger_vote"`.
- Vote digit set is configurable (`vote_keypoints`), default `["index_tip", "middle_tip"]`. Do NOT add ring/pinky.

---

## File Structure

- **Modify:** `integration/src/engine/bin_assignment.py` — add `finger_vote` config fields, the `assign()` branch, and the voting helpers (`_usable_vote_tips`, `_bin_containing`, `_interiority`, `_vote_event`, `_assign_vote`). New code goes in its own `# ── Finger-vote assignment ──` section directly after `_assign_overlap` and before the `# ── Occlusion gate ──` section.
- **Create:** `integration/tests/test_finger_vote.py` — new test module, fixtures mirroring `test_occlusion_gate.py`.
- **Modify:** `integration/config/settings.yaml` — documented, commented `vote_keypoints` / `vote_confidence_floor` under `bin_assignment` (default method unchanged).

---

## Task 1: Core voting (agreement, one-in-one-out, single-tip, center fallback)

**Files:**
- Modify: `integration/src/engine/bin_assignment.py` (`__init__` ~line 70; `assign()` branch ~lines 127-132; new helpers after `_assign_overlap` ~line 211)
- Test: `integration/tests/test_finger_vote.py` (create)

**Interfaces:**
- Consumes: `BinAssignmentEngine(config: dict)`, `set_bin_map_from_geofences(dict)`, `assign(hands, frame_shape=None) -> list[BinEvent]`; `HandDetection`/`HandLandmark` from `hand_models.common`.
- Produces:
  - `self._vote_keypoints: list[str]`
  - `_usable_vote_tips(self, hand, frame_shape) -> list[tuple[float, float]]`
  - `_bin_containing(self, point: tuple[float, float]) -> Optional[BinRegion]`
  - `_vote_event(self, hand, b: Optional[BinRegion], point: tuple[float, float], hand_area: float) -> BinEvent`
  - `_assign_vote(self, hand, hand_area: float, frame_shape) -> BinEvent`

- [ ] **Step 1: Write the failing tests**

Create `integration/tests/test_finger_vote.py`:

```python
"""Unit tests for the finger_vote method in BinAssignmentEngine.

Pure geometry — no camera, no model, no cv2. Mirrors test_occlusion_gate.

Grid used throughout (y increases downward; row 0 = top):

      x:  0----100----200
  y 0  +--------+--------+
       | bin_0_0| bin_0_1|   top row  (single cells)
  100  +--------+--------+
       | bin_1_0| bin_1_1|   bottom row
  200  +--------+--------+
"""
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from hand_models.common import HandDetection, HandLandmark  # noqa: E402
from integration.src.engine.bin_assignment import BinAssignmentEngine  # noqa: E402

FRAME = (200, 200, 3)  # (h, w, channels)


# ── Fixtures ────────────────────────────────────────────────

def two_row_geofences():
    return {
        "bin_0_0": {"x_min": 0,   "x_max": 100, "y_min": 0,   "y_max": 100, "confidence": 0.9},
        "bin_0_1": {"x_min": 100, "x_max": 200, "y_min": 0,   "y_max": 100, "confidence": 0.9},
        "bin_1_0": {"x_min": 0,   "x_max": 100, "y_min": 100, "y_max": 200, "confidence": 0.8},
        "bin_1_1": {"x_min": 100, "x_max": 200, "y_min": 100, "y_max": 200, "confidence": 0.8},
    }


def make_engine(geofences=None, gate=False,
                vote_keypoints=("index_tip", "middle_tip"), floor=0.5):
    if geofences is None:
        geofences = two_row_geofences()
    eng = BinAssignmentEngine({
        "method": "finger_vote",
        "vote_keypoints": list(vote_keypoints),
        "vote_confidence_floor": floor,
        "occlusion_gate": {"enabled": gate},
    })
    eng.set_bin_map_from_geofences(geofences)
    return eng


def make_hand(index=None, middle=None, index_conf=1.0, middle_conf=1.0,
              bbox=None, mcps=None, wrist=None, hand_id=0, handedness="right"):
    """Build a synthetic hand. index/middle are (x, y) of the two vote tips."""
    lms = []
    if index is not None:
        lms.append(HandLandmark(name="index_tip", x=index[0], y=index[1], confidence=index_conf))
    if middle is not None:
        lms.append(HandLandmark(name="middle_tip", x=middle[0], y=middle[1], confidence=middle_conf))
    if mcps is not None:
        for n in ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"):
            lms.append(HandLandmark(name=n, x=mcps[0], y=mcps[1]))
    if wrist is not None:
        lms.append(HandLandmark(name="wrist", x=wrist[0], y=wrist[1]))
    return HandDetection(hand_id=hand_id, handedness=handedness,
                         landmarks=lms, bounding_box=bbox)


# ── Task 1: core voting ─────────────────────────────────────

def test_both_tips_same_bin():
    """Both vote tips in the same bin → that bin, via finger_vote."""
    eng = make_engine()
    hand = make_hand(index=(40, 40), middle=(60, 60))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "finger_vote"


def test_index_outside_middle_inside():
    """The core pain case: index outside every bin, middle inside → middle's bin."""
    eng = make_engine()
    hand = make_hand(index=(250, 250), middle=(50, 50))  # index off the grid
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "finger_vote"


def test_single_tip_only_index():
    """Only the index tip is present → single-point behavior on it."""
    eng = make_engine()
    hand = make_hand(index=(150, 50))  # middle absent
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_1"
    assert ev.method == "finger_vote"


def test_both_tips_nan_center_fallback():
    """Both tips non-finite → fall back to the hand center (from bbox)."""
    eng = make_engine()
    hand = make_hand(index=(math.nan, math.nan), middle=(math.nan, math.nan),
                     bbox=(20, 20, 80, 80))  # center (50, 50) → bin_0_0
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "finger_vote"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest integration/tests/test_finger_vote.py -v`
Expected: All 4 FAIL. With no `finger_vote` branch, `assign()` falls through to `_assign_pip` (method `"point_in_polygon"`), so the `method == "finger_vote"` assertions fail (and `test_index_outside_middle_inside` / `test_both_tips_nan_center_fallback` resolve to `bin_id None`).

- [ ] **Step 3: Add the config field**

In `bin_assignment.py` `__init__`, immediately after the `self._keypoint = ...` line (~line 70), add:

```python
        self._vote_keypoints: list[str] = config.get(
            "vote_keypoints", ["index_tip", "middle_tip"]
        )
```

- [ ] **Step 4: Add the assign() branch**

In `assign()`, replace this block (~lines 127-132):

```python
            if self._method == "area_overlap":
                event = self._assign_overlap(hand, point, hand_area)
            elif self._method == "nearest_centroid":
                event = self._assign_nearest(hand, point, hand_area)
            else:
                event = self._assign_pip(hand, point, hand_area)
```

with:

```python
            if self._method == "area_overlap":
                event = self._assign_overlap(hand, point, hand_area)
            elif self._method == "nearest_centroid":
                event = self._assign_nearest(hand, point, hand_area)
            elif self._method == "finger_vote":
                event = self._assign_vote(hand, hand_area, frame_shape)
            else:
                event = self._assign_pip(hand, point, hand_area)
```

- [ ] **Step 5: Add the voting helpers**

In `bin_assignment.py`, directly after the `_assign_overlap` method (before the `# ── Occlusion gate ──` comment, ~line 213), insert:

```python
    # ── Finger-vote assignment ───────────────────────────────

    def _usable_vote_tips(self, hand, frame_shape):
        """Configured vote fingertips as (x, y), dropping missing/non-finite ones.

        ``frame_shape`` is threaded through for the off-frame filter added later;
        it is unused here.
        """
        tips = []
        for name in self._vote_keypoints:
            lm = hand.get_landmark(name)
            if lm is None:
                continue
            if not (math.isfinite(lm.x) and math.isfinite(lm.y)):
                continue
            tips.append((lm.x, lm.y))
        return tips

    def _bin_containing(self, point):
        """First bin whose axis-aligned bounds contain ``point``, or None."""
        px, py = point
        for b in self._bins:
            if b.x_min <= px <= b.x_max and b.y_min <= py <= b.y_max:
                return b
        return None

    def _vote_event(self, hand, b, point, hand_area):
        """Build a finger_vote BinEvent for bin ``b`` (None → no-match event)."""
        return BinEvent(
            hand_id=hand.hand_id, handedness=hand.handedness,
            bin_id=(b.bin_id if b else None), bin_label=(b.label if b else None),
            hand_point=point, hand_area=hand_area,
            confidence=(b.confidence if b else 0.0), method="finger_vote",
        )

    def _assign_vote(self, hand, hand_area, frame_shape):
        """Index+middle fingertip voting (tiebreak/centroid added in later tasks).

        Each usable tip inside a bin casts a vote. A single inside tip → its bin.
        No tip inside any bin → no match (centroid fallback added in Task 3).
        No usable tips → hand center.
        """
        tips = self._usable_vote_tips(hand, frame_shape)
        if not tips:
            point = hand.center
            if point is None:
                return self._vote_event(hand, None, (0.0, 0.0), hand_area)
            return self._vote_event(hand, self._bin_containing(point), point, hand_area)

        inside = []
        for t in tips:
            b = self._bin_containing(t)
            if b is not None:
                inside.append((t, b))

        if not inside:
            return self._vote_event(hand, None, tips[0], hand_area)

        tip, b = inside[0]
        return self._vote_event(hand, b, tip, hand_area)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest integration/tests/test_finger_vote.py -v`
Expected: All 4 PASS.

- [ ] **Step 7: Commit**

```bash
git add integration/src/engine/bin_assignment.py integration/tests/test_finger_vote.py
git commit -m "feat(v2): finger_vote bin assignment — core index+middle voting

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Deepest-penetration tiebreak for split votes

**Files:**
- Modify: `integration/src/engine/bin_assignment.py` (`_assign_vote` final pick; add `_interiority`)
- Test: `integration/tests/test_finger_vote.py` (append)

**Interfaces:**
- Consumes: `_assign_vote`, `BinRegion` (`.x_min/.x_max/.y_min/.y_max/.bin_id`) from Task 1.
- Produces: `_interiority(point, b) -> float` (staticmethod).

- [ ] **Step 1: Write the failing tests**

Append to `integration/tests/test_finger_vote.py`:

```python
# ── Task 2: split-vote tiebreak ─────────────────────────────

def test_split_deeper_tip_wins():
    """Tips in different top bins → the deeper-planted tip's bin wins.

    index at (95,50) → bin_0_0, interiority 5 (clips the right edge).
    middle at (140,50) → bin_0_1, interiority 40 (planted deeper).
    """
    eng = make_engine()
    hand = make_hand(index=(95, 50), middle=(140, 50))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_1"
    assert ev.method == "finger_vote"


def test_split_equal_interiority_lower_bin():
    """Equal interiority → deterministic: lower bin id wins.

    index at (105,50) → bin_0_1, interiority 5.
    middle at (95,50) → bin_0_0, interiority 5.
    """
    eng = make_engine()
    hand = make_hand(index=(105, 50), middle=(95, 50))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest integration/tests/test_finger_vote.py -k split -v`
Expected: Both FAIL. The Task-1 impl returns `inside[0]` (index tip first), so it yields `bin_0_0` for the first test (expects `bin_0_1`) and `bin_0_1` for the second (expects `bin_0_0`).

- [ ] **Step 3: Add the tiebreak**

In `bin_assignment.py`, replace the final two lines of `_assign_vote`:

```python
        tip, b = inside[0]
        return self._vote_event(hand, b, tip, hand_area)
```

with:

```python
        inside.sort(key=lambda tb: (-self._interiority(tb[0], tb[1]), tb[1].bin_id))
        tip, b = inside[0]
        return self._vote_event(hand, b, tip, hand_area)
```

Then add the `_interiority` staticmethod directly after `_bin_containing`:

```python
    @staticmethod
    def _interiority(point, b):
        """Distance from ``point`` to the nearest edge of bin ``b`` (deeper = larger)."""
        px, py = point
        return min(px - b.x_min, b.x_max - px, py - b.y_min, b.y_max - py)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest integration/tests/test_finger_vote.py -v`
Expected: All 6 PASS (Task 1's 4 still pass — agreement and single-tip are unaffected by the sort).

- [ ] **Step 5: Commit**

```bash
git add integration/src/engine/bin_assignment.py integration/tests/test_finger_vote.py
git commit -m "feat(v2): finger_vote — deepest-penetration tiebreak for split votes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Centroid fallback when no tip is inside a bin

**Files:**
- Modify: `integration/src/engine/bin_assignment.py` (`_assign_vote` no-inside branch)
- Test: `integration/tests/test_finger_vote.py` (append)

**Interfaces:**
- Consumes: `_assign_vote`, `_bin_containing`, `_vote_event` from Tasks 1-2.
- Produces: no new symbols (behavior change only).

- [ ] **Step 1: Write the failing tests**

Append to `integration/tests/test_finger_vote.py`:

```python
# ── Task 3: centroid fallback ───────────────────────────────

def test_no_inside_centroid_maps_to_bin():
    """Neither tip inside any bin, but their centroid is → centroid's bin.

    index (-20,50) and middle (220,50) are both off the grid horizontally;
    centroid (100,50) lands on bin_0_0's right edge (inclusive).
    """
    eng = make_engine()
    hand = make_hand(index=(-20, 50), middle=(220, 50))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "finger_vote"


def test_no_inside_centroid_outside_returns_none():
    """Neither tip nor their centroid is inside any bin → no match."""
    eng = make_engine()
    hand = make_hand(index=(-20, -20), middle=(-40, -40))  # centroid (-30,-30)
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id is None
    assert ev.method == "finger_vote"
```

- [ ] **Step 2: Run the new tests to verify the first fails**

Run: `python -m pytest integration/tests/test_finger_vote.py -k centroid -v`
Expected: `test_no_inside_centroid_maps_to_bin` FAILS (Task-2 impl returns `bin_id None` for the no-inside case). `test_no_inside_centroid_outside_returns_none` already PASSES (it guards the centroid-outside branch).

- [ ] **Step 3: Add the centroid fallback**

In `bin_assignment.py`, replace this block in `_assign_vote`:

```python
        if not inside:
            return self._vote_event(hand, None, tips[0], hand_area)
```

with:

```python
        if not inside:
            cx = sum(t[0] for t in tips) / len(tips)
            cy = sum(t[1] for t in tips) / len(tips)
            centroid = (cx, cy)
            return self._vote_event(hand, self._bin_containing(centroid), centroid, hand_area)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest integration/tests/test_finger_vote.py -v`
Expected: All 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add integration/src/engine/bin_assignment.py integration/tests/test_finger_vote.py
git commit -m "feat(v2): finger_vote — centroid fallback when no tip is inside a bin

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Confidence-floor and off-frame filtering

**Files:**
- Modify: `integration/src/engine/bin_assignment.py` (`__init__` config; `_usable_vote_tips` body)
- Test: `integration/tests/test_finger_vote.py` (append)

**Interfaces:**
- Consumes: `_usable_vote_tips(self, hand, frame_shape)`, `self._vote_keypoints` from Task 1.
- Produces: `self._vote_confidence_floor: float`; `_usable_vote_tips` now also drops low-confidence and (when `frame_shape` given) off-frame tips.

- [ ] **Step 1: Write the failing tests**

Append to `integration/tests/test_finger_vote.py`:

```python
# ── Task 4: confidence floor + off-frame filtering ──────────

def test_low_confidence_tip_dropped():
    """A below-floor tip is ignored, so the confident tip decides.

    index (150,50)→bin_0_1, interiority 50 but confidence 0.2 (below 0.5 floor).
    middle (95,50)→bin_0_0, interiority 5, confidence 1.0.
    Without the floor the deeper index would win bin_0_1; with it, bin_0_0.
    """
    eng = make_engine(floor=0.5)
    hand = make_hand(index=(150, 50), index_conf=0.2, middle=(95, 50), middle_conf=1.0)
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"


def test_offframe_tip_dropped():
    """An off-frame tip is excluded from the centroid.

    Single-bin grid so there is in-frame space outside the bin.
    index (-120,50) is off-frame (x<0); middle (130,50) is in-frame, outside the bin.
    With index kept, centroid (5,50) would fall in bin_0_0; dropping it leaves
    only (130,50), whose centroid is outside → no match.
    """
    one_bin = {"bin_0_0": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100, "confidence": 0.9}}
    eng = make_engine(geofences=one_bin)
    hand = make_hand(index=(-120, 50), middle=(130, 50))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id is None
    assert ev.method == "finger_vote"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest integration/tests/test_finger_vote.py -k "confidence or offframe" -v`
Expected: Both FAIL. The Task-3 `_usable_vote_tips` keeps both tips: the low-confidence index wins on depth (`bin_0_1`, expected `bin_0_0`), and the off-frame index pulls the centroid into `bin_0_0` (expected `None`).

- [ ] **Step 3: Add the config field**

In `bin_assignment.py` `__init__`, immediately after the `self._vote_keypoints = ...` block, add:

```python
        self._vote_confidence_floor: float = config.get("vote_confidence_floor", 0.5)
```

- [ ] **Step 4: Extend the tip filter**

Replace the entire `_usable_vote_tips` method with:

```python
    def _usable_vote_tips(self, hand, frame_shape):
        """Configured vote fingertips that are finite, confident, and in-frame.

        Drops a tip when its landmark is missing, non-finite, below
        ``vote_confidence_floor``, or (when ``frame_shape`` is given) outside the
        frame. Returns a list of (x, y).
        """
        h = w = None
        if frame_shape is not None:
            h, w = frame_shape[0], frame_shape[1]
        tips = []
        for name in self._vote_keypoints:
            lm = hand.get_landmark(name)
            if lm is None:
                continue
            if not (math.isfinite(lm.x) and math.isfinite(lm.y)):
                continue
            if lm.confidence < self._vote_confidence_floor:
                continue
            if w is not None and not (0 <= lm.x <= w and 0 <= lm.y <= h):
                continue
            tips.append((lm.x, lm.y))
        return tips
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest integration/tests/test_finger_vote.py -v`
Expected: All 10 PASS. (Task 1's `test_both_tips_nan_center_fallback` still passes — NaN tips remain dropped.)

- [ ] **Step 6: Commit**

```bash
git add integration/src/engine/bin_assignment.py integration/tests/test_finger_vote.py
git commit -m "feat(v2): finger_vote — drop low-confidence and off-frame vote tips

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Occlusion-gate composition + config documentation

**Files:**
- Modify: `integration/config/settings.yaml` (`bin_assignment` section)
- Test: `integration/tests/test_finger_vote.py` (append)

**Interfaces:**
- Consumes: `assign()` ordering (vote → occlusion gate) and the existing occlusion-gate code; no new symbols.
- Produces: none (documentation + regression guard).

- [ ] **Step 1: Write the composition guard test**

Append to `integration/tests/test_finger_vote.py`:

```python
# ── Task 5: composition with the occlusion gate ─────────────

def test_vote_then_occlusion_gate_reassigns():
    """Voting picks a top bin; the occlusion gate (run after) reassigns it to the
    bottom bin beneath when the proximal anchor is below the rim.

    Both tips → bin_0_0 (top); MCP centroid at (50,150) is below the bottom rim
    (y=100) and over bin_1_0 → gate reassigns to bin_1_0.
    """
    eng = make_engine(gate=True)
    hand = make_hand(index=(50, 50), middle=(50, 55), mcps=(50, 150))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_1_0"
    assert ev.method == "occlusion_gate"


def test_gate_disabled_keeps_vote_result():
    """With the gate off, the same hand keeps the voted top bin."""
    eng = make_engine(gate=False)
    hand = make_hand(index=(50, 50), middle=(50, 55), mcps=(50, 150))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "finger_vote"
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `python -m pytest integration/tests/test_finger_vote.py -k gate -v`
Expected: Both PASS. These guard the documented ordering (voting first, gate second) wired in Task 1 — they confirm the gate still fires after `finger_vote` and is inert when disabled.

- [ ] **Step 3: Document the config keys**

In `integration/config/settings.yaml`, in the `bin_assignment` block, immediately after the `overlap_threshold:` lines (and before the `occlusion_gate:` comment block, ~line 47), insert:

```yaml

  # Finger-vote method (set method: "finger_vote" above to use it). The index
  # and middle fingertips each vote for the bin they fall in; agreement (or a
  # single inside tip) wins, a split is broken by deepest penetration (the tip
  # planted furthest from a bin edge), and if neither tip is inside a bin their
  # centroid is used. Tips below the confidence floor or off-frame are ignored.
  # Index + middle is the human tripod-pick set; do not add ring/pinky (they
  # flail in precision picks and add false cross-bin votes).
  vote_keypoints: [index_tip, middle_tip]
  vote_confidence_floor: 0.5
```

- [ ] **Step 4: Run the full suite to verify nothing regressed**

Run: `python -m pytest integration/tests/test_finger_vote.py integration/tests/test_occlusion_gate.py -v`
Expected: All finger_vote tests (12) and all occlusion_gate tests PASS.

- [ ] **Step 5: Commit**

```bash
git add integration/src/engine/bin_assignment.py integration/tests/test_finger_vote.py integration/config/settings.yaml
git commit -m "feat(v2): finger_vote — gate composition test + documented config keys

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Method `finger_vote` added alongside existing methods → Task 1.
- Vote with index + middle tips → Task 1 (`_vote_keypoints` default).
- Both agree / one-in-one-out / single-tip degrade → Task 1.
- Split → deepest-penetration tiebreak; equal → lower bin id → Task 2.
- Neither inside → centroid fallback; centroid outside → None → Task 3.
- Filter NaN / off-frame / low-confidence → Task 1 (NaN) + Task 4 (confidence, off-frame).
- No usable tips → hand-center fallback → Task 1.
- Occlusion gate runs after voting, unchanged → Task 1 wiring, verified Task 5.
- Config `method` / `vote_keypoints` / `vote_confidence_floor`; default method unchanged → Task 1 + Task 4 + Task 5 docs.
- Tests mirroring `test_occlusion_gate.py` for all listed cases → Tasks 1-5.

**Placeholder scan:** None — every step has concrete code, commands, and expected output.

**Type consistency:** `_usable_vote_tips(hand, frame_shape)`, `_bin_containing(point)`, `_interiority(point, b)`, `_vote_event(hand, b, point, hand_area)`, `_assign_vote(hand, hand_area, frame_shape)` are used with identical signatures everywhere they appear. `_assign_vote` is called as `self._assign_vote(hand, hand_area, frame_shape)` in `assign()` (Task 1) and never re-signed. `BinEvent` fields match the existing dataclass.
