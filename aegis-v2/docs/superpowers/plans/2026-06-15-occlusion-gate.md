# Occlusion Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop MediaPipe's extrapolated fingertips from falsely lighting a *top* bin during a genuine *bottom* reach, by reassigning such hits to the bottom bin beneath using the hand's proximal anchor and the bottom-band cut-off rim.

**Architecture:** A stateless occlusion gate is added *inside* `BinAssignmentEngine.assign()` (where the full `HandDetection` with all 21 landmarks already lives). After each per-hand `BinEvent` is computed, if it lands in a top-row bin but the hand's most-proximal reliable landmark (wrist, else MCP-knuckle centroid) sits at/below the bottom bin's top rim, the event is reassigned to the bottom bin beneath (or suppressed when none exists). The grid structure (top rows, bottom bins, fallback line) is precomputed whenever the bin map changes. The existing `OcclusionHold` post-processor runs after, unchanged.

**Tech Stack:** Python, pytest. Files: `integration/src/engine/bin_assignment.py`, `integration/src/pipeline.py`, `integration/config/settings.yaml`, `integration/tests/test_occlusion_gate.py`.

**Spec:** `docs/superpowers/specs/2026-06-15-occlusion-gate-design.md`

**Test run command (from `aegis-v2/`):**
`python -m pytest integration/tests/test_occlusion_gate.py -v`

---

## File Structure

- `integration/src/engine/bin_assignment.py` — **modify**. Add `import math`; read `occlusion_gate.enabled` in `__init__`; add grid-structure fields + `_recompute_grid_structure()` called from both bin-map setters; add `_bin_row()`, `_occlusion_anchor()`, `_bottom_bin_at()`, `_apply_occlusion_gate()`; add `frame_shape` param to `assign()` and call the gate per hand.
- `integration/src/pipeline.py` — **modify** (1 line). Pass `frame.shape` to `assign()`.
- `integration/config/settings.yaml` — **modify**. Add `occlusion_gate` block under `bin_assignment`.
- `integration/tests/test_occlusion_gate.py` — **create**. Pure-geometry tests driving `BinAssignmentEngine` with synthetic hands.

---

## Task 1: Grid-structure precompute + gate config

Add the config flag and the precomputed grid structure the gate needs. No behavior change yet (gate not wired into `assign()`).

**Files:**
- Modify: `integration/src/engine/bin_assignment.py`
- Test: `integration/tests/test_occlusion_gate.py`

- [ ] **Step 1: Write the failing test**

Create `integration/tests/test_occlusion_gate.py`:

```python
"""Unit tests for the occlusion gate inside BinAssignmentEngine.

Pure geometry — no camera/model/cv2. HandDetection/HandLandmark are imported
straight from base_hand_tracker (numpy-only) so the detectors package __init__
(which pulls cv2 / ultralytics) is never touched.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

import pytest  # noqa: E402
from integration.src.engine.bin_assignment import (  # noqa: E402
    BinAssignmentEngine, BinRegion,
)
from hand_models.common.base_hand_tracker import (  # noqa: E402
    HandDetection, HandLandmark,
)


# Bin map: top row (row 0) of 4 single bins across x, bottom row (row 1) of
# 2 wide bins. Top bins occupy y 0..100; bottom bins y 200..400 (rim at y=200).
def make_bins():
    top = [
        BinRegion(bin_id=f"bin_0_{c}", label=f"bin_0_{c}",
                  x_min=c * 100, x_max=c * 100 + 100,
                  y_min=0, y_max=100, confidence=0.9)
        for c in range(4)
    ]
    bottom = [
        BinRegion(bin_id="bin_1_0", label="bin_1_0",
                  x_min=0, x_max=200, y_min=200, y_max=400, confidence=0.8),
        BinRegion(bin_id="bin_1_1", label="bin_1_1",
                  x_min=200, x_max=400, y_min=200, y_max=400, confidence=0.7),
    ]
    return top + bottom


def make_engine(enabled=True):
    return BinAssignmentEngine({
        "method": "point_in_polygon",
        "hand_keypoint": "index_tip",
        "occlusion_gate": {"enabled": enabled},
    })


def test_grid_structure_precomputed_on_set_bin_map():
    eng = make_engine()
    eng.set_bin_map(make_bins())
    assert eng._top_rows == {0}
    assert [b.bin_id for b in eng._bottom_bins] == ["bin_1_0", "bin_1_1"]
    assert eng._global_occ_y == 200


def test_single_row_layout_makes_gate_inert():
    eng = make_engine()
    eng.set_bin_map([
        BinRegion(bin_id="bin_0_0", label="bin_0_0",
                  x_min=0, x_max=100, y_min=0, y_max=100),
    ])
    assert eng._top_rows == set()
    assert eng._bottom_bins == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest integration/tests/test_occlusion_gate.py -v`
Expected: FAIL — `AttributeError: 'BinAssignmentEngine' object has no attribute '_top_rows'`.

- [ ] **Step 3: Add `import math`**

In `integration/src/engine/bin_assignment.py`, the imports currently are:

```python
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
```

Change to add `math`:

```python
import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
```

- [ ] **Step 4: Extend `__init__` with gate config + grid fields**

Replace the existing `__init__`:

```python
    def __init__(self, config: dict):
        self._method: str = config.get("method", "point_in_polygon")
        self._keypoint: str = config.get("hand_keypoint", "index_tip")
        self._overlap_threshold: float = config.get("overlap_threshold", 0.3)
        self._bins: list[BinRegion] = []
```

with:

```python
    def __init__(self, config: dict):
        self._method: str = config.get("method", "point_in_polygon")
        self._keypoint: str = config.get("hand_keypoint", "index_tip")
        self._overlap_threshold: float = config.get("overlap_threshold", 0.3)
        gate_cfg = config.get("occlusion_gate", {}) or {}
        self._gate_enabled: bool = gate_cfg.get("enabled", True)
        self._bins: list[BinRegion] = []
        # Grid structure for the occlusion gate, recomputed on every bin-map
        # change. Empty until a multi-row bin map is set.
        self._top_rows: set[int] = set()
        self._bottom_bins: list[BinRegion] = []
        self._global_occ_y: float = 0.0
```

- [ ] **Step 5: Add `_bin_row` and `_recompute_grid_structure`**

Add these methods to `BinAssignmentEngine` (place them right after `set_bin_map_from_geofences`, before `assign`):

```python
    @staticmethod
    def _bin_row(bin_id: str) -> int:
        """Row index from 'bin_{row}_{col}'. -1 when unparseable."""
        parts = bin_id.split("_")
        try:
            return int(parts[-2])
        except (ValueError, IndexError):
            return -1

    def _recompute_grid_structure(self) -> None:
        """Precompute the top rows, bottom-row bins (sorted by x), and the
        global bottom-band rim used by the occlusion gate. Inert (everything
        empty) for single-row or unparseable layouts."""
        rows = {self._bin_row(b.bin_id) for b in self._bins}
        rows.discard(-1)
        if len(rows) < 2:
            self._top_rows = set()
            self._bottom_bins = []
            self._global_occ_y = 0.0
            return
        bottom_row = max(rows)
        self._top_rows = {r for r in rows if r < bottom_row}
        self._bottom_bins = sorted(
            (b for b in self._bins if self._bin_row(b.bin_id) == bottom_row),
            key=lambda b: b.x_min,
        )
        self._global_occ_y = min(b.y_min for b in self._bottom_bins)
```

- [ ] **Step 6: Call `_recompute_grid_structure` from both setters**

In `set_bin_map`, after `self._bins = bins`:

```python
    def set_bin_map(self, bins: list[BinRegion]) -> None:
        self._bins = bins
        self._recompute_grid_structure()
        logger.info("Bin map set: %d region(s)", len(bins))
```

In `set_bin_map_from_geofences`, after the `self._bins = [...]` assignment and before the `logger.info`:

```python
        self._recompute_grid_structure()
        logger.info("Bin map set from geofences: %d region(s)", len(self._bins))
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest integration/tests/test_occlusion_gate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 8: Commit**

```bash
git add integration/src/engine/bin_assignment.py integration/tests/test_occlusion_gate.py
git commit -m "feat(v2): precompute grid structure + occlusion_gate config in BinAssignmentEngine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Proximal anchor selection

Add `_occlusion_anchor()` — wrist if finite and in-frame, else MCP-knuckle centroid.

**Files:**
- Modify: `integration/src/engine/bin_assignment.py`
- Test: `integration/tests/test_occlusion_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `integration/tests/test_occlusion_gate.py`:

```python
def hand_with(landmarks, hand_id=0, handedness="right"):
    """Build a HandDetection from a {name: (x, y)} dict."""
    lms = [HandLandmark(name=n, x=xy[0], y=xy[1]) for n, xy in landmarks.items()]
    return HandDetection(hand_id=hand_id, handedness=handedness, landmarks=lms)


def test_anchor_prefers_in_frame_wrist():
    eng = make_engine()
    eng.set_bin_map(make_bins())
    hand = hand_with({
        "wrist": (150, 350),
        "index_mcp": (150, 250), "middle_mcp": (160, 250),
        "ring_mcp": (170, 250), "pinky_mcp": (180, 250),
    })
    assert eng._occlusion_anchor(hand, frame_shape=(720, 1280)) == (150, 350)


def test_anchor_falls_back_to_knuckle_centroid_when_wrist_off_frame():
    eng = make_engine()
    eng.set_bin_map(make_bins())
    hand = hand_with({
        "wrist": (150, 9999),       # off the bottom of a 720-tall frame
        "index_mcp": (100, 250), "middle_mcp": (200, 250),
        "ring_mcp": (100, 350), "pinky_mcp": (200, 350),
    })
    # Centroid of the four knuckles = (150, 300).
    assert eng._occlusion_anchor(hand, frame_shape=(720, 1280)) == (150, 300)


def test_anchor_none_when_nothing_usable():
    eng = make_engine()
    eng.set_bin_map(make_bins())
    hand = hand_with({"thumb_tip": (10, 10)})   # no wrist, no knuckles
    assert eng._occlusion_anchor(hand, frame_shape=(720, 1280)) is None


def test_anchor_skips_in_frame_check_when_no_frame_shape():
    eng = make_engine()
    eng.set_bin_map(make_bins())
    hand = hand_with({
        "wrist": (150, 9999),
        "index_mcp": (100, 250), "middle_mcp": (200, 250),
        "ring_mcp": (100, 350), "pinky_mcp": (200, 350),
    })
    # No frame_shape → in-frame check skipped → wrist used as-is.
    assert eng._occlusion_anchor(hand, frame_shape=None) == (150, 9999)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest integration/tests/test_occlusion_gate.py -k anchor -v`
Expected: FAIL — `AttributeError: ... has no attribute '_occlusion_anchor'`.

- [ ] **Step 3: Implement `_occlusion_anchor`**

Add to `BinAssignmentEngine` (after `_recompute_grid_structure`):

```python
    def _occlusion_anchor(self, hand, frame_shape):
        """Most-proximal reliably-available landmark for the gate: the wrist if
        finite and (when frame_shape is given) in-frame, else the centroid of the
        finite MCP knuckles. Returns (x, y) or None."""
        wrist = hand.get_landmark("wrist")
        if wrist is not None and math.isfinite(wrist.x) and math.isfinite(wrist.y):
            in_frame = True
            if frame_shape is not None:
                fh, fw = frame_shape[0], frame_shape[1]
                in_frame = (0 <= wrist.x <= fw) and (0 <= wrist.y <= fh)
            if in_frame:
                return (wrist.x, wrist.y)

        pts = []
        for name in ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"):
            lm = hand.get_landmark(name)
            if lm is not None and math.isfinite(lm.x) and math.isfinite(lm.y):
                pts.append((lm.x, lm.y))
        if not pts:
            return None
        n = len(pts)
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest integration/tests/test_occlusion_gate.py -k anchor -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add integration/src/engine/bin_assignment.py integration/tests/test_occlusion_gate.py
git commit -m "feat(v2): proximal anchor selection for occlusion gate (wrist -> knuckle centroid)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: The gate + wire into `assign()`

Add `_bottom_bin_at()` and `_apply_occlusion_gate()`, give `assign()` a `frame_shape` param, and call the gate per hand.

**Files:**
- Modify: `integration/src/engine/bin_assignment.py`
- Test: `integration/tests/test_occlusion_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `integration/tests/test_occlusion_gate.py`:

```python
def reaching_hand(tip_xy, anchor_xy, handedness="right"):
    """Hand whose index_tip is at tip_xy and whose knuckles (the anchor) cluster
    at anchor_xy. No wrist → anchor resolves to the knuckle centroid = anchor_xy."""
    ax, ay = anchor_xy
    return hand_with({
        "index_tip": tip_xy,
        "index_mcp": (ax, ay), "middle_mcp": (ax, ay),
        "ring_mcp": (ax, ay), "pinky_mcp": (ax, ay),
    }, handedness=handedness)


def assign_one(eng, hand, frame_shape=(720, 1280)):
    events = eng.assign([hand], frame_shape=frame_shape)
    assert len(events) == 1
    return events[0]


def test_false_top_hit_with_low_knuckles_is_reassigned():
    eng = make_engine()
    eng.set_bin_map(make_bins())
    # index_tip extrapolated into top bin bin_0_1 (x150,y50); knuckles down in
    # the bottom band under x150 (>= rim y=200) → reassign to bin_1_0.
    hand = reaching_hand(tip_xy=(150, 50), anchor_xy=(150, 300))
    ev = assign_one(eng, hand)
    assert ev.bin_id == "bin_1_0"
    assert ev.method == "occlusion_gate"
    assert ev.confidence == 0.8       # taken from the bottom bin


def test_genuine_top_reach_is_untouched():
    eng = make_engine()
    eng.set_bin_map(make_bins())
    # Knuckles up near the top bin (y=60, above rim) → real top pick, no change.
    hand = reaching_hand(tip_xy=(150, 50), anchor_xy=(150, 60))
    ev = assign_one(eng, hand)
    assert ev.bin_id == "bin_0_1"
    assert ev.method == "point_in_polygon"


def test_angled_reach_with_no_bottom_bin_is_suppressed():
    eng = make_engine()
    eng.set_bin_map(make_bins())
    # Knuckles low (y=300 >= global rim 200) but at x=500 — beyond every bottom
    # bin (max x_max=400) and beyond the fingertip's bin too → suppress.
    hand = reaching_hand(tip_xy=(350, 50), anchor_xy=(500, 300))
    ev = assign_one(eng, hand)
    assert ev.bin_id is None
    assert ev.method == "occlusion_gate"


def test_disabled_gate_is_passthrough():
    eng = make_engine(enabled=False)
    eng.set_bin_map(make_bins())
    hand = reaching_hand(tip_xy=(150, 50), anchor_xy=(150, 300))
    ev = assign_one(eng, hand)
    assert ev.bin_id == "bin_0_1"
    assert ev.method == "point_in_polygon"


def test_assign_without_frame_shape_still_works():
    eng = make_engine()
    eng.set_bin_map(make_bins())
    hand = reaching_hand(tip_xy=(150, 50), anchor_xy=(150, 300))
    events = eng.assign([hand])          # no frame_shape
    assert events[0].bin_id == "bin_1_0"


def test_single_row_layout_no_reassignment():
    eng = make_engine()
    eng.set_bin_map([
        BinRegion(bin_id="bin_0_0", label="bin_0_0",
                  x_min=0, x_max=200, y_min=0, y_max=100, confidence=0.9),
    ])
    hand = reaching_hand(tip_xy=(100, 50), anchor_xy=(100, 300))
    ev = assign_one(eng, hand)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "point_in_polygon"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest integration/tests/test_occlusion_gate.py -k "reassigned or untouched or suppressed or passthrough or without_frame or single_row" -v`
Expected: FAIL — `TypeError: assign() got an unexpected keyword argument 'frame_shape'` (or reassignment not happening).

- [ ] **Step 3: Add `_bottom_bin_at` and `_apply_occlusion_gate`**

Add to `BinAssignmentEngine` (after `_occlusion_anchor`):

```python
    def _bottom_bin_at(self, x: float):
        """Bottom-row bin whose x-range contains x, or None."""
        for b in self._bottom_bins:
            if b.x_min <= x <= b.x_max:
                return b
        return None

    def _apply_occlusion_gate(self, hand, event, frame_shape):
        """Reject an extrapolated fingertip that fell into a top bin while the
        hand's proximal anchor sits at/below the bottom-band rim. Reassign to the
        bottom bin beneath, or suppress when there is clearly a bottom reach but
        no bin to assign to. Returns the (possibly rewritten) event."""
        if not self._gate_enabled or event.bin_id is None or not self._top_rows:
            return event
        if self._bin_row(event.bin_id) not in self._top_rows:
            return event

        anchor = self._occlusion_anchor(hand, frame_shape)
        if anchor is None:
            return event
        ax, ay = anchor

        below = self._bottom_bin_at(ax)
        if below is None:
            below = self._bottom_bin_at(event.hand_point[0])

        # Per-column rim (primary): anchor at/below the bin's cut-off rim.
        if below is not None and ay >= below.y_min:
            return BinEvent(
                hand_id=event.hand_id, handedness=event.handedness,
                bin_id=below.bin_id, bin_label=below.label,
                hand_point=event.hand_point, hand_area=event.hand_area,
                confidence=below.confidence, method="occlusion_gate",
            )

        # Global fallback: clearly a bottom reach but no bin beneath → suppress.
        if below is None and ay >= self._global_occ_y:
            logger.info(
                "Occlusion gate suppressed false top hit %s (anchor x=%.1f off all bottom bins)",
                event.bin_id, ax,
            )
            return BinEvent(
                hand_id=event.hand_id, handedness=event.handedness,
                bin_id=None, bin_label=None,
                hand_point=event.hand_point, hand_area=event.hand_area,
                confidence=0.0, method="occlusion_gate",
            )

        return event
```

- [ ] **Step 4: Add `frame_shape` to `assign()` and call the gate**

Replace the `assign` signature line:

```python
    def assign(self, hands: list) -> list[BinEvent]:
```

with:

```python
    def assign(self, hands: list, frame_shape=None) -> list[BinEvent]:
```

Then in the per-hand loop, replace:

```python
            else:
                event = self._assign_pip(hand, point, hand_area)

            events.append(event)
```

with:

```python
            else:
                event = self._assign_pip(hand, point, hand_area)

            event = self._apply_occlusion_gate(hand, event, frame_shape)
            events.append(event)
```

- [ ] **Step 5: Run the full gate test file**

Run: `python -m pytest integration/tests/test_occlusion_gate.py -v`
Expected: PASS (all tests from Tasks 1–3, 12 total).

- [ ] **Step 6: Run the existing engine tests to confirm no regression**

Run: `python -m pytest integration/tests/test_occlusion_hold.py -v`
Expected: PASS (unchanged — `OcclusionHold` and the old `assign(hands)` calls still work).

- [ ] **Step 7: Commit**

```bash
git add integration/src/engine/bin_assignment.py integration/tests/test_occlusion_gate.py
git commit -m "feat(v2): occlusion gate reassigns extrapolated top-bin hits to the bottom bin

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Wire `frame.shape` through the pipeline + config

Pass the frame shape from the main loop and document the new config flag.

**Files:**
- Modify: `integration/src/pipeline.py:467`
- Modify: `integration/config/settings.yaml`

- [ ] **Step 1: Pass `frame.shape` in the main loop**

In `integration/src/pipeline.py`, `_main_loop`, replace:

```python
            events = self._assignment.assign(hands)
            events = self._hold.apply(events, hands)
```

with:

```python
            events = self._assignment.assign(hands, frame.shape)
            events = self._hold.apply(events, hands)
```

- [ ] **Step 2: Add the config block**

In `integration/config/settings.yaml`, under `bin_assignment:`, replace:

```yaml
bin_assignment:
  method: "point_in_polygon"      # "point_in_polygon" | "nearest_centroid" | "area_overlap"
  hand_keypoint: "index_tip"      # Which landmark to use as the "reaching" point
  overlap_threshold: 0.3          # Min overlap ratio for area_overlap method
```

with:

```yaml
bin_assignment:
  method: "point_in_polygon"      # "point_in_polygon" | "nearest_centroid" | "area_overlap"
  hand_keypoint: "index_tip"      # Which landmark to use as the "reaching" point
  overlap_threshold: 0.3          # Min overlap ratio for area_overlap method

  # Occlusion gate. MediaPipe extrapolates occluded fingertips, so a hand
  # reaching into a BOTTOM bin (fingers hidden under the top-shelf lip) can have
  # its index_tip hallucinated into a TOP bin. The gate detects this — fingertip
  # in a top bin but the hand's proximal anchor (wrist, else knuckles) at/below
  # the bottom bin's cut-off rim — and reassigns the hit to the bottom bin
  # beneath (or suppresses it when there is no bottom bin under the hand).
  occlusion_gate:
    enabled: true                 # off → assignment behaves exactly as before
```

- [ ] **Step 3: Verify the config still loads**

Run: `python -c "import yaml; yaml.safe_load(open('integration/config/settings.yaml'))" && echo OK`
Expected: `OK`.

- [ ] **Step 4: Re-run all engine tests**

Run: `python -m pytest integration/tests/test_occlusion_gate.py integration/tests/test_occlusion_hold.py -v`
Expected: PASS (all green).

- [ ] **Step 5: Commit**

```bash
git add integration/src/pipeline.py integration/config/settings.yaml
git commit -m "feat(v2): wire frame shape into assign() + occlusion_gate config

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** §2 grid precompute → Task 1. §3 anchor → Task 2. §4 on-fire (reassign + suppress fallback) → Task 3. Direction safety (rim from grid) → Task 1 `_recompute_grid_structure`. Pipeline wiring + config → Task 4. Testing matrix (false hit, genuine reach, wrist-in-frame, wrist-off-frame, angled suppress, disabled, single-row, no-frame_shape) → Tasks 2–3.
- **Type consistency:** method names (`_bin_row`, `_recompute_grid_structure`, `_occlusion_anchor`, `_bottom_bin_at`, `_apply_occlusion_gate`) and fields (`_top_rows`, `_bottom_bins`, `_global_occ_y`, `_gate_enabled`) are used identically across tasks. `assign(hands, frame_shape=None)` signature matches all call sites (pipeline + tests).
- **Note on the "anchor above rim, fingertip in top bin" spec case:** covered by `test_genuine_top_reach_is_untouched` (anchor y=60 < rim 200 → untouched).
