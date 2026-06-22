"""Unit tests for the occlusion gate in BinAssignmentEngine.

Pure geometry — no camera, no model, no cv2. Drives the engine with a
hand-built bin map and synthetic HandDetections, mirroring test_occlusion_hold.

Grid used throughout (y increases downward; row 0 = top):

      x:  0----100----200
  y 0  +--------+--------+
       | bin_0_0| bin_0_1|   top row  (single cells)
  100  +--------+--------+   <- occlusion line (bottom-bin top rim)
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


# ── Fixtures ────────────────────────────────────────────────

def two_row_geofences():
    return {
        "bin_0_0": {"x_min": 0,   "x_max": 100, "y_min": 0,   "y_max": 100, "confidence": 0.9},
        "bin_0_1": {"x_min": 100, "x_max": 200, "y_min": 0,   "y_max": 100, "confidence": 0.9},
        "bin_1_0": {"x_min": 0,   "x_max": 100, "y_min": 100, "y_max": 200, "confidence": 0.8},
        "bin_1_1": {"x_min": 100, "x_max": 200, "y_min": 100, "y_max": 200, "confidence": 0.8},
    }


def make_engine(geofences=None, enabled=True):
    if geofences is None:
        geofences = two_row_geofences()
    eng = BinAssignmentEngine({
        "method": "point_in_polygon",
        "hand_keypoint": "index_tip",
        "occlusion_gate": {"enabled": enabled},
    })
    eng.set_bin_map_from_geofences(geofences)
    return eng


def make_hand(index_tip, mcps=None, wrist=None, hand_id=0, handedness="right"):
    """Build a synthetic hand.

    index_tip : (x, y) of the reaching keypoint.
    mcps      : (x, y) shared by all four MCP knuckles, or None to omit them.
    wrist     : (x, y) of the wrist, or None to omit it. Use math.nan coords
                to model a present-but-non-finite wrist.
    """
    lms = [HandLandmark(name="index_tip", x=index_tip[0], y=index_tip[1])]
    if mcps is not None:
        for name in ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"):
            lms.append(HandLandmark(name=name, x=mcps[0], y=mcps[1]))
    if wrist is not None:
        lms.append(HandLandmark(name="wrist", x=wrist[0], y=wrist[1]))
    return HandDetection(hand_id=hand_id, handedness=handedness, landmarks=lms)


FRAME = (200, 200, 3)  # (h, w, channels) — like frame.shape


# ── Tests ───────────────────────────────────────────────────

def test_false_top_hit_knuckles_low_reassigned():
    """Wrist hidden, MCP centroid below the bottom rim, fingertip extrapolated
    into a top bin → reassigned to the bottom bin beneath."""
    eng = make_engine()
    hand = make_hand(index_tip=(50, 50), mcps=(50, 150))  # no wrist
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_1_0"
    assert ev.method == "occlusion_gate"
    assert ev.confidence == 0.8       # taken from the bottom bin
    assert ev.hand_point == (50, 50)  # fingertip preserved


def test_genuine_top_reach_untouched():
    """Knuckles up near the top bin (above the rim), fingertip in top bin →
    event untouched."""
    eng = make_engine()
    hand = make_hand(index_tip=(50, 30), mcps=(50, 60))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "point_in_polygon"


def test_knuckles_win_over_low_wrist_top_reach_untouched():
    """Over-the-shelf reach: fingertip and knuckles are up in the top bin, but
    the wrist trails back down over the bottom bin. The knuckles — not the
    wrist — decide where the hand is, so the genuine top pick is left untouched.

    (Regression: the wrist used to win here and wrongly reassigned to bin_1_0.)"""
    eng = make_engine()
    hand = make_hand(index_tip=(50, 50), mcps=(50, 40), wrist=(50, 150))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "point_in_polygon"


def test_knuckles_low_reassigned_even_with_high_wrist():
    """Mirror: knuckles below the rim (hand in the bottom band, fingertip
    extrapolated up) → reassigned, regardless of a wrist that reads above."""
    eng = make_engine()
    hand = make_hand(index_tip=(50, 50), mcps=(50, 150), wrist=(50, 40))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_1_0"
    assert ev.method == "occlusion_gate"


def test_wrist_fallback_when_no_knuckles():
    """No knuckles available → wrist is the fallback anchor. Wrist below rim →
    reassigned to the bottom bin beneath."""
    eng = make_engine()
    hand = make_hand(index_tip=(50, 50), wrist=(50, 150))  # no mcps
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_1_0"
    assert ev.method == "occlusion_gate"


def test_wrist_fallback_offframe_cannot_judge():
    """No knuckles and the wrist is off-frame → no usable anchor → the gate
    cannot judge and leaves the event untouched."""
    eng = make_engine()
    hand = make_hand(index_tip=(50, 50), wrist=(50, -50))  # no mcps, wrist off-frame
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "point_in_polygon"


def test_wrist_fallback_non_finite_cannot_judge():
    """No knuckles and a present-but-NaN wrist → no usable anchor → untouched."""
    eng = make_engine()
    hand = make_hand(index_tip=(50, 50), wrist=(math.nan, math.nan))  # no mcps
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "point_in_polygon"


def test_angled_reach_no_bottom_bin_suppressed():
    """Anchor below the global line but its x is under no bottom bin (reaching in
    at an angle) → suppressed, not reassigned."""
    eng = make_engine()
    hand = make_hand(index_tip=(50, 50), mcps=(250, 150))  # anchor x off the grid
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id is None
    assert ev.method == "occlusion_gate"


def test_anchor_above_rim_steep_top_pick_untouched():
    """True top pick at a steep angle: fingertip in top bin, anchor x under a
    bottom bin but anchor y above the rim → untouched."""
    eng = make_engine()
    hand = make_hand(index_tip=(150, 30), mcps=(150, 80))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_1"
    assert ev.method == "point_in_polygon"


def test_disabled_is_passthrough():
    """enabled: false → identical to pre-gate behavior (false top hit stays top)."""
    eng = make_engine(enabled=False)
    hand = make_hand(index_tip=(50, 50), mcps=(50, 150))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "point_in_polygon"


def test_single_row_layout_gate_inert():
    """Only one row → no top rows → gate never reassigns."""
    single = {
        "bin_0_0": {"x_min": 0,   "x_max": 100, "y_min": 0, "y_max": 200, "confidence": 0.9},
        "bin_0_1": {"x_min": 100, "x_max": 200, "y_min": 0, "y_max": 200, "confidence": 0.9},
    }
    eng = make_engine(single)
    hand = make_hand(index_tip=(50, 50), mcps=(50, 150))
    [ev] = eng.assign([hand], FRAME)
    assert ev.bin_id == "bin_0_0"
    assert ev.method == "point_in_polygon"


def test_assign_without_frame_shape_still_gates():
    """assign(hands) with no frame_shape skips the in-frame check but still
    runs the gate via the MCP centroid."""
    eng = make_engine()
    hand = make_hand(index_tip=(50, 50), mcps=(50, 150))
    [ev] = eng.assign([hand])  # no frame_shape
    assert ev.bin_id == "bin_1_0"
    assert ev.method == "occlusion_gate"
