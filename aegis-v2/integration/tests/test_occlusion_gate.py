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
    # Top row extends to x=500 (bin_0_4 at 400..500) but the bottom row only
    # covers x 0..400, so a hit on bin_0_4 has no bottom bin beneath it.
    bins = make_bins() + [
        BinRegion(bin_id="bin_0_4", label="bin_0_4",
                  x_min=400, x_max=500, y_min=0, y_max=100, confidence=0.9),
    ]
    eng.set_bin_map(bins)
    # Both anchor x (450) and fingertip x (450) are off every bottom bin, and the
    # anchor is low (y=300 >= global rim 200) → clearly a bottom reach, no target.
    hand = reaching_hand(tip_xy=(450, 50), anchor_xy=(450, 300))
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
