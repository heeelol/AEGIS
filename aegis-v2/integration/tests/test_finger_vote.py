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
