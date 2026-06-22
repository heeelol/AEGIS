"""Unit tests for OcclusionHold — pure logic, no camera/model/cv2."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from integration.src.engine.bin_assignment import BinEvent  # noqa: E402
from integration.src.engine.occlusion_hold import OcclusionHold  # noqa: E402


class FakeHand:
    """Minimal stand-in: OcclusionHold only reads .handedness off a hand."""
    def __init__(self, handedness, hand_id=0):
        self.handedness = handedness
        self.hand_id = hand_id


def event(handedness, bin_id, hand_id=0):
    return BinEvent(
        hand_id=hand_id, handedness=handedness,
        bin_id=bin_id, bin_label=bin_id,
        hand_point=(0.0, 0.0), hand_area=0.0,
        confidence=0.5, method="point_in_polygon",
    )


def make_hold(eligible=("bin_1_0", "bin_1_1"), enabled=True):
    h = OcclusionHold({"enabled": enabled})
    h.set_eligible_bins(set(eligible))
    return h


def test_disabled_is_passthrough():
    h = make_hold(enabled=False)
    evs = [event("Right", "bin_1_0")]
    assert h.apply(evs, [FakeHand("Right")]) is evs


def test_holds_bin_while_hand_occluded():
    h = make_hold()
    # Frame 1: right hand picking the eligible bottom bin.
    h.apply([event("Right", "bin_1_0")], [FakeHand("Right")])
    # Frame 2: hand vanished (occluded). Bin must stay active via a synthetic event.
    out = h.apply([], [])
    held = [e for e in out if e.bin_id == "bin_1_0"]
    assert len(held) == 1
    assert held[0].method == "occlusion_hold"
    assert held[0].handedness == "Right"


def test_release_when_hand_seen_again():
    h = make_hold()
    h.apply([event("Right", "bin_1_0")], [FakeHand("Right")])
    h.apply([], [])  # occluded — held
    # Hand reappears but is no longer in the held bin → hold released.
    out = h.apply([event("Right", None)], [FakeHand("Right")])
    assert all(e.method != "occlusion_hold" for e in out)
    # And it stays released on a subsequent occluded frame.
    assert h.apply([], []) == []


def test_no_hold_for_non_eligible_bin():
    h = make_hold()
    h.apply([event("Right", "bin_0_0")], [FakeHand("Right")])  # top bin, not eligible
    assert h.apply([], []) == []


def test_no_duplicate_when_bin_already_active():
    h = make_hold()
    h.apply([event("Right", "bin_1_0")], [FakeHand("Right")])
    # Left hand now also in the same bin while right is gone — no synthetic dup.
    out = h.apply([event("Left", "bin_1_0", hand_id=1)], [FakeHand("Left", 1)])
    assert sum(1 for e in out if e.bin_id == "bin_1_0") == 1


def test_two_hands_independent_holds():
    h = make_hold()
    h.apply(
        [event("Right", "bin_1_0", 0), event("Left", "bin_1_1", 1)],
        [FakeHand("Right", 0), FakeHand("Left", 1)],
    )
    # Both occluded → both bins held.
    out = h.apply([], [])
    held_bins = {e.bin_id for e in out if e.method == "occlusion_hold"}
    assert held_bins == {"bin_1_0", "bin_1_1"}


def test_set_eligible_bins_drops_stale_hold():
    h = make_hold()
    h.apply([event("Right", "bin_1_0")], [FakeHand("Right")])
    h.set_eligible_bins({"bin_1_1"})  # bin_1_0 no longer eligible
    assert h.apply([], []) == []
