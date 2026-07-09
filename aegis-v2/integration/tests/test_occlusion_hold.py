"""Unit tests for OcclusionHold — bin-keyed, handedness-independent.

A bottom bin is held while it has been visibly picked AND still has a forearm in
it (occupied). Handedness is ignored (MediaPipe's left/right label is unreliable
on occluded hands). Occluded fingertip events are dropped as unreliable. Pure
logic — no camera/model/cv2.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from integration.src.engine.bin_assignment import BinEvent  # noqa: E402
from integration.src.engine.occlusion_hold import OcclusionHold  # noqa: E402


class FakeHand:
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
    evs = [event("right", "bin_1_0")]
    assert h.apply(evs, [FakeHand("right")]) is evs


def test_visible_pick_passes_through_not_doubled():
    h = make_hold()
    out = h.apply([event("left", "bin_1_0", 0)], [FakeHand("left", 0)],
                  occluded_ids=set(), occupied_bins={"bin_1_0"})
    assert len(out) == 1
    assert out[0].method == "point_in_polygon"  # live event kept, no synthetic


def test_holds_bin_while_occluded():
    h = make_hold()
    h.apply([event("left", "bin_1_0", 0)], [FakeHand("left", 0)],
            occluded_ids=set(), occupied_bins={"bin_1_0"})           # arm (visible)
    out = h.apply([event("left", None, 0)], [FakeHand("left", 0)],
                  occluded_ids={0}, occupied_bins={"bin_1_0"})       # occluded
    held = [e for e in out if e.bin_id == "bin_1_0"]
    assert len(held) == 1 and held[0].method == "occlusion_hold"


def test_occluded_event_is_dropped():
    """The unreliable occluded fingertip event is removed (no stray 0.0 box)."""
    h = make_hold()
    h.apply([event("left", "bin_1_0", 0)], [FakeHand("left", 0)],
            occluded_ids=set(), occupied_bins={"bin_1_0"})
    out = h.apply([event("left", "bin_1_0", 0)], [FakeHand("left", 0)],
                  occluded_ids={0}, occupied_bins={"bin_1_0"})
    assert len(out) == 1
    assert out[0].method == "occlusion_hold"  # only the synthetic, the live one is gone


def test_handedness_flip_does_not_create_phantom():
    """The reported bug: one hand whose label flips to 'right' while occluded must
    not leave a stuck 'right' box — the bin is held once, label-agnostic."""
    h = make_hold()
    h.apply([event("left", "bin_1_0", 0)], [FakeHand("left", 0)],         # seen as left
            occluded_ids=set(), occupied_bins={"bin_1_0"})
    out = h.apply([event("right", "bin_1_0", 0)], [FakeHand("right", 0)],  # flipped + occluded
                  occluded_ids={0}, occupied_bins={"bin_1_0"})
    lit = [e for e in out if e.bin_id == "bin_1_0"]
    assert len(lit) == 1
    assert lit[0].method == "occlusion_hold"


def test_release_when_bin_empty():
    h = make_hold()
    h.apply([event("left", "bin_1_0", 0)], [FakeHand("left", 0)],
            occluded_ids=set(), occupied_bins={"bin_1_0"})
    h.apply([event("left", None, 0)], [FakeHand("left", 0)],
            occluded_ids={0}, occupied_bins={"bin_1_0"})            # held
    out = h.apply([], [], occluded_ids=set(), occupied_bins=set())  # forearm gone
    assert out == []


def test_no_hold_if_bin_never_visibly_picked():
    """An occluded event in a bin never seen visibly → dropped, no hold."""
    h = make_hold()
    out = h.apply([event("left", "bin_1_0", 0)], [FakeHand("left", 0)],
                  occluded_ids={0}, occupied_bins={"bin_1_0"})
    assert out == []


def test_two_bins_independent():
    h = make_hold()
    h.apply([event("left", "bin_1_0", 0), event("right", "bin_1_1", 1)],
            [FakeHand("left", 0), FakeHand("right", 1)],
            occluded_ids=set(), occupied_bins={"bin_1_0", "bin_1_1"})
    out = h.apply([], [], occluded_ids=set(), occupied_bins={"bin_1_0", "bin_1_1"})
    held = {e.bin_id for e in out if e.method == "occlusion_hold"}
    assert held == {"bin_1_0", "bin_1_1"}


def test_non_eligible_bin_not_held():
    h = make_hold()
    h.apply([event("left", "bin_0_0", 0)], [FakeHand("left", 0)],  # top bin
            occluded_ids=set(), occupied_bins=set())
    out = h.apply([], [], occluded_ids=set(), occupied_bins=set())
    assert out == []


def test_set_eligible_bins_drops_stale_arm():
    h = make_hold()
    h.apply([event("left", "bin_1_0", 0)], [FakeHand("left", 0)],
            occluded_ids=set(), occupied_bins={"bin_1_0"})
    h.set_eligible_bins({"bin_1_1"})
    out = h.apply([], [], occluded_ids=set(), occupied_bins={"bin_1_0"})
    assert out == []


def test_no_double_activation_when_hand_moves_to_top_bin():
    """Reported bug: the forearm transits the bottom bin (arming it) on the way up,
    then the hand settles in the top bin. The pipeline drops that bottom bin from
    occupied (pass-through under a top pick), so it releases — only the top bin is
    active, never two at once."""
    h = make_hold()
    h.apply([event("left", "bin_1_0", 0)], [FakeHand("left", 0)],          # transit arms bin_1_0
            occluded_ids=set(), occupied_bins={"bin_1_0"})
    out = h.apply([event("left", "bin_0_0", 0)], [FakeHand("left", 0)],     # now up in top bin
                  occluded_ids=set(), occupied_bins=set())  # bin_1_0 excluded (pass-through)
    assert {e.bin_id for e in out} == {"bin_0_0"}


def test_no_duplicate_when_live_event_covers_bin():
    h = make_hold()
    h.apply([event("left", "bin_1_0", 0)], [FakeHand("left", 0)],
            occluded_ids=set(), occupied_bins={"bin_1_0"})
    out = h.apply([event("left", "bin_1_0", 0)], [FakeHand("left", 0)],
                  occluded_ids=set(), occupied_bins={"bin_1_0"})  # visible again
    assert sum(1 for e in out if e.bin_id == "bin_1_0") == 1
