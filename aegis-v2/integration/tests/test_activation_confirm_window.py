"""Tests for the activation-confirm window in placement.py.

A bin only becomes ACTIVE once it's been the leading candidate continuously
for activation_confirm_s — a single noisy frame crossing activation_frac used
to activate (and visually "focus") a bin instantly, with zero real hand
interaction. These tests use mocked time (not real sleeps) so they stay fast
and deterministic.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # integration/

from src.sensing.placement import PlacementTracker

UNITS = {"bin_a": 16.6, "bin_b": 16.6}
TARGETS = {"bin_a": 0, "bin_b": 0}
BOX = "kit_box"


def w(a=0.0, b=0.0, box=0.0):
    return {"bin_a": a, "bin_b": b, "kit_box": box}


def test_momentary_candidate_never_activates():
    with patch("src.sensing.placement.time.time") as mock_time:
        mock_time.return_value = 1000.0
        t = PlacementTracker(UNITS, TARGETS, BOX, fault_settle_s=0, activation_confirm_s=0.4)
        t.tare({"bin_a": 0.0, "bin_b": 0.0, "kit_box": 0.0})
        for _ in range(5):
            s = t.update(w(a=-16.6))   # a brief blip over the activation threshold
        assert s.active is None        # not confirmed yet (0s elapsed on this candidate)
        # blip released before the confirm window would ever elapse
        for _ in range(25):
            s = t.update(w())
        assert s.active is None


def test_sustained_candidate_activates_after_window():
    with patch("src.sensing.placement.time.time") as mock_time:
        mock_time.return_value = 1000.0
        t = PlacementTracker(UNITS, TARGETS, BOX, fault_settle_s=0, activation_confirm_s=0.4)
        t.tare({"bin_a": 0.0, "bin_b": 0.0, "kit_box": 0.0})
        for _ in range(5):
            s = t.update(w(a=-16.6))
        assert s.active is None
        mock_time.return_value = 1000.5   # past the 0.4s window; same candidate still holds
        s = t.update(w(a=-16.6))
        assert s.active == "bin_a"


def test_confirm_window_restarts_on_a_different_candidate():
    # A near-confirmed bin_a candidate must NOT hand its elapsed time to a
    # later, different bin_b candidate — each candidate gets its own clock.
    with patch("src.sensing.placement.time.time") as mock_time:
        mock_time.return_value = 1000.0
        t = PlacementTracker(UNITS, TARGETS, BOX, fault_settle_s=0, activation_confirm_s=0.4)
        t.tare({"bin_a": 0.0, "bin_b": 0.0, "kit_box": 0.0})
        for _ in range(5):
            s = t.update(w(a=-16.6))
        mock_time.return_value = 1000.3     # 0.3s elapsed on bin_a, not yet confirmed
        s = t.update(w(a=-16.6))
        assert s.active is None

        for _ in range(5):                   # switch to bin_b before bin_a confirms
            s = t.update(w(b=-16.6))
        mock_time.return_value = 1000.5     # only ~0.2s since bin_b's candidate first appeared —
        s = t.update(w(b=-16.6))            # would have wrongly activated if the clock carried over
        assert s.active is None
        mock_time.return_value = 1000.8     # now >=0.4s since bin_b's OWN candidate started
        s = t.update(w(b=-16.6))
        assert s.active == "bin_b"


def test_real_reach_still_activates_promptly():
    # Sanity check: a genuine sustained pick clears the window easily — this
    # isn't a delay on legitimate picks, only a filter on single-frame noise.
    with patch("src.sensing.placement.time.time") as mock_time:
        mock_time.return_value = 1000.0
        t = PlacementTracker(UNITS, TARGETS, BOX, fault_settle_s=0, activation_confirm_s=0.4)
        t.tare({"bin_a": 0.0, "bin_b": 0.0, "kit_box": 0.0})
        s = t.update(w(a=-16.6))
        mock_time.return_value = 1000.45
        s = t.update(w(a=-16.6))
        assert s.active == "bin_a"
