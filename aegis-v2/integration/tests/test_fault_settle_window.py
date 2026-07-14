"""Tests for the post-tare fault-settle window in placement.py.

Right after tare() (boot, empty-box confirm, cycle restart), individual bins'
own EMAs are starting fresh and can still be settling for a moment — the same
class of issue as the box-baseline timing fix, just never gated for per-bin
values. Without a settle window, a bin whose reading hasn't fully converged
yet can spuriously read as "added" and fault instantly, with zero real
operator interaction. These tests use mocked time (not real sleeps) so they
stay fast and deterministic.

Needs an active/holding bin to exercise return-to-wrong-bin under the current
holding-based design (see placement.py module docstring) — bin_a is activated
(activation_confirm_s=0, immediate) and bin_b receives a matching addition.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # integration/

from src.sensing.placement import PlacementTracker

UNITS = {"bin_a": 16.6, "bin_b": 16.6}
# bin_a must be in the BOM (target > 0) to activate and hold; bin_b then
# receives the matching item as the wrong destination (return-to-wrong-bin).
TARGETS = {"bin_a": 5, "bin_b": 5}
BOX = "kit_box"


def w(a=0.0, b=0.0, box=0.0):
    return {"bin_a": a, "bin_b": b, "kit_box": box}


def test_spurious_reading_suppressed_immediately_after_tare():
    with patch("src.sensing.placement.time.time", return_value=1000.0):
        t = PlacementTracker(UNITS, TARGETS, BOX, fault_settle_s=2.5, activation_confirm_s=0)
        t.tare({"bin_a": 0.0, "bin_b": 0.0, "kit_box": 0.0})
        s = None
        for _ in range(25):
            # bin_a active/holding, bin_b matches -- would normally be a
            # return-to-wrong-bin fault
            s = t.update(w(a=-16.6, b=16.6))
    assert s.alert is None


def test_same_condition_faults_once_settle_window_elapses():
    with patch("src.sensing.placement.time.time") as mock_time:
        mock_time.return_value = 1000.0
        t = PlacementTracker(UNITS, TARGETS, BOX, fault_settle_s=2.5, activation_confirm_s=0)
        t.tare({"bin_a": 0.0, "bin_b": 0.0, "kit_box": 0.0})

        mock_time.return_value = 1003.0   # past the 2.5s window
        s = None
        for _ in range(25):
            s = t.update(w(a=-16.6, b=16.6))
    assert s.alert is not None and s.alert["type"] == "return-to-wrong-bin"


def test_settle_window_restarts_on_a_later_tare():
    # A second tare (e.g. empty-box confirm) must re-arm its own settle
    # window, not inherit whatever time already elapsed since boot.
    with patch("src.sensing.placement.time.time") as mock_time:
        mock_time.return_value = 1000.0
        t = PlacementTracker(UNITS, TARGETS, BOX, fault_settle_s=2.5, activation_confirm_s=0)
        t.tare({"bin_a": 0.0, "bin_b": 0.0, "kit_box": 0.0})

        mock_time.return_value = 1010.0   # well past the first window
        t.tare({"bin_a": 0.0, "bin_b": 0.0, "kit_box": 0.0})   # re-tare resets it

        mock_time.return_value = 1010.5   # only 0.5s after the SECOND tare
        s = None
        for _ in range(25):
            s = t.update(w(a=-16.6, b=16.6))
    assert s.alert is None
