"""Tests for the cycle / set sequencing (CycleManager)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # integration/

from src.engine.cycle import CycleManager


def test_advances_through_sets_then_completes():
    cm = CycleManager([{"bin_0_4": 3, "bin_1_2": 2}, {"bin_0_5": 5}])
    assert cm.total_sets == 2
    assert cm.set_number == 1
    assert not cm.is_complete
    assert cm.current_targets() == {"bin_0_4": 3, "bin_1_2": 2}

    assert cm.advance() is False          # confirm set 1 -> set 2
    assert cm.set_number == 2
    assert cm.current_targets() == {"bin_0_5": 5}

    assert cm.advance() is True           # confirm last set -> cycle complete
    assert cm.is_complete
    assert cm.current_targets() == {}
    assert cm.advance() is False          # idempotent once complete


def test_restart_begins_same_cycle():
    cm = CycleManager([{"bin_0_4": 1}])
    assert cm.advance() is True
    assert cm.is_complete
    cm.restart()
    assert not cm.is_complete
    assert cm.set_number == 1
    assert cm.current_targets() == {"bin_0_4": 1}


def test_sanitizes_zero_and_empty_sets():
    cm = CycleManager([{"bin_0_4": 0}, {}, {"bin_1_2": 2, "bin_0_5": 0}])
    assert cm.total_sets == 1
    assert cm.current_targets() == {"bin_1_2": 2}


def test_no_sets_is_immediately_complete():
    cm = CycleManager([])
    assert cm.total_sets == 0
    assert cm.is_complete
    assert cm.current_targets() == {}
    assert cm.snapshot() == {"set_number": 0, "total_sets": 0, "complete": True}
