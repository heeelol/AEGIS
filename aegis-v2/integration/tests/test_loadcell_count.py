"""Unit tests for load-cell -> pick-count wiring.

Pure — no serial, no camera. Uses a temp inventory.yaml and a fake reader.
Pipeline._apply_loadcell_counts is exercised via object.__new__ (we set only
the attributes the method touches; __init__ would open a camera/config).
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from integration.src.pipeline import Pipeline, derive_pick_counts  # noqa: E402
from integration.src.sensing.inventory import InventoryTracker  # noqa: E402
from integration.src.ui.state import PipelineState  # noqa: E402


def _tracker(tmp_path):
    inv = tmp_path / "inventory.yaml"
    inv.write_text(
        "items:\n"
        "  amp: { unit_g: 3.8 }\n"
        "bins:\n"
        "  bin_0_5: amp\n"
    )
    return InventoryTracker(str(inv))


class _FakeReader:
    def __init__(self, weights, connected):
        self._weights, self._connected = weights, connected

    def get_weights(self):
        return dict(self._weights)

    def is_connected(self):
        return self._connected


def test_derive_counts_when_connected(tmp_path):
    tracker = _tracker(tmp_path)
    # -7.6 g / 3.8 g = 2 AMPs taken
    assert derive_pick_counts({"bin_0_5": -7.6}, tracker, True) == {"bin_0_5": 2}


def test_derive_counts_empty_when_disconnected(tmp_path):
    tracker = _tracker(tmp_path)
    assert derive_pick_counts({"bin_0_5": -7.6}, tracker, False) == {}


def test_derive_counts_clamps_and_ignores_unmapped(tmp_path):
    tracker = _tracker(tmp_path)
    # small positive (noise) -> 0; unmapped bin not in inventory -> omitted
    assert derive_pick_counts({"bin_0_5": 1.0, "bin_9_9": -100.0}, tracker, True) \
        == {"bin_0_5": 0}


def _pipeline_with(state, reader, tracker):
    p = object.__new__(Pipeline)        # bypass __init__ (no camera/config)
    p._state, p._loadcells, p._inventory = state, reader, tracker
    return p


def test_apply_counts_sets_state_when_connected(tmp_path):
    state = PipelineState()
    state.update_bins({"bin_0_5": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}})
    p = _pipeline_with(state, _FakeReader({"bin_0_5": -7.6}, True), _tracker(tmp_path))

    p._apply_loadcell_counts()

    current = {b["id"]: b["current"] for b in state.get_bins()}
    assert current["bin_0_5"] == 2


def test_apply_counts_noop_when_disconnected(tmp_path):
    state = PipelineState()
    state.update_bins({"bin_0_5": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}})
    state.set_pick_count("bin_0_5", 4)  # pre-existing manual value
    p = _pipeline_with(state, _FakeReader({"bin_0_5": -7.6}, False), _tracker(tmp_path))

    p._apply_loadcell_counts()

    current = {b["id"]: b["current"] for b in state.get_bins()}
    assert current["bin_0_5"] == 4  # untouched — guard held
