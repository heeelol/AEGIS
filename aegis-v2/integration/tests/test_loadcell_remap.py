"""Unit tests for LoadCellReader's bin_remap (firmware key -> canonical bin id).

Pure — no serial. enabled defaults to False so __init__ opens no port.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from integration.src.sensing.loadcell import LoadCellReader  # noqa: E402


def _reader(remap=None):
    cfg = {}
    if remap is not None:
        cfg["bin_remap"] = remap
    return LoadCellReader(cfg)  # enabled False -> no serial port opened


def test_apply_remap_renames_known_key():
    r = _reader({"bin_0_0": "bin_0_5"})
    assert r._apply_remap({"bin_0_0": -7.6}) == {"bin_0_5": -7.6}


def test_apply_remap_passes_through_unknown_key():
    r = _reader({"bin_0_0": "bin_0_5"})
    assert r._apply_remap({"bin_1_2": -1.0}) == {"bin_1_2": -1.0}


def test_apply_remap_identity_when_no_remap():
    r = _reader()
    assert r._apply_remap({"bin_0_0": -7.6}) == {"bin_0_0": -7.6}


def test_get_weights_and_layout_reflect_remap():
    # Simulate the read loop: parse a real firmware line, then cache it remapped.
    r = _reader({"bin_0_0": "bin_0_5"})
    parsed = LoadCellReader._parse_line(b'{"bins":{"bin_0_0":-7.6}}\n')
    r._weights = r._apply_remap(parsed)
    assert r.get_weights() == {"bin_0_5": -7.6}
    layout = r.get_layout()
    # bin_0_5 -> row 0, col 5 -> 6 bins in row 0
    assert layout.bins_per_layer == {0: 6}
