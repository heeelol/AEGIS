"""Smoke test for OverlayUI.render_foreground_debug (the tuning view)."""
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from integration.src.engine.bin_assignment import BinRegion  # noqa: E402
from integration.src.ui.overlay import OverlayUI  # noqa: E402


def _overlay():
    bins = [BinRegion(bin_id="bin_0_0", label="bin_0_0",
                      x_min=0, x_max=100, y_min=0, y_max=100, confidence=1.0)]
    return OverlayUI({}, bins)


def test_debug_view_returns_bgr_image_of_mask_shape():
    ov = _overlay()
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[40:80, 60:100] = 1
    samples = [(80, 60, 0.9), (10, 10, 0.0)]
    view = ov.render_foreground_debug(mask, samples, present_ratio=0.10, patch_size=21)
    assert view.shape == (120, 160, 3)
    assert view.dtype == np.uint8


def test_debug_view_handles_no_samples():
    ov = _overlay()
    mask = np.zeros((120, 160), dtype=np.uint8)
    view = ov.render_foreground_debug(mask, [], present_ratio=0.10,
                                      patch_size=21, ready=False)
    assert view.shape == (120, 160, 3)
