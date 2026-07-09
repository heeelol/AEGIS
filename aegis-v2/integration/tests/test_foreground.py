"""Unit tests for ForegroundModel (background-subtraction hand-presence oracle).

Uses synthetic frames — a static gray background plus a bright blob — to verify
the model learns the background and reports foreground only where new content
appears. cv2 + numpy required.
"""
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from integration.src.detectors.foreground import ForegroundModel  # noqa: E402


def _bg(value=127):
    return np.full((120, 160, 3), value, dtype=np.uint8)


def test_not_ready_until_warmup_frames_seen():
    m = ForegroundModel(warmup_frames=5)
    assert not m.ready
    for _ in range(4):
        m.update(_bg())
    assert not m.ready
    m.update(_bg())
    assert m.ready


def test_foreground_high_inside_new_blob_zero_in_background():
    m = ForegroundModel(warmup_frames=3, patch_size=21)
    for _ in range(15):          # learn the static background
        m.update(_bg())

    frame = _bg()
    frame[40:80, 60:100] = 255   # a bright blob appears (a "hand")
    mask = m.update(frame)

    assert m.patch_ratio(mask, 80, 60) > 0.5   # inside the blob
    assert m.patch_ratio(mask, 10, 10) < 0.05  # far away, still background


def test_patch_ratio_clips_at_frame_edge():
    m = ForegroundModel(warmup_frames=1, patch_size=21)
    mask = np.ones((120, 160), dtype=np.uint8)  # all foreground
    # A point in the corner: patch is clipped to the frame but still valid.
    assert m.patch_ratio(mask, 0, 0) == 1.0


def test_update_returns_binary_mask():
    m = ForegroundModel(warmup_frames=1)
    mask = m.update(_bg())
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


def test_region_ratio_measures_box_foreground():
    m = ForegroundModel(warmup_frames=1)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:50, :] = 1  # top half is foreground
    assert m.region_ratio(mask, 0, 0, 100, 50) == 1.0     # fully inside the fg band
    assert m.region_ratio(mask, 0, 50, 100, 100) == 0.0    # fully in the bg band


def test_region_ratio_empty_box_is_zero():
    m = ForegroundModel(warmup_frames=1)
    mask = np.ones((100, 100), dtype=np.uint8)
    assert m.region_ratio(mask, 50, 50, 50, 50) == 0.0     # degenerate box
