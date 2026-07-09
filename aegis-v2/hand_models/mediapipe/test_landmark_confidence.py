"""Unit tests for MediaPipe per-landmark confidence extraction.

Regression guard: the Tasks-API hand model exposes a ``visibility`` attribute
that is ``None``. The old ``getattr(lm, "visibility", 1.0)`` returned that
``None``, producing ``HandLandmark.confidence = None`` and a ``TypeError`` when
compared against ``vote_confidence_floor`` in bin assignment.
"""
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from hand_models.mediapipe.tracker import _landmark_confidence  # noqa: E402


class _LM:
    def __init__(self, visibility):
        self.visibility = visibility


def test_visibility_none_is_fully_confident():
    """A landmark whose visibility is None must yield a usable float."""
    conf = _landmark_confidence(_LM(visibility=None))
    assert isinstance(conf, float)
    assert conf == 1.0


def test_missing_visibility_attribute_is_fully_confident():
    conf = _landmark_confidence(object())
    assert conf == 1.0


def test_real_visibility_is_preserved():
    conf = _landmark_confidence(_LM(visibility=0.42))
    assert math.isclose(conf, 0.42)


def test_zero_visibility_is_preserved():
    """0.0 is a real value, not 'missing' — it must survive."""
    conf = _landmark_confidence(_LM(visibility=0.0))
    assert conf == 0.0
