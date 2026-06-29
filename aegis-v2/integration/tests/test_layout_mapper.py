"""
Unit tests for the LayoutMapper.

Pure geometry — no camera, model, or heavy deps. We import the module directly
by path so the test doesn't drag in the detectors package __init__ (which pulls
cv2 / ultralytics).
"""

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "detectors")),
)

import layout_mapper as lm  # noqa: E402

# The rig under test: 6 single-slot top bins, 3 double-slot bottom bins.
LAYOUT = [[1, 1, 1, 1, 1, 1], [2, 2, 2]]
FRAME_W, FRAME_H = 1280, 720
NUM_ROWS = len(LAYOUT)


def make_box(row, slot_start, span, row_slots, conf=0.9):
    """Build a detection box centred on a slot's expected position."""
    ex = (slot_start + span / 2.0) / row_slots          # x centre fraction
    xc = ex * FRAME_W
    half_w = (span / row_slots) * FRAME_W * 0.4
    yc = (row + 0.5) / NUM_ROWS * FRAME_H               # y centre in row's band
    half_h = (FRAME_H / NUM_ROWS) * 0.3
    return (xc - half_w, yc - half_h, xc + half_w, yc + half_h, conf)


def top_row():
    return [make_box(0, c, 1, 6) for c in range(6)]


def bottom_row(slots=(0, 2, 4)):
    return [make_box(1, s, 2, 6) for s in slots]


# ── Skeleton is always complete ──────────────────────────────

def test_skeleton_complete_even_when_empty():
    result = lm.map_detections_to_layout([], LAYOUT, FRAME_W, FRAME_H)
    assert len(result) == 9                              # 6 + 3
    assert all(v["detected"] is False for v in result.values())
    # spans / slot positions still correct
    assert result["bin_1_0"]["span"] == 2 and result["bin_1_0"]["slot_start"] == 0
    assert result["bin_1_1"]["slot_start"] == 2
    assert result["bin_1_2"]["slot_start"] == 4
    assert result["bin_0_5"]["span"] == 1 and result["bin_0_5"]["row_slots"] == 6


# ── Full detection ───────────────────────────────────────────

def test_full_detection():
    boxes = top_row() + bottom_row()
    result = lm.map_detections_to_layout(boxes, LAYOUT, FRAME_W, FRAME_H)
    assert all(result[f"bin_0_{c}"]["detected"] for c in range(6))
    assert all(result[f"bin_1_{c}"]["detected"] for c in range(3))
    # real pixel box attached
    assert "x_min" in result["bin_0_0"] and "x_max" in result["bin_1_2"]
    assert result["bin_1_0"]["confidence"] == 0.9


# ── Missing middle bottom bin — the key robustness case ──────

def test_missing_middle_bin_does_not_shift_neighbours():
    boxes = top_row() + bottom_row(slots=(0, 4))         # skip slot 2 -> bin_1_1
    result = lm.map_detections_to_layout(boxes, LAYOUT, FRAME_W, FRAME_H)
    assert result["bin_1_1"]["detected"] is False        # the gap lands here…
    assert result["bin_1_0"]["detected"] is True          # …not on its neighbours
    assert result["bin_1_2"]["detected"] is True
    assert result["bin_1_2"]["slot_start"] == 4 and result["bin_1_2"]["span"] == 2


# ── Out-of-order detections ──────────────────────────────────

def test_out_of_order_x_is_sorted():
    boxes = list(reversed(top_row()))                    # right-to-left
    result = lm.map_detections_to_layout(boxes, LAYOUT, FRAME_W, FRAME_H)
    assert all(result[f"bin_0_{c}"]["detected"] for c in range(6))


# ── Single row only ──────────────────────────────────────────

def test_single_row_detected():
    result = lm.map_detections_to_layout(top_row(), LAYOUT, FRAME_W, FRAME_H)
    assert all(result[f"bin_0_{c}"]["detected"] for c in range(6))
    assert all(result[f"bin_1_{c}"]["detected"] is False for c in range(3))


# ── Extra / spurious detection ───────────────────────────────

def test_extra_detection_is_dropped():
    spurious = make_box(0, 0, 1, 6)                       # 7th box in a 6-slot row
    boxes = top_row() + [spurious] + bottom_row()
    result = lm.map_detections_to_layout(boxes, LAYOUT, FRAME_W, FRAME_H)
    # still exactly 9 bins, no phantom id created
    assert len(result) == 9
    assert all(result[f"bin_0_{c}"]["detected"] for c in range(6))
