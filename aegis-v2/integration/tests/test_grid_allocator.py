"""Unit tests for grid_allocator — pure geometry, no camera/model/cv2.

Imported by path so the test doesn't drag in the detectors package __init__
(which pulls cv2 / ultralytics).
"""
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "detectors")),
)

import grid_allocator as ga  # noqa: E402

FRAME_W = 1280


def test_skeleton_is_fully_preindexed():
    sk = ga.build_skeleton(ga.DEFAULT_LAYOUT)
    assert len(sk) == 9
    # top row -> indices 1..6, span 1
    assert sk["bin_1"]["index"] == 1 and sk["bin_1"]["span"] == 1
    assert sk["bin_1"]["layer"] == "top" and sk["bin_1"]["slot_start"] == 0
    assert sk["bin_6"]["index"] == 6 and sk["bin_6"]["slot_start"] == 5
    # bottom row -> indices 7..9, span 2, slot_start 0/2/4
    assert sk["bin_7"]["index"] == 7 and sk["bin_7"]["span"] == 2
    assert sk["bin_7"]["layer"] == "bottom" and sk["bin_7"]["slot_start"] == 0
    assert sk["bin_8"]["slot_start"] == 2 and sk["bin_9"]["slot_start"] == 4
    # every cell starts undetected, row_slots 6 for both rows
    assert all(c["detected"] is False for c in sk.values())
    assert sk["bin_1"]["row_slots"] == 6 and sk["bin_9"]["row_slots"] == 6


def make_det(cx, cy, conf=0.9, half=20):
    """A detection dict with a square box centred on (cx, cy)."""
    return {
        "corners": [[cx - half, cy - half], [cx + half, cy - half],
                    [cx + half, cy + half], [cx - half, cy + half]],
        "center": [float(cx), float(cy)],
        "conf": conf,
    }


def test_split_rows_by_largest_y_gap():
    # 6 top dets near y=200, 3 bottom dets near y=560 (big gap between bands)
    top = [make_det(x, 200) for x in (100, 300, 500, 700, 900, 1100)]
    bottom = [make_det(x, 560) for x in (200, 640, 1080)]
    rows = ga.split_rows_by_y(top + bottom, num_rows=2)
    assert len(rows) == 2
    assert len(rows[0]) == 6 and all(d["center"][1] == 200 for d in rows[0])
    assert len(rows[1]) == 3 and all(d["center"][1] == 560 for d in rows[1])


def test_split_rows_single_row_only():
    top = [make_det(x, 200) for x in (100, 500, 900)]
    rows = ga.split_rows_by_y(top, num_rows=2)
    assert len(rows[0]) == 3 and rows[1] == []


def top_dets():
    # one bin centred in each of the 6 equal frame bands (band width = 1280/6)
    return [make_det(int((c + 0.5) * FRAME_W / 6), 200) for c in range(6)]


def bottom_dets(cols=(0, 1, 2)):
    # bottom row has 3 bins -> band width = 1280/3
    return [make_det(int((c + 0.5) * FRAME_W / 3), 560) for c in cols]


def test_full_detection_maps_to_1_through_9():
    res = ga.allocate_grid(top_dets() + bottom_dets(), FRAME_W)
    assert all(res[f"bin_{i}"]["detected"] for i in range(1, 10))
    assert res["bin_1"]["corners"] is not None
    assert res["bin_7"]["confidence"] == 0.9


def test_missing_middle_top_bin_leaves_that_index_empty():
    dets = [d for c, d in enumerate(top_dets()) if c != 2]  # drop band 2 -> index 3
    res = ga.allocate_grid(dets + bottom_dets(), FRAME_W)
    assert res["bin_3"]["detected"] is False        # the gap lands on index 3…
    assert res["bin_2"]["detected"] is True          # …neighbours unaffected
    assert res["bin_4"]["detected"] is True
    assert sum(res[f"bin_{i}"]["detected"] for i in range(1, 7)) == 5


def test_missing_middle_bottom_bin():
    res = ga.allocate_grid(top_dets() + bottom_dets(cols=(0, 2)), FRAME_W)
    assert res["bin_8"]["detected"] is False         # middle bottom = index 8
    assert res["bin_7"]["detected"] is True and res["bin_9"]["detected"] is True


def test_out_of_order_x_is_placed_by_position():
    res = ga.allocate_grid(list(reversed(top_dets())) + bottom_dets(), FRAME_W)
    assert all(res[f"bin_{i}"]["detected"] for i in range(1, 7))


def test_empty_detection_returns_full_placeholder_grid():
    res = ga.allocate_grid([], FRAME_W)
    assert len(res) == 9
    assert all(res[f"bin_{i}"]["detected"] is False for i in range(1, 10))


def test_extra_detection_in_row_is_dropped():
    extra = make_det(int(0.5 * FRAME_W / 6) + 3, 200)   # 7th box in 6-band top row
    res = ga.allocate_grid(top_dets() + [extra] + bottom_dets(), FRAME_W)
    assert len(res) == 9
    assert all(res[f"bin_{i}"]["detected"] for i in range(1, 7))


def test_degenerate_zero_height_single_row_not_split():
    # boxes with zero usable height sharing one y must stay one row, not force-split
    dets = [{"corners": [], "center": [x, 200.0], "conf": 0.9}
            for x in (100, 500, 900)]
    rows = ga.split_rows_by_y(dets, num_rows=2)
    assert len(rows[0]) == 3 and rows[1] == []


def test_non_finite_center_detections_are_dropped():
    bad = {"corners": [[0, 0]], "center": [float("nan"), 200.0], "conf": 0.9}
    res = ga.allocate_grid(top_dets() + [bad] + bottom_dets(), FRAME_W)
    # the NaN detection is ignored; the real 6+3 grid still maps cleanly to 1..9
    assert all(res[f"bin_{i}"]["detected"] for i in range(1, 10))


def test_index_bins_pure_transform():
    import snapshot_obb as so
    payload = {
        "frame_w": FRAME_W, "frame_h": 720,
        "bins": [
            {"id": 0, "corners": [[10, 190], [30, 190], [30, 210], [10, 210]],
             "center": [int(0.5 * FRAME_W / 6), 200], "conf": 0.8},
            {"id": 1, "corners": [[10, 550], [30, 550], [30, 570], [10, 570]],
             "center": [int(0.5 * FRAME_W / 3), 560], "conf": 0.7},
        ],
    }
    out = so.index_bins(payload)
    assert out["bins"]["bin_1"]["detected"] is True       # top-left small bin
    assert out["bins"]["bin_7"]["detected"] is True       # bottom-left big bin
    assert out["bins"]["bin_2"]["detected"] is False      # nothing there
    assert out["rule"] == "rotate180_then_band_split"
