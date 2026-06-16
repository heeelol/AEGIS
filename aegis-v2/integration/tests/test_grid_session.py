"""Unit tests for grid_session — pure geometry, no camera/model/cv2.

Imported by path so the test doesn't drag in the detectors package __init__.
grid_session itself puts the detectors dir on sys.path for grid_calibrator.
"""
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "detectors")),
)

import pytest  # noqa: E402
import grid_session as gs  # noqa: E402

FRAME_W = 1280


def make_det(cx, cy, conf=0.9, half=20):
    return {
        "corners": [[cx - half, cy - half], [cx + half, cy - half],
                    [cx + half, cy + half], [cx - half, cy + half]],
        "center": [float(cx), float(cy)],
        "conf": conf,
    }


def top_dets():
    # 6 bins across the top band (y=200), evenly spaced
    return [make_det(int((c + 0.5) * FRAME_W / 6), 200) for c in range(6)]


def bottom_dets():
    # 3 big bins across the bottom band (y=560)
    return [make_det(int((c + 0.5) * FRAME_W / 3), 560) for c in range(3)]


def full():
    return top_dets() + bottom_dets()


# ── Calibrate / init_kit lifecycle ───────────────────────────

def test_calibrate_then_init_kit_all_present():
    session = gs.GridSession()
    session.calibrate(full())
    assert session.calibrated
    session.init_kit(full())
    assert session.present_slots == list(range(1, 10))
    assert session.missing_slots == []

    geo = session.to_geofences()
    assert len(geo) == 9
    assert all(g["detected"] for g in geo.values())


def test_init_kit_with_missing_bin():
    session = gs.GridSession()
    session.calibrate(full())
    # Drop the middle bottom bin from the kit snapshot -> slot 8 (bin_1_1) missing.
    kit = top_dets() + [bottom_dets()[0], bottom_dets()[2]]
    session.init_kit(kit)

    assert 8 in session.missing_slots
    assert 8 not in session.present_slots

    geo = session.to_geofences()
    assert geo["bin_1_1"]["detected"] is False
    assert geo["bin_1_0"]["detected"] is True
    assert geo["bin_1_2"]["detected"] is True


def test_init_kit_before_calibrate_raises():
    session = gs.GridSession()
    with pytest.raises(RuntimeError):
        session.init_kit(full())


def test_recalibration_clears_kit():
    session = gs.GridSession()
    session.calibrate(full())
    kit = top_dets() + [bottom_dets()[0], bottom_dets()[2]]
    session.init_kit(kit)
    assert session.missing_slots == [8]

    session.calibrate(full())          # recalibrate
    assert session.occupancy is None
    assert session.missing_slots == []           # no kit -> nothing missing
    assert session.present_slots == list(range(1, 10))


# ── to_geofences contract ────────────────────────────────────

def test_to_geofences_slot_mapping_and_boxes():
    session = gs.GridSession()
    session.calibrate(full())
    session.init_kit(full())
    geo = session.to_geofences()

    # Top row: 6 single-slot bins, slot_start == col, row_slots 6.
    top = geo["bin_0_0"]
    assert top["span"] == 1 and top["slot_start"] == 0 and top["row_slots"] == 6
    assert geo["bin_0_5"]["slot_start"] == 5
    # bbox of a half=20 box centred at x=(0.5)*1280/6≈106, y=200
    cx = int(0.5 * FRAME_W / 6)
    assert top["x_min"] == cx - 20 and top["x_max"] == cx + 20
    assert top["y_min"] == 180 and top["y_max"] == 220
    assert len(top["polygon"]) == 4

    # Bottom row: 3 double-slot bins, slot_start 0/2/4.
    assert geo["bin_1_0"]["span"] == 2 and geo["bin_1_0"]["slot_start"] == 0
    assert geo["bin_1_1"]["slot_start"] == 2
    assert geo["bin_1_2"]["slot_start"] == 4
    assert geo["bin_1_2"]["row_slots"] == 6


def test_present_only_filters_missing():
    session = gs.GridSession()
    session.calibrate(full())
    kit = top_dets() + [bottom_dets()[0], bottom_dets()[2]]
    session.init_kit(kit)

    all_bins = session.to_geofences(present_only=False)
    present = session.to_geofences(present_only=True)

    assert "bin_1_1" in all_bins
    assert "bin_1_1" not in present
    assert len(present) == 8


def test_calibrated_only_renders_locked_grid():
    """Before any kit, to_geofences shows all 9 slots as the locked grid."""
    session = gs.GridSession()
    session.calibrate(full())
    geo = session.to_geofences()
    assert len(geo) == 9
    assert all(g["detected"] for g in geo.values())
    # present_only is a no-op when nothing is missing
    assert len(session.to_geofences(present_only=True)) == 9
