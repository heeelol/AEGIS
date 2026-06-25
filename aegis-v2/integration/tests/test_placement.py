"""Tests for box-verified, conservation-based kitting counting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # integration/

from src.sensing.placement import PlacementTracker

UNITS = {"bin_0_4": 3.6, "bin_0_5": 16.6, "bin_1_2": 67.1}
TARGETS = {"bin_0_4": 3, "bin_0_5": 3, "bin_1_2": 3}
BOX = "kit_box"
EXPECTED = 3 * 3.6 + 3 * 16.6 + 3 * 67.1  # 261.9


def tracker():
    return PlacementTracker(UNITS, TARGETS, BOX, ema_alpha=0.4, hysteresis=0.25,
                            box_step_tolerance_g=1.8)


def settle(t, weights, n=25):
    """Drive the same weights until the EMA converges; return the last state."""
    s = None
    for _ in range(n):
        s = t.update(weights)
    return s


def w(b04=0.0, b05=0.0, b12=0.0, box=0.0):
    return {"bin_0_4": b04, "bin_0_5": b05, "bin_1_2": b12, "kit_box": box}


def test_empty_is_init():
    s = settle(tracker(), w())
    assert s.placed == {"bin_0_4": 0, "bin_0_5": 0, "bin_1_2": 0}
    assert s.state == "INIT" and not s.complete


def test_removed_but_not_in_box_does_not_count():
    # Item lifted out of the bin but NOT placed in the box -> no count (verified-only).
    s = settle(tracker(), w(b04=-3.6, box=0))
    assert s.removed["bin_0_4"] == 1
    assert s.placed["bin_0_4"] == 0          # not counted until verified in the box
    assert s.state == "PICKING"              # work started (item out), just not placed yet


def test_small_item_counts_when_verified_in_box():
    s = settle(tracker(), w(b04=-3.6, box=3.6))
    assert s.placed["bin_0_4"] == 1 and s.state == "PICKING"


def test_small_item_counts_on_top_of_heavy_box():
    # Robustness: 3 big items already in the box (201 g); a 3.6 g step still counts.
    t = tracker()
    settle(t, w(b12=-201.3, box=201.3))
    s = settle(t, w(b04=-3.6, b12=-201.3, box=204.9))
    assert s.placed == {"bin_0_4": 1, "bin_0_5": 0, "bin_1_2": 3}


def test_mixed_conservation():
    s = settle(tracker(), w(b04=-3.6, b05=-16.6, b12=-67.1, box=87.3))
    assert s.placed == {"bin_0_4": 1, "bin_0_5": 1, "bin_1_2": 1}


def test_held_then_placed():
    t = tracker()
    s = settle(t, w(b12=-67.1, box=0))          # removed, held in hand
    assert s.placed["bin_1_2"] == 0
    s = settle(t, w(b12=-67.1, box=67.1))        # now in the box
    assert s.placed["bin_1_2"] == 1


def test_return_to_bin_uncredits():
    t = tracker()
    settle(t, w(b04=-3.6, box=3.6))
    assert t.update(w(b04=-3.6, box=3.6)).placed["bin_0_4"] == 1
    s = settle(t, w(b04=0, box=0))               # item back in the bin, box empty
    assert s.placed["bin_0_4"] == 0


def test_hysteresis_no_flicker():
    t = tracker()
    settle(t, w(b04=-3.6, box=3.6))
    for jb, jx in [(-2.9, 3.0), (-4.3, 4.2), (-3.1, 3.1), (-4.0, 4.0)]:
        s = t.update(w(b04=jb, box=jx))
        assert s.placed["bin_0_4"] == 1


def test_overpick_returns_to_bin_not_red():
    # 4 taken out of the bin but only 3 placed in the box (one held back).
    # removed > target -> overpick (return to bin); placed == target -> NO overpack, NO red.
    s = settle(tracker(), w(b12=-4 * 67.1, box=3 * 67.1))
    assert s.removed["bin_1_2"] == 4 and s.placed["bin_1_2"] == 3
    assert s.overpick == {"bin_1_2": 1} and s.overpack == {}
    assert s.alert is None


def test_overpack_triggers_red_alert():
    # 4 placed in the box (target 3) -> overpack -> full-red event.
    s = settle(tracker(), w(b12=-4 * 67.1, box=4 * 67.1))
    assert s.placed["bin_1_2"] == 4
    assert s.overpack == {"bin_1_2": 1}
    assert s.state == "OVERPACK" and s.alert and s.alert["type"] == "overpack-kit"


def test_large_item_counts_with_noisy_box():
    # The box reads ~6 g low for the 67 g item; the per-item step tolerance
    # (scales with unit) still credits it — the "large bin inaccurate" fix.
    s = settle(tracker(), w(b12=-67.1, box=61.0))
    assert s.placed["bin_1_2"] == 1


def test_full_kit_completes():
    s = settle(tracker(), w(b04=-10.8, b05=-49.8, b12=-201.3, box=EXPECTED))
    assert s.placed == {"bin_0_4": 3, "bin_0_5": 3, "bin_1_2": 3}
    assert s.complete and s.state == "KIT_COMPLETE" and s.box_verified


def test_software_tare_resets():
    t = tracker()
    settle(t, w(b04=-10.8, b05=-49.8, b12=-201.3, box=EXPECTED))
    t.tare(w(b04=-10.8, b05=-49.8, b12=-201.3, box=EXPECTED))
    s = settle(t, w(b04=-10.8, b05=-49.8, b12=-201.3, box=EXPECTED))
    assert s.placed == {"bin_0_4": 0, "bin_0_5": 0, "bin_1_2": 0}
    assert s.box_grams == 0.0


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
