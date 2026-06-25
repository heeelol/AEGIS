"""Tests for the sequential single-bin kitting FSM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # integration/

from src.sensing.placement import PlacementTracker

UNITS = {"bin_0_4": 3.6, "bin_0_5": 16.6, "bin_1_2": 67.1}
TARGETS = {"bin_0_4": 3, "bin_0_5": 3, "bin_1_2": 3}
BOX = "kit_box"


def tracker():
    return PlacementTracker(UNITS, TARGETS, BOX, ema_alpha=0.4, hysteresis=0.25,
                            box_step_tolerance_g=1.8, box_step_fraction=0.15,
                            activation_frac=0.5)


def w(b04=0.0, b05=0.0, b12=0.0, box=0.0):
    return {"bin_0_4": b04, "bin_0_5": b05, "bin_1_2": b12, "kit_box": box}


def settle(t, weights, n=25):
    s = None
    for _ in range(n):
        s = t.update(weights)
    return s


def test_idle_then_activation():
    t = tracker()
    assert settle(t, w()).state == "IDLE"
    s = settle(t, w(b04=-3.6))           # first clear drop activates bin_0_4
    assert s.active == "bin_0_4" and s.state == "PICKING"
    assert s.placed["bin_0_4"] == 0      # removed but not yet in the box


def test_only_active_bin_counts_on_placement():
    t = tracker()
    settle(t, w(b04=-3.6))               # activate bin_0_4
    s = settle(t, w(b04=-3.6, box=3.6))  # place it
    assert s.placed["bin_0_4"] == 1 and s.placed["bin_0_5"] == 0


def test_weight_variance_tolerated():
    # 67 g item reads 65 g in the box -> still exactly 1 (no cross-bin confusion).
    t = tracker()
    s = settle(t, w(b12=-67.1, box=65.0))
    assert s.active == "bin_1_2" and s.placed["bin_1_2"] == 1


def test_bank_and_advance_to_next_bin():
    t = tracker()
    s = settle(t, w(b04=-10.8, box=10.8))     # finish bin_0_4 (3/3)
    assert s.done == ["bin_0_4"] and s.active is None and s.state == "IDLE"
    # now start bin_0_5; box delta measured from the banked baseline
    s = settle(t, w(b04=-10.8, b05=-16.6, box=27.4))
    assert s.active == "bin_0_5" and s.placed["bin_0_5"] == 1


def test_overpick_badge_not_fault():
    # 4 taken from active bin, 3 placed -> overpick (return), not a fault.
    t = tracker()
    s = settle(t, w(b12=-4 * 67.1, box=3 * 67.1))
    assert s.overpick == {"bin_1_2": 1} and s.alert is None and s.state == "PICKING"


def test_overpack_fault():
    t = tracker()
    s = settle(t, w(b12=-4 * 67.1, box=4 * 67.1))
    assert s.state == "FAULT" and s.alert["type"] == "overpack-kit"


def test_pick_from_wrong_bin_fault():
    t = tracker()
    settle(t, w(b04=-3.6))                       # bin_0_4 active
    s = settle(t, w(b04=-3.6, b05=-16.6))        # touch a locked bin
    assert s.state == "FAULT" and s.alert["type"] == "pick-from-wrong-bin"
    assert s.alert["bin"] == "bin_0_5"


def test_return_to_wrong_bin_fault():
    t = tracker()
    settle(t, w(b04=-3.6))                        # bin_0_4 active
    s = settle(t, w(b04=-3.6, b05=+20.0))         # item dropped INTO a locked bin
    assert s.state == "FAULT" and s.alert["type"] == "return-to-wrong-bin"


def test_remove_from_kit_fault():
    t = tracker()
    settle(t, w(b12=-67.1, box=67.1))             # place 1 (holding==0)
    s = settle(t, w(b12=-67.1, box=0.0))          # box emptied with empty hands
    assert s.state == "FAULT" and s.alert["type"] == "remove-from-kit"


def test_fault_auto_clears():
    t = tracker()
    settle(t, w(b04=-3.6))
    assert settle(t, w(b04=-3.6, b05=-16.6)).state == "FAULT"   # wrong bin
    s = settle(t, w(b04=-3.6, b05=0.0))           # extra returned to the locked bin
    assert s.alert is None and s.state == "PICKING"


def test_full_kit_completes():
    t = tracker()
    settle(t, w(b04=-10.8, box=10.8))                          # bin_0_4 done
    settle(t, w(b04=-10.8, b05=-49.8, box=60.6))               # bin_0_5 done
    s = settle(t, w(b04=-10.8, b05=-49.8, b12=-201.3, box=261.9))  # bin_1_2 done
    assert sorted(s.done) == ["bin_0_4", "bin_0_5", "bin_1_2"]
    assert s.complete and s.state == "KIT_COMPLETE"


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
