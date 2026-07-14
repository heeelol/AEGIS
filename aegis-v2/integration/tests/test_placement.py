"""Tests for the sequential single-bin kitting FSM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # integration/

from src.sensing.placement import PlacementTracker

UNITS = {"bin_0_4": 3.6, "bin_0_5": 16.6, "bin_1_2": 67.1}
TARGETS = {"bin_0_4": 3, "bin_0_5": 3, "bin_1_2": 3}
BOX = "kit_box"


def tracker():
    # fault_settle_s=0 / activation_confirm_s=0: tests run in milliseconds of
    # real wall-clock time, well under any production window (see
    # test_fault_settle_window.py / test_activation_confirm_window.py for
    # those windows' own dedicated tests) — would otherwise suppress every
    # fault, and prevent any bin from ever activating, in this file.
    return PlacementTracker(UNITS, TARGETS, BOX, ema_alpha=0.4, hysteresis=0.25,
                            box_step_tolerance_g=1.8, activation_frac=0.5,
                            fault_settle_s=0, activation_confirm_s=0)


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


def test_overpack_triggers_fault_and_clears():
    # 4 taken AND 4 placed (target 3) -> real overpack, not just an overpick badge.
    t = tracker()
    s = settle(t, w(b12=-4 * 67.1, box=4 * 67.1))
    assert s.state == "FAULT" and s.alert["type"] == "overpack-kit" and s.alert["bin"] == "bin_1_2"
    # message (corrective-action subtitle; the UI adds the "OVERPACK" keyword)
    # names the specific bin to return the excess to, not just "the kit"
    assert s.alert["message"] == "Remove 1 from kit, return to bin_1_2"
    # correct it: return the extra item to ITS BIN (both removed and box drop
    # together) -> fault auto-clears and the bin completes cleanly. Merely
    # lifting the item out of the box without a bin gaining it back doesn't
    # clear — see test_overpack_recovery_requires_returning_to_the_bin.
    s = settle(t, w(b12=-3 * 67.1, box=3 * 67.1))
    assert s.alert is None and s.done == ["bin_1_2"]


def test_overpack_clears_to_overpick_when_only_the_box_is_corrected():
    # Taking the extra item out of the box WITHOUT it landing back in the bin's
    # own cell (removed count unchanged) still resolves cleanly: `placed` just
    # tracks the box's decrease like any other change (no more freeze-on-decrease
    # — see the removed remove-from-kit check), dropping the hard overpack fault
    # down to a soft overpick badge (the bin still shows 1 outstanding to return).
    t = tracker()
    settle(t, w(b12=-4 * 67.1, box=4 * 67.1))
    s = settle(t, w(b12=-4 * 67.1, box=3 * 67.1))
    assert s.alert is None and s.state == "PICKING" and s.overpick == {"bin_1_2": 1}


def test_overpack_blocks_completion_while_faulted():
    # Even though placed==removed==target would normally complete the bin, an
    # active overpack fault must block it (bin isn't actually correct yet).
    t = tracker()
    s = settle(t, w(b04=-4 * 3.6, box=4 * 3.6))   # 4/4, target 3 -> overpack
    assert s.alert["type"] == "overpack-kit" and "bin_0_4" not in s.done


def test_out_of_sequence_triggers_fault_and_clears():
    # bin_0_5 IS in this kit (target 3) but is locked while bin_0_4 is active.
    # Picking from it is out-of-sequence, NOT "wrong bin" — it's a required bin,
    # just not right now.
    t = tracker()
    settle(t, w(b04=-3.6))                       # bin_0_4 active
    s = settle(t, w(b04=-3.6, b05=-16.6))        # pick from a locked in-BOM bin
    assert (s.state == "FAULT" and s.alert["type"] == "out-of-sequence"
            and s.alert["bin"] == "bin_0_5")
    # subtitle: how many to put back, and to finish the active bin first
    assert s.alert["message"] == "Return 1 to bin_0_5 — finish bin_0_4 first"
    assert s.alert["count"] == 1
    # correct it: return the item to bin_0_5 -> fault auto-clears
    s = settle(t, w(b04=-3.6, b05=0.0))
    assert s.alert is None and s.state == "PICKING"


def test_pick_from_completed_green_bin_is_wrong_bin_not_out_of_sequence():
    # Once a bin is finished (green/done) it must never be picked from again —
    # that's a real WRONG BIN, not a timing/out-of-sequence slip.
    t = tracker()
    s = settle(t, w(b04=-3 * 3.6, box=3 * 3.6))   # complete bin_0_4 (3/3) -> done
    assert s.done == ["bin_0_4"] and s.active is None
    s = settle(t, w(b04=-4 * 3.6, box=3 * 3.6))   # take a 4th from the done bin
    assert s.state == "FAULT" and s.alert["type"] == "pick-from-wrong-bin"
    assert s.alert["bin"] == "bin_0_4" and s.alert["count"] == 1
    assert s.alert["message"] == "Return 1 to bin_0_4"


def test_pick_from_not_in_bom_bin_always_faults_and_never_activates():
    # A bin NOT in this kit (target 0) must always error when picked from — even
    # as the very first action, with nothing active — and must never silently
    # become the active bin (which used to swallow the first wrong pick).
    units = {"bin_0_4": 3.6, "bin_0_5": 16.6}
    targets = {"bin_0_4": 3, "bin_0_5": 0}       # bin_0_5 not in this kit
    t = PlacementTracker(units, targets, BOX, ema_alpha=0.4, hysteresis=0.25,
                         box_step_tolerance_g=1.8, activation_frac=0.5,
                         fault_settle_s=0, activation_confirm_s=0)
    s = settle(t, {"bin_0_4": 0.0, "bin_0_5": -16.6, "kit_box": 0.0})  # first action
    assert s.state == "FAULT" and s.alert["type"] == "pick-from-wrong-bin"
    assert s.alert["bin"] == "bin_0_5" and s.alert["count"] == 1
    assert s.active is None                       # never activated the wrong bin
    # correct it: return the item -> fault clears, still idle
    s = settle(t, {"bin_0_4": 0.0, "bin_0_5": 0.0, "kit_box": 0.0})
    assert s.alert is None and s.active is None


def test_return_to_wrong_bin_triggers_fault_and_clears():
    t = tracker()
    settle(t, w(b04=-3.6))                        # bin_0_4 active
    s = settle(t, w(b04=-3.6, b05=+20.0))         # item dropped INTO a locked bin
    assert (s.state == "FAULT" and s.alert["type"] == "return-to-wrong-bin"
            and s.alert["bin"] == "bin_0_5")
    # subtitle says REMOVE (not "return to", which would misread as instructing
    # the operator to put it there — it's already there, wrongly), how many
    # (5 units of the held item's weight landed here), and names bin_0_4
    # (currently active) as where it actually belongs
    assert s.alert["message"] == "Remove 5 from bin_0_5, belongs in bin_0_4"
    assert s.alert["count"] == 5
    # correct it: take the item back out of bin_0_5 -> fault auto-clears
    s = settle(t, w(b04=-3.6, b05=0.0))
    assert s.alert is None and s.state == "PICKING"


def test_return_to_wrong_done_bin_fires_even_for_a_much_lighter_held_item():
    # Regression: bin_1_2 (67.1g item) completes first. Later, bin_0_4's much
    # lighter item (3.6g) is held and wrongly returned to the now-DONE
    # bin_1_2. Since bin_1_2's own resting point (post-completion) is well
    # below its original zero tare, a naive "is the reading above zero"
    # check would never see the light item's small increase as anything --
    # it must be measured against bin_1_2's own EXPECTED (already-emptied)
    # resting point instead.
    t = tracker()
    settle(t, w(b12=-3 * 67.1))                     # bin_1_2 active, target 3
    s = settle(t, w(b12=-3 * 67.1, box=3 * 67.1))   # bin_1_2 completes
    assert s.done == ["bin_1_2"]
    settle(t, w(b12=-3 * 67.1, b04=-3.6, box=3 * 67.1))          # bin_0_4 active, holding
    s = settle(t, w(b12=-3 * 67.1 + 3.6, b04=-3.6, box=3 * 67.1))  # wrongly returned to done bin_1_2
    assert s.alert is not None and s.alert["type"] == "return-to-wrong-bin"
    assert s.alert["bin"] == "bin_1_2"


def test_no_fault_while_idle_since_nothing_is_in_hand():
    # With nothing active, "holding" is always 0 -- there's no known item to
    # misplace, so a weight change on an available bin is never flagged (an
    # intentional trade-off: fewer false positives, at the cost of not
    # catching an out-of-band event with no operator activity to explain it).
    t = tracker()
    s = settle(t, w(b05=+20.0))    # dropped into bin_0_5 while station is idle
    assert s.alert is None


def test_small_cross_talk_blip_does_not_match_held_item_and_is_ignored():
    # A neighbor's weight increase that settles to something much smaller than
    # one unit of what's actually in-hand (classic mechanical cross-talk, e.g.
    # a real pick flexing a shared shelf) must NOT be mistaken for a genuine
    # return -- it only matches bin_0_5's own (much larger) item weight, not
    # the held bin_0_4 item's.
    t = tracker()
    settle(t, w(b04=-3.6))                      # bin_0_4 active, 1 unit (3.6g) in hand
    s = settle(t, w(b04=-3.6, b05=+2.0))        # bin_0_5 settles at a tiny +2g blip
    assert s.alert is None and s.active == "bin_0_4"


def test_steady_state_gate_blocks_fault_while_bin_still_fluctuating():
    # A bin whose reading is actively oscillating (raw never catches up to its
    # own EMA) is invisible to the wrong-bin check entirely, even if its
    # average would otherwise match the held item's weight -- this is the
    # "still mid-bump/press" case, not a settled reading.
    t = tracker()
    settle(t, w(b04=-3.6))    # bin_0_4 active, holding=1
    s = None
    for i in range(20):
        s = t.update(w(b04=-3.6, b05=(20.0 if i % 2 == 0 else -20.0)))
    assert s.alert is None


def test_steady_state_does_not_flap_under_small_ongoing_noise():
    # Regression: real sensor noise near the steady-state tolerance used to
    # make the settled flag flip every frame, flashing the alert on and off
    # (see the 3x hysteresis dead-band in update()). Alternating a small
    # in-tolerance-then-just-over-tolerance reading on bin_0_5 must not toggle
    # the alert once it's genuinely settled at a value matching the held item.
    t = tracker()
    settle(t, w(b04=-3.6))    # bin_0_4 active, holding=1
    s = settle(t, w(b04=-3.6, b05=+3.6))    # bin_0_5 settles at a matching return
    assert s.alert is not None
    flips = 0
    prev = s.alert is not None
    for i in range(20):
        jitter = 0.35 if i % 2 == 0 else -0.35   # straddles the raw tolerance, not the 3x exit band
        s = t.update(w(b04=-3.6, b05=3.6 + jitter))
        now = s.alert is not None
        if now != prev:
            flips += 1
        prev = now
    assert flips == 0 and s.alert is not None


def test_settled_return_matching_held_item_weight_still_flags():
    # Sanity check alongside the above: a genuine return, settling at roughly
    # one unit of the ACTIVE item's weight, still flags correctly.
    t = tracker()
    settle(t, w(b04=-3.6))
    s = settle(t, w(b04=-3.6, b05=+3.6))        # matches bin_0_4's OWN unit weight
    assert s.alert is not None and s.alert["type"] == "return-to-wrong-bin"


def test_wrong_bin_fault_does_not_corrupt_active_bin_and_it_still_completes():
    t = tracker()
    settle(t, w(b04=-3.6))
    assert settle(t, w(b04=-3.6, b05=-16.6)).state == "FAULT"   # wrong-bin pick
    settle(t, w(b04=-3.6, b05=0.0))                             # corrected, fault clears
    # bin_0_4 can still be finished normally afterward
    s = settle(t, w(b04=-10.8, b05=0.0, box=10.8))
    assert s.alert is None and s.done == ["bin_0_4"]


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
