"""Tests for the per-receptor kitting placement tracker (robust per-bin counting)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # integration/

from src.sensing.placement import PlacementTracker

# Demo config: each bin counted from its OWN cell; box is cross-check only.
UNITS = {"bin_0_4": 3.6, "bin_0_5": 16.6, "bin_1_2": 67.1}
TARGETS = {"bin_0_4": 3, "bin_0_5": 3, "bin_1_2": 3}
BOX = "kit_box"
EXPECTED = 3 * 3.6 + 3 * 16.6 + 3 * 67.1  # 261.9


def tracker():
    return PlacementTracker(UNITS, TARGETS, BOX, ema_alpha=0.4, hysteresis=0.25)


def settle(t, weights, n=20):
    """Feed the same weights repeatedly so the EMA converges; return last state."""
    s = None
    for _ in range(n):
        s = t.update(weights)
    return s


def test_empty_is_init():
    s = settle(tracker(), {"bin_0_4": 0, "bin_0_5": 0, "bin_1_2": 0, "kit_box": 0})
    assert s.placed == {"bin_0_4": 0, "bin_0_5": 0, "bin_1_2": 0}
    assert s.state == "INIT" and not s.complete


def test_small_item_counts_from_own_cell():
    # The whole point of the rewrite: a 3.6 g item must count, with NO box weight.
    s = settle(tracker(), {"bin_0_4": -3.6, "bin_0_5": 0, "bin_1_2": 0, "kit_box": 0})
    assert s.placed["bin_0_4"] == 1
    assert s.state == "PICKING"


def test_all_three_bins_count_mixed():
    s = settle(tracker(), {"bin_0_4": -7.2, "bin_0_5": -33.2, "bin_1_2": -67.1, "kit_box": 0})
    assert s.placed == {"bin_0_4": 2, "bin_0_5": 2, "bin_1_2": 1}


def test_hysteresis_no_flicker():
    t = tracker()
    settle(t, {"bin_0_4": -3.6, "bin_0_5": 0, "bin_1_2": 0, "kit_box": 0})
    assert t.update({"bin_0_4": -3.6}).placed["bin_0_4"] == 1
    # Jitter ±1 g around one item must not move the count.
    for w in (-2.8, -4.4, -3.1, -4.1, -3.6, -2.9):
        s = t.update({"bin_0_4": w})
        assert s.placed["bin_0_4"] == 1, (w, s.placed)


def test_half_item_does_not_count():
    # A reading at ~0.5 item (below the 0.75 hysteresis threshold) stays at 0.
    s = settle(tracker(), {"bin_0_4": -1.8, "bin_0_5": 0, "bin_1_2": 0, "kit_box": 0})
    assert s.placed["bin_0_4"] == 0


def test_full_kit_completes_and_box_verifies():
    s = settle(tracker(), {"bin_0_4": -10.8, "bin_0_5": -49.8, "bin_1_2": -201.3, "kit_box": EXPECTED})
    assert s.placed == {"bin_0_4": 3, "bin_0_5": 3, "bin_1_2": 3}
    assert s.complete and s.state == "KIT_COMPLETE"
    assert s.box_verified


def test_completion_independent_of_box():
    # Counts met but box empty -> still complete (box never gates), just not verified.
    s = settle(tracker(), {"bin_0_4": -10.8, "bin_0_5": -49.8, "bin_1_2": -201.3, "kit_box": 0})
    assert s.complete
    assert not s.box_verified


def test_overpick_flagged():
    s = settle(tracker(), {"bin_0_4": 0, "bin_0_5": 0, "bin_1_2": -4 * 67.1, "kit_box": 0})
    assert s.placed["bin_1_2"] == 4
    assert s.overpick == {"bin_1_2": 1}
    assert s.state == "OVERPICK" and not s.complete


def test_software_tare_resets():
    t = tracker()
    settle(t, {"bin_0_4": -10.8, "bin_0_5": -49.8, "bin_1_2": -201.3, "kit_box": EXPECTED})
    t.tare({"bin_0_4": -10.8, "bin_0_5": -49.8, "bin_1_2": -201.3, "kit_box": EXPECTED})
    s = settle(t, {"bin_0_4": -10.8, "bin_0_5": -49.8, "bin_1_2": -201.3, "kit_box": EXPECTED})
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
