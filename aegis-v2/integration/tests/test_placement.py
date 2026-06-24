"""Tests for the kitting-box placement tracker (3-load-receptor demo)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # integration/

from src.sensing.placement import PlacementTracker

# Demo config: bin_1_0 = 63.7 g item (x3), bin_0_0 = 3.6 g item (x3), box = kit_box.
UNITS = {"bin_1_0": 63.7, "bin_0_0": 3.6}
TARGETS = {"bin_1_0": 3, "bin_0_0": 3}
BOX = "kit_box"
EXPECTED = 3 * 63.7 + 3 * 3.6  # 201.9


def tracker():
    return PlacementTracker(UNITS, TARGETS, BOX)


def test_empty_is_init():
    s = tracker().update({"bin_1_0": 0, "bin_0_0": 0, "kit_box": 0})
    assert s.placed == {"bin_1_0": 0, "bin_0_0": 0}
    assert s.state == "INIT" and not s.complete


def test_removed_but_not_placed_does_not_count():
    # One big item lifted out of bin_1_0, but nothing in the box yet.
    s = tracker().update({"bin_1_0": -63.7, "bin_0_0": 0, "kit_box": 0})
    assert s.removed["bin_1_0"] == 1
    assert s.placed["bin_1_0"] == 0          # counter only moves on placement
    assert s.state == "INIT"


def test_single_placement_attributed_to_correct_bin():
    s = tracker().update({"bin_1_0": -63.7, "bin_0_0": 0, "kit_box": 63.7})
    assert s.placed == {"bin_1_0": 1, "bin_0_0": 0}
    assert s.state == "PICKING" and not s.complete


def test_distinct_weights_disambiguate():
    # Box ~63.7 g with only a big item removed must read as 1 big, not ~18 small.
    s = tracker().update({"bin_1_0": -63.7, "bin_0_0": 0, "kit_box": 63.5})
    assert s.placed == {"bin_1_0": 1, "bin_0_0": 0}


def test_full_kit_completes():
    s = tracker().update({"bin_1_0": -191.1, "bin_0_0": -10.8, "kit_box": EXPECTED})
    assert s.placed == {"bin_1_0": 3, "bin_0_0": 3}
    assert s.complete and s.state == "KIT_COMPLETE"


def test_targets_met_but_box_short_is_not_complete():
    # Bins emptied to target, but the box weight is missing an item -> not done.
    s = tracker().update({"bin_1_0": -191.1, "bin_0_0": -10.8, "kit_box": EXPECTED - 63.7})
    assert not s.complete


def test_overpick_flagged():
    # 4 big + 3 small placed; bin_1_0 target is 3.
    box = 4 * 63.7 + 3 * 3.6
    s = tracker().update({"bin_1_0": -4 * 63.7, "bin_0_0": -10.8, "kit_box": box})
    assert s.placed["bin_1_0"] == 4
    assert s.overpick == {"bin_1_0": 1}
    assert s.state == "OVERPICK" and not s.complete


def test_noise_tolerance():
    # ~1 g of noise on each cell should not change integer counts.
    s = tracker().update({"bin_1_0": -190.5, "bin_0_0": -11.2, "kit_box": 202.6})
    assert s.placed == {"bin_1_0": 3, "bin_0_0": 3}
    assert s.complete


def test_three_bins_with_near_equal_weights():
    # Real demo config: bin_0_4=3.6 g, bin_0_5=3.1 g (0.5 g apart), bin_1_2=67.1 g.
    units = {"bin_0_4": 3.6, "bin_0_5": 3.1, "bin_1_2": 67.1}
    tg = {"bin_0_4": 3, "bin_0_5": 3, "bin_1_2": 3}
    t = PlacementTracker(units, tg, "kit_box", 1.5)
    assert abs(t.expected_grams - 221.4) < 0.05

    # 2 from bin_0_4 + 1 from bin_0_5 placed; each bin's own cell bounds the count
    # so the near-equal weights don't cross-credit.
    s = t.update({"bin_0_4": -7.2, "bin_0_5": -3.1, "bin_1_2": 0, "kit_box": 10.3})
    assert s.placed == {"bin_0_4": 2, "bin_0_5": 1, "bin_1_2": 0}

    # Full kit completes.
    s = t.update({"bin_0_4": -10.8, "bin_0_5": -9.3, "bin_1_2": -201.3, "kit_box": 221.4})
    assert s.complete and s.placed == {"bin_0_4": 3, "bin_0_5": 3, "bin_1_2": 3}


def test_software_tare_resets_baseline():
    t = tracker()
    # Pretend the box is sitting at 202 g; tare makes that the new zero.
    t.tare({"bin_1_0": -191.1, "bin_0_0": -10.8, "kit_box": EXPECTED})
    s = t.update({"bin_1_0": -191.1, "bin_0_0": -10.8, "kit_box": EXPECTED})
    assert s.placed == {"bin_1_0": 0, "bin_0_0": 0}
    assert s.box_grams == 0.0 and s.state == "INIT"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
