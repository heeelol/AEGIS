#!/usr/bin/env python3
"""
Buzzer discovery / test — run ON THE MIC.
=========================================
Finds which gpiochip/line is your DO channel and confirms the buzzer beeps,
using the SAME driver the pipeline uses. Once you know the right values, paste
them into settings.yaml under `sensing.buzzer`.

Examples
--------
  # 1) List the GPIO chips and their lines (look for the DO / DIO lines):
  python3 integration/tools/buzzer_test.py --list

  # 2) Beep 3× on a candidate channel:
  python3 integration/tools/buzzer_test.py --chip gpiochip0 --line 5

  # 3) If it beeps when IDLE and goes quiet when driven, add --active-low:
  python3 integration/tools/buzzer_test.py --chip gpiochip0 --line 5 --active-low

Electrical read-out
-------------------
  * Silent on connect, beeps during the test  → wiring/polarity correct.
  * Beeps immediately on connect (before/after the test, line released)
    → that pin conducts at idle (sink-type DO or a GND pin); MIC-direct
      can't give silent-from-boot — use the ESP32-switched fallback.
  * Audible but weak → the DO is current-limited; drive the buzzer through
    an NPN transistor (DO → base). No software change.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make integration.src importable when run from the aegis-v2 root or elsewhere.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from integration.src.actuators.buzzer import Buzzer  # noqa: E402


def list_chips() -> None:
    """Print each gpiochip and its lines (name + consumer) via libgpiod."""
    try:
        import gpiod
    except ImportError:
        print("libgpiod (python3-gpiod) not installed. Try:\n"
              "  sudo apt-get install python3-libgpiod gpiod\n"
              "Then `gpiodetect` and `gpioinfo` also list chips/lines.")
        return

    # libgpiod's chip-enumeration API differs across versions; fall back to
    # scanning /dev for gpiochip* if the convenience iterator isn't present.
    names = []
    if hasattr(gpiod, "ChipIter"):
        names = [c.name() for c in gpiod.ChipIter()]
    else:
        names = sorted(n for n in os.listdir("/dev") if n.startswith("gpiochip"))

    for name in names:
        try:
            chip = gpiod.Chip(name)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: (could not open: {e})")
            continue
        num = chip.num_lines() if hasattr(chip, "num_lines") else "?"
        print(f"\n{name}  ({num} lines)")
        try:
            for i in range(chip.num_lines()):
                ln = chip.get_line(i)
                label = ln.name() or "-"
                consumer = ln.consumer() or "-"
                print(f"  line {i:3d}: name={label!s:20s} consumer={consumer}")
        except Exception as e:  # noqa: BLE001
            print(f"  (could not enumerate lines: {e})")
        finally:
            chip.close()


def beep(chip: str, line: int, active_low: bool, times: int) -> None:
    cfg = {
        "enabled": True,
        "gpiochip": chip,
        "line": line,
        "active_low": active_low,
        "pulse_hz": 2.0,
    }
    buz = Buzzer(cfg).start()
    print(f"Beeping {times}× on {chip} line {line} "
          f"(active_low={active_low}). Ctrl-C to stop early.")
    try:
        for i in range(times):
            print(f"  beep {i + 1}/{times} — ON")
            buz.set_alarm(True)
            time.sleep(0.5)
            print(f"  beep {i + 1}/{times} — off")
            buz.set_alarm(False)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        buz.close()
        print("Line released and driven silent.")


def main() -> None:
    p = argparse.ArgumentParser(description="MIC DIO buzzer discovery / test")
    p.add_argument("--list", action="store_true",
                   help="list gpiochips and their lines, then exit")
    p.add_argument("--chip", default="gpiochip0", help="gpiochip name")
    p.add_argument("--line", type=int, help="line offset to beep")
    p.add_argument("--active-low", action="store_true",
                   help="buzzer sounds on a LOW line (inverts drive)")
    p.add_argument("--times", type=int, default=3, help="number of beeps")
    args = p.parse_args()

    if args.list:
        list_chips()
        return
    if args.line is None:
        p.error("give --line N to beep, or --list to discover channels")
    beep(args.chip, args.line, args.active_low, args.times)


if __name__ == "__main__":
    main()
