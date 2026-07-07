"""Unit tests for the fault Buzzer driver.

Pure — no real GPIO. A fake backend records every raw line level written, so
we can assert on silence-at-start, pulsing, forced-off, polarity, and
fail-soft behaviour without any hardware.
"""
import os
import sys
import threading
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from integration.src.actuators.buzzer import Buzzer  # noqa: E402


class FakeBackend:
    """Records raw line levels; thread-safe for the pulser thread."""

    def __init__(self):
        self.levels = []          # every value written, in order
        self.released = False
        self._lock = threading.Lock()

    def set(self, value):
        with self._lock:
            self.levels.append(int(value))

    def release(self):
        self.released = True

    # Helpers for assertions
    def last(self):
        with self._lock:
            return self.levels[-1] if self.levels else None

    def any_high(self):
        with self._lock:
            return any(v == 1 for v in self.levels)


def _buzzer(backend, **over):
    cfg = {"enabled": True, "gpiochip": "gpiochip0", "line": 0,
           "active_low": False, "pulse_hz": 10.0}
    cfg.update(over)
    return Buzzer(cfg, backend=backend)


def test_disabled_never_touches_backend():
    fake = FakeBackend()
    buz = Buzzer({"enabled": False}, backend=fake).start()
    buz.set_alarm(True)
    time.sleep(0.05)
    buz.close()
    assert fake.levels == []            # nothing written at all


def test_starts_silent():
    fake = FakeBackend()
    buz = _buzzer(fake).start()
    try:
        time.sleep(0.05)
        # No alarm yet: everything written so far is the silent level (0).
        assert fake.any_high() is False
        assert fake.last() == 0
    finally:
        buz.close()


def test_alarm_pulses_the_line():
    fake = FakeBackend()
    buz = _buzzer(fake).start()
    try:
        buz.set_alarm(True)
        time.sleep(0.3)                 # several half-periods at 10 Hz
        # Pulsing means both levels appear.
        assert fake.any_high() is True
        assert 0 in fake.levels
    finally:
        buz.close()


def test_alarm_off_forces_silent():
    fake = FakeBackend()
    buz = _buzzer(fake).start()
    try:
        buz.set_alarm(True)
        time.sleep(0.2)
        buz.set_alarm(False)
        time.sleep(0.1)
        assert fake.last() == 0         # settled silent
    finally:
        buz.close()


def test_close_forces_silent_and_releases():
    fake = FakeBackend()
    buz = _buzzer(fake).start()
    buz.set_alarm(True)
    time.sleep(0.1)
    buz.close()
    assert fake.last() == 0
    assert fake.released is True


def test_active_low_inverts_levels():
    fake = FakeBackend()
    buz = _buzzer(fake, active_low=True).start()
    try:
        time.sleep(0.05)
        # Silent on an active-low line is a HIGH raw level.
        assert fake.last() == 1
        assert 0 not in fake.levels     # never driven low while silent
    finally:
        buz.close()


def test_no_backend_is_a_silent_noop():
    # enabled but backend open fails → start() finds nothing and no-ops.
    buz = Buzzer({"enabled": True, "gpiochip": "gpiochip-does-not-exist",
                  "line": 999})
    buz.start()                         # must not raise
    buz.set_alarm(True)                 # must not raise
    buz.close()                         # must not raise


def test_set_alarm_before_start_is_ignored():
    fake = FakeBackend()
    buz = _buzzer(fake)
    buz.set_alarm(True)                 # not started yet
    assert fake.levels == []
    buz.start()
    buz.close()
