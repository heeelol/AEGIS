"""
AEGIS v2 — Fault Buzzer (MIC-733-AO DIO)
========================================
Drives an active buzzer wired directly to the MIC's DIO terminal (buzzer live
→ a DO channel, buzzer ground → DIO GND). The buzzer sounds ONLY while the
kitting FSM is in a FAULT state; it is silent from power-on because a DO
channel idles low/floating and nothing sounds until the pipeline actively
drives the line to its "on" level.

Usage:
    buzzer = Buzzer(cfg)          # cfg = settings.yaml `sensing.buzzer` block
    buzzer.start()                # claims the line at the SILENT level
    ...
    buzzer.set_alarm(True)        # pulse at pulse_hz while a fault persists
    buzzer.set_alarm(False)       # back to silent
    ...
    buzzer.close()                # force silent + release the line

Fail-soft: any GPIO problem (missing libgpiod, wrong chip/line, no
permission) logs one warning and degrades to a no-op. The buzzer must NEVER
take the pipeline down. Set ``enabled: false`` in settings.yaml to skip GPIO
entirely.

Wire protocol / polarity
------------------------
``active_low: false`` (default): line HIGH = buzzer on, LOW = silent — the
expected wiring (DO sources current into the buzzer). ``active_low: true``
inverts, for the case where the hardware turns out to sound on a LOW line.

GPIO backend
------------
Tries ``gpiod`` (libgpiod v1 Python API, present on JetPack 6) first, then
the sysfs ``/sys/class/gpio`` interface. The backend is injectable for tests.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Protocol

logger = logging.getLogger("aegis.buzzer")


class GpioBackend(Protocol):
    """Minimal line-output interface the buzzer needs from a GPIO backend."""

    def set(self, value: int) -> None:
        """Drive the line to ``value`` (0 or 1), in raw electrical terms."""
        ...

    def release(self) -> None:
        """Release the line / free the backend."""
        ...


# ── GPIO backends ────────────────────────────────────────────────────────

def _open_backend(gpiochip: str, line: int) -> Optional[GpioBackend]:
    """Open the first working GPIO backend for (gpiochip, line), or None.

    Order: libgpiod (preferred on JetPack 6), then sysfs. Each attempt is
    isolated so a broken/absent backend just falls through to the next, and a
    total failure returns None (caller becomes a silent no-op).
    """
    for opener in (_open_gpiod, _open_sysfs):
        try:
            backend = opener(gpiochip, line)
            if backend is not None:
                return backend
        except Exception as e:  # noqa: BLE001 — every backend failure is soft
            logger.debug("Buzzer GPIO backend %s failed: %s", opener.__name__, e)
    return None


def _open_gpiod(gpiochip: str, line: int) -> Optional[GpioBackend]:
    import gpiod  # may be absent → caller falls through to sysfs

    chip = gpiod.Chip(gpiochip)
    line_obj = chip.get_line(int(line))
    # Request as output, starting LOW (raw). The Buzzer sets the true silent
    # level right after open via its first _drive() call.
    line_obj.request(consumer="aegis-buzzer",
                     type=gpiod.LINE_REQ_DIR_OUT,
                     default_vals=[0])

    class _GpiodBackend:
        def set(self, value: int) -> None:
            line_obj.set_value(1 if value else 0)

        def release(self) -> None:
            try:
                line_obj.release()
            finally:
                chip.close()

    logger.info("Buzzer GPIO via libgpiod: %s line %d", gpiochip, line)
    return _GpiodBackend()


def _open_sysfs(gpiochip: str, line: int) -> Optional[GpioBackend]:
    """sysfs /sys/class/gpio fallback.

    The sysfs interface addresses lines by a global integer, not
    (chip, offset). We can only use it when the caller passes a plain global
    number as ``line`` and the chip's base offset is 0 — good enough as a
    fallback; the libgpiod path is preferred and chip-aware.
    """
    import os

    base = "/sys/class/gpio"
    num = int(line)
    gpio_dir = f"{base}/gpio{num}"

    if not os.path.isdir(gpio_dir):
        with open(f"{base}/export", "w") as f:
            f.write(str(num))
    with open(f"{gpio_dir}/direction", "w") as f:
        f.write("out")

    value_path = f"{gpio_dir}/value"

    class _SysfsBackend:
        def set(self, value: int) -> None:
            with open(value_path, "w") as f:
                f.write("1" if value else "0")

        def release(self) -> None:
            try:
                with open(f"{base}/unexport", "w") as f:
                    f.write(str(num))
            except OSError:
                pass

    logger.info("Buzzer GPIO via sysfs: gpio%d", num)
    return _SysfsBackend()


# ── Buzzer ───────────────────────────────────────────────────────────────

class Buzzer:
    """Fault buzzer driver with a background pulser thread.

    Thread model: ``set_alarm`` only flips an ``Event``; a single daemon
    thread owns all line writes, so toggling is decoupled from the pipeline's
    frame rate (the beep tempo stays steady even if the main loop stalls —
    exactly when a fault is likely on screen).
    """

    def __init__(self, cfg: Optional[dict] = None, backend: Optional[GpioBackend] = None):
        cfg = cfg or {}
        self._enabled = bool(cfg.get("enabled", False))
        self._gpiochip = str(cfg.get("gpiochip", "gpiochip0"))
        self._line = int(cfg.get("line", 0))
        self._active_low = bool(cfg.get("active_low", False))
        hz = float(cfg.get("pulse_hz", 2.0))
        # Half-period of the on/off pulse. Guard against 0/negative from config.
        self._half_period = 1.0 / (2.0 * hz) if hz > 0 else 0.25

        self._backend = backend                 # may be injected (tests) or None
        self._alarm = threading.Event()         # set = fault active (pulse)
        self._stop = threading.Event()          # set = shut the thread down
        self._wake = threading.Event()          # alarm-state changed → re-evaluate
        self._thread: Optional[threading.Thread] = None
        self._started = False

    # Raw electrical level for a desired buzzer state, honouring active_low.
    def _level(self, on: bool) -> int:
        return int(on) ^ int(self._active_low)

    def _drive(self, on: bool) -> None:
        if self._backend is None:
            return
        try:
            self._backend.set(self._level(on))
        except Exception as e:  # noqa: BLE001 — a bad write must not crash us
            logger.debug("Buzzer line write failed: %s", e)

    def start(self) -> "Buzzer":
        """Claim the line at the SILENT level and start the pulser thread.

        Called during pipeline init so the buzzer is quiet the instant the
        pipeline comes up, regardless of the DO's power-on default. Disabled
        or backend-less → no-op (stays a silent stub).
        """
        if not self._enabled:
            logger.info("Buzzer disabled in config — not started")
            return self
        if self._backend is None:
            self._backend = _open_backend(self._gpiochip, self._line)
        if self._backend is None:
            logger.warning("Buzzer enabled but no GPIO backend available "
                           "(%s line %d) — running without a buzzer",
                           self._gpiochip, self._line)
            return self

        self._drive(False)  # silent immediately
        self._stop.clear()
        self._alarm.clear()
        self._thread = threading.Thread(
            target=self._run, name="aegis-buzzer", daemon=True)
        self._thread.start()
        self._started = True
        logger.info("Buzzer armed: %s line %d (active_low=%s, %.1f Hz pulse)",
                    self._gpiochip, self._line, self._active_low,
                    0.5 / self._half_period)
        return self

    def set_alarm(self, on: bool) -> None:
        """Turn the fault alarm on (pulsing) or off (silent). Idempotent."""
        if not self._started:
            return
        if on and not self._alarm.is_set():
            self._alarm.set()
            self._wake.set()
        elif not on and self._alarm.is_set():
            self._alarm.clear()
            self._wake.set()

    def _run(self) -> None:
        """Pulser loop: toggle while alarm is set, hold silent while it isn't."""
        while not self._stop.is_set():
            if self._alarm.is_set():
                # Pulse: on for a half-period, off for a half-period. A state
                # change (alarm cleared / stop) interrupts the wait promptly.
                self._drive(True)
                if self._wait(self._half_period):
                    continue
                self._drive(False)
                if self._wait(self._half_period):
                    continue
            else:
                self._drive(False)
                # Sleep until the alarm state changes or we're told to stop.
                self._wake.wait()
                self._wake.clear()
        self._drive(False)  # leave silent on the way out

    def _wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; return True early if state changed/stop."""
        if self._wake.wait(seconds):
            self._wake.clear()
            return True
        return self._stop.is_set()

    def close(self) -> None:
        """Force the line silent, stop the thread, release the line.

        No-op if the buzzer never started (disabled, or no GPIO backend) — it
        never claimed the line, so there's nothing to silence or release.
        """
        if not self._started:
            return
        self._stop.set()
        self._wake.set()  # unblock the pulser so it can exit
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._drive(False)
        if self._backend is not None:
            try:
                self._backend.release()
            except Exception as e:  # noqa: BLE001
                logger.debug("Buzzer backend release failed: %s", e)
            self._backend = None
        self._started = False
