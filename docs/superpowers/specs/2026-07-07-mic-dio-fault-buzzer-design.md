# MIC DIO Fault Buzzer — Design

**Date:** 2026-07-07
**Status:** Approved (brainstormed with operator)

## Goal

An active buzzer wired directly to the MIC-733-AO's DIO terminal (live → DO
channel, ground → DIO GND) sounds **only while the kitting FSM is in FAULT**
(`overpack-kit`, `pick-from-wrong-bin`, `return-to-wrong-bin`). It is silent
from power-on: a DO channel idles low/floating, so the buzzer cannot sound
until the pipeline actively drives the line high. The ESP32 is not involved
(no firmware change, no reupload) — its earlier role as buzzer power (J10 VCC)
is retired.

Rejected alternatives:
- **ESP32-switched buzzer over serial** — works, but needs a firmware
  reupload and rewiring to GPIO 27; MIC-direct wiring is simpler.
- **MIC DO as ground-side sink** (original wiring) — the DO conducted at
  idle, so the buzzer beeped continuously from power-on with no software fix.

## Electrical precondition (verified on device with the test script)

The plan assumes the MIC's DO channels are push-pull (can **source**
current). Outcomes when wiring up:
- Silent on connect, beeps when the line is driven high → works as designed.
- Beeps immediately on connect → that pin conducts at idle; fall back to the
  ESP32-switched design.
- Audible but weak → DO is current-limited; add an NPN transistor
  (GPIO → base) as the driver. Software unchanged.

## Components

### 1. `integration/src/actuators/buzzer.py` — `Buzzer`

- `Buzzer(cfg)` reads: `enabled` (bool, default false), `gpiochip`
  (e.g. `"gpiochip0"`), `line` (int offset), `active_low` (default false;
  escape hatch if the hardware inverts), `pulse_hz` (default 2.0).
- `start()` claims the line as output at the **silent** level immediately,
  and spawns a daemon pulser thread.
- `set_alarm(bool)` is the whole runtime API — thread-safe, idempotent.
  While alarm is on, the pulser toggles the line at `pulse_hz` (on/off each
  half-period); while off, the line is held silent.
- `close()` forces the line silent, stops the thread, releases the line.
- GPIO backends tried in order: `gpiod` (libgpiod v1 API, present on
  JetPack 6), then sysfs `/sys/class/gpio`. Any failure (missing lib, bad
  chip/line, permissions) → log a warning once and become a no-op — the
  pipeline must never die because of the buzzer. `enabled: false` skips GPIO
  entirely (hard off switch in settings.yaml).
- The backend is injectable (constructor arg) so tests drive a fake.

### 2. Pipeline wiring (`pipeline.py`)

- Build + `start()` the buzzer in init (alongside `_init_loadcells`).
- In `_apply_loadcell_counts`:
  - normal path: `self._buzzer.set_alarm(kit.state == "FAULT")`
  - WAITING_EMPTY branch, not-connected, and placement-disabled paths:
    `set_alarm(False)` — the buzzer can never stick on when data stops.
- `_cleanup()`: `close()` — Ctrl-C never leaves it sounding.

### 3. Config (`settings.yaml`, under `sensing:`)

```yaml
buzzer:
  enabled: true          # false = buzzer feature fully off (no GPIO touched)
  gpiochip: "gpiochip0"  # from buzzer_test.py discovery on the MIC
  line: 0                # DO line offset — ditto
  active_low: false      # flip if the hardware inverts (beeps when idle)
  pulse_hz: 2.0          # on/off toggles per second while in FAULT
```

Repo default: `enabled: true` with placeholder chip/line; the MIC keeps its
device-specific values as a local override (same convention as `device:` and
the serial `port:`).

### 4. `integration/tools/buzzer_test.py` — one-time discovery on the MIC

- `--list`: enumerate gpiochips and line names/count.
- `--chip gpiochip0 --line N [--active-low]`: beep 3× (0.5 s on/off) using
  the same driver class the pipeline uses, then leave the line silent.
- Doubles as the electrical sanity check above.

### 5. Docs

`MIC_GUIDE.md` gains a short "Fault buzzer" section: wiring, running the
discovery script, config keys, and the transistor note.

## Testing

- Unit tests (`integration/tests/test_buzzer.py`) with a fake GPIO backend:
  starts silent; alarm on → line toggles; alarm off mid-pulse → line forced
  silent; close → silent + released; disabled → no backend calls;
  `active_low` inverts levels; backend failure → no-op without raising.
- Pipeline-level: FAULT sets alarm, WAITING_EMPTY/disconnect clears it
  (direct call test on `_apply_loadcell_counts`' helper path if feasible).
- On-device: `buzzer_test.py`, then a real wrong-bin fault end-to-end.
