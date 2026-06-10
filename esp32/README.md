# AEGIS v2 — ESP32 Load-Cell Firmware

ESP32 firmware that reads an array of HX711 load-cell amplifiers and streams
per-bin weights to the host PC over USB serial. It is the device-side
counterpart to the host reader at
`aegis-v2/integration/src/sensing/loadcell.py`.

## What it does

Each load cell maps to one bin (`bin_{row}_{col}`). The ESP32 reads every cell,
applies per-bin calibration, and emits one line of JSON per cycle (~10 Hz):

```json
{"bins": {"bin_0_0": 123.4, "bin_0_1": 80.1, "bin_1_0": 0.0}}
```

Weights are in **grams**. The format, keys, and `115200` baud exactly match the
contract the host parser expects.

## Hardware

- 1× ESP32 dev board (any variant).
- 1× HX711 amplifier + load cell per bin (7 bins for the default `[4, 3]`
  layout).
- Each HX711 needs **VCC (3.3V), GND, DT (data), SCK (clock)**. DT/SCK go to the
  GPIO pins listed in `BINS[]`; VCC/GND share the ESP32's 3.3V and GND rails.

The example pins in `aegis_loadcells/aegis_loadcells.ino` are a starting point —
edit the `BINS[]` table to match your actual wiring and layout.

## Build & flash (Arduino IDE)

1. Install ESP32 board support: **File → Preferences →** add
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   to *Additional Board Manager URLs*, then install **esp32** in
   *Tools → Board → Boards Manager*.
2. Install the HX711 library: **Tools → Manage Libraries → "HX711" by Bogdan
   Necula** (bogde/HX711).
3. Open `aegis_loadcells/aegis_loadcells.ino`.
4. Select your board under **Tools → Board** and the serial port under
   **Tools → Port**, then click **Upload**.

(arduino-cli equivalent:)

```bash
arduino-cli core install esp32:esp32
arduino-cli lib install "HX711"
arduino-cli compile -b esp32:esp32:esp32 esp32/aegis_loadcells
arduino-cli upload  -b esp32:esp32:esp32 -p /dev/ttyUSB0 esp32/aegis_loadcells
```

## Calibration

Open the Arduino **Serial Monitor** at 115200 baud (newline line-ending). The
firmware streams JSON, and also accepts these one-line commands:

| Command       | Action                                                              |
|---------------|---------------------------------------------------------------------|
| `t`           | Tare (zero) every cell. Do this with all bins **empty**.            |
| `c <i> <g>`   | Calibrate cell index `<i>` using a known mass of `<g>` grams on it. |
| `r`           | Print the current pin/scale table.                                  |

To calibrate a cell:

1. Send `t` with empty bins.
2. Place a known weight (e.g. 500 g) on bin index `i`.
3. Send `c i 500`. The firmware prints the computed scale factor.
4. Paste that value into the `scale` column of `BINS[]` and re-flash so it
   persists across power cycles.

Command replies start with `#`, so the host parser ignores them harmlessly.

## Connect to the host pipeline

Enable load cells in `aegis-v2/integration/config/settings.yaml`:

```yaml
sensing:
  loadcells:
    enabled: true
    port: "/dev/ttyUSB0"   # ESP32 USB device (Windows: "COM3", macOS: "/dev/tty.usbserial-*")
    baudrate: 115200
    stale_after: 2.0
```

Find the port with `ls /dev/ttyUSB* /dev/ttyACM*` (Linux) or Device Manager
(Windows). The host reads weights on a background thread and the dashboard shows
live per-bin grams.

## Quick test without the pipeline

Any serial monitor at 115200 baud will show the JSON stream:

```bash
# Linux/macOS
screen /dev/ttyUSB0 115200      # Ctrl-A then K to quit
# or
python3 -m serial.tools.miniterm /dev/ttyUSB0 115200
```
