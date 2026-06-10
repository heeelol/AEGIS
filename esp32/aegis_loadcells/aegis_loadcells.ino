/*
 * AEGIS v2 — ESP32 Load-Cell Reader
 * =================================
 * Reads an array of HX711 load-cell amplifiers, applies per-bin calibration,
 * and streams ready-made per-bin weights to the host PC as newline-delimited
 * JSON over USB serial. This is the firmware counterpart to the host-side
 * reader in aegis-v2/integration/src/sensing/loadcell.py.
 *
 * Wire protocol (must match loadcell.py)
 * --------------------------------------
 *   - 115200 baud, USB serial.
 *   - One JSON object per line, '\n' terminated:
 *
 *       {"bins": {"bin_0_0": 123.4, "bin_0_1": 80.1, "bin_1_0": 0.0}}\n
 *
 *   - Weights are in grams, keyed by bin_{row}_{col} to match the rest of the
 *     system. Layout below mirrors settings.yaml `manual_layout: [4, 3]`
 *     (layer 0 → 4 bins, layer 1 → 3 bins = 7 load cells).
 *
 * Bin layout
 * ----------
 * Edit BINS[] to match your physical layout. Each entry is one load cell:
 * {row, col, HX711 DOUT pin, HX711 SCK pin, calibration scale}.
 *
 * Serial commands (type into the Arduino Serial Monitor, newline-terminated)
 * -------------------------------------------------------------------------
 *   t          → tare (zero) every load cell. Do this with empty bins.
 *   c <i> <g>  → calibrate cell index <i>: put a known mass of <g> grams on
 *                that bin first, then send e.g. `c 0 500` for a 500 g weight.
 *                The new scale is printed; copy it into BINS[] to persist it.
 *   r          → print current raw/calibration table (for debugging).
 *
 * Library dependency
 * ------------------
 *   HX711 by Bogdan Necula ("HX711" in the Arduino Library Manager / bogde/HX711)
 *
 * Board
 * -----
 *   Any ESP32 dev board. Tools → Board → "ESP32 Dev Module" (or your variant).
 */

#include <HX711.h>

// ── Bin / load-cell table ────────────────────────────────────────────────
// One row per load cell. Pins are ESP32 GPIO numbers. `scale` is the HX711
// scale factor (raw counts per gram) found via the `c` calibration command;
// the placeholder 1.0 means "uncalibrated — reports raw counts" until set.
struct BinCell {
  uint8_t row;
  uint8_t col;
  uint8_t pin_dout;   // HX711 DT  (data)
  uint8_t pin_sck;    // HX711 SCK (clock)
  float   scale;      // counts per gram (calibrate, then paste back here)
};

// Layout for manual_layout: [4, 3]  → 7 bins.
// Pin assignments below are EXAMPLES — change them to match your wiring.
BinCell BINS[] = {
  // row, col, DOUT, SCK, scale
  {0, 0,  16, 17,  1.0f},
  {0, 1,  18, 19,  1.0f},
  {0, 2,  21, 22,  1.0f},
  {0, 3,  23, 25,  1.0f},
  {1, 0,  26, 27,  1.0f},
  {1, 1,  32, 33,  1.0f},
  {1, 2,  13, 14,  1.0f},
};

const size_t NUM_BINS = sizeof(BINS) / sizeof(BINS[0]);

// How often to emit a reading (ms). 10 Hz is plenty for the host pipeline.
const unsigned long REPORT_INTERVAL_MS = 100;

HX711 scales[NUM_BINS];
unsigned long last_report = 0;

// ── Setup ────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(200);

  for (size_t i = 0; i < NUM_BINS; i++) {
    scales[i].begin(BINS[i].pin_dout, BINS[i].pin_sck);
    scales[i].set_scale(BINS[i].scale);
    // Tare on boot — make sure the bins are empty at power-up. Send `t` later
    // to re-zero once the rig is settled.
    if (scales[i].wait_ready_timeout(1000)) {
      scales[i].tare();
    }
  }
}

// ── Main loop ────────────────────────────────────────────────────────────
void loop() {
  handleSerialCommands();

  unsigned long now = millis();
  if (now - last_report >= REPORT_INTERVAL_MS) {
    last_report = now;
    emitReading();
  }
}

// Read every cell and emit one line of JSON the host expects.
void emitReading() {
  Serial.print("{\"bins\": {");
  for (size_t i = 0; i < NUM_BINS; i++) {
    float grams = 0.0f;
    if (scales[i].is_ready()) {
      // Average a few samples to cut HX711 jitter without blocking long.
      grams = scales[i].get_units(3);
    }
    Serial.print("\"bin_");
    Serial.print(BINS[i].row);
    Serial.print("_");
    Serial.print(BINS[i].col);
    Serial.print("\": ");
    Serial.print(grams, 1);   // 1 decimal place, grams
    if (i + 1 < NUM_BINS) Serial.print(", ");
  }
  Serial.println("}}");
}

// ── Calibration / control over serial ────────────────────────────────────
void handleSerialCommands() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  char cmd = line.charAt(0);

  if (cmd == 't') {
    for (size_t i = 0; i < NUM_BINS; i++) {
      if (scales[i].wait_ready_timeout(1000)) scales[i].tare();
    }
    // Comment, not JSON — the host parser skips non-JSON lines safely.
    Serial.println("# tared all cells");

  } else if (cmd == 'c') {
    // c <index> <known_grams>
    int idx = -1;
    float known = 0.0f;
    if (sscanf(line.c_str(), "c %d %f", &idx, &known) == 2 &&
        idx >= 0 && (size_t)idx < NUM_BINS && known != 0.0f) {
      if (scales[idx].wait_ready_timeout(1000)) {
        // raw average (offset already removed by tare) / known mass = counts/gram
        long raw = scales[idx].get_value(10);
        float new_scale = raw / known;
        scales[idx].set_scale(new_scale);
        BINS[idx].scale = new_scale;
        Serial.print("# cell ");
        Serial.print(idx);
        Serial.print(" scale = ");
        Serial.print(new_scale, 4);
        Serial.println("  (paste into BINS[] to persist)");
      }
    } else {
      Serial.println("# usage: c <index> <known_grams>");
    }

  } else if (cmd == 'r') {
    for (size_t i = 0; i < NUM_BINS; i++) {
      Serial.print("# [");
      Serial.print(i);
      Serial.print("] bin_");
      Serial.print(BINS[i].row);
      Serial.print("_");
      Serial.print(BINS[i].col);
      Serial.print(" dout=");
      Serial.print(BINS[i].pin_dout);
      Serial.print(" sck=");
      Serial.print(BINS[i].pin_sck);
      Serial.print(" scale=");
      Serial.println(BINS[i].scale, 4);
    }
  }
}
