/* ============================================================
   Multi-Bin Load Cell Scale
   - Each bin = 1..MAX_CELLS HX711 channels, summed into one weight
   - All HX711s SHARE one SCK (clock); each needs its own DOUT
   - Streams per-bin weights as newline-delimited JSON for loadcell.py:
       {"bins":{"bin_0_0":123.40,"bin_0_1":80.10}}
   - Currently configured for 2 bins x 3 cells. Add bins by extending
     the bins[] array (and bumping NUM_BINS). A bin may have fewer cells
     (set numCells = 1) for single-load-cell bins later.
   ============================================================ */

#include "HX711.h"

// ---------- SHARED CLOCK ----------
// One SCK pin drives ALL HX711s. Each cell still needs its own DOUT (below).
const int SCK_PIN = 5;          // TODO: set your shared SCK GPIO

// ---------- BIN TABLE ----------
const int NUM_BINS  = 2;         // number of bins (sets of load cells)
const int MAX_CELLS = 3;         // most cells any single bin has

struct Bin {
  const char* id;                // JSON key, must match loadcell.py (bin_row_col)
  int   numCells;                // cells in this bin (3 = triple, 1 = single)
  int   dout[MAX_CELLS];         // DOUT GPIO per cell (only numCells used)
  float scale[MAX_CELLS];        // counts per gram, per cell (from calibrate())
  long  offset[MAX_CELLS];       // tare offset, per cell (refreshed at boot)
};

// TODO: fill in dout pins, and the scale/offset values from calibrate().
// Until calibrated, scale=1 / offset=0 just gives raw-ish numbers.
//
Bin bins[NUM_BINS] = {
  // id          numCells  dout{c0,c1,c2}     scale{c0,c1,c2}          offset{c0,c1,c2}
  // { "bin_0_0",   3,        { 18, 19, 21 },    { 1072.3745, 1055.9662, 1010.8351 },    { 0, 0, 0 } },
  { "bin_0_1",   3,        { 18, 19, 21 },    { 417.7680, 420.1367, 437.0460 },    { 0, 0, 0 } },
};

HX711 cell[NUM_BINS][MAX_CELLS];

// ---------- CALIBRATION ----------
const bool  DO_CALIBRATION = false;  // true to (re)calibrate; false to run
const float CAL_MASS_G     = 182.8;  // known reference mass, grams

// ---------- FILTER ----------
const int N_SAMPLES = 3;    // readings averaged per cell per cycle
                            // (low keeps the JSON cadence up; HX711 is ~10 SPS)

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  for (int b = 0; b < NUM_BINS; b++) {
    for (int c = 0; c < bins[b].numCells; c++) {
      cell[b][c].begin(bins[b].dout[c], SCK_PIN);   // shared SCK
    }
  }

  Serial.println(F("Multi-bin scale starting..."));

  if (DO_CALIBRATION) {
    calibrate();
  } else {
    // Apply stored counts-per-gram for every cell.
    for (int b = 0; b < NUM_BINS; b++) {
      for (int c = 0; c < bins[b].numCells; c++) {
        cell[b][c].set_scale(bins[b].scale[c]);
      }
    }
    // Re-tare at boot instead of trusting frozen offsets. Load-cell zero
    // drifts with temperature, mounting stress, creep and supply voltage,
    // so a hardcoded offset reads non-zero at no load. Keep platforms EMPTY
    // at power-up (or FULL, if you want weight to read negative as items leave).
    Serial.println(F("Remove ALL load. Taring in 5s..."));
    delay(5000);
    for (int b = 0; b < NUM_BINS; b++) {
      for (int c = 0; c < bins[b].numCells; c++) {
        cell[b][c].tare(20);                       // 20-reading average as zero
        bins[b].offset[c] = cell[b][c].get_offset();
      }
    }
    Serial.println(F("Boot tare done. Using stored scale."));
  }
}

void loop() {
  // --- emit all bins as one newline-delimited JSON object ---
  // {"bins":{"bin_0_0":<g>,"bin_0_1":<g>}}

  Serial.print(F("{\"bins\":{"));
  for (int b = 0; b < NUM_BINS; b++) {
    // bin total = sum of its cells (position-independent when calibrated)
    float total = 0;
    for (int c = 0; c < bins[b].numCells; c++) {
      total += cell[b][c].get_units(N_SAMPLES);    // (raw - offset)/scale = grams
    }
    if (b > 0) Serial.print(',');
    Serial.print('"'); Serial.print(bins[b].id); Serial.print(F("\":"));
    Serial.print(total, 2);
  }
  Serial.println(F("}}"));

  delay(200);   // ~5 Hz output
}

/* ============================================================
   CALIBRATION ROUTINE  (per-cell, position-independent)

   Rationale: a bin's total = sum of its cells is only position-independent
   if every cell reports the TRUE grams it carries -- i.e. each cell's scale
   must be its own true counts-per-gram. Assuming a centred mass splits evenly
   is wrong (it never does) and makes the total drift with object position.
   So we calibrate each cell against a known mass applied DIRECTLY to it.

   1. Remove all load when prompted (tare, captures per-cell offset)
   2. For each cell of each bin, place CAL_MASS_G directly over THAT cell
   3. Copy the printed scale/offset values into the bins[] table above
   4. Set DO_CALIBRATION = false and re-upload
   ============================================================ */
void calibrate() {
  Serial.println(F("\n--- CALIBRATION ---"));
  Serial.println(F("Remove ALL load. Taring in 5s..."));
  delay(5000);

  for (int b = 0; b < NUM_BINS; b++) {
    for (int c = 0; c < bins[b].numCells; c++) {
      cell[b][c].set_scale();      // scale = 1 for raw
      cell[b][c].tare(20);
      bins[b].offset[c] = cell[b][c].get_offset();
    }
  }
  Serial.println(F("Tare done."));

  // --- per-cell span: load each cell of each bin individually ---
  for (int b = 0; b < NUM_BINS; b++) {
    for (int c = 0; c < bins[b].numCells; c++) {
      Serial.print(F("\nPlace ")); Serial.print(CAL_MASS_G);
      Serial.print(F(" g DIRECTLY over ")); Serial.print(bins[b].id);
      Serial.print(F(" cell ")); Serial.print(c + 1);
      Serial.println(F(". Measuring in 8s..."));
      delay(8000);

      long raw = cell[b][c].read_average(20);
      float countsAboveZero = (float)(raw - bins[b].offset[c]);
      bins[b].scale[c] = countsAboveZero / CAL_MASS_G;   // true counts/gram
      cell[b][c].set_scale(bins[b].scale[c]);
      cell[b][c].set_offset(bins[b].offset[c]);

      Serial.print(F("  scale = ")); Serial.println(bins[b].scale[c], 4);
    }
  }

  // --- print pasteable values for the bins[] table ---
  Serial.println(F("\n--- PASTE THESE INTO bins[] ---"));
  for (int b = 0; b < NUM_BINS; b++) {
    Serial.print(F("  ")); Serial.print(bins[b].id);
    Serial.print(F("  scale { "));
    for (int c = 0; c < bins[b].numCells; c++) {
      Serial.print(bins[b].scale[c], 4);
      Serial.print(c < bins[b].numCells - 1 ? F(", ") : F(" }"));
    }
    Serial.print(F("  offset { "));
    for (int c = 0; c < bins[b].numCells; c++) {
      Serial.print(bins[b].offset[c]);
      Serial.print(c < bins[b].numCells - 1 ? F(", ") : F(" }\n"));
    }
  }
  Serial.println(F("Set DO_CALIBRATION = false, then re-upload.\n"));
  delay(3000);
}
