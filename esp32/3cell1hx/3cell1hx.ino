/* ============================================================
   Multi-Bin Load Cell Scale  (ONE HX711 per bin)
   - Each bin = a set of load cells wired in PARALLEL into a SINGLE HX711
     (all E+ joined, all E- joined, all A+ joined, all A- joined).
     The bin's cells are summed/averaged in HARDWARE, so the ESP32 sees
     one combined signal per bin -> one DOUT pin per bin.
   - All HX711s SHARE one SCK (clock); each needs its own DOUT.
   - Streams per-bin weights as newline-delimited JSON for loadcell.py:
       {"bins":{"bin_0_0":123.40,"bin_0_1":80.10}}
   - Pin budget: NUM_BINS DOUT pins + 1 shared SCK. (vs. the per-cell
     version which needed one DOUT per individual load cell.)

   NOTE vs. tripleCell.ino:
     tripleCell.ino  = 1 HX711 PER CELL, summed in software (position-
                       independent even with mismatched cells, but uses
                       many pins).
     this file       = 1 HX711 PER BIN, cells combined in hardware
                       (few pins; total weight is correct regardless of
                       item position ONLY if the cells are reasonably
                       matched -- otherwise expect some corner error).
   ============================================================ */

#include "HX711.h"

// ---------- PIN TABLE ----------
// One SCK pin drives ALL HX711s. Each bin still needs its own DOUT (below).
const int SCK_PIN = 5;          // TODO: set your shared SCK GPIO

const int J1 = 4;
const int J2 = 13; //NOT IN USE
const int J3 = 18;
const int J4 = 19;
const int J5 = 21;
const int J6 = 22;
const int J7 = 23;
const int J8 = 25;
const int J9 = 26; //NOT IN USE
const int J10 = 27; //NOT IN USE
const int J11 = 32;
const int J12 = 33;

// ---------- BIN TABLE ----------
const int NUM_BINS = 9;          // number of bins (each = one HX711)

// ------- LOADCELL CONST -------
const int CELL1 = 0;
const int CELL2 = 0;
const int CELL3 = 0;
const int CELL4 = 0;
const int CELL5 = 0;
const int CELL6 = 0;
const int CELL7 = 0;
const int CELL8 = 0;
const int CELL9 = 0;
const int CELL10 = 0;

struct Bin {
  const char* id;                // JSON key, must match loadcell.py (bin_row_col)
  int   dout;                    // DOUT GPIO for this bin's HX711
  float scale;                   // counts per gram for the combined bin
  long  offset;                  // tare offset (refreshed at boot)
};

// TODO: fill in dout pins, and the scale/offset values from calibrate().
// Until calibrated, scale=1 / offset=0 just gives raw-ish numbers.
//
Bin bins[NUM_BINS] = {
  // id          dout    scale        offset
  { "bin_0_0", J1, CELL, 0 }, //71.5
  { "bin_0_1",  J3, CELL,  0 },
  { "bin_0_2",  J4, CELL, 0 },
  { "bin_0_3",  J5, CELL,  0 },
  { "bin_0_4", J6, CELL, 0 },
  { "bin_0_5", J7, CELL, 0 },
  { "bin_1_0", J8, CELL, 0 },
  // { "bin_1_1", J10, CELL, 0 }, //TODO solder the jst for this
  { "bin_1_2", J11, CELL, 0 },
  { "bin_2_1", J12, CELL, 0 },
};

HX711 cell[NUM_BINS];

// ---------- CALIBRATION ----------
const bool  DO_CALIBRATION = false;  // true to (re)calibrate; false to run
const float CAL_MASS_G     = 506.8;  // known reference mass, grams

// ---------- FILTER ----------
const int N_SAMPLES = 10;    // readings averaged per bin per cycle
                            // (low keeps the JSON cadence up; HX711 is ~10 SPS)

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  for (int b = 0; b < NUM_BINS; b++) {
    cell[b].begin(bins[b].dout, SCK_PIN);   // shared SCK
  }

  Serial.println(F("Multi-bin scale starting (1 HX711 per bin)..."));

  if (DO_CALIBRATION) {
    calibrate();
  } else {
    // Apply stored counts-per-gram for every bin.
    for (int b = 0; b < NUM_BINS; b++) {
      cell[b].set_scale(bins[b].scale);
    }
    // Re-tare at boot instead of trusting frozen offsets. Load-cell zero
    // drifts with temperature, mounting stress, creep and supply voltage,
    // so a hardcoded offset reads non-zero at no load. Keep platforms EMPTY
    // at power-up (or FULL, if you want weight to read negative as items leave).
    Serial.println(F("Remove ALL load. Taring in 5s..."));
    delay(5000);
    for (int b = 0; b < NUM_BINS; b++) {
      Serial.print(F("Taring ")); Serial.print(bins[b].id);
      // Serial.print(F(" (DOUT GPIO=")); Serial.print(bins[b].dout);
      // Serial.print(F(", SCK GPIO=")); Serial.print(SCK_PIN); Serial.println(F(")..."));
      // // Don't block forever if the HX711 never signals "ready" (DOUT stuck
      // // high = bad power/GND/wiring or wrong pin). Bail with a message.
      // if (!cell[b].wait_ready_timeout(1000)) {
      //   Serial.println(F("  HX711 NOT RESPONDING -> check power, GND, DT/SCK wiring & pin numbers"));
      //   continue;
      // }
      cell[b].tare(20);                       // 20-reading average as zero
      bins[b].offset = cell[b].get_offset();
      Serial.println(F("  tared OK"));
    }
    Serial.println(F("Boot tare done. Using stored scale."));
  }
}

void loop() {
  // --- emit all bins as one newline-delimited JSON object ---
  // {"bins":{"bin_0_0":<g>,"bin_0_1":<g>}}

  Serial.print(F("{\"bins\":{"));
  for (int b = 0; b < NUM_BINS; b++) {
    // One combined reading per bin (cells already summed in hardware).
    float total = cell[b].get_units(N_SAMPLES);   // (raw - offset)/scale = grams
    if (b > 0) Serial.print(',');
    Serial.print('"'); Serial.print(bins[b].id); Serial.print(F("\":"));
    Serial.print(total, 2);
  }
  Serial.println(F("}}"));

  delay(200);   // ~5 Hz output
}

/* ============================================================
   CALIBRATION ROUTINE  (per-bin, hardware-combined cells)

   Because the cells are paralleled into ONE HX711, you calibrate the
   whole bin as a single scale -- there is no per-cell access. Place a
   known mass on the bin platform and we derive one counts-per-gram for
   the combined signal.

   Tip for least corner error: place the calibration mass at the CENTRE
   of the platform so it loads the cells roughly evenly. If your cells
   are well matched, position won't matter much; if not, centring gives
   the most representative scale factor.

   1. Remove all load when prompted (tare, captures per-bin offset)
   2. For each bin, place CAL_MASS_G on the platform (centred)
   3. Copy the printed scale/offset values into the bins[] table above
   4. Set DO_CALIBRATION = false and re-upload
   ============================================================ */
void calibrate() {
  Serial.println(F("\n--- CALIBRATION ---"));
  Serial.println(F("Remove ALL load. Taring in 5s..."));
  delay(5000);

  for (int b = 0; b < NUM_BINS; b++) {
    cell[b].set_scale();      // scale = 1 for raw
    cell[b].tare(20);
    bins[b].offset = cell[b].get_offset();
  }
  Serial.println(F("Tare done."));

  // --- per-bin span: load each bin platform with the known mass ---
  for (int b = 0; b < NUM_BINS; b++) {
    Serial.print(F("\nPlace ")); Serial.print(CAL_MASS_G);
    Serial.print(F(" g on ")); Serial.print(bins[b].id);
    Serial.println(F(" (centred). Measuring in 8s..."));
    delay(8000);

    long raw = cell[b].read_average(20);
    float countsAboveZero = (float)(raw - bins[b].offset);
    bins[b].scale = countsAboveZero / CAL_MASS_G;   // counts/gram for the bin
    cell[b].set_scale(bins[b].scale);
    cell[b].set_offset(bins[b].offset);

    Serial.print(F("  scale = ")); Serial.println(bins[b].scale, 4);
  }

  // --- print pasteable values for the bins[] table ---
  Serial.println(F("\n--- PASTE THESE INTO bins[] ---"));
  for (int b = 0; b < NUM_BINS; b++) {
    Serial.print(F("  { \"")); Serial.print(bins[b].id);
    Serial.print(F("\", ")); Serial.print(bins[b].dout);
    Serial.print(F(", ")); Serial.print(bins[b].scale, 4);
    Serial.print(F(", ")); Serial.print(bins[b].offset);
    Serial.println(F(" },"));
  }
  Serial.println(F("Set DO_CALIBRATION = false, then re-upload.\n"));
  delay(3000);
}
