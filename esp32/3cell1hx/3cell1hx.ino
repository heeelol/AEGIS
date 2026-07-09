/* ============================================================
   Multi-Bin Load Cell Scale  (ONE HX711 per bin)  -- GANGED READ
   - Each bin = a set of load cells wired in PARALLEL into a SINGLE HX711
     (all E+ joined, all E- joined, all A+ joined, all A- joined).
     The bin's cells are summed/averaged in HARDWARE, so the ESP32 sees
     one combined signal per bin -> one DOUT pin per bin.
   - All HX711s SHARE one SCK (clock); each needs its own DOUT.
   - Streams per-bin weights as newline-delimited JSON for loadcell.py:
       {"bins":{"bin_1":123.40,"bin_2":80.10}}

   WHY GANGED (parallel) INSTEAD OF PER-BIN get_units():
     Because every HX711 shares SCK, clocking ANY chip kicks ALL of them
     into a fresh ~100 ms conversion. Reading one bin at a time therefore
     forces a fresh wait for each of the next bins -> time grows with bin
     count (~55 ms x 10 bins ~= 550 ms -> 2 Hz).
     The chips are already synchronized by the shared clock: they finish
     converting together and shift data out on the SAME pulses. So we read
     the whole bank at once:
         wait ONCE for ready -> pulse SCK 24x -> on each pulse sample ALL
         10 DOUT pins -> done. Total ~= one conversion wait (~100 ms) for
         ALL bins. Rate stops depending on bin count (~10 Hz regardless).
     The bogde/HX711 library can only read one chip per call, so this file
     bit-bangs the shared-clock read itself (readGanged / readAveraged).

   FLAKY / UNWIRED PINS:
     DOUT pins use INPUT_PULLDOWN. A disconnected DOUT (e.g. bin_6 J9 if
     not yet soldered) then reads LOW -> looks "ready", yields ~0 counts,
     and CANNOT stall the bank. A real HX711 drives DOUT push-pull and
     easily overpowers the ~45k pulldown, so wired chips are unaffected.
     waitReadyAll() also has a timeout so nothing hangs the loop.

   NOTE vs. tripleCell.ino:
     tripleCell.ino  = 1 HX711 PER CELL, summed in software.
     this file       = 1 HX711 PER BIN, cells combined in hardware.
   ============================================================ */

// ---------- PIN TABLE ----------
// One SCK pin drives ALL HX711s. Each bin still needs its own DOUT (below).
const int SCK_PIN = 5; // shared SCK GPIO

const int J1 = 4;
const int J2 = 13; // NOT IN USE
const int J3 = 18;
const int J4 = 19;
const int J5 = 21;
const int J6 = 22;
const int J7 = 23;
const int J8 = 25;
const int J9 = 26;  // NOT IN USE, TODO: SOLDER IT
const int J10 = 27; // NOT IN USE
const int J11 = 32;
const int J12 = 33;

// ------- LOADCELL CONST -------
const float CELL1 = 74.1207;
const float CELL2 = 68.1992;
const float CELL3 = 71.6398;
const float CELL4 = 143.7663;
const float CELL5 = 141.1965;
const float CELL6 = 118.5296;
const float CELL7 = 1009.8922; // TODO FIX MECH SIDE THEN RECALIB
const float CELL8 = 1056.6241; // TODO FIX MECH SIDE THEN RECALIB
const float CELL9 = 1000.00;   // NOT IN USE, PENDING PORT 9
const float CELL10 = 140.5147;

struct Bin
{
  const char *id; // JSON key, must match loadcell.py (bin_row_col)
  int dout;       // DOUT GPIO for this bin's HX711
  float scale;    // counts per gram for the combined bin
};

const int NUM_BINS = 10; // number of bins (each = one HX711)
Bin bins[NUM_BINS] = {
    // id          dout    scale
    {"bin_1", J4, CELL1},
    {"bin_2", J3, CELL2},
    {"bin_3", J1, CELL3},
    {"bin_4", J6, CELL4},
    {"bin_5", J8, CELL5},
    {"bin_6", J9, CELL6},
    {"bin_7", J5, CELL7},
    {"bin_8", J7, CELL8},
    {"bin_9", J11, CELL9},
    {"bin_10", J12, CELL10},
    // { "bin_1_1", J10, CELL }, //TODO solder the jst for this
};

// Per-bin tare offset (raw counts at zero load), filled by tareAll().
long offset[NUM_BINS] = {0};

// ---------- CALIBRATION ----------
const bool DO_CALIBRATION = false; // true to (re)calibrate; false to run
const float CAL_MASS_G = 677.8;    // known reference mass, grams

// ---------- FILTER ----------
const int N_SAMPLES = 1; // ganged passes averaged per output cycle
                         // (1 keeps ~10 Hz; 2-3 trades rate for less noise)

// ---------- GANGED-READ TUNING ----------
const unsigned long READY_TIMEOUT_MS = 150; // max wait for the bank to be ready
const int GAIN_PULSES = 1;                  // 1 = channel A gain 128 (next reading)

// ---------- BUZZER / RINGTONE ----------
// Passive buzzer on J10 (GPIO 27). tone() drives it by FREQUENCY (the pitch),
// not analogWrite duty. Melody = a table of {frequency Hz, duration ms};
// freq 0 = a rest.
const int BUZZER_PIN = J10;

struct Note
{
  int freq;
  int dur;
};

// "ring" tune: McDonald's "I'm Lovin' It" hook ("ba da ba ba ba" -> last note
// jumps up). Pitches are approximate and easy to tweak -- edit the table.
const Note CORR_TUNE[] = {
    {1046, 90}, // C5
    {1318, 90}, // E5
    {1568, 90}, // G5
    {2092, 90}, // C6
};

// "err" tune: short low double-beep.
const Note ERR_TUNE[] = {
    {164, 90}, // E2
    {164, 90}, // E2
};

const int CORR_LEN = sizeof(CORR_TUNE) / sizeof(CORR_TUNE[0]);
const int ERR_LEN = sizeof(ERR_TUNE) / sizeof(ERR_TUNE[0]);
const int NOTE_GAP_MS = 40; // brief silence between notes so repeats are distinct

// Play any tune table once. The delay()s are vTaskDelay under the hood, so this
// is meant to run in its own FreeRTOS task (buzzerTask) -- NOT in loop(). tone()
// drives the LEDC hardware, so the melody sounds while loop() keeps reading load
// cells on the other core: buzzer and streaming run in parallel.
void playTune(const Note *tune, int len)
{
  for (int i = 0; i < len; i++)
  {
    if (tune[i].freq > 0)
      tone(BUZZER_PIN, tune[i].freq, tune[i].dur);
    else
      noTone(BUZZER_PIN);
    delay(tune[i].dur + NOTE_GAP_MS); // note length + gap (yields the task)
  }
  noTone(BUZZER_PIN);
}

// Which tune loop() is requesting -- carried in the task-notification VALUE so
// the single buzzer task can play EITHER tune.
enum
{
  TUNE_CORR = 1,
  TUNE_ERR = 2
};

// Dedicated buzzer task, pinned to core 0 (loop() runs on core 1). Sleeps at
// zero CPU cost until loop() notifies it, then plays the requested tune. This is
// what lets the buzzer ring "in parallel" with the JSON stream.
TaskHandle_t buzzerTaskHandle = NULL;

void buzzerTask(void *param)
{
  uint32_t which;
  for (;;)
  {
    // Block until notified; `which` receives the tune id loop() sent.
    if (xTaskNotifyWait(0, 0xFFFFFFFF, &which, portMAX_DELAY) == pdTRUE)
    {
      if (which == TUNE_ERR)
        playTune(ERR_TUNE, ERR_LEN);
      else
        playTune(CORR_TUNE, CORR_LEN);
    }
  }
}

String cmd = "null";

// ---- Is every HX711's DOUT low (conversion ready)? Timeout so one flaky /
//      unwired chip can't stall the bank. Returns true only if all went low. ----
bool waitReadyAll(unsigned long timeout_ms)
{
  unsigned long start = millis();
  while (millis() - start < timeout_ms)
  {
    bool allReady = true;
    for (int b = 0; b < NUM_BINS; b++)
    {
      if (digitalRead(bins[b].dout) == HIGH)
      {
        allReady = false;
        break;
      }
    }
    if (allReady)
      return true;
    yield(); // keep RTOS/WiFi housekeeping alive while polling
  }
  return false;
}

// ---- One ganged conversion: clock the shared SCK 24x and sample every
//      DOUT per pulse, so all bins are read in a single burst. Fills raw[]
//      with sign-extended 24-bit two's-complement counts. ----
void readGanged(long raw[NUM_BINS])
{
  waitReadyAll(READY_TIMEOUT_MS);

  uint32_t value[NUM_BINS];
  for (int b = 0; b < NUM_BINS; b++)
    value[b] = 0;

  // Timing-sensitive: keep SCK pulses < 60 us so no HX711 powers down.
  noInterrupts();
  for (int i = 0; i < 24; i++)
  {
    digitalWrite(SCK_PIN, HIGH);
    delayMicroseconds(1);
    for (int b = 0; b < NUM_BINS; b++)
    {
      value[b] = (value[b] << 1) | (uint32_t)digitalRead(bins[b].dout);
    }
    digitalWrite(SCK_PIN, LOW);
    delayMicroseconds(1);
  }
  // Extra pulses set the channel/gain used for the NEXT conversion.
  for (int p = 0; p < GAIN_PULSES; p++)
  {
    digitalWrite(SCK_PIN, HIGH);
    delayMicroseconds(1);
    digitalWrite(SCK_PIN, LOW);
    delayMicroseconds(1);
  }
  interrupts();

  // Sign-extend 24-bit two's complement into a signed long.
  for (int b = 0; b < NUM_BINS; b++)
  {
    if (value[b] & 0x00800000UL)
      value[b] |= 0xFF000000UL;
    raw[b] = (long)((int32_t)value[b]);
  }
}

// ---- Average n ganged passes into raw[] (each pass is one full bank read). ----
void readAveraged(long raw[NUM_BINS], int n)
{
  if (n < 1)
    n = 1;
  long sum[NUM_BINS];
  for (int b = 0; b < NUM_BINS; b++)
    sum[b] = 0;
  long one[NUM_BINS];
  for (int s = 0; s < n; s++)
  {
    readGanged(one);
    for (int b = 0; b < NUM_BINS; b++)
      sum[b] += one[b];
  }
  for (int b = 0; b < NUM_BINS; b++)
    raw[b] = sum[b] / n;
}

// ---- Capture per-bin zero (average of n ganged passes) into offset[]. ----
void tareAll(int n)
{
  long raw[NUM_BINS];
  readAveraged(raw, n);
  for (int b = 0; b < NUM_BINS; b++)
    offset[b] = raw[b];
}

void setup()
{
  Serial.begin(115200);
  while (!Serial)
  {
  }

  pinMode(SCK_PIN, OUTPUT);
  digitalWrite(SCK_PIN, LOW);
  for (int b = 0; b < NUM_BINS; b++)
  {
    // Pulldown: an unwired DOUT reads LOW (never stalls the bank); a real
    // HX711's push-pull output overpowers it.
    pinMode(bins[b].dout, INPUT_PULLDOWN);
  }

  Serial.println(F("Multi-bin scale starting (ganged read, 1 HX711 per bin)..."));

  if (DO_CALIBRATION)
  {
    calibrate();
  }
  else
  {
    Serial.println(F("Remove ALL load. Taring in 5s..."));
    delay(5000);
    tareAll(20); // 20-pass average as zero for every bin
    Serial.println(F("Boot tare done. Using stored scale."));
  }

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW); // tone() manages the pin from here on

  // Spawn the buzzer on core 0 so it plays independently of the loop() (core 1)
  // that streams load-cell data. 2048-word stack, low priority.
  xTaskCreatePinnedToCore(buzzerTask, "buzzer", 2048, NULL, 1, &buzzerTaskHandle, 0);
}

void loop()
{
  // --- read the whole bank, then emit as one newline-delimited JSON object ---
  // {"bins":{"bin_1":<g>,"bin_2":<g>, ...}}
  long raw[NUM_BINS];
  readAveraged(raw, N_SAMPLES);

  Serial.print(F("{\"bins\":{"));
  for (int b = 0; b < NUM_BINS; b++)
  {
    float grams = (bins[b].scale != 0.0f)
                      ? (float)(raw[b] - offset[b]) / bins[b].scale
                      : 0.0f;
    if (b > 0)
      Serial.print(',');
    Serial.print('"');
    Serial.print(bins[b].id);
    Serial.print(F("\":"));
    Serial.print(grams, 2);
  }
  Serial.println(F("}}"));

  if (Serial.available())
  {
    cmd = Serial.readStringUntil('\n');
    cmd.trim();
  }
  if (cmd == "ring")
  {
    // Non-blocking: tell the buzzer task WHICH tune to play, then move on. The
    // tune plays on core 0 while this loop keeps reading and streaming.
    if (buzzerTaskHandle)
      xTaskNotify(buzzerTaskHandle, TUNE_CORR, eSetValueWithOverwrite);
    cmd = "null"; // one-shot: clear so it doesn't retrigger every loop
  }
  if (cmd == "err")
  {
    if (buzzerTaskHandle)
      xTaskNotify(buzzerTaskHandle, TUNE_ERR, eSetValueWithOverwrite);
    cmd = "null";
  }

  // No fixed delay needed: the ganged conversion wait (~100 ms) paces the
  // loop at ~10 Hz on its own. A tiny yield keeps serial/RTOS happy.
  yield();
}

/* ============================================================
   CALIBRATION ROUTINE  (per-bin, hardware-combined cells)

   Because the cells are paralleled into ONE HX711, you calibrate the
   whole bin as a single scale -- there is no per-cell access. Place a
   known mass on the bin platform and we derive one counts-per-gram for
   the combined signal.

   Tip for least corner error: place the calibration mass at the CENTRE
   of the platform so it loads the cells roughly evenly.

   1. Remove all load when prompted (tare, captures per-bin offset)
   2. For each bin, place CAL_MASS_G on the platform (centred)
   3. Copy the printed scale values into the bins[] table above
   4. Set DO_CALIBRATION = false and re-upload
   ============================================================ */
void calibrate()
{
  Serial.println(F("\n--- CALIBRATION ---"));
  Serial.println(F("Remove ALL load. Taring in 5s..."));
  delay(5000);
  tareAll(20);
  Serial.println(F("Tare done."));

  // --- per-bin span: load each bin platform with the known mass ---
  for (int b = 0; b < NUM_BINS; b++)
  {
    Serial.print(F("\nPlace "));
    Serial.print(CAL_MASS_G);
    Serial.print(F(" g on "));
    Serial.print(bins[b].id);
    Serial.println(F(" (centred). Measuring in 8s..."));
    delay(8000);

    long raw[NUM_BINS];
    readAveraged(raw, 20);
    float countsAboveZero = (float)(raw[b] - offset[b]);
    bins[b].scale = countsAboveZero / CAL_MASS_G; // counts/gram for the bin
    Serial.print(F("  scale = "));
    Serial.println(bins[b].scale, 4);
  }

  // --- print pasteable values for the bins[] table ---
  Serial.println(F("\n--- PASTE THESE SCALE VALUES INTO bins[] ---"));
  for (int b = 0; b < NUM_BINS; b++)
  {
    Serial.print(F("  "));
    Serial.print(bins[b].id);
    Serial.print(F(" scale = "));
    Serial.println(bins[b].scale, 4);
  }
  Serial.println(F("Set DO_CALIBRATION = false, then re-upload.\n"));
  delay(3000);
}
