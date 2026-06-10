#include <Arduino.h>
#include "HX711.h"

// --- Pin definitions ---
// Change these to match whichever GPIO pins you used
const int LOADCELL_DOUT_PIN = 13;
const int LOADCELL_SCK_PIN  = 5;

HX711 scale;

// --- Calibration ---
// You'll fill this in during the calibration step below
float calib_factor = 142.3; //417.7680 420.1367 437.0460
const bool doCalib = false;

// Known reference weight (in grams) you will place on the cell during calibration.
const float KNOWN_WEIGHT_G = 470;
// How many samples to average when measuring.
const int CALIB_SAMPLES = 50;


// Runs the calibration routine and returns the computed calibration factor.
// Procedure: tare empty, place the known weight, then average the raw output.
// calib_factor = average_raw_reading / known_weight
float runCalibration() {
  Serial.println();
  Serial.println("=== Calibration ===");
  Serial.println("Remove all weight from the load cell.");
  delay(3000);

  scale.set_scale();   // reset scale factor to 1.0 so we read raw counts
  scale.tare();        // zero out the empty-cell offset
  Serial.println("Tared with empty cell.");

  Serial.print("Place the known weight (");
  Serial.print(KNOWN_WEIGHT_G, 1);
  Serial.println(" g) on the load cell now.");
  delay(5000);

  // Average the initial output over CALIB_SAMPLES readings.
  Serial.print("Measuring (averaging ");
  Serial.print(CALIB_SAMPLES);
  Serial.println(" readings)...");
  float avg_reading = scale.get_units(CALIB_SAMPLES);

  // calib_factor maps raw counts to grams.
  float factor = avg_reading / KNOWN_WEIGHT_G;

  Serial.print("Average reading: ");
  Serial.println(avg_reading, 2);
  Serial.print(">>> Calibration factor: ");
  Serial.println(factor, 2);
  Serial.println("Copy this value into 'calib_factor' and set doCalib = false.");
  Serial.println("===================");
  Serial.println();

  return factor;
}


void setup() {
  Serial.begin(115200);
  Serial.println("HX711 load cell initialising...");

  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);

  // Wait until the HX711 is ready
  while (!scale.is_ready()) {
    Serial.println("Waiting for HX711...");
    delay(500);
  }

  if (doCalib) {
    calib_factor = runCalibration();
    scale.set_scale(calib_factor);
  } else {
    scale.set_scale(calib_factor);
  }

  delay(2000);
  scale.tare();  // Reset to zero with nothing on the scale

  Serial.println("Tare complete. Place a known weight to calibrate.");
}

void loop() {
  if (scale.is_ready()) {
    double reading = scale.get_units(10);  // average of 10 readings
    Serial.print("Weight: ");
    Serial.print(reading, 2);
    Serial.println(" g");
  } else {
    Serial.println("HX711 not ready");
  }
  delay(100);
}
