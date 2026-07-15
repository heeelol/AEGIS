# AEGIS Kitting Station — Final Report

> **Status:** Working draft consolidating the project to date. Sections marked
> `[TO ADD]` need your own data, figures, or findings. Drop supporting
> images / Excel / exported Google Docs into [`./sources/`](./sources) and link
> them from the relevant section.

---

## 1. Overview

AEGIS is a computer-vision + load-cell assisted **kitting station** that guides an
operator to pick the correct parts, in the correct quantities, into a kit — and
verifies every placement in real time. The goal is to error-proof manual kitting:
tell the operator *what* to pick and *how many*, catch mistakes (wrong bin, wrong
count, wrong order) as they happen, and confirm each kit before it moves on.

The system combines three sensing/deciding layers:

1. **Vision** — an overhead camera + a trained detector locate the 9 bins.
2. **Weight** — load cells under each bin and the kit box verify picks and placements.
3. **HMI** — an operator console (touch display) shows the live pick list, highlights
   the active bin and where the operator's hand is, and raises faults.

---

## 2. Physical Setup & Hardware

### 2.1 Key design features
- **Camera elevation** — the camera is mounted above human height so it never
  obstructs the operator, and high enough that the **entire kitting area and all
  bins sit inside the field of view (FOV)**.
- **Anti-glare surface** — the stations use **matte ESD mats** on the table to cut
  glare and improve computer-vision accuracy.
- **Bin configuration** — a **9-bin grid**: three large bins on the lower tier, with
  three medium and three small bins above them.
- **Kit box** — a fourth load receptor (the "final kitting box") the operator packs
  into; a separate load cell verifies the packed contents.

### 2.2 Edge compute
- Advantech **MIC-733-AO**, which houses an **NVIDIA Jetson AGX Orin** (aarch64,
  JetPack 6 / L4T r36.4, CUDA 12.6). The full pipeline (camera decode, hand
  tracking, detection, overlay, dashboard) runs on-device at **~30 fps**.

### 2.3 Sensing hardware
- **Camera** — Logitech C270, 1280×720 MJPG.
- **Load cells** — an ESP32 (CP210x serial) reads an HX711 array and streams 10
  channels (9 bins + kit box) as JSON over USB serial at ~10 Hz.

`[TO ADD]` Photo of the physical station / rig → `sources/`.

---

## 3. Software Stack

| Tool | Role |
|---|---|
| **Ultralytics** | Computer-vision model training (YOLO bin detector) |
| **Roboflow** | Data annotation and augmentation |
| **AEGIS-CORE** | Testing pipeline to validate the trained CV models |

---

## 4. Computer-Vision Data Pipeline

The bin-detection model was built in three stages:

1. **Capture** — overhead camera above operator height, whole 9-bin grid in frame;
   matte ESD mats to cut glare; images taken with bins at varied fill levels and
   with a layer-by-layer permutation of bin positions (see §5).
2. **Annotate (Roboflow)** — bin **rims / edges** are labelled as the detection
   targets, then augmentation expands and diversifies the dataset. The design intent
   is a **rim-centric** detector that keys on bin edges, not on the objects inside.
3. **Train (Ultralytics)** — the YOLO bin detector is trained on the labelled set,
   iterated across the setup versions below, and exported for on-device deployment.

Every trained model is then validated in the **AEGIS-CORE** test pipeline.

`[TO ADD]` Pipeline diagram → `sources/` (a rendered 3-stage flowchart slide exists).

---

## 5. Model Development & Testing

**Test methodology — layer-by-layer permutation.** Each bin is treated as unique
(different fill levels, regardless of size). One bin is shifted at a time, tracking
its index: bin 1 is moved and swapped left-to-right one grid position at a time until
it reaches the end, then bin 2 does the same, and so on until all bins return to
their initial positions. Bins are tested **empty, partially filled, and fully
filled**, repeated for each bin size (S, M, L).

### 5.1 FINAL SETUP V1 — items in bins (same colour/reflectiveness)
- **Items:** all the same colour/reflectiveness (black, matte), differing in shape
  and size, repeated per bin size (S/M/L).
- **Objective:** determine whether training with items in the bins eliminates
  hallucinations — i.e. whether the model trains on the **bin edges/rims** and not on
  the objects inside.
- **Findings:** the model is **rim-centric** (regardless of colour) — it detects bins
  accurately despite same-colour items inside, and also correctly detected **silver
  items**.

### 5.2 FINAL SETUP V2 — screws and nuts
- **Change:** training items switched to screws and nuts.
- **Objective:** determine whether training with different items remains equally robust.
- **Findings:** `[TO ADD]`

### 5.3 FINAL SETUP V3 — varying fullness levels
- **Change:** trained on varying levels of fullness (0%, 20%, 40%, 60%, 80%, 100%).
- **Objective:** determine whether accuracy increases with more/richer data — i.e.
  whether more quality data yields a more robust model.
- **Findings:** `[TO ADD]`

`[TO ADD]` Per-version metrics (mAP / precision / recall), confusion, sample
detections → `sources/`.

---

## 6. Kitting Pipeline & Operator Console

### 6.1 Sequential single-bin FSM
Only **one bin is active at a time**. Because just one item type can be leaving a bin
at any moment, the kit box's weight change is unambiguous (no cross-bin weight
confusion). The operator starts a bin, packs it, and it locks green when complete;
the next bin can then be started.

### 6.2 Load-cell placement verification
A bin's count is driven by what actually lands in the **kit box**, not merely what
left the bin — the counter only advances on a genuine placement. Completed bins are
banked and frozen so their counters don't fluctuate.

### 6.3 Fault types
| Fault | Meaning |
|---|---|
| **Overpack** | More placed in the kit box than the target for the active bin |
| **Picked from wrong bin** | Taken from a bin not in this kit, *or* one already completed |
| **Returned to wrong bin** | An item placed into a bin it doesn't belong in |
| **Out of sequence** | Taken from an in-kit bin that's locked while another is active |

Each fault shows a big keyword + a corrective subtitle naming the specific bin and
quantity, highlights the physical bin, and sounds a buzzer; all auto-clear when
corrected.

### 6.4 HMI
9-bin grid mirroring the physical layout; the active bin pulses; a bold
colour-coded halo shows where the operator's hand is (teal = valid bin, red = a bin
they shouldn't pick from); a sets-remaining progress indicator; and an auto-appearing
"empty the kit box" confirmation between sets.

`[TO ADD]` HMI screenshots → `sources/`.

---

## 7. User Testing

### 7.1 Method
Six operators (Benny, Chloe, Ming Zhan, Song Yi, Royce, Jack) completed the full
protocol between **8–13 July 2026**: a within-subjects, counterbalanced comparison of
a paper **Control** block against the **AEGIS System** block, three BOMs per block.

### 7.2 Ratings summary
(1 = poor / strongly disagree, 5 = excellent / strongly agree; averaged across all 6
operators unless noted.)

| Dim | Question | Avg | Range | n |
|---|---|---|---|---|
| R1 | Screen wording clear and easy to understand | 3.3 | 2–5 | 6 |
| R2 | Placement / layout of on-screen info easy to follow | 4.2 | 3–5 | 6 |
| R3 | Easy to understand what to pack | 4.9 | 4.5–5 | 6 |
| R4 | Easy to understand how many to pack | 5.0 | 5–5 | 6 |
| R5 | Running count per bin helped keep track | 3.8 | 1–5 | 6 |
| R6 | Warnings / error alerts easy to understand | 3.3 | 2–5 | 6 |
| R7 | Corrective instructions clearly explained the fix | 3.0 | 1–4 | 5* |
| R8 | Overall trust the system would catch mistakes | 4.0 | 3–5 | 6 |
| R9 | Overall, the system made kitting easier | 4.3 | 2–5 | 6 |

\* Benny triggered no errors during his live runs, so R7 was not scored for him (n = 5).

**Strongest:** knowing *what* and *how many* to pack (R3–R4, at ceiling).
**Weakest:** on-screen wording, alert legibility, corrective-instruction completeness
(R1, R6, R7).

### 7.3 Key findings
1. **Screen wording & alert legibility** — small font, especially the overpick
   warning, was the single most repeated complaint (4 of 6 operators).
2. **Warning vs. error differentiation** — operators could tell an alert fired but
   often couldn't distinguish severity/type at a glance; inconsistent colours
   (white-on-red vs red-on-black) were called out.
3. **Corrective-instruction completeness** — lowest-rated dimension (R7 = 3.0); the
   wrong-bin-return correction didn't say *how many* to move or *which* bin.
4. **Load-cell responsiveness on small/light parts** — lag / missed reads on the
   smallest, lightest bins; lower trust for those items.
5. **Physical interaction & consolidation** — emptying a completed bin into the kit
   container was a friction/fatigue point; on-screen "complete" taps didn't always
   register.
6. **Progress / pacing visibility** — no indicator of how many sets remain.
7. **Strengths to preserve** — the build-list (what + how many) was rated at/near
   ceiling by every operator; the system was seen as easier than the paper baseline.

`[TO ADD]` Full per-operator notes / raw sheets → `sources/`.

### 7.4 Data-collection gap ⚠️
None of the 6 completed sessions have the **quantitative** run data filled in
(per-set timing, whole-order cycle time, flag/event log, escaped-defect counts). This
is needed for the cycle-time and defect-catch analysis and must still be reconciled
from the room-camera footage. `[TO ADD]`

---

## 8. Improvements Implemented (post user-testing)

Changes made to the operator console in direct response to the findings above:

- **Legibility** — larger fonts throughout; the fault banner is now a big keyword
  (`OVERPACK` / `PICKED FROM WRONG BIN` / `RETURNED TO WRONG BIN` / `OUT OF SEQUENCE`)
  with the corrective action as a subtitle.
- **Consistent alert language** — one red/white error palette (removed the
  white-on-red / red-on-black inconsistency).
- **Corrective completeness** — wrong-bin messages now name the specific bin **and
  quantity** to move.
- **Fault reclassification** — a finished (green) bin picked from now reads "wrong
  bin" (not merely out-of-sequence); a not-in-kit bin always errors and can never
  silently activate.
- **Hand location** — a bold, colour-coded hand-in-bin halo (teal = valid, red =
  wrong bin); removed the separate L/R chip.
- **Flow** — completion goes straight to the empty-box confirmation (no extra button);
  a mistake before emptying interrupts with the fault.
- **Bins & layers relabelled** — `BIN 1`–`BIN 9`, `LAYER 1` / `LAYER 2` (1-indexed).
- **Progress** — a sets-remaining indicator.
- **Counting robustness** — fixed a count flicker (box count no longer yanked to 0 by
  a transient bin-cell dip).

---

## 9. Known Limitations & Hardware Findings

- **Load-cell precision for light items.** The kit box is a coarse high-capacity
  cell; resolving a ~3 g item on top of a box already holding everything else is at
  the sensor's limit and can cause phantom counts / false overpacks — worst when the
  lightest bin is packed **last**. Mitigations: sequence the lightest bin earlier, or
  fit a finer kit-box load cell.
- **Sensor drift.** Bin cells are tared once at boot and not re-zeroed, so they drift
  over a session; this can nudge fault thresholds. `[TO ADD]` drift measurements.
- **Dead / disconnected cells.** Diagnosed three channels reading a flat 0.00 (grid
  BIN 1, BIN 2, BIN 9) — those load cells / wires need attention. `[TO ADD]`
- **Flaky ESP32 USB link.** The load-cell ESP32 repeatedly reset (kernel USB resets),
  dropping the serial link and crashing the pipeline — a cabling / power issue.
- **Lighting sensitivity.** A separate lux-vs-confidence characterization was run (bin
  detection stayed reliable across the tested range, with weaker confidence in the
  dimmest and one mid conditions). `[TO ADD]` reference the lighting workbook →
  `sources/`.

---

## 10. Conclusions & Future Work

- **Scalability.** The MIC (Jetson AGX Orin) runs a single station at ~10 % load, so a
  **second station** fits on the same box with no new compute — just another
  camera/ESP32 and config.
- **Kit-box sensing.** A finer load cell would make the lightest items reliable.
- **Close the data gap.** Reconcile the quantitative user-test metrics (cycle time,
  defect-catch) from the room-camera footage to complete the study.

---

## Appendix — Sources

Supporting material lives in [`./sources/`](./sources): station/rig photos, HMI
screenshots, model metrics, the CV pipeline diagram, user-test raw sheets, the
lighting-characterization workbook, and any exported Google Docs findings.
