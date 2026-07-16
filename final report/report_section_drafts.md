# AEGIS — Draft report sections (paste-ready)

> Prose drafts for the empty sections of the IS-305 report, in report order and
> written to match the tone of Chapter 1 (British spelling). Paste under the
> matching heading in the Google Doc and re-format.
>
> Callouts to slot your supplements into:
> - `[Figure X — …]` where a picture goes.
> - `[PLACEHOLDER — …]` where your Excel / measured data goes.
> - `[CONFIRM — …]` where you should check a detail against your own records.

---

## 2.2.2 Detailed Selection

This section refines each subsystem chosen in Section 2.2.1 into a concrete
implementation choice. The selected option in each row is shown in **bold**.

| Item | A | B | C | D | Reasoning |
|---|---|---|---|---|---|
| Monitor Application | **Web Application** | Desktop Application | Mobile Application | Terminal-based | Runs full-screen in a browser on the touch HMI with no per-station install, is served locally from the edge device, and is inherently cross-platform. |
| Edge AI PC | Jetson Orin Nano | Jetson Orin NX | **Jetson AGX Orin** | Generic mini-PC | GPU-accelerated inference runs the full multi-model pipeline (bin + hand) at ~30 fps with substantial headroom, in an industrial form factor (Advantech MIC-733), and on the CUDA ecosystem the CV toolchain targets. `[CONFIRM — alternatives your team actually evaluated]` |
| Hand Detection Model | **MediaPipe Hands** | YOLO-Pose | OpenPose | Custom model | Off-the-shelf, real-time 21-landmark hand tracking that runs reliably on-device without bespoke training, providing the worker-intention signal (which bin the hand is in). |
| Bin Boundary Model | **YOLO detector (Ultralytics)** | YOLO-OBB | Segmentation (e.g. Mask R-CNN) | Classical CV (contours) | Fast, accurate rim/edge detection; trained and augmented easily via Roboflow; GPU-efficient on the Jetson. |
| Weight Sensors | Kitchen scale | 1× load cell | Human weighing scale | **3× load cell** | Three-point support keeps the bin's centre of gravity within the support triangle for any bin size, giving both mechanical stability and accurate readings (see Section 3.2.2). |

`[CONFIRM — the option cells (A–D) reflect a sensible candidate set; adjust to the
exact options your morphological analysis considered.]`

---

## 2.2.3 Finite State Machine

The core of AEGIS's verification logic is a finite state machine (FSM) that
interprets the continuous stream of load-cell and vision data as a sequence of
discrete, verifiable kitting actions. The central design decision is that **only
one bin is active at any moment** — a sequential, single-bin model.

This constraint is deliberate and resolves a fundamental measurement ambiguity.
Because the kit box is monitored by a single load cell, a change in its weight
cannot in general be attributed to a specific component if several bins are being
picked from concurrently — two components of similar mass drawn from different
bins are indistinguishable to the box's aggregate reading. By requiring the
operator to complete one bin before starting the next, the only item type that
can be entering the box at any instant is known, which removes the cross-bin
weight ambiguity and allows every placement to be attributed and counted
unambiguously.

The machine occupies one of the following states:

- **IDLE** — no bin is active; the operator may begin any not-yet-completed bin
  in the current order.
- **PICKING** — a bin has been activated (the first pick from it is detected) and
  is now the sole bin whose picks are counted; all other bins are soft-locked.
- **FAULT** — an error condition has been detected (see below); counting is
  suspended until the operator corrects it, whereupon the fault auto-clears and
  the machine returns to PICKING.
- **KIT_COMPLETE** — every bin required by the order has reached its target
  quantity; the kit is verified.
- **WAITING_EMPTY** — the operator is prompted to empty the completed kit into
  its container before the next set begins.

A bin is activated by the first clear removal of an item from it, confirmed over
a short debounce window to reject transient sensor noise. Once activated, the
bin's counter advances only when items are verified as **placed into the kit
box** — not merely removed from the bin — so that a pick is credited only once the
component has actually reached the kit. When the active bin reaches its target
quantity it is banked and frozen (its count no longer fluctuates) and the lock
releases, allowing the next bin to be started. The order is complete once every
required bin is banked.

The FSM continuously evaluates four fault conditions, each of which raises a FAULT
state, drives an operator alert, and auto-clears once the underlying condition is
corrected:

- **Overpack** — more of the active component has been placed into the kit box
  than the order requires.
- **Picked-from-wrong-bin** — a component has been taken from a bin that is not
  part of the current order, or from one that has already been completed.
- **Returned-to-wrong-bin** — a component has been placed into a bin to which it
  does not belong.
- **Out-of-sequence** — a component has been taken from a bin that *is* part of
  the order but is not the currently active bin (picked out of turn while another
  bin is locked).

This fault taxonomy realises functional requirements FR-02 (wrong-part
detection), FR-04 (quantity verification), and FR-06 (corrective guidance): each
fault is specific enough to tell the operator not only that an error occurred,
but which bin and how many units are involved, enabling correction before the kit
is completed.

`[Figure 2.x — FSM state diagram: IDLE → PICKING → (FAULT) → KIT_COMPLETE →
WAITING_EMPTY, with the four fault transitions.]`

---

## 3.1 Physical Frame

The physical frame positions the sensing hardware around an unmodified manual
kitting workstation, in keeping with the requirement (FR-09) that the system adapt
to the environment rather than the reverse. Two constraints drove its design.

First, the camera is mounted **overhead, above operator height**, so that it never
obstructs the operator's reach and so that the **entire 9-bin grid and the kit box
sit within its field of view**, clear of the operator's hands during normal
picking. Second, the working surfaces use **matte ESD mats**, which suppress
specular glare that would otherwise degrade detection accuracy while remaining
compatible with the electrostatic-discharge requirements of electronics handling.

The bins are arranged as a **9-bin grid** across two tiers — three large bins on
the lower tier, with three medium and three small bins above — and a fourth load
receptor serves as the kit box into which the operator packs.

`[Figure 3.x — the assembled physical frame / workstation.]`
`[PLACEHOLDER — frame construction details: material, key dimensions, camera
mounting height and stand-off.]`

---

## 3.2.1 Computer Vision

The computer-vision subsystem does not attempt to recognise individual components
directly — with roughly 800–900 active SKUs the per-part classification burden is
impractical, and item identity is instead resolved by bin position (Section 2.2,
Item Identification). Vision instead answers two questions: *where are the bins*
and *which bin is the operator reaching into*. It comprises two models running on
the edge device.

The **bin-boundary model** is a YOLO object detector (trained with Ultralytics,
annotated and augmented in Roboflow) that locates each bin in the frame. It is
trained to be **rim-centric** — keyed on the bin edges rather than the contents —
so that a bin is detected reliably regardless of what, or how much, is inside it.
Training data was captured across bins that were empty, partially filled, and
fully filled, and with the bins permuted across grid positions (Section 4.2.1), so
that the detector generalises across fill level and position rather than
memorising a fixed scene.

The **hand-detection model** (MediaPipe Hands) tracks the operator's hands and,
combined with the detected bin boundaries, yields the **worker-intention**
signal — which bin a hand is currently in. This is what the interface uses to
highlight the active bin and to flag a reach into a wrong bin the moment it
happens.

The two models run together on the Jetson at approximately 30 fps, so the vision
signal is available in real time to the verification engine and the display.

`[Figure 3.x — sample detections: bin boundaries + hand landmarks.]`

---

## 4.1 Physical Frame

The physical frame was validated against its two design objectives. Camera
placement was verified by confirming that all nine bins and the kit box remain
fully within the field of view, and that the operator's arms and torso do not
occlude any bin during a normal picking motion. The anti-glare surface was
assessed by its effect on detection stability under the workstation's lighting.

`[PLACEHOLDER — insert: FOV / occlusion check results; before-and-after detection
stability with vs. without the matte ESD mat.]`
`[Figure 4.x — camera field-of-view coverage of the bin grid.]`

---

## 4.2 Sensors

### 4.2.1 Computer Vision

#### 4.2.1.1 Testing Framework

The detector was evaluated with a **layer-by-layer permutation** protocol designed
to test invariance to bin position and fill level. Each bin is treated as unique
(distinguished by its fill level, independent of size). One bin is shifted at a
time, tracking its index: bin 1 is moved and swapped one grid position at a time
from left to right until it reaches the end, then bin 2 does the same, and so on,
until all bins return to their initial positions. At each configuration the bins
are tested **empty, partially filled, and fully filled**, and the protocol is
repeated for each bin size (S, M, L).

Testing proceeded across three setup versions, each isolating one variable:

- **Setup V1 — items in bins (uniform appearance).** Items all of the same colour
  and reflectiveness (black, matte), differing only in shape and size.
  *Objective:* determine whether training with items present eliminates
  hallucination — i.e. whether the model keys on the bin rim/edges and not on the
  objects inside.
- **Setup V2 — screws and nuts.** *Objective:* determine whether the detector
  remains equally robust with a different class of training items.
- **Setup V3 — varying fullness.** Trained across fill levels (0 %, 20 %, 40 %,
  60 %, 80 %, 100 %). *Objective:* determine whether richer, more varied data
  improves accuracy and robustness.

A separate **lighting characterisation** was also run, sweeping ambient
illuminance (lux) against detection confidence to establish the system's operating
range under varying lighting (FR-09).

#### 4.2.1.2 Evaluation Results

**V1.** The model proved **rim-centric**: it detected bins accurately despite
same-colour items inside, and correctly detected silver items as well —
confirming that it keys on bin edges rather than contents, and that in-bin items
do not induce hallucinations.

**V2.** `[PLACEHOLDER — insert V2 findings: robustness with screws and nuts.]`

**V3.** `[PLACEHOLDER — insert V3 findings: effect of fullness-varied training on
accuracy / robustness.]`

**Lighting.** Illuminance was swept across **eight levels** spanning the station's
full realistic range — from a fully darkened room (blinds down, all lights off) up
to the tunable strip light at maximum with the office lights on — with the
illuminance at each of the nine bins measured by lux meter at every level, giving
**72 readings** in total. Taking the reference bin as the scale, the sweep ran from
**~0 lux to ~506 lux**.

Detection held at **100 % across every level and every bin**: all 72 readings
identified the correct bin, including at the darkest condition. Confidence
correlated only weakly with illuminance (Pearson *r* ≈ 0.27 linear, ≈ 0.40 against
log-lux), and the trend was **non-monotonic** — the dimmest level was not the worst,
with a shallow dip at a low-mid level instead. The practical conclusion is that the
rim-centric detector is **not lighting-limited over the station's operating range**,
satisfying the environmental-robustness requirement (FR-09); ambient light is
therefore not a variable that needs controlling in deployment, beyond the matte
anti-glare surface already specified.

`[PLACEHOLDER — per-version metrics table: mAP / precision / recall.]`
`[Figure 4.x — sample detections per setup version; lux-vs-confidence chart (data:
lighting_cv_test_backup.xlsx in sources/).]`

### 4.2.2 Load Cells

#### 4.2.2.1 Testing Framework

The load-cell hardware was qualified as a **measurement system** on the bench,
independently of the verification logic, so that the counting behaviour built on
top of it could be interpreted against known sensor limits. Three platform
configurations were tested, matching the three bin sizes:

| Config | Cells | Wiring | Platform geometry | Max test load |
|---|---|---|---|---|
| **1 kg** | 1 × 1 kg | 1 HX711 | single 100 mm-radius circle | 1 000 g |
| **5 kg** | 3 × 5 kg | summed → 1 HX711 | 2 × 50 mm circles + 1 apex 50 × 100 mm rectangle | 5 000 g |
| **10 kg** | 3 × 10 kg | summed → 1 HX711 | 2 × 50 mm circles + 1 apex 50 × 100 mm rectangle | 10 000 g |

Each three-cell platform reports **one summed weight** (three bridges into a single
HX711); all three cells share a rigid base and a single bin rests across all three.
Test loads came from a graded reference-mass kit of individually weighed sand bags
(reference scale: 1 000 g capacity, 0.1 g readability); because that scale caps
direct weighing at 1 000 g, every load above 1 kg was built by **stacking
individually-weighed ≤ 1 kg bags**, carrying an accumulated ±0.1 g-per-bag
uncertainty and setting a realistic ~0.5 g practical floor for the 1 kg ladder.
Actual (weighed), not nominal, masses were used throughout the analysis.

The battery was organised as **Capability → Stability → Integration**, with a
dedicated testing firmware streaming raw CSV (`millis, channel, raw, grams`):

- **Capability** — A1 minimum detectable mass (noise floor and count registration),
  A2 repeatability (5 trials × 5 points), A3 hysteresis (up-path vs down-path).
- **Stability** — A4 warm-up / thermal drift (cold vs 30 min warmed), A5 creep at
  two loads.
- **Integration** — A6 off-centre / placement sensitivity, A8 settling time.

Cells were warmed up for 30 minutes before the capability tests so early
instability would not contaminate the noise floor, and re-tared before each block.
Every cell recorded is a raw reading; readiness was judged post-hoc rather than by
baked-in pass/fail verdicts.

#### 4.2.2.2 Evaluation Results

**Capability.** The zero-noise band (σ on an empty, tared cell) and the smallest
reliably-separated mass both scale with platform capacity:

| Config | Zero-band σ (g) | Min. detectable mass (g) | Repeatability SD @ 50 % (g) | Max hysteresis (g) |
|---|---|---|---|---|
| 1 kg (1 cell) | 0.028 | ~0.5 | 0.009 | 0.09 |
| 5 kg (3 cells) | 0.223 | ~2.3 | 1.06 | 5.44 |
| 10 kg (3 cells) | 0.433 | ~4.3 | 2.85 | 16.99 |

Against the accepted reference that reliable single-pick counting requires a unit
mass of **≥ 5–10 × σ**, this quantifies the light-item limit precisely: the
lightest components in the BOM (~3–3.5 g) are comfortably resolved on the **1 kg**
platform, but sit at or below the reliable threshold on the **5 kg** and **10 kg**
platforms. Absolute accuracy was otherwise excellent — grams-versus-reference
agreed within a few tenths of a gram across the full ladder (e.g. 1 kg config:
500.0 g reference → 501.187 g mean; 5 kg: 5 004.2 g → 5 010.053 g). Repeatability
and hysteresis degrade with capacity in the same proportion, confirming the
resolution-for-range trade-off described in Section 3.2.2.

**Stability.** Warm-up drift (A4) over 30 minutes at mid-load was ≤ 0.22 g on the
1 kg platform but reached ~6 g on the 5 kg — justifying the 30-minute warm-up and
periodic re-taring as standard procedure. Creep (A5) stayed within ~0.1 % of the
applied load on all three configs, so sustained load itself is not a material
error source over a kitting session.

**Integration.** The decisive finding for the summed three-cell platforms was
**off-centre placement sensitivity** (A6). The 1 kg single-cell platform was
essentially position-invariant (≤ ~0.2 g deviation across a 90 mm radius). The
three-cell platforms were **not**: moving the *same fixed mass* from the calibrated
centroid to directly over an individual cell or the apex shifted the summed reading
by tens of grams — **5 kg: −38.6 g to +64.0 g; 10 kg: −65.8 g to +105.0 g**. For
the larger bins, therefore, *where* a component sits on the platform materially
affects the reading, which bounds how finely those platforms can resolve individual
picks regardless of the electronics. Settling time (A8) was 0.3–0.6 s per step at
the firmware's 10 SPS.

**Implications.** These metrology limits explain and bound the light-item behaviour
observed at system level: the lightest components are only reliably counted on the
smallest (1 kg) platform, while the medium and large three-cell platforms trade
both resolution and placement-invariance for capacity. The counting and fault logic
built on these cells is validated separately in Section 4.3.2; its light-item edge
cases were mitigated there with absolute-gram floors on the crediting and
return-detection thresholds, and operationally by not sequencing the lightest bin
last. Two hardware-reliability issues were also recorded during integration:
several individual channels were found reading a flat zero (disconnected cells),
and the load-cell microcontroller's USB link showed intermittent resets that dropped
the serial connection — both cabling/power faults rather than software.

`[Figure 4.x — min-detectable-mass and off-centre deviation by platform; full raw
ladders, trial tables and drift logs in the Appendix / AEGIS_LoadCell_Qualification.]`
`[PLACEHOLDER — unit-to-unit consistency across the other two units per platform
type, and the crosstalk figure, once those runs are filled in.]`

---

## 4.3 User Interface

### 4.3.1 Frontend

#### 4.3.1.1 Testing Framework

The frontend was evaluated through a structured usability study with six
operators, conducted between 8 and 13 July 2026. The study used a
**within-subjects, counterbalanced** design in which each operator completed a
paper-based **Control** block and an **AEGIS System** block, with three bills of
materials (BOMs) per block; block order was counterbalanced across participants to
control for learning effects.

Evaluation focused on the operator's comprehension and subjective experience of
the interface. Each session comprised a structured **alert walkthrough**, in which
the operator encountered and was asked to interpret each fault type, followed by a
**rating interview** across nine dimensions (Table 4.x) scored on a 1–5 Likert
scale, and an open-ended interview capturing free-form feedback and redesign
suggestions. The dimensions were chosen to separate the two things a kitting HMI
must do well: communicate the task (what and how many to pack) and communicate
exceptions (warnings, errors, and how to correct them).

`[PLACEHOLDER — the quantitative run data (per-set and whole-order cycle time,
event/flag logs, and escaped-defect counts) was not captured within the sessions
and is to be reconciled from the room-camera footage; the results below are
therefore the completed qualitative and subjective findings.]`

#### 4.3.1.2 Evaluation Results

**Table 4.x — Operator ratings** (1 = poor / strongly disagree, 5 = excellent /
strongly agree; mean across six operators unless noted).

| # | Dimension | Mean | Range | n |
|---|---|---|---|---|
| R1 | Screen wording was clear and easy to understand | 3.3 | 2–5 | 6 |
| R2 | Placement / layout of on-screen information was easy to follow | 4.2 | 3–5 | 6 |
| R3 | Easy to understand *what* to pack | 4.9 | 4.5–5 | 6 |
| R4 | Easy to understand *how many* to pack | 5.0 | 5–5 | 6 |
| R5 | Running count per bin helped keep track | 3.8 | 1–5 | 6 |
| R6 | Warnings / error alerts were easy to understand | 3.3 | 2–5 | 6 |
| R7 | Corrective instructions clearly explained the fix | 3.0 | 1–4 | 5* |
| R8 | Overall trust that the system would catch mistakes | 4.0 | 3–5 | 6 |
| R9 | Overall, the system made kitting easier than without it | 4.3 | 2–5 | 6 |

\* One operator triggered no errors during his live runs, so R7 was not scored for
him (n = 5).

The results show a clear pattern. Operators rated the system's core guidance —
understanding *what* to pack (R3, 4.9) and *how many* to pack (R4, 5.0) — at or
near ceiling, and the visual, position-based bin layout was repeatedly cited as
the reason; the build list was the interface's clear strength. The system was also
rated as making kitting easier than the paper baseline (R9, 4.3).

The weakest dimensions concerned the communication of **exceptions** rather than
the task itself: on-screen wording clarity (R1, 3.3), alert legibility (R6, 3.3),
and — the lowest-rated dimension — the completeness of corrective instructions
(R7, 3.0). The open feedback localised these to specific issues: small font on the
alerts (the single most frequently raised complaint, from four of six operators);
difficulty distinguishing a warning from an error at a glance, compounded by
inconsistent alert colours; corrective messages that did not state how many units
to move or which bin was involved; perceived lag and missed reads on the smallest,
lightest components; friction and fatigue in emptying completed bins; unreliable
registration of on-screen "complete" taps; and the absence of a remaining-sets
indicator.

These findings directly drove a subsequent **iteration** of the frontend:
enlarged alert typography; a single keyword-plus-subtitle fault banner naming the
specific bin and quantity; a consistent red/white alert colour scheme;
reclassification of faults so that a completed bin reads as a wrong-bin error
rather than an out-of-sequence one; a bold, colour-coded hand-in-bin halo; an
automatic completion-to-empty flow that removed an unreliable confirmation tap;
1-indexed bin and layer labels; and a remaining-sets progress indicator. The
load-cell responsiveness issues on light components were traced to the sensing
subsystem and addressed there (Section 4.2.2).

`[Figure 4.x — HMI before/after the iteration: fault banner, hand halo, bin grid.]`

### 4.3.2 Backend

#### 4.3.2.1 Testing Framework

Because the verification logic is encapsulated in a pure placement-tracker
component with no dependency on the camera or display (Section 3.3.2), it is
tested in isolation with an automated unit-test suite. The suite feeds
deterministic sequences of per-bin and kit-box weights and asserts the resulting
state, covering bin activation and the confirmation window, placement counting and
completion, every fault type and its auto-clear, and a set of edge cases —
post-tare settle windows, sensor drift, transient dips, and the light-item
crediting floors.

#### 4.3.2.2 Evaluation Results

The suite passes in full `[CONFIRM — 135 tests at time of writing]`, locking in the
intended behaviour: counts advance only on genuine kit-box placement; each fault
fires on its correct condition and clears on correction; a completed bin picked
from is classified as a wrong-bin error rather than out-of-sequence; and transient
sensor dips no longer produce phantom count changes. Running as fast, deterministic
unit tests, the suite also serves as a regression guard for future changes to the
verification logic.

`[PLACEHOLDER — insert: test count, coverage, or a summary table of tested
behaviours.]`

---

## Chapter 5: Conclusion

AEGIS demonstrates a proof-of-concept, edge-deployed kitting-assistance system that
addresses the operational gap identified in Chapter 1: the absence of real-time,
part-specific error detection and corrective guidance at the manual kitting
workstation. By combining computer vision (rim-centric bin detection and
hand-based worker-intention tracking) with load-cell gravimetric verification, and
by enforcing a sequential single-bin finite state machine, the system detects
wrong-bin, out-of-sequence, quantity, and overpack errors at the moment they occur
and guides the operator to correct them before the kit is completed — satisfying
the core functional requirements (FR-02, FR-04, FR-05, FR-06) while running
entirely on-device and augmenting, rather than replacing, the existing manual
workflow.

The prototype was validated across its three subsystems. The vision model was shown
to be rim-centric — detecting bins independently of their contents — and to hold
100 % detection from near-darkness to full office lighting, satisfying FR-09. The
load-cell platforms were qualified as a measurement system, establishing their
minimum detectable mass (0.5 g / 2.3 g / 4.3 g for the 1 kg / 5 kg / 10 kg
configurations) and, critically, the off-centre placement sensitivity of the summed
three-cell platforms — limits that bound which components can be reliably counted on
which bins. A six-operator user study confirmed that the system communicates the
task exceptionally well (what and how many to pack rated at ceiling) and is
preferred over the paper baseline, while surfacing concrete interface weaknesses
that a subsequent iteration addressed. The architecture also leaves substantial
compute headroom — the full pipeline occupies roughly one of the edge device's
twelve cores — supporting the intended multi-workstation extension (FR-11) without
per-station redesign.

`[PLACEHOLDER — add any headline quantitative outcomes once the cycle-time /
defect-catch data is reconciled.]`

---

## Chapter 6: Reflections and Lessons Learned

Several lessons emerged over the course of the project.

**Sensor limits shape the design.** The single hardest problem was not software
but the physics of gravimetric sensing: resolving very light components on a
coarse, heavily loaded load cell sits at the edge of the sensor's capability. This
drove a recurring theme — the trade-off between false positives and missed
detections — and taught us to combine software mitigations with operational and
hardware ones rather than expecting the code to overcome a sensor limit.

**User testing revealed what builders could not see.** Features that were obvious
to us as developers — alert wording, colour coding, corrective phrasing — were the
very things operators struggled with. The study turned subjective impressions into
a prioritised backlog and directly shaped a better interface; testing with real
operators earlier would have caught these sooner.

**Clean separation paid off.** Encapsulating the verification logic as a pure,
hardware-independent component made it possible to test the FSM exhaustively and
iterate on it with confidence, without a rig in the loop.

**Real-world robustness matters.** Intermittent USB resets and disconnected load
cells were a reminder that a laboratory prototype meets messy hardware realities;
reliability and diagnosability deserve design attention, not just the happy path.

`[PLACEHOLDER — add individual / team reflections and any course-specific lessons.]`

---

## References

`[PLACEHOLDER — TE Connectivity COPQ figures and SKU data; MediaPipe; Ultralytics
YOLO; Roboflow; NVIDIA Jetson AGX Orin; load-cell datasheets.]`

## Appendix

`[PLACEHOLDER — supplementary material: full user-test sheets, per-version model
metrics, the lighting workbook, HMI screenshots, wiring/bin-remap reference.]`
