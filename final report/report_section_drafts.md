# AEGIS — Draft report sections (paste-ready)

> Prose drafts for the empty sections of the IS-305 report, written to match the
> tone of Chapter 1. Paste into the Google Doc under the matching heading and
> re-format. `[PLACEHOLDER — …]` marks data only your team can supply.

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

---

## 3.3.1 Frontend

The frontend is the operator-facing HMI: a full-screen web dashboard rendered on
the workstation's touch display and served locally from the edge device. It is
designed for **glanceability** — an operator working at production pace must
absorb the current instruction, and any fault, in a fraction of a second and
without reading dense text — and for a keyboard-free touch environment.

The central element is a **bin grid that mirrors the physical 9-bin layout** (one
row per tier), so that the on-screen position of each bin corresponds directly to
its physical position. Each tile shows its identifier and a running count against
target (e.g. 2 / 3). Colour and motion encode state: the active bin pulses,
completed bins turn green and freeze, and bins not part of the current order are
dimmed.

To make the operator's hand position unmistakable, the tile the operator's hand
is currently over is given a bold, animated **halo**, colour-coded by
correctness — teal when the hand is in a valid bin (the active or an available
one) and red when it is in a bin the operator should not be picking from (already
completed, locked out of sequence, or not in the order). A misdirected reach is
therefore signalled the instant the hand enters the wrong bin, before any item is
removed.

Faults are surfaced through a full-width **banner** presenting a single large
keyword (OVERPACK, PICKED FROM WRONG BIN, RETURNED TO WRONG BIN, OUT OF SEQUENCE)
with the specific corrective action beneath it as a subtitle — naming the bin and
the quantity to move — together with a direct highlight of the offending tile on
the grid. A consistent red/white colour language is used across all alerts to
avoid ambiguity between severities. Between sets, completion flows directly into
an "empty the kit box" confirmation, and a progress indicator shows how many sets
remain in the order.

The frontend holds no state of its own: it polls the backend for the live system
state and renders it, so the display is always a faithful view of the
verification engine's current understanding.

---

## 3.3.2 Backend

The backend runs entirely on the edge device and couples the real-time sensing
pipeline to the operator display without exposing either to the other's timing.
It is organised around a **thread-safe shared state object**: the vision and
load-cell processing loop writes its latest interpretation of the scene into this
object each frame, while a lightweight web server reads from it to answer requests
from the frontend. This separation lets the sensing loop run at full frame rate
independently of how often the display refreshes.

The verification logic is encapsulated in a **placement tracker**, which
implements the finite state machine of Section 2.2.3. On each update it ingests
the current per-bin and kit-box weights, debounces them, and derives the discrete
kitting state — which bin is active, how many items have been removed from each
bin and placed into the kit, whether the order is complete, and whether any fault
condition holds. The tracker is deliberately a pure, self-contained component with
no direct dependency on the camera or the display, which allows it to be exercised
in isolation under an automated test suite (Section 4.3.2).

The web server exposes the system state to the frontend through a small set of
read endpoints (bin states and counts, grid layout, kit and cycle status, and
system statistics) and a few action endpoints (for example, confirming that the
kit box has been emptied to advance to the next set). All processing and storage
are performed on-device; no production data is transmitted to external networks,
satisfying the data-sensitivity constraint established in the non-functional
requirements of Chapter 1.

---

## 4.3.1 Frontend

### 4.3.1.1 Testing Framework

The frontend was evaluated through a structured usability study with six
operators, conducted between 8 and 13 July 2026. The study used a **within-subjects,
counterbalanced** design in which each operator completed a paper-based **Control**
block and an **AEGIS System** block, with three bills of materials (BOMs) per
block; block order was counterbalanced across participants to control for learning
effects.

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

### 4.3.1.2 Evaluation Results

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
rather than an out-of-sequence one; a bold, colour-coded hand-in-bin halo;
an automatic completion-to-empty flow that removed an unreliable confirmation tap;
1-indexed bin and layer labels; and a remaining-sets progress indicator. The
load-cell responsiveness issues on light components were traced to the sensing
subsystem and addressed there (Section 4.2.2).
