# Kitting Operator UI — Design Spec

**Project:** IS-305 — AI machine-vision kitting workstation
**Screen:** Operator view (workstation touchscreen)
**Version:** v1 (first iteration — minimal)
**Status:** Layout + principles locked. Legends, colour palette, and full status set are open (see [Extension points](#extension-points)).

---

## 1. Purpose & scope

This is the single screen an operator watches while assembling one kit at the workstation. A fixed overhead camera runs YOLO11-seg over the bin rack; the model's per-bin output drives everything on screen. The screen's only job is to answer one question at a glance: **what do I do next?**

**In scope (v1):**
- Show the source bins and their per-bin detection status.
- Show the destination kit box and overall assembly progress.
- Show one clear status message and gate kit completion on the vision result.
- Show batch progress for the current run.

**Out of scope (v1)** — deferred, do not build yet:
- Manager-side kit builder / recipe configurator
- Bill of materials, cost allocation, weight, feasibility calculator
- Barcode-scan fallback
- Label printing and staging build-out (steps exist as stubs only)
- Mobile / RF-gun responsive layout, theme switching

---

## 2. Design principles (do not violate when extending)

These are the rules that make the screen calm. Any added legend, colour, or status must respect them.

1. **Verified is silent.** A bin that passes shows a quiet tick and nothing else — no numbers, no item name on the face. Detail is available on tap, not on the face.
2. **Attention is scarce.** Exactly one accent colour (amber) signals "needs action." Reserving it means it always means something. Do **not** introduce new bright colours that compete for the eye unless they carry an equally urgent meaning.
3. **The status bar carries the instruction.** It is the only element that tells the operator what to do. Every system state must map to one unambiguous status-bar message.
4. **The completion gate is sacred.** The "Complete kit" action stays locked until every bin reaches a verified state. This is the whole point of the vision system in the workflow.
5. **Design around real model outputs.** The UI consumes only fields the model actually produces (see [Data contract](#5-data-contract)). Do not invent fields (e.g. exact item counts in cluttered bins, which the model cannot reliably deliver).

---

## 3. Layout

Three content zones plus a thin progress bar, top to bottom:

```
+--------------------------------------------------------------+
|  Batch progress            Kit 45 of 200                     |  <- progress bar (top)
|  [====------------------------------------------]            |
+--------------------------------------------------------------+
|  Bins                                |  Kit box               |
|  +------+ +------+ +------+          |  +------------------+   |
|  |  A1  | |  A2  | |  A3  |          |  |                  |   |
|  |  ok  | |  ok  | |  ok  |          |  |      5 / 6       |   |
|  +------+ +------+ +------+          |  |   (fill level)   |   |
|  +------+ +======+ +------+          |  |                  |   |
|  |  B1  | | B2!! | |  B3  |          |  +------------------+   |
|  |  ok  | |CHECK | |  ok  |          |                        |
|  +------+ +======+ +------+          |                        |
+--------------------------------------------------------------+
|  (•) Check bin B2 before completing       [ Complete kit ]   |  <- status bar (bottom)
+--------------------------------------------------------------+
```

### Layout values (current implementation)

| Element | Value |
|---|---|
| Screen container | padding 20px, radius `lg`, 0.5px border |
| Body grid columns | `minmax(0, 1fr)` (bins) + `230px` (kit box), gap 20px |
| Bins grid | `repeat(3, 1fr)`, gap 10px — 2 rows × 3 cols, mirrors the two-tier rack |
| Bin tile | min-height 80px, radius `md` |
| Kit box | height 170px, radius `md`, bottom-anchored fill |
| Status bar | separated by 0.5px top border; flex, space-between |
| Action button | min-height 48px (touch target ≥ 44px) |

The bin grid maps 1:1 to the physical rack: top row = upper tier (A1–A3), bottom row = lower tier (B1–B3). Because the overhead geometry cannot give equally clean views of both tiers, expect detection quality to differ by row; the per-bin status surfaces this honestly.

---

## 4. Zones

### 4.1 Batch progress bar (top)
- **Shows:** progress through the current production run, e.g. `Kit 45 of 200`.
- **Why batch, not current kit:** the kit box already shows this kit's count; the top bar adds the only thing nothing else shows — how far through the shift the operator is.
- **Colour:** neutral grey fill. Deliberately not coloured, to keep amber unique to attention.
- **Value source:** `kit.batch.done / kit.batch.target`.

### 4.2 Bins
- **Shows:** one tile per source bin, in rack order.
- **Verified tile:** neutral border, quiet tick. No text beyond the bin label.
- **Needs-action tile:** amber border, amber icon + one-word state. This is the only coloured tile on screen.
- **Value source:** per-bin `status` (see state model).

### 4.3 Kit box
- **Shows:** the destination container with aggregate fill — `placed / total`.
- **Role split:** bins answer *"which one is wrong"*; the kit box answers *"how close to done."* No per-item detail here.
- **Value source:** `kit.placed / kit.total`; fill height = placed/total.

### 4.4 Status bar (bottom)
- **Shows:** a status dot + one sentence (the next action), and the gated completion button.
- **Value source:** the worst current bin state (drives message + dot), plus the completion gate.

---

## 5. Data contract

The UI is a pure function of the vision system's output. Per-bin object the front end consumes:

```json
{
  "bin_id": "B2",
  "position": { "tier": "lower", "slot": 2 },
  "expected": { "sku": "SP-200", "name": "Screen protector", "qty": 2 },
  "detected": { "class": "screen_protector", "confidence": 0.62 },
  "flags": ["protrusion"],
  "status": "needs_check"
}
```

Kit-level object:

```json
{
  "kit_sku": "KIT-SB100",
  "name": "Starter bundle",
  "placed": 5,
  "total": 6,
  "batch": { "done": 45, "target": 200 },
  "bins": [ /* per-bin objects */ ]
}
```

### Status derivation (tunable)
`status` is computed from the model output, not sent by hand:

| Condition | Resulting status |
|---|---|
| `detected.class == expected` AND `confidence >= T_high` AND no blocking flags | `verified` |
| `T_low <= confidence < T_high` OR any flag in `flags` | `needs_check` |
| `detected.class != expected` AND `confidence >= T_high` | `mismatch` |
| no detection OR `confidence < T_low` | `missing` |

`T_high` and `T_low` are confidence thresholds. **Defaults to confirm:** `T_high = 0.90`, `T_low = 0.50`. These are placeholders — validate against real detections before locking (the same caveat that applies to the Δp assumption elsewhere in the project).

---

## 6. State model

### 6.1 Bin states

Each status needs: a name, a trigger, a visual treatment, and a legend label. Implemented in v1:

| Status | Meaning | Visual | Legend label |
|---|---|---|---|
| `verified` | Correct item, model confident | neutral tile, success tick | (implicit — silent) |
| `needs_check` | Low confidence or a flag (protrusion/occlusion) | amber border, amber alert + word | "Check" |

> **TODO (you will specify):** `mismatch`, `missing`, plus any others (`empty`, `idle`, `overfilled`…). Define each with the four fields above. See [Extension points](#extension-points).

### 6.2 Completion gate
- `Complete kit` is **enabled only when every bin is `verified`.**
- In all other states the button is locked (greyed, lock icon).

### 6.3 Status-bar states

| State | Trigger | Dot | Message (template) | Button |
|---|---|---|---|---|
| `checking` | detection in progress / all pending | neutral | "Checking bins…" | locked |
| `action_required` | any bin not `verified` | amber | "Check bin {id} before completing the kit" | locked |
| `ready` | all bins `verified` | success | "All bins verified — ready to complete" | enabled |

> **TODO (you will specify):** `paused`, `mismatch`/`shortage` wording, multi-bin messaging (when more than one bin needs action). Status-bar wording matters most here — it is the only instruction the operator gets.

---

## 7. Colour tokens

Map these **semantic roles** to your repo's design tokens. v1 uses only four roles. Suggested defaults below are starting points — replace/extend as you finalise the palette.

| Role | Used for | Suggested default (light) |
|---|---|---|
| `neutral / surface` | tiles, kit box bg, progress track | greys |
| `success` | verified tick, kit-box fill, ready dot | green |
| `warning` | needs-check border/icon, action dot | amber |
| `danger` | (reserved) mismatch / missing | red |

> **TODO (you will specify):** the full palette and the complete status→colour legend. Keep principle #2 in mind — adding more accent colours dilutes the amber signal, so confirm each new colour earns its place.

---

## 8. Extension points

The handoff list — what to tell Claude Code to add on top of this spec:

- [ ] Additional bin statuses (`mismatch`, `missing`, `empty`, …) — fill the table in §6.1.
- [ ] Full colour palette and status→colour mapping — §7.
- [ ] On-screen legend block (status word → meaning) — currently the design relies on a single "Check" word; decide whether an explicit legend is shown or kept off-screen.
- [ ] Tap-to-expand bin detail (confidence number, expected vs detected, crop) — required because the face intentionally hides detail.
- [ ] Full status-bar wording for every state, including multi-bin and paused — §6.3.
- [ ] Confirm confidence thresholds `T_high` / `T_low` against real detections — §5.
- [ ] Pause / resume and exception-flag behaviour.

---

## 9. Notes for implementation

- Framework-agnostic. The reference was built in HTML/CSS; the layout values in §3 translate directly to fl/grid in any framework.
- Touch targets ≥ 44px; the completion button is 48px.
- A deployed touchscreen should lock to a single high-contrast theme (the station runs under controlled lighting), rather than following a system light/dark setting.
