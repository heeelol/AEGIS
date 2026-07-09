# AEGIS — Future Tasks / Roadmap

> **Always consult this file when working in this repo.** It holds deferred work that is
> intentionally out of scope for the *current* iteration. Guiding principle from the user:
> **make the solution work entirely first, then make it adaptable.** Do not pull these
> items into a task unless explicitly asked.

---

## A. Adaptability & Scalability (the big direction)

The current bin grid allocator (`aegis-v2/integration/src/detectors/grid_allocator.py`)
is built for the **fixed 6×2 rig** with hardcoded spans. The goal is to evolve it into a
**geometry-driven, multi-workstation** system that infers structure from the detections
themselves, requiring no per-station hardcoding.

- [ ] **Geometry-driven layout inference (Method B, generalized).** Replace the fixed
  top-6 / bottom-3 assumption with:
  - rows inferred by clustering detection y-centers (largest-gap / DBSCAN) → any number of layers;
  - column positions inferred from a per-snapshot **unit width** (median narrowest-box width),
    so it auto-calibrates to camera zoom/distance at each station;
  - gaps detected by spacing (gap > ~1 unit width → empty slot), valid in any position
    because the rig fills the frame.
- [ ] **Multi-workstation scalability.** Drop the hardcoded grid; support arbitrary bin
  counts / arrangements across different workstations. Optional per-station config used
  only as a sanity check / override, not as a requirement.

## B. Robust bin-size differentiation

Right now we cannot tell a small/medium bin from a large one from the image — size is
**hardcoded by row** (top row = 1-cell bins, bottom row = 2-cell bins). A future system
should detect bin size directly (e.g. from the OBB box dimensions / box width, or a size
class) and assign 1-cell vs 2-cell spans automatically, instead of assuming it from the
row. *(This falls out of the Method-B geometry inference above — the same unit-width step
that places bins also classifies their span.)*

## C. Downstream integration (from the original plan)

- [ ] **Wire `bins_indexed.json` into the live aegis-v2 dashboard/pipeline** (slot-proportional
  rendering already supports the contract).
- [ ] **Link ESP32 load-cell weights to bin indices 1–9.** When a hand reaches into a bin
  (CV) and an item is removed, the ESP detects the weight change and sends it to the
  computer; match that weight event to the correct bin index.
- [ ] **Hand-in-bin event attribution** against the indexed bins.
- [ ] **Distinguish "undetected-but-expected" grey from "not-in-job" grey** in the dashboard
  (currently both render identically).

---

_Source design: `aegis-v2/docs/superpowers/specs/2026-06-10-obb-grid-allocator-design.md`_
