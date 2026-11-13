"""Grid Session
===============
In-memory state holder for the two-snapshot operator flow:

  * ``calibrate(detections)`` — snapshot 1: all 9 bins -> the fixed grid (slots).
  * ``init_kit(detections)``  — snapshot 2: which calibrated slots hold a bin.

Pure geometry: it wraps ``grid_calibrator`` (no camera/model/cv2), so it
unit-tests in milliseconds. The pipeline turns its output into the
``bin_{row}_{col}`` geofence dict the dashboard and overlay already consume via
``to_geofences``.

Slot -> dashboard-bin mapping (the preprogrammed 6-top + 3-bottom grid):

    slot 1..6 (top)    -> row 0, bin_0_0..bin_0_5, span 1, slot_start 0..5
    slot 7..9 (bottom) -> row 1, bin_1_0..bin_1_2, span 2, slot_start 0,2,4

Both rows render against a 6-slot track (``row_slots = 6``); a bottom bin spans
two slots so the 3 bottom bins line up under the 6 top bins.
"""
from __future__ import annotations

import logging
import os
import sys

# grid_calibrator does a bare ``import grid_allocator``; both live in this
# directory. Mirror the path setup grid_calibrator/snapshot_obb already use so
# this module imports the same way whether loaded as a package submodule or by
# path in a test.
sys.path.insert(0, os.path.dirname(__file__))
import grid_calibrator as gc  # noqa: E402

logger = logging.getLogger("aegis.detectors.grid_session")

DEFAULT_LAYOUT = [6, 3]  # per-row bin counts; the classic 6-top + 3-bottom grid


def _row_slots(layout: list[int]) -> int:
    """Width of the slot track every row renders against (the widest row)."""
    return max(layout) if layout else 1


def _slot_layout(index: int, layout: list[int]) -> tuple[int, int, int, int]:
    """Map a grid slot index (1..N) to (row, col, span, slot_start) for ``layout``
    (per-row bin counts). Every row renders against a ``max(layout)``-slot track,
    so a row with fewer bins spans wider and the rows stay aligned
    (e.g. ``[6, 3]`` -> top span 1, bottom span 2; ``[6]`` -> a single span-1 row)."""
    track = _row_slots(layout)
    start = 1
    for row, count in enumerate(layout):
        if index < start + count:
            col = index - start
            span = track // count if count and track % count == 0 else 1
            return row, col, span, col * span
        start += count
    # index out of range -> best-effort single-row placement
    return 0, index - 1, 1, index - 1


def _bbox_and_polygon(corners) -> tuple[int, int, int, int, list]:
    """Axis-aligned bbox + integer polygon from 4 OBB corners (np array or list)."""
    pts = [(float(x), float(y)) for x, y in corners]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    polygon = [[int(round(x)), int(round(y))] for x, y in pts]
    return int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys)), polygon


class GridSession:
    """Holds the calibrated grid and the current kit occupancy."""

    def __init__(self, layout: list[int] | None = None):
        # Per-row bin counts, e.g. [6, 3] (default) or [6] for a single top row.
        self._layout = list(layout) if layout else list(DEFAULT_LAYOUT)
        self._row_slots = _row_slots(self._layout)
        self._calibration: dict | None = None   # slots dict from calibrate_grid
        self._occupancy: dict | None = None      # per-slot result from match_to_grid

    # ── State ────────────────────────────────────────────────
    @property
    def calibrated(self) -> bool:
        return self._calibration is not None

    @property
    def calibration(self) -> dict | None:
        return self._calibration

    @property
    def occupancy(self) -> dict | None:
        return self._occupancy

    # ── Snapshot 1: workstation grid ─────────────────────────
    def calibrate(self, detections: list) -> dict:
        """Lock the fixed slots from a full-grid snapshot.

        Wraps ``calibrate_grid`` (raises ``ValueError`` if the snapshot doesn't
        match this session's ``layout``, e.g. 6 top + 3 bottom by default).
        Recalibrating clears any prior kit.
        """
        self._calibration = gc.calibrate_grid(detections, self._layout)
        self._occupancy = None
        return self._calibration

    # ── Snapshot 2: kit occupancy ────────────────────────────
    def init_kit(self, detections: list) -> dict:
        """Match a kit snapshot's detections to calibrated slots -> occupancy.

        Raises ``RuntimeError`` if called before ``calibrate``.
        """
        if not self.calibrated:
            raise RuntimeError("Calibrate the workstation grid before initialising the kit.")
        self._occupancy = gc.match_to_grid(detections, self._calibration)
        return self._occupancy

    # ── Derived views ────────────────────────────────────────
    def _slot_present(self, slot_id: str) -> bool:
        """A slot is 'present' if the kit matched a bin to it. Before a kit is
        initialised, every calibrated slot counts as present (the locked grid)."""
        if self._occupancy is None:
            return True
        return bool(self._occupancy.get(slot_id, {}).get("present", False))

    @property
    def present_slots(self) -> list[int]:
        if self._calibration is None:
            return []
        return sorted(info["index"] for sid, info in self._calibration.items()
                      if self._slot_present(sid))

    @property
    def missing_slots(self) -> list[int]:
        if self._calibration is None:
            return []
        return sorted(info["index"] for sid, info in self._calibration.items()
                      if not self._slot_present(sid))

    def to_geofences(self, present_only: bool = False) -> dict:
        """Convert slots to the ``bin_{row}_{col}`` geofence dict.

        Present bins use the live (snapshot-2) box; missing bins fall back to the
        calibrated (snapshot-1) box with ``detected=False`` so the dashboard greys
        them. With no kit yet, all calibrated slots render as the locked grid.
        ``present_only=True`` drops missing bins entirely (used for the assignment
        engine so a hand can't be assigned to an empty slot).
        """
        if self._calibration is None:
            return {}

        geofences: dict = {}
        for sid, info in self._calibration.items():
            present = self._slot_present(sid)
            if present_only and not present:
                continue

            occ = self._occupancy.get(sid) if self._occupancy is not None else None
            if present and occ is not None and occ.get("corners") is not None:
                corners = occ["corners"]
                confidence = float(occ.get("confidence", 0.0))
            else:
                corners = info["corners"]
                confidence = 1.0 if self._occupancy is None else 0.0

            row, col, span, slot_start = _slot_layout(info["index"], self._layout)
            x_min, x_max, y_min, y_max, polygon = _bbox_and_polygon(corners)
            geofences[f"bin_{row}_{col}"] = {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "confidence": confidence,
                "span": span,
                "slot_start": slot_start,
                "row_slots": self._row_slots,
                "detected": present,
                "polygon": polygon,
            }
        return geofences
