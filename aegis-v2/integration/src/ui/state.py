"""
Shared Pipeline State
======================
Thread-safe state object that the pipeline loop writes to and the
FastAPI dashboard reads from. This is the bridge between the real-time
CV pipeline and the web UI.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from ..sensing import LayerLayout


@dataclass
class BinStatus:
    """Status of a single bin for the dashboard."""
    bin_id: str
    label: str
    x_min: float = 0
    x_max: float = 0
    y_min: float = 0
    y_max: float = 0
    confidence: float = 0.0
    span: int = 1                    # slots this bin occupies in its row
    slot_start: int = 0              # first slot index in the row's track
    row_slots: int = 0               # total slots in this bin's row (0 = unknown)
    detected: bool = True            # False → declared slot the model didn't find
    is_active: bool = False          # Hand currently in this bin
    hand_id: Optional[int] = None
    handedness: str = ""
    pick_count: int = 0              # Successful picks from this bin
    target_count: int = 0            # Expected picks (from work order)
    using: bool = True               # Is this bin part of the current job


@dataclass
class HandStatus:
    """Current status of a detected hand."""
    hand_id: int
    handedness: str
    position: tuple[float, float] = (0.0, 0.0)
    assigned_bin: Optional[str] = None


@dataclass
class PipelineState:
    """
    Thread-safe container for the live pipeline state.

    The pipeline loop calls update_*() methods.
    The FastAPI endpoints call get_*() methods.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._bins: dict[str, BinStatus] = {}
        self._hands: list[HandStatus] = []
        self._errors: list[dict] = []
        self._fps: float = 0.0
        self._last_update: float = 0.0
        self._frame_count: int = 0
        self._start_time: float = time.time()

        # Load-cell view of the layout (rows × bins) and per-bin weights.
        # Empty until the load-cell driver is implemented; the layout merge
        # falls back to the CV-detected grid in that case.
        self._loadcell_layout: LayerLayout = LayerLayout()
        self._loadcell_weights: dict[str, float] = {}

        # Kitting-box (3-load-receptor demo): latest snapshot from the placement
        # tracker, plus a one-shot "complete kit" request raised by the UI and
        # consumed by the pipeline (which re-tares the receptors for the next kit).
        self._kit: dict = {}
        self._complete_requested: bool = False

        # Sequential pick lock: the one bin currently in progress. The first pick on
        # a bin locks it (only it counts); hitting its target completes it (added to
        # `_done`) and releases the lock so any other bin can be started next.
        self._active_bin: Optional[str] = None
        self._done: set[str] = set()

    # ── Writers (called by the pipeline loop) ────────────────

    def update_bins(self, geofences: dict, active_bin_ids: set[str] | None = None) -> None:
        """Update bin map from detected geofences."""
        with self._lock:
            active = active_bin_ids or set()
            for bid, coords in geofences.items():
                if bid not in self._bins:
                    self._bins[bid] = BinStatus(bin_id=bid, label=bid)
                b = self._bins[bid]
                b.x_min = coords.get("x_min", 0)
                b.x_max = coords.get("x_max", 0)
                b.y_min = coords.get("y_min", 0)
                b.y_max = coords.get("y_max", 0)
                b.confidence = coords.get("confidence", 0.0)
                b.span = coords.get("span", 1)
                b.slot_start = coords.get("slot_start", 0)
                b.row_slots = coords.get("row_slots", 0)
                b.detected = coords.get("detected", True)
                b.is_active = bid in active

    def update_hands(self, hands: list, events: list) -> None:
        """Update hand positions and bin assignments."""
        with self._lock:
            self._hands = []
            event_map = {ev.hand_id: ev for ev in events}
            for hand in hands:
                center = getattr(hand, "center", None)
                if center is None:
                    center = (0.0, 0.0)
                ev = event_map.get(hand.hand_id)
                self._hands.append(HandStatus(
                    hand_id=hand.hand_id,
                    handedness=hand.handedness,
                    position=center,
                    assigned_bin=ev.bin_id if ev else None,
                ))

            # Update bin active states
            active_ids = {ev.bin_id for ev in events if ev.bin_id is not None}
            for bid, b in self._bins.items():
                b.is_active = bid in active_ids
                matching = [ev for ev in events if ev.bin_id == bid]
                if matching:
                    b.hand_id = matching[0].hand_id
                    b.handedness = matching[0].handedness
                else:
                    b.hand_id = None
                    b.handedness = ""

    def record_pick(self, bin_id: str) -> None:
        """Increment pick count, enforcing the sequential one-bin-at-a-time lock."""
        with self._lock:
            b = self._bins.get(bin_id)
            if b is None:
                return
            # First pick locks this bin; while locked, only this bin counts.
            if self._active_bin is None:
                self._active_bin = bin_id
            elif self._active_bin != bin_id:
                return  # another bin is in progress → ignore out-of-order pick
            b.pick_count += 1
            # Target met → complete the bin and release the lock for the next one.
            if b.target_count > 0 and b.pick_count >= b.target_count:
                self._done.add(bin_id)
                self._active_bin = None

    def adjust_pick_count(self, bin_id: str, delta: int) -> Optional[int]:
        """Nudge a bin's pick count by `delta`. Clamped to ≥0. Returns new value."""
        with self._lock:
            b = self._bins.get(bin_id)
            if b is None:
                return None
            b.pick_count = max(0, b.pick_count + int(delta))
            return b.pick_count

    def set_pick_count(self, bin_id: str, count: int) -> Optional[int]:
        """Set a bin's pick count to `count`. Clamped to ≥0. Returns new value."""
        with self._lock:
            b = self._bins.get(bin_id)
            if b is None:
                return None
            b.pick_count = max(0, int(count))
            return b.pick_count

    def add_error(self, bin_id: str, message: str) -> None:
        """Record an error event."""
        with self._lock:
            self._errors.append({
                "bin_id": bin_id,
                "message": message,
                "timestamp": time.time(),
            })
            # Keep last 50 errors
            if len(self._errors) > 50:
                self._errors = self._errors[-50:]

    def clear_errors(self) -> None:
        with self._lock:
            self._errors.clear()

    def update_fps(self, fps: float) -> None:
        with self._lock:
            self._fps = fps
            self._last_update = time.time()
            self._frame_count += 1

    def update_loadcells(
        self,
        layout: LayerLayout,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Update the load-cell layout and per-bin weights (called by pipeline)."""
        with self._lock:
            self._loadcell_layout = layout or LayerLayout()
            if weights is not None:
                self._loadcell_weights = dict(weights)

    def update_kit(self, kit: dict) -> None:
        """Store the latest kitting-box snapshot (called by the pipeline)."""
        with self._lock:
            self._kit = dict(kit)

    def request_complete(self) -> None:
        """UI asks to close the current kit (one-shot; consumed by the pipeline)."""
        with self._lock:
            self._complete_requested = True

    def consume_complete_request(self) -> bool:
        """Pipeline checks/clears the close-kit request. True if one was pending."""
        with self._lock:
            pending = self._complete_requested
            self._complete_requested = False
            return pending

    # ── Readers (called by FastAPI endpoints) ────────────────

    def get_bins(self) -> list[dict]:
        with self._lock:
            result = []
            removed_map = self._kit.get("removed", {}) if self._kit else {}
            for b in self._bins.values():
                status = self._calculate_bin_status(b)
                layer, col = self._parse_bin_id(b.bin_id)
                result.append({
                    "id": b.bin_id,
                    "label": b.label,
                    "layer": layer,
                    "col": col,
                    "current": b.pick_count,           # placed (verified in box)
                    "removed": removed_map.get(b.bin_id, b.pick_count),  # taken out of bin
                    "total": b.target_count,
                    "status": status,
                    "using": b.using,
                    "is_active": b.is_active,
                    "handedness": b.handedness,
                    "confidence": b.confidence,
                    "weight": self._loadcell_weights.get(b.bin_id, 0.0),
                })
            return result

    def get_kit(self) -> dict:
        """Latest kitting-box state for the dashboard, incl. the sequential lock.

        ``active`` = the bin currently in progress (the UI locks/dims the others);
        ``done`` = completed bins (shown green/frozen). Read by ``binUiState`` in app.js.
        """
        with self._lock:
            kit = dict(self._kit)
            kit["active"] = self._active_bin
            kit["done"] = sorted(self._done)
            return kit

    def get_hands(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "hand_id": h.hand_id,
                    "handedness": h.handedness,
                    "x": h.position[0],
                    "y": h.position[1],
                    "assigned_bin": h.assigned_bin,
                }
                for h in self._hands
            ]

    def get_errors(self) -> list[dict]:
        with self._lock:
            return list(self._errors)

    def get_stats(self) -> dict:
        with self._lock:
            uptime = time.time() - self._start_time
            return {
                "fps": self._fps,
                "frame_count": self._frame_count,
                "uptime_seconds": uptime,
                "last_update": self._last_update,
                "num_bins": len(self._bins),
                "num_hands": len(self._hands),
            }

    def get_layout(self) -> dict:
        """
        Return the bin grid layout (layers × bins) for the dashboard to render.

        Merges two sources:
          - CV: bins detected by the vision model, grouped by their row (layer)
                from the ``bin_{row}_{col}`` id.
          - Load cells: the layout reported by the weight-sensor array.

        For each layer present in either source, the bin count is the larger of
        the two. The dashboard renders one row per layer so empty/not-yet-seen
        slots still appear as placeholders.
        """
        with self._lock:
            # Group bins by layer (row), carrying their slot placement.
            rows: dict[int, list] = {}
            for b in self._bins.values():
                layer, col = self._parse_bin_id(b.bin_id)
                rows.setdefault(layer, []).append((col, b))

            lc_counts = dict(self._loadcell_layout.bins_per_layer)
            lc_connected = self._loadcell_layout.num_layers > 0
            all_layers = set(rows) | set(lc_counts)

            layers = []
            for layer in sorted(all_layers):
                bins_in_row = sorted(
                    rows.get(layer, []), key=lambda t: (t[1].slot_start, t[0])
                )
                if bins_in_row:
                    # Explicit slot layout (from LayoutMapper) when row_slots is
                    # known; otherwise fall back to one uniform slot per bin so
                    # the legacy layout still renders correctly.
                    explicit = max((b.row_slots for _, b in bins_in_row), default=0)
                    if explicit > 0:
                        row_slots = explicit
                        bin_list = [
                            {"id": b.bin_id, "slot_start": b.slot_start,
                             "span": b.span, "detected": b.detected}
                            for _, b in bins_in_row
                        ]
                    else:
                        row_slots = len(bins_in_row)
                        bin_list = [
                            {"id": b.bin_id, "slot_start": i,
                             "span": 1, "detected": b.detected}
                            for i, (_, b) in enumerate(bins_in_row)
                        ]
                else:
                    # Layer known only from load cells — placeholders.
                    n = lc_counts.get(layer, 0)
                    row_slots = n
                    bin_list = [
                        {"id": f"bin_{layer}_{c}", "slot_start": c,
                         "span": 1, "detected": False}
                        for c in range(n)
                    ]

                layers.append({
                    "layer": layer,
                    "row_slots": row_slots,
                    "num_bins": len(bin_list),
                    "bins": bin_list,
                })

            return {
                "num_layers": len(all_layers),
                "num_bins": sum(l["num_bins"] for l in layers),
                "layers": layers,
                "source": {"cv": bool(rows), "loadcells": lc_connected},
            }

    def set_work_order(self, bin_targets: dict[str, int]) -> None:
        """Set target pick counts per bin from a work order."""
        with self._lock:
            for bid, target in bin_targets.items():
                if bid in self._bins:
                    self._bins[bid].target_count = target
                    self._bins[bid].using = target > 0

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_bin_id(bin_id: str) -> tuple[int, int]:
        """Parse 'bin_{row}_{col}' into (layer, col). Returns (0, 0) on failure."""
        parts = bin_id.split("_")
        try:
            return int(parts[-2]), int(parts[-1])
        except (ValueError, IndexError):
            return 0, 0

    @staticmethod
    def _calculate_bin_status(b: BinStatus) -> str:
        """Determine display status for a bin."""
        if not b.detected:
            return "grey"

        if not b.using:
            if b.is_active:
                return "wrong_bin"
            return "grey"

        if b.is_active:
            return "active"

        if b.target_count == 0:
            return "white"
        elif b.pick_count == 0:
            return "white"
        elif b.pick_count < b.target_count:
            return "orange"
        elif b.pick_count == b.target_count:
            return "green"
        else:
            return "warn"
