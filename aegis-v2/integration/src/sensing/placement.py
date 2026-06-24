"""
Placement Tracker (kitting-box driven counting)
===============================================
Counts items **as they are placed into the kitting box**, not as they leave a
bin. Three load receptors: 2 source bins (each one item type) + 1 kitting box.

Design (see docs/superpowers/specs/2026-06-24-3-load-receptor-kitting-demo-design.md):

* Each source bin is tared FULL; its weight goes *negative* as items leave, so
  ``removed[bin] = round(max(0, -bin_weight) / unit_g)``.
* The kitting box is tared EMPTY; its weight goes *positive* as items arrive.
* A bin's **placed count** is how many of its items the box currently holds. We
  recover it by decomposing the box weight into per-item counts, bounded by what
  was actually removed from each bin (you can't place more than you took). With
  distinct unit weights this decomposition is unambiguous and tolerant of the
  load cell's ~gram noise — no event/timing correlation needed.

The kit is complete when every BOM bin's placed count equals its target AND the
box total matches the expected total within tolerance.

A software ``tare()`` re-baselines all receptors so the demo can be repeated
without re-flashing/rebooting the ESP32.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product


@dataclass
class KitState:
    """Snapshot of the kitting state for the dashboard/FSM."""
    placed: dict[str, int]           # bin_id -> items of that bin now in the box
    removed: dict[str, int]          # bin_id -> items currently out of that bin
    box_grams: float                 # current (tared) box weight
    expected_grams: float            # target total weight in the box
    targets: dict[str, int]          # bin_id -> target count
    complete: bool                   # all targets met AND box total matches
    state: str                       # FSM label: INIT|PICKING|OVERPICK|KIT_COMPLETE
    overpick: dict[str, int] = field(default_factory=dict)  # bin_id -> extra placed


class PlacementTracker:
    """Derives per-bin placed counts from the 3 load-receptor weights."""

    def __init__(
        self,
        units_g: dict[str, float],
        targets: dict[str, int],
        box_id: str,
        tolerance_g: float | None = None,
        max_search: int = 60,
    ):
        # bin_id -> unit weight (grams). Only BOM source bins.
        self._units = {b: float(u) for b, u in units_g.items() if u and u > 0}
        self._targets = {b: int(t) for b, t in targets.items()}
        self._box_id = box_id
        # Default tolerance: half the smallest unit, so you can't be a whole
        # (smallest) item off and still read "matched".
        self._tol = (
            float(tolerance_g)
            if tolerance_g is not None
            else 0.5 * min(self._units.values(), default=1.0)
        )
        self._max_search = max_search
        # Software tare offsets (grams), subtracted from raw readings.
        self._offsets: dict[str, float] = {}

    @property
    def box_id(self) -> str:
        return self._box_id

    @property
    def expected_grams(self) -> float:
        return sum(self._targets.get(b, 0) * u for b, u in self._units.items())

    def tare(self, weights: dict[str, float]) -> None:
        """Re-baseline every receptor to the current reading (software zero)."""
        self._offsets = {k: float(v) for k, v in weights.items()}

    def _adj(self, weights: dict[str, float], bin_id: str) -> float:
        return float(weights.get(bin_id, 0.0)) - self._offsets.get(bin_id, 0.0)

    def removed_counts(self, weights: dict[str, float]) -> dict[str, int]:
        """Items currently out of each BOM bin (bin weight went negative)."""
        out: dict[str, int] = {}
        for b, unit in self._units.items():
            grams = self._adj(weights, b)
            out[b] = int(round(max(0.0, -grams) / unit))
        return out

    def _decompose(self, box_grams: float, removed: dict[str, int]) -> dict[str, int]:
        """Best per-item counts explaining box_grams, bounded by removed counts.

        Picks the combination minimising |Σ count·unit − box_grams|; ties broken
        toward the larger total count (so a matched weight reads as 'placed').
        """
        bins = list(self._units)
        ranges = [
            range(0, min(removed.get(b, 0), self._max_search) + 1) for b in bins
        ]
        best_combo = tuple(0 for _ in bins)
        best_key = None
        for combo in product(*ranges):
            total = sum(combo[i] * self._units[bins[i]] for i in range(len(bins)))
            key = (abs(total - box_grams), -sum(combo))
            if best_key is None or key < best_key:
                best_key = key
                best_combo = combo
        return {bins[i]: best_combo[i] for i in range(len(bins))}

    def update(self, weights: dict[str, float]) -> KitState:
        """Compute the current kitting state from raw receptor weights."""
        box_grams = max(0.0, self._adj(weights, self._box_id))
        removed = self.removed_counts(weights)
        placed = self._decompose(box_grams, removed)

        overpick = {
            b: placed[b] - self._targets.get(b, 0)
            for b in placed
            if placed[b] > self._targets.get(b, 0)
        }
        targets_met = all(
            placed.get(b, 0) == self._targets.get(b, 0) for b in self._units
        )
        box_matches = abs(box_grams - self.expected_grams) <= self._tol
        complete = bool(self._units) and targets_met and box_matches and not overpick

        if complete:
            state = "KIT_COMPLETE"
        elif overpick:
            state = "OVERPICK"
        elif any(placed.values()):
            state = "PICKING"
        else:
            state = "INIT"

        return KitState(
            placed=placed,
            removed=removed,
            box_grams=round(box_grams, 1),
            expected_grams=round(self.expected_grams, 1),
            targets=dict(self._targets),
            complete=complete,
            state=state,
            overpick=overpick,
        )
