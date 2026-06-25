"""
Placement Tracker (per-receptor kitting counter)
================================================
Counts items for the kitting demo across 4 load receptors: 3 source bins (each
one item type, on its own cell) + 1 kitting box.

Design (revised 2026-06-25 for robustness — see the spec):

* **Each source bin is counted from its OWN cell.** The bin is tared FULL, so its
  weight goes negative as items leave: ``count = round(-weight / unit_g)``. Each
  cell is range-matched to its item (1 kg cell for 3.6 g parts, etc.), so this is
  reliable for small AND large items.
* **The box is NOT used to attribute per-item counts.** A single 5 kg box cell
  cannot resolve a 3.6 g item added on top of a few hundred grams — small items
  vanish in its noise. So the box weight is shown only as a *verification* total
  (``box_verified`` = box ≈ expected), it does not drive or gate the counters.
* **Smoothing + hysteresis** keep the counter steady: weights are EMA-filtered and
  each integer count only changes once the reading is clearly past the midpoint
  (a deadband), so sensor jitter never makes the count flicker.

Kit is complete when every BOM bin's count equals its target (no overpick).
``tare()`` re-baselines all receptors so the demo can repeat without rebooting.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KitState:
    """Snapshot of the kitting state for the dashboard/FSM."""
    placed: dict[str, int]           # bin_id -> items counted out of that bin
    removed: dict[str, int]          # alias of placed (kept for the watcher/UI)
    box_grams: float                 # current (tared, smoothed) box weight
    expected_grams: float            # target total weight in the box
    targets: dict[str, int]
    complete: bool                   # all targets met, no overpick
    state: str                       # INIT|PICKING|OVERPICK|KIT_COMPLETE
    box_verified: bool = False       # box total ≈ expected (cross-check only)
    overpick: dict[str, int] = field(default_factory=dict)


class PlacementTracker:
    """Per-receptor counter: each BOM bin counted from its own cell, with
    EMA smoothing + hysteresis for a steady, robust count."""

    def __init__(
        self,
        units_g: dict[str, float],
        targets: dict[str, int],
        box_id: str,
        tolerance_g: float | None = None,
        ema_alpha: float = 0.4,
        hysteresis: float = 0.25,
        box_tolerance_g: float | None = None,
    ):
        self._units = {b: float(u) for b, u in units_g.items() if u and u > 0}
        self._targets = {b: int(t) for b, t in targets.items()}
        self._box_id = box_id
        self._alpha = float(ema_alpha)        # EMA weight on the newest sample
        self._h = float(hysteresis)           # deadband (fraction of a unit)
        # Box verification tolerance: generous, since the box cell is coarse for
        # small items. Defaults to ~1.5× the smallest unit (or 5 g, whichever is
        # larger). Cross-check only — never blocks completion.
        smallest = min(self._units.values(), default=1.0)
        self._box_tol = (
            float(box_tolerance_g) if box_tolerance_g is not None
            else max(5.0, 1.5 * smallest)
        )
        self._tol = tolerance_g  # retained for API compatibility (unused here)

        self._offsets: dict[str, float] = {}   # software tare
        self._ema: dict[str, float] = {}       # smoothed, tared weights
        self._counts: dict[str, int] = {b: 0 for b in self._units}  # hysteresis state

    @property
    def box_id(self) -> str:
        return self._box_id

    @property
    def expected_grams(self) -> float:
        return sum(self._targets.get(b, 0) * u for b, u in self._units.items())

    def tare(self, weights: dict[str, float]) -> None:
        """Re-baseline every receptor and reset the smoothed/committed state."""
        self._offsets = {k: float(v) for k, v in weights.items()}
        self._ema = {}
        self._counts = {b: 0 for b in self._units}

    def _adj(self, weights: dict[str, float], key: str) -> float:
        return float(weights.get(key, 0.0)) - self._offsets.get(key, 0.0)

    def _smooth(self, weights: dict[str, float]) -> dict[str, float]:
        """EMA-filter the tared weights of every receptor we care about."""
        keys = set(self._units) | {self._box_id}
        for k in keys:
            adj = self._adj(weights, k)
            self._ema[k] = (adj if k not in self._ema
                            else self._alpha * adj + (1 - self._alpha) * self._ema[k])
        return self._ema

    def _hysteretic_count(self, removed_raw: float, current: int) -> int:
        """Integer count that only changes once clearly past the .5 boundary."""
        n = current
        # increase while well above the next boundary
        while removed_raw >= n + 0.5 + self._h:
            n += 1
        # decrease while well below the previous boundary
        while n > 0 and removed_raw <= n - 0.5 - self._h:
            n -= 1
        return n

    def update(self, weights: dict[str, float]) -> KitState:
        sm = self._smooth(weights)

        # Per-bin counts from each bin's own cell (negative = removed).
        for b, unit in self._units.items():
            removed_raw = max(0.0, -sm.get(b, 0.0)) / unit
            self._counts[b] = self._hysteretic_count(removed_raw, self._counts[b])
        placed = dict(self._counts)

        box_grams = max(0.0, sm.get(self._box_id, 0.0))

        overpick = {
            b: placed[b] - self._targets.get(b, 0)
            for b in placed if placed[b] > self._targets.get(b, 0)
        }
        targets_met = all(
            placed.get(b, 0) == self._targets.get(b, 0) for b in self._units
        )
        box_verified = abs(box_grams - self.expected_grams) <= self._box_tol
        # Completion is driven by the reliable per-bin cells, not the coarse box.
        complete = bool(self._units) and targets_met and not overpick

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
            removed=dict(placed),
            box_grams=round(box_grams, 1),
            expected_grams=round(self.expected_grams, 1),
            targets=dict(self._targets),
            complete=complete,
            state=state,
            box_verified=box_verified,
            overpick=overpick,
        )
