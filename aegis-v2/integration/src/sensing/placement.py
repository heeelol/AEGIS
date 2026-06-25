"""
Placement Tracker (box-verified, conservation-based counting)
=============================================================
Counts items for the kitting demo across 4 load receptors: 3 source bins (each
one item type, on its own cell) + 1 kitting box.

A bin's **placed** count rises only when the box VERIFIES the item is in it: the
weight that left the bin must arrive in the box. We track the box weight's change
from a committed baseline, so small items count even on top of a heavy box.

Two distinct "too many" conditions (do NOT conflate them):
  * **overpick**  = removed > target  (too many taken OUT of the bin). The bin
    shows "↩ RETURN N" so the extras go back. Not a red-screen event.
  * **overpack**  = placed  > target  (too many IN the kit box). One of the four
    full-red-screen faults — surfaced via ``alert``.

A bin is "done" (green) only when ``placed == target AND removed == target``
(right amount in the box AND all extras returned).

Per-item step tolerance: the box step for a big item is noisier in absolute grams
than for a small one, so the credit tolerance scales with the item
(``max(floor, fraction · unit)``) — fixes the large bin being inaccurate while
keeping small items crisp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KitState:
    placed: dict[str, int]           # items verified in the box, per bin
    removed: dict[str, int]          # items currently out of the bin, per bin
    box_grams: float
    expected_grams: float
    targets: dict[str, int]
    complete: bool
    state: str                       # INIT|PICKING|OVERPACK|KIT_COMPLETE
    box_verified: bool = False
    overpick: dict[str, int] = field(default_factory=dict)   # removed-target (return to bin)
    overpack: dict[str, int] = field(default_factory=dict)   # placed-target (red: remove from kit)
    alert: Optional[dict] = None     # active full-red event, e.g. {type,message,bins}


class PlacementTracker:
    def __init__(
        self,
        units_g: dict[str, float],
        targets: dict[str, int],
        box_id: str,
        tolerance_g: float | None = None,
        ema_alpha: float = 0.4,
        hysteresis: float = 0.25,
        box_tolerance_g: float | None = None,
        box_step_tolerance_g: float | None = None,
        box_step_fraction: float = 0.15,
    ):
        self._units = {b: float(u) for b, u in units_g.items() if u and u > 0}
        self._targets = {b: int(t) for b, t in targets.items()}
        self._box_id = box_id
        self._alpha = float(ema_alpha)
        self._h = float(hysteresis)

        smallest = min(self._units.values(), default=1.0)
        # Credit tolerance = max(floor, fraction · unit): big items get a larger
        # absolute slack (their box step is noisier), small items stay tight.
        self._step_floor = (
            float(box_step_tolerance_g) if box_step_tolerance_g is not None
            else 0.5 * smallest
        )
        self._step_frac = float(box_step_fraction)
        self._box_tol = (
            float(box_tolerance_g) if box_tolerance_g is not None
            else max(5.0, 1.5 * smallest)
        )
        self._tol = tolerance_g  # API compat (unused)

        self._offsets: dict[str, float] = {}
        self._ema: dict[str, float] = {}
        self._removed: dict[str, int] = {b: 0 for b in self._units}
        self._placed: dict[str, int] = {b: 0 for b in self._units}

    @property
    def box_id(self) -> str:
        return self._box_id

    @property
    def expected_grams(self) -> float:
        return sum(self._targets.get(b, 0) * u for b, u in self._units.items())

    def _step_tol(self, b: str) -> float:
        return max(self._step_floor, self._step_frac * self._units[b])

    def tare(self, weights: dict[str, float]) -> None:
        self._offsets = {k: float(v) for k, v in weights.items()}
        self._ema = {}
        self._removed = {b: 0 for b in self._units}
        self._placed = {b: 0 for b in self._units}

    def _adj(self, weights, key):
        return float(weights.get(key, 0.0)) - self._offsets.get(key, 0.0)

    def _smooth(self, weights):
        for k in set(self._units) | {self._box_id}:
            adj = self._adj(weights, k)
            self._ema[k] = (adj if k not in self._ema
                            else self._alpha * adj + (1 - self._alpha) * self._ema[k])
        return self._ema

    def _hysteretic(self, raw, current):
        n = current
        while raw >= n + 0.5 + self._h:
            n += 1
        while n > 0 and raw <= n - 0.5 - self._h:
            n -= 1
        return n

    def _accounted(self):
        return sum(self._placed[b] * self._units[b] for b in self._units)

    def update(self, weights: dict[str, float]) -> KitState:
        sm = self._smooth(weights)

        for b, unit in self._units.items():
            raw = max(0.0, -sm.get(b, 0.0)) / unit
            self._removed[b] = self._hysteretic(raw, self._removed[b])

        box_grams = max(0.0, sm.get(self._box_id, 0.0))
        order = sorted(self._units, key=lambda b: -self._units[b])  # largest unit first

        # CREDIT: item left the bin AND the box rose to account for it.
        changed = True
        while changed:
            changed = False
            unaccounted = box_grams - self._accounted()
            for b in order:
                if (self._placed[b] < self._removed[b]
                        and unaccounted >= self._units[b] - self._step_tol(b)):
                    self._placed[b] += 1
                    changed = True
                    break

        # UN-CREDIT: item left the box, or was returned to the bin.
        changed = True
        while changed:
            changed = False
            overshoot = self._accounted() - box_grams
            for b in order:
                if self._placed[b] > 0 and (
                    self._placed[b] > self._removed[b]
                    or overshoot >= self._units[b] - self._step_tol(b)
                ):
                    self._placed[b] -= 1
                    changed = True
                    break

        placed = dict(self._placed)
        removed = dict(self._removed)
        tgt = self._targets

        # overpick = too many OUT of the bin (return to bin); overpack = too many IN the box (red).
        overpick = {b: removed[b] - tgt.get(b, 0) for b in removed if removed[b] > tgt.get(b, 0)}
        overpack = {b: placed[b] - tgt.get(b, 0) for b in placed if placed[b] > tgt.get(b, 0)}

        # Done only when the right amount is in the box AND all extras are returned.
        complete = bool(self._units) and all(
            placed[b] == tgt.get(b, 0) and removed[b] == tgt.get(b, 0) for b in self._units
        )
        box_verified = abs(box_grams - self.expected_grams) <= self._box_tol

        alert = None
        if overpack:
            worst = max(overpack, key=overpack.get)
            alert = {
                "type": "overpack-kit",
                "bins": sorted(overpack),
                "message": f"OVER-PACKED — remove {overpack[worst]} from the kit ({worst})",
            }

        if complete:
            state = "KIT_COMPLETE"
        elif overpack:
            state = "OVERPACK"
        elif any(placed.values()) or any(removed.values()):
            state = "PICKING"
        else:
            state = "INIT"

        return KitState(
            placed=placed, removed=removed,
            box_grams=round(box_grams, 1),
            expected_grams=round(self.expected_grams, 1),
            targets=dict(tgt), complete=complete, state=state,
            box_verified=box_verified, overpick=overpick, overpack=overpack,
            alert=alert,
        )
