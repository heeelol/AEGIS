"""
Placement Tracker (box-verified, conservation-based counting)
=============================================================
Counts items for the kitting demo across 4 load receptors: 3 source bins (each
one item type, on its own cell) + 1 kitting box.

A bin's count only rises when the box **verifies** the item is in it — by
conservation of mass: the weight that LEFT the bin must ARRIVE in the box.

  qty[bin] = (matched box increase) / unit_g[bin]

Why this is robust for small items (the previous absolute-decomposition wasn't):
we track the box weight **relative to a committed baseline** (the weight already
accounted for by counted items). A newly-placed item is a small *delta* from that
baseline — a +3.6 g step is resolvable even when the box already holds 200 g —
whereas decomposing the absolute 203.6 g loses the 3.6 g in noise.

Mechanics per update (all weights EMA-smoothed; bin counts hysteresis-debounced):
  * removed[bin]  = round(-bin_weight / unit)         — items out of each bin (its own cell)
  * unaccounted   = box_weight - Σ placed·unit        — box weight not yet credited
  * CREDIT a bin (largest unit first, so the right item matches the box step) when
    an item left it (placed < removed) AND unaccounted ≥ its unit  → item verified in box.
  * UN-CREDIT when the item leaves the box (box drops) or returns to the bin.

Overpick = placed > target → the bin shows how many to RETURN. Kit completes when
every BOM bin's verified count equals its target. ``tare()`` re-baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KitState:
    placed: dict[str, int]           # bin_id -> items verified in the box from that bin
    removed: dict[str, int]          # bin_id -> items currently out of that bin
    box_grams: float                 # current (tared, smoothed) box weight
    expected_grams: float            # target total weight in the box
    targets: dict[str, int]
    complete: bool
    state: str                       # INIT|PICKING|OVERPICK|KIT_COMPLETE
    box_verified: bool = False       # box total ≈ expected (display cross-check)
    overpick: dict[str, int] = field(default_factory=dict)  # bin_id -> items to return


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
    ):
        self._units = {b: float(u) for b, u in units_g.items() if u and u > 0}
        self._targets = {b: int(t) for b, t in targets.items()}
        self._box_id = box_id
        self._alpha = float(ema_alpha)
        self._h = float(hysteresis)

        smallest = min(self._units.values(), default=1.0)
        # How much the box step may fall short of a unit and still credit it.
        self._step_tol = (
            float(box_step_tolerance_g) if box_step_tolerance_g is not None
            else 0.5 * smallest
        )
        # Generous tolerance for the whole-box "verified" display flag.
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

        # Items out of each bin (own cell, debounced).
        for b, unit in self._units.items():
            raw = max(0.0, -sm.get(b, 0.0)) / unit
            self._removed[b] = self._hysteretic(raw, self._removed[b])

        box_grams = max(0.0, sm.get(self._box_id, 0.0))
        # Largest unit first so a big item's box step isn't eaten by small items.
        order = sorted(self._units, key=lambda b: -self._units[b])

        # CREDIT: item left the bin AND the box has risen to account for it.
        changed = True
        while changed:
            changed = False
            unaccounted = box_grams - self._accounted()
            for b in order:
                if (self._placed[b] < self._removed[b]
                        and unaccounted >= self._units[b] - self._step_tol):
                    self._placed[b] += 1
                    changed = True
                    break

        # UN-CREDIT: item left the box (box dropped) or was returned to the bin.
        changed = True
        while changed:
            changed = False
            overshoot = self._accounted() - box_grams
            for b in order:
                if self._placed[b] > 0 and (
                    self._placed[b] > self._removed[b]
                    or overshoot >= self._units[b] - self._step_tol
                ):
                    self._placed[b] -= 1
                    changed = True
                    break

        placed = dict(self._placed)
        overpick = {
            b: placed[b] - self._targets.get(b, 0)
            for b in placed if placed[b] > self._targets.get(b, 0)
        }
        targets_met = all(placed.get(b, 0) == self._targets.get(b, 0) for b in self._units)
        complete = bool(self._units) and targets_met and not overpick
        box_verified = abs(box_grams - self.expected_grams) <= self._box_tol

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
            removed=dict(self._removed),
            box_grams=round(box_grams, 1),
            expected_grams=round(self.expected_grams, 1),
            targets=dict(self._targets),
            complete=complete,
            state=state,
            box_verified=box_verified,
            overpick=overpick,
        )
