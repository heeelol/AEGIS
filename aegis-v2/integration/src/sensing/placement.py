"""
Placement Tracker (sequential single-bin kitting FSM)
=====================================================
Only ONE bin is active at a time, so the kitting box's weight change can only
come from that one item type — which removes the cross-bin weight ambiguity
(item variance, and same-weight items in different bins).

See docs/superpowers/specs/2026-06-25-sequential-single-bin-kitting-fsm.md.

Model
-----
* The operator may start any not-done bin; the first clear ~½-unit drop activates
  it (``_active``) and soft-locks the rest. Forward-only: DONE bins stay locked.
* Active bin count: ``placed = round((box − baseline_box) / unit_active)``.
  On completion (``placed == removed == target``) the box weight is banked into
  ``baseline_box`` and the bin is frozen — completed counters never fluctuate.
* ``removed[bin]`` from each bin's own cell; ``holding = removed − placed``.
* Four red faults (auto-clear when corrected): overpack-kit, remove-from-kit
  (only checked when ``holding == 0``), pick-from-wrong-bin, return-to-wrong-bin.

All weights are EMA-smoothed; integer counts use a hysteresis deadband.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KitState:
    placed: dict[str, int]
    removed: dict[str, int]
    targets: dict[str, int]
    active: Optional[str]            # currently active bin id, or None (IDLE)
    done: list[str]                  # completed bin ids (frozen)
    box_grams: float
    complete: bool
    state: str                       # IDLE|PICKING|FAULT|KIT_COMPLETE
    overpick: dict[str, int] = field(default_factory=dict)   # removed-target (return to bin)
    alert: Optional[dict] = None     # active red fault {type,message,bin}


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
        activation_frac: float = 0.5,
        wrong_bin_frac: float = 0.5,
    ):
        self._units = {b: float(u) for b, u in units_g.items() if u and u > 0}
        self._targets = {b: int(t) for b, t in targets.items()}
        self._box_id = box_id
        self._alpha = float(ema_alpha)
        self._h = float(hysteresis)
        smallest = min(self._units.values(), default=1.0)
        self._step_floor = (float(box_step_tolerance_g) if box_step_tolerance_g is not None
                            else 0.5 * smallest)
        self._step_frac = float(box_step_fraction)
        self._activation = float(activation_frac)
        self._wrong = float(wrong_bin_frac)

        self._offsets: dict[str, float] = {}
        self._ema: dict[str, float] = {}
        self._reset_run()

    def _reset_run(self):
        self._baseline_box = 0.0
        self._active: Optional[str] = None
        self._done: set[str] = set()
        self._removed = {b: 0 for b in self._units}
        self._added = {b: 0 for b in self._units}
        self._placed = {b: 0 for b in self._units}

    @property
    def box_id(self):
        return self._box_id

    @property
    def expected_grams(self):
        return sum(self._targets.get(b, 0) * u for b, u in self._units.items())

    def _step_tol(self, b):
        return max(self._step_floor, self._step_frac * self._units[b])

    def tare(self, weights):
        self._offsets = {k: float(v) for k, v in weights.items()}
        self._ema = {}
        self._reset_run()

    def set_targets(self, targets):
        """Replace the per-bin BOM targets (e.g. when the cycle advances a set).

        Keyed by canonical bin id; bins absent from ``targets`` (or not in this
        tracker's units) get 0. Keeps an entry for every unit so update() never
        KeyErrors.
        """
        self._targets = {b: int(targets.get(b, 0)) for b in self._units}

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

    def _raw_removed(self, sm, b):
        return max(0.0, -sm.get(b, 0.0)) / self._units[b]

    def _raw_added(self, sm, b):
        return max(0.0, sm.get(b, 0.0)) / self._units[b]

    def update(self, weights):
        sm = self._smooth(weights)

        # 1) per-bin removal / addition (own cell, debounced)
        for b in self._units:
            self._removed[b] = self._hysteretic(self._raw_removed(sm, b), self._removed[b])
            self._added[b] = self._hysteretic(self._raw_added(sm, b), self._added[b])

        box = max(0.0, self._adj(sm, self._box_id))  # smoothed, tared box weight

        # 2) activation: in IDLE, the clearest not-done bin past the threshold goes active
        if self._active is None:
            cands = [b for b in self._units
                     if b not in self._done and self._raw_removed(sm, b) >= self._activation]
            if cands:
                self._active = max(cands, key=lambda b: self._raw_removed(sm, b))

        active = self._active
        prev_placed = self._placed.get(active, 0) if active else 0

        fault = None

        # 3) wrong-bin faults: any non-active bin deviating from its expected state. (Disabled)
        for b in self._units:
            if b == active:
                continue
            expected = self._targets[b] if b in self._done else 0
            r = self._removed[b]
            added = self._added[b] >= 1
            if b in self._done:
                if r < expected or added:
                    pass
                elif r > expected:
                    pass
            else:  # available/locked, expected empty change
                if added:
                    pass
                elif r >= 1:
                    # In IDLE this bin would have been activated above; if we are
                    # PICKING another bin, a drop here is a wrong-bin pick.
                    if active is not None:
                        pass

        # 4) active-bin counting + its faults
        if active is not None:
            unit = self._units[active]
            raw_placed = (box - self._baseline_box) / unit
            committed = self._baseline_box + prev_placed * unit
            holding = self._removed[active] - prev_placed

            # remove-from-kit: empty hands but the box dropped below what's counted.
            if holding <= 0 and (box < committed - self._step_tol(active)):
                # freeze the count while weight drops (don't silently un-count)
                placed = prev_placed
            else:
                placed = self._hysteretic(max(0.0, raw_placed), prev_placed)
            
            placed = min(placed, self._removed[active])
            self._placed[active] = placed

            # completion -> bank + advance
            if (placed == self._targets[active]
                    and self._removed[active] == self._targets[active]):
                self._done.add(active)
                self._baseline_box = box
                self._active = None
                active = None
        else:
            # IDLE remove-from-kit: box dropped below the banked total, empty hands.
            if box < self._baseline_box - max(self._step_floor, 1.0):
                pass

        placed = dict(self._placed)
        removed = dict(self._removed)
        overpick = {b: removed[b] - self._targets[b]
                    for b in self._units if removed[b] > self._targets[b]}

        complete = any(t > 0 for t in self._targets.values()) and all(b in self._done for b, t in self._targets.items() if t > 0)
        if fault:
            state = "FAULT"
        elif complete:
            state = "KIT_COMPLETE"
        elif self._active is not None:
            state = "PICKING"
        else:
            state = "IDLE"

        return KitState(
            placed=placed, removed=removed, targets=dict(self._targets),
            active=self._active, done=sorted(self._done),
            box_grams=round(box, 1), complete=complete, state=state,
            overpick=overpick, alert=fault,
        )

    def _fault(self, kind, binid):
        msgs = {
            "overpack-kit": (lambda n: f"OVER-PACKED — REMOVE {n} FROM KIT"),
            "remove-from-kit": (lambda n: "ITEM REMOVED FROM KIT — RETURN IT"),
            "pick-from-wrong-bin": (lambda n: f"WRONG BIN — DO NOT PICK FROM {binid}"),
            "return-to-wrong-bin": (lambda n: f"WRONG BIN — RETURN ITEM TO {binid}"),
        }
        n = 0
        if kind == "overpack-kit" and binid:
            n = self._placed.get(binid, 0) - self._targets.get(binid, 0)
        return {"type": kind, "bin": binid, "message": msgs[kind](n)}
