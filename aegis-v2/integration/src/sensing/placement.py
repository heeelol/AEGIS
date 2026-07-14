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
  The candidate must hold for ``activation_confirm_s`` before it actually
  commits — a single noisy frame crossing the threshold shouldn't focus a bin
  with zero real hand interaction; the clock restarts if the candidate changes
  or drops back below threshold.
* Active bin count: ``placed = round((box − baseline_box) / unit_active)``.
  On completion (``placed == removed == target``) the box weight is banked into
  ``baseline_box`` and the bin is frozen — completed counters never fluctuate.
  This rounding uses ``box_hysteresis`` (defaults to ``hysteresis``) rather
  than the bin-level fraction — the kit box is a physically different, coarser
  load cell than the individual bin cells, so its own noise floor need not
  match theirs; tune independently if light items credit unreliably.
* ``removed[bin]`` from each bin's own cell.
* Four red faults (auto-clear when corrected): overpack-kit, pick-from-wrong-bin
  (a bin not in this kit, or one already completed), return-to-wrong-bin, and
  out-of-sequence (a not-yet-done kit bin reached too early). Operators never take items back out of the kit box once
  placed, so there's no separate remove-from-kit check — a box-weight decrease
  (e.g. correcting an overpack) just tracks normally, in either direction.
* Wrong-bin detection knows what's actually in the operator's hand instead of
  reacting to any weight blip on another bin:
  - A bin is only even considered once its OWN reading reaches steady state
    (``abs(raw − smoothed) <= count_h(bin) * unit_weight(bin)``) — the same
    "has the EMA caught up" test already used for the box baseline, just
    applied per-bin, and scaled to that bin's own item weight (the same
    relative precision already accepted for counting) rather than a flat gram
    value — a flat tolerance makes heavier items take much longer, in
    absolute time, to converge tightly enough to even be considered, so a
    mistake corrected quickly would never get a chance to fault. A bin still
    moving from a bump/press or mechanical cross-talk (a real removal on one
    bin flexing a shared shelf enough for a neighbor's cell to blip) is
    invisible to the check until it settles; transient cross-talk that decays
    back out is never seen "elevated" at all. Latched with a wide (3x)
    dead-band between entering vs. leaving settled — a single memoryless
    threshold test flickers true/false every frame under ordinary sensor
    noise once that noise approaches the tolerance, which flashed the alert
    itself on and off. Other bins are also skipped entirely while the ACTIVE
    bin itself hasn't settled — genuine, ongoing physical activity there (a
    real pick or place in progress) is the most likely source of mechanical
    vibration bleeding into a neighbor's cell.
  - Since only one bin is active at a time, "what's in hand" is always a
    specific, known item — the active bin's, whenever ``holding =
    removed[active] − placed[active] > 0``. A weight increase on some OTHER
    bin (done or not) only counts as a genuine return-to-wrong-bin if it
    settles to roughly ONE UNIT OF THE ACTIVE BIN'S ITEM — never that other
    bin's own item weight. A stray blip elsewhere almost never coincidentally
    matches an unrelated item's specific mass, and matching against the OTHER
    bin's own weight would miss real mistakes: a light held item wrongly
    dropped into an already-done HEAVY-item bin would never reach even one
    unit of that bin's own (much heavier) weight, so it would never register
    if checked that way. While nothing is actually in hand (``holding == 0``),
    there's nothing to misplace, so this check doesn't fire on other bins at
    all — including while the station is fully idle.
  - ``pick-from-wrong-bin`` (an unexpected removal from a non-active bin) is
    unchanged — measured against that bin's own item weight, since a pick
    creates new in-hand content rather than needing to match existing content.
    It only gains the steady-state gate.

All weights are EMA-smoothed; integer counts use a hysteresis deadband, floored
in absolute grams (``count_tolerance_g``) so light items keep a real margin
instead of a near-zero percentage-of-unit deadband.
"""

from __future__ import annotations

import time
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
        box_hysteresis: float | None = None,
        box_tolerance_g: float | None = None,
        box_step_tolerance_g: float | None = None,
        activation_frac: float = 0.5,
        wrong_bin_frac: float = 0.5,
        count_tolerance_g: float | None = None,
        fault_settle_s: float = 2.5,
        activation_confirm_s: float = 0.4,
        labels: dict[str, str] | None = None,
    ):
        # Human-facing bin names ("BIN 1"…"BIN 9") for the fault messages, keyed
        # by canonical bin id. Falls back to the raw id when unmapped so unit
        # tests (which don't pass labels) still read sensibly.
        self._labels = dict(labels or {})
        self._units = {b: float(u) for b, u in units_g.items() if u and u > 0}
        self._targets = {b: int(t) for b, t in targets.items()}
        self._box_id = box_id
        self._alpha = float(ema_alpha)
        self._h = float(hysteresis)
        # Box-crediting (placed) uses a SEPARATE hysteresis fraction from
        # bin-level removed/added counting, defaulting to the same value (no
        # behavior change). The kit box is a physically different, coarser
        # (higher-capacity) load cell than the individual bin cells, so its own
        # noise floor in grams need not match theirs -- if light items register
        # unreliably in the box specifically while their own bin's removed
        # count is accurate, this is the knob to tune independently (lower it
        # for more sensitivity) rather than changing `hysteresis`, which would
        # also affect the bin cells' own, already-reliable counting.
        self._box_h = float(box_hysteresis) if box_hysteresis is not None else self._h
        smallest = min(self._units.values(), default=1.0)
        self._step_floor = (float(box_step_tolerance_g) if box_step_tolerance_g is not None
                            else 0.5 * smallest)
        self._activation = float(activation_frac)
        self._wrong = float(wrong_bin_frac)
        # Absolute-gram floor for count rounding (removed/added/placed). A pure
        # percentage of unit weight (self._h) shrinks to near-nothing for light
        # items (e.g. 0.05 * 3.6g = 0.18g) — well below the scale's real noise
        # floor — so small items round on noise instead of real picks/placements.
        # NOT defaulted to box_step_tolerance_g: that's the box-baseline
        # stability gate's noise floor, a much coarser tolerance, and is large
        # enough (e.g. 1.8g on a 3.6g item) to push the rounding threshold to
        # ~100% of the unit weight, which real items at that size can't
        # reliably reach at all. 0.3g is a
        # modest, independent default — verified to leave clean single-item
        # rounding unchanged while still giving small items more margin than the
        # bare percentage deadband against real scale noise. Also reused as the
        # per-bin steady-state tolerance for wrong-bin detection (see module
        # docstring) — the same "is this reading trustworthy yet" floor.
        self._count_floor = (float(count_tolerance_g) if count_tolerance_g is not None
                             else 0.3)
        self._fault_settle_s = float(fault_settle_s)
        self._activation_confirm_s = float(activation_confirm_s)

        self._offsets: dict[str, float] = {}
        self._ema: dict[str, float] = {}
        self._settle_until = 0.0
        self._activation_candidate: Optional[str] = None
        self._activation_candidate_since = 0.0
        self._reset_run()

    def _reset_run(self):
        self._baseline_box = 0.0
        self._active: Optional[str] = None
        self._done: set[str] = set()
        self._removed = {b: 0 for b in self._units}
        # High-water mark of `removed` per bin since the last tare. The placed
        # (box) count is capped at this, NOT the instantaneous `removed`, so a
        # transient dip in a bin's own cell (an arm pressing the bin, mechanical
        # settle) can't momentarily un-count an item already in the box — the box
        # count only falls when the BOX itself drops (the item genuinely leaves).
        self._removed_peak = {b: 0 for b in self._units}
        self._placed = {b: 0 for b in self._units}
        self._activation_candidate = None
        self._activation_candidate_since = 0.0
        # Hysteretic "how many active-bin-units have accumulated on this OTHER
        # bin" counters, keyed by bin id — valid only while _return_match_active
        # matches the current active bin (reset lazily in update() whenever the
        # active bin changes, since the unit being matched against changes too).
        self._return_match = {b: 0 for b in self._units}
        self._return_match_active: Optional[str] = None
        # Latched per-bin steady-state flag (Schmitt-trigger style, same idiom
        # as _hysteretic()'s dead-band elsewhere in this file): starts
        # "unsettled" until proven otherwise, and once settled won't flip back
        # from a single noisy frame — see the steady-state comment in update().
        self._settled_state = {b: False for b in self._units}

    @property
    def box_id(self):
        return self._box_id

    @property
    def expected_grams(self):
        return sum(self._targets.get(b, 0) * u for b, u in self._units.items())

    def _lbl(self, bid):
        """Human-facing name for a bin id ("BIN 6"), or the raw id if unmapped."""
        return self._labels.get(bid, bid) if bid else bid

    def _count_h(self, b):
        """Hysteresis for item-count rounding, in units of one item — floored in
        absolute grams so tiny items keep a real margin (see __init__ comment)."""
        return max(self._h, self._count_floor / self._units[b])

    def _box_count_h(self, b):
        """Hysteresis for box-crediting (placed), in units of one item — see
        box_hysteresis in __init__ for why this is separate from _count_h."""
        return max(self._box_h, self._count_floor / self._units[b])

    def tare(self, weights):
        self._offsets = {k: float(v) for k, v in weights.items()}
        self._ema = {}
        self._reset_run()
        # Grace period before wrong-bin faults can fire. Right after a tare,
        # individual bins' own EMAs are starting fresh and can still be
        # settling for a moment (same class of issue as the box-baseline
        # timing fix — just never gated for per-bin values, which have no
        # equivalent "wait for stability" check). Without this, a bin whose
        # reading hasn't fully settled yet can spuriously read as `added` and
        # fault instantly at boot, with zero real operator interaction.
        self._settle_until = time.time() + self._fault_settle_s

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

    def _hysteretic(self, raw, current, h=None):
        if h is None:
            h = self._h
        n = current
        while raw >= n + 0.5 + h:
            n += 1
        while n > 0 and raw <= n - 0.5 - h:
            n -= 1
        return n

    def _raw_removed(self, sm, b):
        return max(0.0, -sm.get(b, 0.0)) / self._units[b]

    def update(self, weights):
        sm = self._smooth(weights)

        # 1) per-bin removal (own cell, debounced)
        for b in self._units:
            self._removed[b] = self._hysteretic(self._raw_removed(sm, b), self._removed[b], self._count_h(b))
            if self._removed[b] > self._removed_peak[b]:
                self._removed_peak[b] = self._removed[b]

        # Per-bin steady-state: has this bin's own EMA actually caught up to its
        # raw reading yet? A bin mid-bump/press, or riding out mechanical
        # cross-talk from a neighbor, is still visibly diverging here — the
        # wrong-bin checks below skip it entirely until it settles (see module
        # docstring). Latched with a wide dead-band (Schmitt-trigger style, like
        # _hysteretic() elsewhere): a single memoryless threshold test flickers
        # true/false every frame under ordinary sensor noise once that noise is
        # comparable to the tolerance, which made the alert itself flash on/off.
        # Entering settled still requires a genuinely small gap; LEAVING settled
        # requires the gap to grow well past it (3x), so borderline noise can't
        # flip the decision every frame in either direction.
        #
        # Tolerance is scaled per-bin (count_h(b) as a fraction of that bin's OWN
        # unit weight, same relative precision already accepted for counting)
        # rather than a flat count_tolerance_g. A flat gram value makes heavier
        # items take much longer, in absolute time, to converge tightly enough to
        # even be considered -- up to ~2s+ for a 100g+ item at ema_alpha=0.4 -- so
        # a mistake corrected faster than that would never get a chance to fault.
        raw_bin = {b: self._adj(weights, b) for b in self._units}
        settled = {}
        for b in self._units:
            tol = self._count_h(b) * self._units[b]
            gap = abs(raw_bin[b] - sm.get(b, 0.0))
            if self._settled_state[b]:
                if gap > tol * 3:
                    self._settled_state[b] = False
            else:
                if gap <= tol:
                    self._settled_state[b] = True
            settled[b] = self._settled_state[b]

        raw_box = max(0.0, self._adj(weights, self._box_id))    # unsmoothed, tared
        box = max(0.0, self._adj(sm, self._box_id))              # smoothed, tared

        # 2) activation: in IDLE, the clearest not-done bin past the threshold goes
        # active — but only once it's been the leading candidate continuously for
        # activation_confirm_s. A single noisy frame crossing the threshold used to
        # activate (and visually "focus") a bin instantly, with zero real hand
        # interaction — the same class of gap as the fault checks (raw threshold
        # cross, no persistence requirement), just never covered here. If the
        # candidate changes or drops below threshold, the clock restarts — this
        # is a genuine "did it actually happen" check, not a fixed delay before
        # a real pick is allowed to register (a real reach clears it easily).
        #
        # Only bins that are in THIS kit (target > 0) can activate. A bin not in
        # the BOM must never silently become the active bin when picked from —
        # that pick is a mistake and has to fault (see the wrong-bin block below),
        # not quietly focus the wrong bin and swallow the error.
        if self._active is None:
            now = time.time()
            cands = [b for b in self._units
                     if b not in self._done and self._targets.get(b, 0) > 0
                     and self._raw_removed(sm, b) >= self._activation]
            if not cands:
                self._activation_candidate = None
            else:
                b = max(cands, key=lambda b: self._raw_removed(sm, b))
                if b != self._activation_candidate:
                    self._activation_candidate = b
                    self._activation_candidate_since = now
                if now - self._activation_candidate_since >= self._activation_confirm_s:
                    self._active = b
                    self._activation_candidate = None

        # While idle, keep the box baseline tracking the settled EMA, instead of
        # freezing it once at completion. The EMA needs a few frames to catch up
        # after a step change, and completion can fire before it has; freezing
        # then banks a stale value, so the next bin's raw_placed reads the tail
        # of THIS bin's convergence as if it were newly placed weight. Continuous
        # tracking absorbs that settling during the operator's natural
        # bin-to-bin transition — no added wait, since it just piggybacks on
        # time that already elapses.
        #
        # Two guards, not just "no bin active":
        #  - Runs AFTER the activation check, not before: on the exact frame a
        #    bin activates, the box may already reflect a genuine same-frame
        #    placement (a single observed sample, or a fast remove-then-place
        #    within one poll interval) — snapping the baseline to that already-
        #    elevated value would erase the very placement being measured.
        #  - Only commits once the smoothed box has actually caught up to its
        #    raw reading (within the box's own noise floor, self._step_floor):
        #    the box's OWN EMA can still be mid-transition from an earlier event
        #    (e.g. right as it starts rising for this bin's first placement) even
        #    while no bin is active yet — banking an unconverged mid-transition
        #    value is just as wrong as banking one at the instant of activation.
        if self._active is None and abs(raw_box - box) <= self._step_floor:
            self._baseline_box = box

        active = self._active
        prev_placed = self._placed.get(active, 0) if active else 0

        # The active bin changed since last frame (new pick, or a completion) ->
        # the "what's in hand" unit changed too, so any accumulated match progress
        # against the OLD unit is meaningless now.
        if self._return_match_active != active:
            self._return_match = {b: 0 for b in self._units}
            self._return_match_active = active

        # Units of the active bin's item genuinely in-hand: removed but not yet
        # placed. Zero means hands are empty -- nothing to misplace.
        holding = (self._removed[active] - prev_placed) if active is not None else 0

        fault = None

        # 3) wrong-bin faults: any non-active bin deviating from its expected state.
        # Suppressed during the post-tare settle window (see tare()) so a bin
        # still catching up from a fresh zero can't spuriously fault, per-bin
        # until that bin reaches steady state (see module docstring), and
        # entirely while the ACTIVE bin itself hasn't settled -- genuine, ongoing
        # physical activity there (a real pick or place in progress) is the most
        # likely source of mechanical vibration bleeding into a neighbor's cell,
        # so other bins aren't evaluated until that activity itself settles.
        active_settled = active is None or settled[active]
        if time.time() >= self._settle_until and active_settled:
            for b in self._units:
                if b == active or not settled[b]:
                    continue
                expected = self._targets[b] if b in self._done else 0
                r = self._removed[b]

                # Did a settled increase here match roughly one unit of the
                # ACTIVE bin's item weight (not b's own)? Same check for a
                # done or not-yet-done bin -- either way, the only thing that
                # could legitimately be landing on b right now is what's
                # actually in hand, so that's what has to match, not b's own
                # nominal item weight (a light held item would otherwise never
                # cross a much heavier done bin's own threshold). The increase
                # is measured against b's own EXPECTED resting point
                # (-expected*unit, zero for a not-yet-done bin), not against
                # the original zero tare -- a done bin already sits well below
                # zero (its own item fully removed), so comparing to zero would
                # never see a foreign item landing there as an "increase" at
                # all unless it happened to outweigh everything already taken.
                returned = False
                expected_baseline = -expected * self._units[b]
                deviation = sm.get(b, 0.0) - expected_baseline
                if active is not None and holding > 0 and deviation > 0:
                    raw_match = deviation / self._units[active]
                    self._return_match[b] = self._hysteretic(
                        raw_match, self._return_match[b], self._count_h(active))
                    returned = self._return_match[b] >= 1

                # A mispick is classified by whether the bin is still a valid target:
                #   - already DONE (green)  -> pick-from-wrong-bin: it's finished, you
                #     must never take from it again — a real "wrong bin", not a timing slip.
                #   - in the BOM, not done, but locked while another bin is active
                #                           -> out-of-sequence: a required bin, just not yet;
                #     put it back and finish the active one first.
                #   - not in the BOM at all -> pick-from-wrong-bin: faults even while idle
                #     (it can never activate, so nothing else flags the mistake).
                in_bom = self._targets.get(b, 0) > 0
                if b in self._done:
                    if returned:
                        fault = fault or self._fault("return-to-wrong-bin", b)
                    elif r > expected:
                        # a completed/green bin — never pick from it again
                        fault = fault or self._fault("pick-from-wrong-bin", b)
                else:  # available/locked, expected empty change
                    if returned:
                        fault = fault or self._fault("return-to-wrong-bin", b)
                    elif r >= 1:
                        if in_bom:
                            # in the kit but locked while another bin is active;
                            # when idle it would have activated above instead.
                            if active is not None:
                                fault = fault or self._fault("out-of-sequence", b)
                        else:
                            # not in this kit — always a wrong-bin pick, idle or not.
                            fault = fault or self._fault("pick-from-wrong-bin", b)

        # 4) active-bin counting + its faults
        if active is not None:
            unit = self._units[active]
            raw_placed = (box - self._baseline_box) / unit
            # No freeze on a decrease: operators never take items back out of
            # the kit box, so a box-weight drop is never itself a fault — most
            # commonly it's the operator correcting an overpack, which should
            # just track straight back down like any other change.
            placed = self._hysteretic(max(0.0, raw_placed), prev_placed, self._box_count_h(active))

            # Cap at the HIGH-WATER removed, not the instantaneous value, so a
            # transient dip in the active bin's own cell can't yank the box count
            # back down to 0 while the item is sitting in the box (the flicker
            # operators saw). A genuine removal from the box still lowers `placed`
            # via `raw_placed` above — this only blocks the spurious dips.
            placed = min(placed, self._removed_peak[active])
            self._placed[active] = placed

            if placed > self._targets[active]:
                fault = self._fault("overpack-kit", active)  # overrides; highest severity

            # completion -> bank + advance (only when clean)
            if (fault is None and placed == self._targets[active]
                    and self._removed[active] == self._targets[active]):
                self._done.add(active)
                self._baseline_box = box
                self._active = None
                active = None

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
        # The UI shows a big keyword (from `type`) plus this message as the
        # corrective-action subtitle — so the message is the action only, no
        # "WRONG BIN —" prefix (the keyword carries that). Each names the SPECIFIC
        # bin to act on; `binid` means something different per fault:
        #   overpack-kit         -> the bin the excess belongs back in (== binid)
        #   pick-from-wrong-bin  -> a bin NOT in this kit, OR one already completed
        #                           (green) — wrongly taken FROM; put it back THERE
        #   return-to-wrong-bin  -> the bin something was wrongly placed INTO;
        #                           the fix is to take it back OUT (and, if a bin
        #                           is active, it belongs there)
        #   out-of-sequence      -> a bin that IS in this kit and not yet done, but
        #                           is locked while another bin is active; put it
        #                           back and finish the active bin first
        active = self._active
        lbl = self._lbl(binid)
        active_lbl = self._lbl(active)

        # Count of units the operator must move — the "how many?" operators said
        # the corrections were missing. Derived per fault kind:
        if kind == "overpack-kit":
            n = self._placed.get(binid, 0) - self._targets.get(binid, 0)
        elif kind in ("pick-from-wrong-bin", "out-of-sequence"):
            # units wrongly taken OUT of binid = removed beyond what it should give
            # (a completed bin should have given its whole target; any other, zero).
            expected = self._targets.get(binid, 0) if binid in self._done else 0
            n = self._removed.get(binid, 0) - expected
        elif kind == "return-to-wrong-bin":
            # foreign units the active item's weight says landed in binid.
            n = self._return_match.get(binid, 0)
        else:
            n = 0
        n = max(1, int(round(n)))  # always concrete + actionable (never "0"/"ITEM")

        belongs = f", belongs in {active_lbl}" if active else ""
        finish = f" — finish {active_lbl} first" if active else ""
        msgs = {
            "overpack-kit": f"Remove {n} from kit, return to {lbl}",
            "pick-from-wrong-bin": f"Return {n} to {lbl}",
            "return-to-wrong-bin": f"Remove {n} from {lbl}{belongs}",
            "out-of-sequence": f"Return {n} to {lbl}{finish}",
        }
        return {"type": kind, "bin": binid, "bin_label": lbl,
                "message": msgs[kind], "count": n}
