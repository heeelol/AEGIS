"""
Occlusion Hold
==============
Keeps an eligible (bottom-layer) bin marked **active** while the hand that was
picking from it is hidden under the rack lip and the hand tracker has lost sight
of it. The stateless :class:`BinAssignmentEngine` drops the bin the instant the
hand disappears; this stateful layer bridges that gap.

A hold is keyed by **handedness** (one left, one right): when a hand is assigned
to an eligible bin it is remembered, and while that handedness is no longer seen
the bin keeps emitting an active event. The hold ends the moment that handedness
is seen again — there is **no timeout** (a bin can stay lit if the operator walks
away; clear it from the dashboard if needed).

This complements — and is distinct from — the *occlusion gate* inside
``BinAssignmentEngine``, which rewrites a single extrapolated fingertip; the hold
deals with the hand vanishing entirely.
"""
from __future__ import annotations

import logging

from .bin_assignment import BinEvent

logger = logging.getLogger("aegis.engine.occlusion_hold")


class OcclusionHold:
    """Stateful hold over eligible bins. Call :meth:`apply` every frame."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self._enabled: bool = config.get("enabled", True)
        # None → pipeline auto-detects the bottom row; otherwise explicit rows.
        self.eligible_rows = config.get("eligible_rows")
        self._eligible_bins: set[str] = set()
        # handedness -> the BinEvent that established the hold.
        self._holds: dict[str, BinEvent] = {}

    def set_eligible_bins(self, bin_ids: set[str]) -> None:
        """Set which bin ids may be held. Holds on no-longer-eligible bins drop."""
        self._eligible_bins = set(bin_ids)
        for handed in [h for h, ev in self._holds.items()
                       if ev.bin_id not in self._eligible_bins]:
            del self._holds[handed]

    def apply(self, events: list[BinEvent], hands: list) -> list[BinEvent]:
        """Augment this frame's events with held bins.

        - A hand currently in an eligible bin (re)arms a hold for its handedness.
        - A hand seen again (its handedness present) but not in its held bin ends
          the hold.
        - While a held handedness is absent, a synthetic ``occlusion_hold`` event
          keeps the bin active.
        """
        if not self._enabled:
            return events

        present_handed = {getattr(h, "handedness", "") for h in hands}

        for ev in events:
            if ev.bin_id is not None and ev.bin_id in self._eligible_bins:
                # Actively picking an eligible bin → arm/refresh the hold.
                self._holds[ev.handedness] = ev
            elif ev.handedness in self._holds and ev.handedness in present_handed:
                # Hand is back in view but no longer in its held bin → release.
                logger.info("Occlusion hold released for %s hand (seen again)", ev.handedness)
                del self._holds[ev.handedness]

        result = list(events)
        active_bins = {ev.bin_id for ev in events if ev.bin_id is not None}
        for handed, held in self._holds.items():
            if handed in present_handed:
                continue  # hand visible this frame; its own event covers the bin
            if held.bin_id in active_bins:
                continue  # bin already active via another hand/event
            result.append(BinEvent(
                hand_id=held.hand_id,
                handedness=held.handedness,
                bin_id=held.bin_id,
                bin_label=held.bin_label,
                hand_point=held.hand_point,
                hand_area=held.hand_area,
                confidence=held.confidence,
                method="occlusion_hold",
            ))
        return result
