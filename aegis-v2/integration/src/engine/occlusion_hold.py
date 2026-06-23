"""
Occlusion Hold
==============
Keeps an eligible (bottom-layer) bin marked **active** while the hand picking from
it is hidden under the rack lip and the stateless :class:`BinAssignmentEngine`
drops it. This stateful layer bridges that gap for visual continuity only — it
does not affect counts.

Bin-keyed, **handedness-independent**. MediaPipe's left/right label is unreliable
on an occluded/extrapolated hand (it flips), so keying a hold by handedness
produces stuck phantom holds. Instead a *bin* is held while:

  1. it has been **visibly picked** (a genuine, non-occluded in-bin assignment
     armed it), and
  2. it is still **occupied** — real foreground (a forearm) is in the bin region.

Occluded fingertip events (the extrapolated tips with no foreground behind them)
are dropped as unreliable, so they never light a bin or leave a stray 0.0 marker;
the occupancy-confirmed hold lights the bin instead. A hold releases the instant
its bin region goes background (the forearm left).

This complements the *occlusion gate* in ``BinAssignmentEngine`` (which rewrites a
single extrapolated fingertip); the hold deals with the bin's continuity over time.
"""
from __future__ import annotations

import logging

from .bin_assignment import BinEvent

logger = logging.getLogger("aegis.engine.occlusion_hold")


class OcclusionHold:
    """Stateful per-bin hold. Call :meth:`apply` every frame."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self._enabled: bool = config.get("enabled", True)
        # None → pipeline auto-detects the bottom row; otherwise explicit rows.
        self.eligible_rows = config.get("eligible_rows")
        self._eligible_bins: set[str] = set()
        # bin_id -> the event that last armed it (a genuine in-bin pick); used as
        # the template for the synthetic hold event.
        self._armed: dict[str, BinEvent] = {}

    def set_eligible_bins(self, bin_ids: set[str]) -> None:
        """Set which bin ids may be held. Drops arms on no-longer-eligible bins."""
        self._eligible_bins = set(bin_ids)
        for bid in [b for b in self._armed if b not in self._eligible_bins]:
            del self._armed[bid]

    def apply(
        self,
        events: list[BinEvent],
        hands: list,
        occluded_ids: set[int] | None = None,
        occupied_bins: set[str] | None = None,
    ) -> list[BinEvent]:
        """Augment this frame's events with held bins.

        ``occluded_ids`` are the ``hand_id``s whose fingertip is occluded this
        frame (foreground ratio below ``occlusion_ratio``). Their events are
        dropped as unreliable extrapolations.

        ``occupied_bins`` are the eligible bins that still have real foreground (a
        forearm reaching in) — the authoritative "is the hand still there" signal.
        When ``None`` (no foreground model / warming up) occupancy release is
        skipped.
        """
        if not self._enabled:
            return events
        occluded_ids = occluded_ids or set()

        # 1. Arm bins from genuine (non-occluded) in-bin picks.
        for ev in events:
            if ev.hand_id in occluded_ids:
                continue
            if ev.bin_id is not None and ev.bin_id in self._eligible_bins:
                self._armed[ev.bin_id] = ev

        # 2. Release armed bins whose forearm has left (region went background).
        if occupied_bins is not None:
            for bid in [b for b in self._armed if b not in occupied_bins]:
                logger.info("Occlusion hold released bin %s (empty)", bid)
                del self._armed[bid]

        # 3. Drop unreliable occluded fingertip events (the stray 0.0 markers).
        result = [ev for ev in events if ev.hand_id not in occluded_ids]

        # 4. Keep each armed bin lit while no live event currently covers it.
        active = {ev.bin_id for ev in result if ev.bin_id is not None}
        for bid, tmpl in self._armed.items():
            if bid in active:
                continue
            result.append(BinEvent(
                hand_id=tmpl.hand_id,
                handedness=tmpl.handedness,
                bin_id=bid,
                bin_label=tmpl.bin_label,
                hand_point=tmpl.hand_point,
                hand_area=tmpl.hand_area,
                confidence=tmpl.confidence,
                method="occlusion_hold",
            ))
        return result
