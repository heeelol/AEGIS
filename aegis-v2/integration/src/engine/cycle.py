"""Cycle / set sequencing for the kitting work order.

A *cycle* is an ordered list of *sets*; each set is a ``{bin_id: quantity}`` pick
list (bin ids are canonical grid ids, after bin_remap). The operator completes the
current set's pick list and confirms it (``advance``); the cycle is done once the
last set is confirmed. ``restart`` begins the same configured cycle from set 1.

Pure logic — no I/O, no deps — so it is unit-tested in isolation and the pipeline
just drives it.
"""

from __future__ import annotations


class CycleManager:
    def __init__(self, sets: list[dict[str, int]] | None):
        # Sanitize: keep positive quantities only, drop empty sets.
        cleaned = []
        for s in (sets or []):
            picks = {b: int(q) for b, q in (s or {}).items() if int(q) > 0}
            if picks:
                cleaned.append(picks)
        self._sets = cleaned
        self._index = 0
        self._complete = len(self._sets) == 0

    @property
    def total_sets(self) -> int:
        return len(self._sets)

    @property
    def set_number(self) -> int:
        """1-based current set number (clamped to total_sets once complete)."""
        if self._complete:
            return self.total_sets
        return self._index + 1

    @property
    def is_complete(self) -> bool:
        return self._complete

    def current_targets(self) -> dict[str, int]:
        """The active set's pick list, or {} when the cycle is complete/empty."""
        if self._complete or not self._sets:
            return {}
        return dict(self._sets[self._index])

    def advance(self) -> bool:
        """Confirm the current set and move on.

        Returns True iff the cycle JUST completed (the confirmed set was the last).
        """
        if self._complete:
            return False
        if self._index + 1 < len(self._sets):
            self._index += 1
            return False
        self._complete = True
        return True

    def restart(self) -> None:
        """Begin the same configured cycle again from set 1."""
        self._index = 0
        self._complete = len(self._sets) == 0

    def snapshot(self) -> dict:
        """Serializable view for the dashboard API."""
        return {
            "set_number": self.set_number,
            "total_sets": self.total_sets,
            "complete": self._complete,
        }
