"""
Inventory Tracker
=================
Turns live per-bin weights into item counts.

Each bin holds exactly one item type. The load cell is tared with the bins
FULL at startup, so each bin's live weight is the signed change from full: it
goes negative as items are removed. For a bin holding an item of unit weight
``unit_g``::

    items_taken = round(-weight / unit_g)      # clamped at >= 0

The item->unit-weight map and bin->item assignment live in their own file
(``config/inventory.yaml``), kept separate from ``settings.yaml``::

    items:
      bolt_m6:   { unit_g: 4.2 }
      washer_m6: { unit_g: 1.1 }
    bins:
      bin_0_0: bolt_m6
      bin_0_1: washer_m6
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("aegis.sensing.inventory")

# Default config path, resolved relative to this file so it works from any cwd.
_DEFAULT_CONFIG = str(Path(__file__).resolve().parents[2] / "config" / "inventory.yaml")


class InventoryTracker:
    """Maps per-bin weights to item counts using a separate inventory.yaml.

    Construct with a path to the YAML (defaults to ``config/inventory.yaml``),
    then feed it the weights dict from :meth:`LoadCellReader.get_weights`.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG):
        import yaml

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        items = cfg.get("items", {}) or {}
        bins = cfg.get("bins", {}) or {}

        # bin_id -> unit weight in grams, flattened for quick lookup.
        self._unit_g: dict[str, float] = {}
        # bin_id -> item name, for reporting.
        self._item: dict[str, str] = {}

        for bin_id, item_name in bins.items():
            spec = items.get(item_name)
            if spec is None:
                logger.warning("Bin %s references unknown item %r — skipped",
                               bin_id, item_name)
                continue
            unit_g = float(spec.get("unit_g", 0.0))
            if unit_g <= 0.0:
                logger.warning("Item %r has non-positive unit_g (%s) — skipped",
                               item_name, unit_g)
                continue
            self._unit_g[str(bin_id)] = unit_g
            self._item[str(bin_id)] = str(item_name)

        if not self._unit_g:
            logger.warning("No usable bin/item entries in %s", config_path)

    # ── Queries ──────────────────────────────────────────────

    def items_taken(self, weights: dict[str, float]) -> dict[str, int]:
        """Items removed per bin, derived from each bin's (negative) weight.

        Bins with no inventory entry or no current weight are omitted.
        """
        taken: dict[str, int] = {}
        for bin_id, unit_g in self._unit_g.items():
            weight = weights.get(bin_id)
            if weight is None:
                continue
            # Tared full -> weight goes negative as items leave. Removed mass
            # is -weight; clamp so noise just above zero doesn't read negative.
            n = round(max(0.0, -weight) / unit_g)
            taken[bin_id] = int(n)
        return taken

    def total_taken(self, weights: dict[str, float]) -> int:
        """Total items removed across all bins."""
        return sum(self.items_taken(weights).values())

    def snapped_grams(self, weights: dict[str, float]) -> dict[str, float]:
        """Removed mass per bin, snapped to a whole number of items.

        i.e. ``items_taken * unit_g`` — the clean weight that corresponds to an
        exact item count, with measurement noise rounded out.
        """
        taken = self.items_taken(weights)
        return {bin_id: n * self._unit_g[bin_id] for bin_id, n in taken.items()}

    def item_for(self, bin_id: str) -> str | None:
        """The item name assigned to a bin, or None if unmapped."""
        return self._item.get(bin_id)

    def units(self) -> dict[str, float]:
        """bin_id -> unit weight (grams) for every inventory-mapped bin."""
        return dict(self._unit_g)
