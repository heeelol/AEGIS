"""
Load Cell Reader
================
Reads per-bin weights from a load-cell array via an ESP32 over USB serial.

Wiring model
------------
Load cells (HX711 amplifiers) are wired to an **ESP32**. The ESP32 firmware
sums the channels for each bin and applies calibration, then streams ready-made
per-bin weights to this host as **newline-delimited JSON** over USB serial::

    {"bins": {"bin_0_0": 123.4, "bin_0_1": 80.1, "bin_1_0": 0.0}}\\n
    {"bins": {"bin_0_0": 121.9, ...}}\\n
    ...

A flat object (``{"bin_0_0": 123.4, ...}``) is also accepted. Weights are in
grams, keyed by ``bin_{row}_{col}`` to match the rest of the system.

This reader opens the port and pulls lines on a background thread, caching the
latest reading. ``get_weights`` / ``get_layout`` return that cache, so the
pipeline's per-frame polling never blocks on serial I/O. When the port is
absent or no data has arrived recently, it reports "not connected" and returns
empty values — and :class:`~integration.src.ui.state.PipelineState` falls back
to the CV-detected layout, exactly as it did with the old stub.

Config (``sensing.loadcells`` in settings.yaml)::

    loadcells:
      enabled: true
      port: "/dev/ttyUSB0"   # ESP32 USB device (Windows: "COM3")
      baudrate: 115200
      stale_after: 2.0       # seconds without data → is_connected() False
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("aegis.sensing.loadcell")


@dataclass
class LayerLayout:
    """Physical bin layout as reported by the load cells."""
    num_layers: int = 0
    bins_per_layer: dict[int, int] = field(default_factory=dict)  # layer index -> bin count


class LoadCellReader:
    """
    Streams per-bin weights from an ESP32 over USB serial.

    The latest reading is cached and served to callers without blocking; a
    daemon thread does the actual serial reads. Disabled (or unable to open the
    port) → behaves like the old stub: not connected, empty weights/layout.
    """

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._enabled = bool(self._config.get("enabled", False))
        self._port = self._config.get("port", "/dev/ttyUSB0")
        self._baudrate = int(self._config.get("baudrate", 115200))
        self._stale_after = float(self._config.get("stale_after", 2.0))
        # Optional {firmware_bin_id: canonical_bin_id} rename, applied to every
        # reading before it is cached, so all downstream consumers (weights,
        # layout, inventory, dashboard) see the canonical id.
        self._remap: dict[str, str] = dict(self._config.get("bin_remap") or {})

        self._serial = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._weights: dict[str, float] = {}
        self._last_rx: float = 0.0

        if self._enabled:
            self._connect()
        else:
            logger.info("Load cells disabled in config — layout from CV only")

    # ── Connection / read loop ───────────────────────────────

    def _connect(self) -> None:
        """Open the serial port and start the background read thread."""
        try:
            import serial  # pyserial — only needed when load cells are enabled
        except ImportError:
            logger.error("pyserial not installed (pip install pyserial); "
                         "load cells disabled")
            return

        try:
            self._serial = serial.Serial(self._port, self._baudrate, timeout=1.0)
        except Exception as exc:  # serial.SerialException and friends
            logger.error("Cannot open load-cell port %s @ %d: %s",
                         self._port, self._baudrate, exc)
            self._serial = None
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="aegis-loadcells")
        self._thread.start()
        logger.info("Load-cell reader listening on %s @ %d baud",
                    self._port, self._baudrate)

    def _read_loop(self) -> None:
        """Read newline-delimited JSON from the ESP32 and cache the latest."""
        while not self._stop.is_set():
            try:
                raw = self._serial.readline()
            except Exception as exc:
                logger.warning("Load-cell serial read failed: %s", exc)
                break
            if not raw:
                continue  # timeout, no data this tick
            weights = self._parse_line(raw)
            if weights is not None:
                weights = self._apply_remap(weights)
                with self._lock:
                    self._weights = weights
                    self._last_rx = time.time()

    @staticmethod
    def _parse_line(raw: bytes) -> dict[str, float] | None:
        """Parse one serial line into a {bin_id: grams} dict, or None if junk.

        Accepts either ``{"bins": {...}}`` or a flat ``{...}`` object.
        """
        try:
            obj = json.loads(raw.decode("utf-8").strip())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None  # partial line / boot noise / garbage — skip it
        if not isinstance(obj, dict):
            return None
        bins = obj.get("bins", obj)
        if not isinstance(bins, dict):
            return None
        out: dict[str, float] = {}
        for bin_id, grams in bins.items():
            try:
                out[str(bin_id)] = float(grams)
            except (TypeError, ValueError):
                continue
        return out

    def _apply_remap(self, weights: dict[str, float]) -> dict[str, float]:
        """Rename bin ids per ``self._remap``; identity when remap/weights empty."""
        if not self._remap or not weights:
            return weights
        return {self._remap.get(bin_id, bin_id): grams
                for bin_id, grams in weights.items()}

    # ── Public interface (unchanged contract) ────────────────

    def is_connected(self) -> bool:
        """True when the port is open and data arrived within ``stale_after``."""
        if self._serial is None:
            return False
        with self._lock:
            return (time.time() - self._last_rx) < self._stale_after

    def get_weights(self) -> dict[str, float]:
        """Latest weight per bin in grams, keyed by ``bin_{row}_{col}``."""
        with self._lock:
            return dict(self._weights)

    def get_layout(self) -> LayerLayout:
        """Bin layout inferred from the bin ids currently reporting weight.

        Empty until data arrives, so :meth:`PipelineState.get_layout` keeps
        using the CV-detected grid; once the ESP32 streams known bins, those
        counts merge in (the state layer takes the max of CV and load cells).
        """
        with self._lock:
            ids = list(self._weights.keys())
        if not ids:
            return LayerLayout()

        bins_per_layer: dict[int, int] = {}
        for bin_id in ids:
            parts = bin_id.split("_")
            try:
                row, col = int(parts[-2]), int(parts[-1])
            except (ValueError, IndexError):
                continue
            bins_per_layer[row] = max(bins_per_layer.get(row, 0), col + 1)
        return LayerLayout(num_layers=len(bins_per_layer),
                           bins_per_layer=bins_per_layer)

    def close(self) -> None:
        """Stop the read thread and release the serial port."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None


# ── Standalone tester ────────────────────────────────────────
# Run directly to verify the ESP32 -> host serial link in isolation:
#     python3 -m sensing.loadcell --port /dev/ttyUSB1
# Prints, once a second until Ctrl-C: how many items have been grabbed per bin
# (via InventoryTracker + config/inventory.yaml) plus the total weight change.
# Needs pyserial: pip install pyserial
if __name__ == "__main__":
    import argparse

    # Works both as `-m integration.src.sensing.loadcell` (package-relative) and
    # as a plain script run from this directory (flat import).
    try:
        from .inventory import InventoryTracker
    except ImportError:
        from inventory import InventoryTracker

    parser = argparse.ArgumentParser(description="Standalone load-cell serial tester")
    parser.add_argument("--port", default="/dev/ttyUSB0",
                        help="serial device (Windows: COM3)")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--stale-after", type=float, default=2.0,
                        help="seconds without data before is_connected() is False")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    reader = LoadCellReader({
        "enabled": True,
        "port": args.port,
        "baudrate": args.baudrate,
        "stale_after": args.stale_after,
    })
    tracker = InventoryTracker()   # loads config/inventory.yaml

    print(f"Listening on {args.port} @ {args.baudrate} baud. Ctrl-C to stop.")
    try:
        while True:
            connected = reader.is_connected()
            weights = reader.get_weights()
            taken = tracker.items_taken(weights)
            # Items grabbed per bin (with the item name), e.g. "bin_0_0(bolt_m6): 5"
            grabbed = " ".join(f"{bin_id}({tracker.item_for(bin_id)}): {n}"
                               for bin_id, n in sorted(taken.items()))
            total_items = tracker.total_taken(weights)
            total_change = sum(weights.values())   # signed grams since tare
            print(f"connected={connected}  {grabbed}  "
                  f"| total_items={total_items}  total_change={total_change:.2f}g")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
