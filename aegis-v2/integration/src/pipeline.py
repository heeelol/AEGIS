"""
AEGIS v2 — Integration Pipeline
=================================
The main orchestrator that wires together:
  1. CV Model   → Bin boundary detection (snapshot at startup)
  2. Hand Model → Real-time hand tracking (any registered backend)
  3. Engine     → Bin assignment + Triple-Gate FSM for kinetic gating
  4. UI         → OpenCV camera overlay + FastAPI web dashboard

Lifecycle:
  INIT  — Open camera, snapshot bins, lock coordinates, load hand tracker,
          build overlay, launch dashboard
  LOOP  — Sense (detect hands) → Analyse (assign bins, run FSM) → Act (UI + gate)
  STOP  — Cleanup resources

Usage:
    from integration.src.pipeline import Pipeline
    Pipeline("config/settings.yaml").run()
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

# Ensure parent paths are importable
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT.parent))  # aegis-v2/
sys.path.insert(0, str(_ROOT))         # integration/

# Default config path, resolved relative to this file so the pipeline can be
# launched from any working directory.
_DEFAULT_CONFIG = str(_ROOT / "integration" / "config" / "settings.yaml")

from integration.src.detectors import BinDetector
from integration.src.engine import BinAssignmentEngine, BinRegion, TripleGateFSM, SensorReading
from integration.src.sensing import LoadCellReader
from integration.src.ui.overlay import OverlayUI
from integration.src.ui.state import PipelineState

# Import hand tracker registry and backends
from hand_models.common import TrackerRegistry, BaseHandTracker

# Auto-register available backends
try:
    import hand_models.mediapipe.tracker  # noqa: F401
except ImportError:
    pass
try:
    import hand_models.yolo_hand.tracker  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger("aegis.pipeline")


def _load_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Pipeline:
    """
    Main AEGIS v2 orchestrator.

    Connects cv-models (bin detection) with hand-models (hand tracking)
    through the bin assignment engine and triple-gate FSM, with both an
    OpenCV camera overlay and a web dashboard for the operator.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG):
        self._config = _load_config(config_path)
        self._setup_logging()

        self._cap: Optional[cv2.VideoCapture] = None
        self._bin_detector: Optional[BinDetector] = None
        self._hand_tracker: Optional[BaseHandTracker] = None
        self._assignment: Optional[BinAssignmentEngine] = None
        self._fsm: Optional[TripleGateFSM] = None
        self._overlay: Optional[OverlayUI] = None
        self._loadcells: Optional[LoadCellReader] = None
        self._geofences: dict = {}

        # Shared state for the web dashboard
        self._state = PipelineState()

    def run(self) -> None:
        """Full lifecycle: init → loop → cleanup."""
        try:
            self._open_camera()
            self._detect_bins()
            self._apply_work_order()
            self._init_loadcells()
            self._load_hand_tracker()
            self._create_engines()
            self._create_overlay()
            self._start_dashboard()
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._cleanup()

    # ── Stage 1: Initialization ──────────────────────────────

    def _open_camera(self) -> None:
        cam = self._config.get("camera", {})
        source = cam.get("source", 0)
        logger.info("Opening camera: %s", source)
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("width", 1280))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 720))
        self._cap.set(cv2.CAP_PROP_FPS, cam.get("fps", 30))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {source}")

    def _detect_bins(self) -> None:
        """Snapshot → detect bin boundaries → lock coordinates for session.

        If ``bin_detector.manual_layout`` is set, CV detection is skipped and
        the bins are laid out as an even grid built from the per-layer counts.
        """
        det_cfg = self._config.get("bin_detector", {})

        logger.info("Taking initialization snapshot for bin detection...")
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture initialization snapshot")

        manual_layout = det_cfg.get("manual_layout")
        if manual_layout:
            h, w = frame.shape[:2]
            self._geofences = self._build_manual_geofences(manual_layout, w, h)
            logger.info("Manual bin layout: %d layer(s), %d bins total",
                        len(manual_layout), len(self._geofences))
        else:
            model_path = det_cfg.get("model_path", "yolov8n.pt")
            conf = det_cfg.get("confidence_threshold", 0.5)
            self._bin_detector = BinDetector(model_path=model_path, conf_threshold=conf)
            self._geofences = self._bin_detector.detect_bins(frame)
            logger.info("Bin map locked: %d regions", len(self._geofences))

        # Push bin map to shared state
        self._state.update_bins(self._geofences)

        # Show detection result (CV path only; manual layout has no detector)
        if self._config.get("debug", {}).get("show_init_snapshot", False) and self._bin_detector:
            vis = self._bin_detector.visualize(frame)
            cv2.imshow("Bin Detection — Initialization", vis)
            cv2.waitKey(2000)
            cv2.destroyWindow("Bin Detection — Initialization")

    @staticmethod
    def _build_manual_geofences(
        bins_per_layer: list, width: int, height: int, gap: int = 6
    ) -> dict:
        """Build geofences from per-layer bin counts, no pixel coords needed.

        ``bins_per_layer[i]`` is the number of bins in layer (row) ``i``. Layers
        split the frame height equally; within a layer the bins split the width
        equally. Produces the same ``bin_{row}_{col}`` ids and geofence dict the
        CV detector would, with confidence 1.0. ``gap`` insets each box a few
        pixels so adjacent bins don't share an edge.
        """
        num_layers = len(bins_per_layer)
        geofences: dict = {}
        if num_layers == 0:
            return geofences
        row_h = height / num_layers
        for row, n_cols in enumerate(bins_per_layer):
            n_cols = int(n_cols)
            if n_cols <= 0:
                continue
            col_w = width / n_cols
            y_min = int(row * row_h) + gap
            y_max = int((row + 1) * row_h) - gap
            for col in range(n_cols):
                x_min = int(col * col_w) + gap
                x_max = int((col + 1) * col_w) - gap
                geofences[f"bin_{row}_{col}"] = {
                    "x_min": max(0, x_min),
                    "x_max": min(width, x_max),
                    "y_min": max(0, y_min),
                    "y_max": min(height, y_max),
                    "confidence": 1.0,
                }
        return geofences

    def _apply_work_order(self) -> None:
        """Load predetermined target pick counts and push them to the dashboard.

        ``work_order.targets`` is a list-of-lists mirroring the bin layout: one
        inner list per layer, one number per bin. Each maps to a ``bin_{row}_{col}``
        id so the dashboard can show "current / target" (e.g. 2/5). Bins with no
        entry keep target 0 and display just the live count.
        """
        wo_cfg = self._config.get("work_order", {}) or {}
        targets = wo_cfg.get("targets")
        if not targets:
            logger.info("No work order configured — bins show live count only")
            return

        # Build a target for EVERY bin on the layout. A bin's target comes from
        # the targets grid by its (row, col); bins with target 0 — or absent
        # from the grid entirely — are marked not-in-use (using=False) by
        # set_work_order, so a hand entering one trips the wrong-bin warning.
        bin_targets: dict = {}
        for bin_id in self._geofences:
            row, col = PipelineState._parse_bin_id(bin_id)
            try:
                bin_targets[bin_id] = int(targets[row][col])
            except (IndexError, TypeError, ValueError):
                bin_targets[bin_id] = 0

        self._state.set_work_order(bin_targets)
        in_use = sum(1 for t in bin_targets.values() if t > 0)
        logger.info("Work order applied: %d/%d bins in use, %d total items",
                    in_use, len(bin_targets), sum(bin_targets.values()))

    def _init_loadcells(self) -> None:
        """Initialise the load-cell reader and merge its layout into shared state.

        The driver is a stub today (returns nothing), so the dashboard layout
        falls back to the CV-detected grid. Once hardware is wired up, the same
        path supplies real layer counts and per-bin weights.
        """
        lc_cfg = self._config.get("sensing", {}).get("loadcells", {})
        self._loadcells = LoadCellReader(lc_cfg)
        layout = self._loadcells.get_layout()
        self._state.update_loadcells(layout, self._loadcells.get_weights())
        if self._loadcells.is_connected():
            logger.info("Load cells connected: %d layer(s)", layout.num_layers)
        else:
            logger.info("Load cells not connected (stub) — layout from CV only")

    def _load_hand_tracker(self) -> None:
        ht_cfg = self._config.get("hand_tracker", {})
        backend = ht_cfg.get("backend", "mediapipe")
        logger.info("Loading hand tracker: %s", backend)
        logger.info("Available backends: %s", TrackerRegistry.available())
        self._hand_tracker = TrackerRegistry.create(backend, ht_cfg)

    def _create_engines(self) -> None:
        # Bin assignment
        assign_cfg = self._config.get("bin_assignment", {})
        self._assignment = BinAssignmentEngine(assign_cfg)
        self._assignment.set_bin_map_from_geofences(self._geofences)

        # FSM
        self._fsm = TripleGateFSM(self._config)
        self._fsm.set_callbacks(
            on_success=self._on_gate_success,
            on_error=self._on_gate_error,
        )

    def _create_overlay(self) -> None:
        """Build the OpenCV overlay renderer from detected bins."""
        ui_cfg = self._config.get("ui", {})
        if not ui_cfg.get("enabled", True):
            return

        # Convert geofences dict to BinRegion objects for the overlay
        bins = [
            BinRegion(
                bin_id=bid, label=bid,
                x_min=c["x_min"], x_max=c["x_max"],
                y_min=c["y_min"], y_max=c["y_max"],
                confidence=c.get("confidence", 0.0),
                polygon=c.get("polygon"),
            )
            for bid, c in self._geofences.items()
        ]
        self._overlay = OverlayUI(ui_cfg, bins)
        logger.info("OpenCV overlay created with %d bins", len(bins))

    def _start_dashboard(self) -> None:
        """Launch the FastAPI web dashboard in a background thread."""
        dash_cfg = self._config.get("dashboard", {})
        if not dash_cfg.get("enabled", True):
            logger.info("Web dashboard disabled in config")
            return

        port = dash_cfg.get("port", 8080)
        try:
            from integration.src.ui.dashboard import start_dashboard
            start_dashboard(self._state, port=port)
            logger.info("Web dashboard available at http://localhost:%d", port)
        except ImportError as e:
            logger.warning("Could not start dashboard (missing deps): %s", e)
        except Exception as e:
            logger.warning("Dashboard failed to start: %s", e)

    # ── Stage 2: Sense → Analyse → Act loop ──────────────────

    def _main_loop(self) -> None:
        logger.info("Entering Sense-Analyse-Act loop (press 'q' to quit)...")
        frame_count = 0
        t0 = time.time()
        show_overlay = self._overlay is not None

        if show_overlay:
            cv2.namedWindow("AEGIS v2 — Bin Tracker", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self._cap.read()
            if not ret:
                continue

            hands = self._hand_tracker.detect(frame)
            events = self._assignment.assign(hands)

            for ev in events:
                if ev.bin_id is not None:
                    hand = next((h for h in hands if h.hand_id == ev.hand_id), None)
                    is_grabbing = getattr(hand, "is_grabbing", False) if hand else False
                    reading = SensorReading(
                        timestamp=time.time(),
                        hand_in_geofence=True,
                        closed_fist_detected=is_grabbing,
                        weight_delta=0.0,
                        bin_id=ev.bin_id,
                    )
                    self._fsm.update(reading)

            fsm_info = self._fsm.get_state_info()
            self._state.update_hands(hands, events)
            self._state.update_fsm(
                state=fsm_info["state"],
                bin_id=fsm_info["bin_id"],
                elapsed=fsm_info["elapsed_time"],
            )
            active_ids = {ev.bin_id for ev in events if ev.bin_id is not None}
            self._state.update_bins(self._geofences, active_ids)

            if show_overlay:
                display = self._overlay.render(
                    frame, hands, events,
                    fsm_state=self._fsm.state,
                    fsm_info=fsm_info,
                )
                cv2.imshow("AEGIS v2 — Bin Tracker", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t0
                fps = frame_count / max(elapsed, 1e-6)
                self._state.update_fps(fps)
                if self._loadcells is not None:
                    self._state.update_loadcells(
                        self._loadcells.get_layout(),
                        self._loadcells.get_weights(),
                    )

            if frame_count % 300 == 0:
                fps = frame_count / (time.time() - t0)
                logger.info("FPS: %.1f | FSM: %s | Hands: %d | Active bins: %s",
                            fps, self._fsm.state.value, len(hands),
                            ", ".join(active_ids) or "none") 

    # ── Callbacks ────────────────────────────────────────────

    def _on_gate_success(self, bin_id: str) -> None:
        """Called when all three gates pass — activate load receptor."""
        logger.info("LOAD RECEPTOR ACTIVATED for %s", bin_id)
        self._state.record_pick(bin_id)
        # TODO: Send signal to hardware (Modbus write / serial command)

    def _on_gate_error(self, bin_id: str, reason: str) -> None:
        """Called when a gate fails."""
        logger.warning("Gate error for %s: %s", bin_id, reason)
        self._state.add_error(bin_id, reason)

    # ── Cleanup ──────────────────────────────────────────────

    def _cleanup(self) -> None:
        logger.info("Shutting down pipeline...")
        if self._hand_tracker:
            self._hand_tracker.release()
        if self._loadcells:
            self._loadcells.close()
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()

    def _setup_logging(self) -> None:
        level = self._config.get("logging", {}).get("level", "INFO")
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AEGIS v2 Pipeline")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    args = parser.parse_args()
    Pipeline(args.config).run()
