"""
Pipeline Orchestrator
=====================
Ties together the AEGIS Sense-Analyse-Act loop:

  1. SENSE   — Capture frames, run hand tracker, poll Modbus weight sensors
  2. ANALYSE — Assign hands to bins, run Triple-Gate FSM verification
  3. ACT     — Update inventory, trigger warnings, publish UI events

This is the single entry point that wires every module together.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2

from src.detectors import BinDetector, DynamicGeofenceManager
from src.trackers import TrackerRegistry
from src.trackers.base_hand_tracker import BaseHandTracker
from src.engine import BinAssignmentEngine, TripleGateFSM, SensorReading
from src.ui.overlay import OverlayUI
from src.engine.bin_assignment import BinRegion
from src.utils import load_config, setup_logger

# Ensure built-in tracker backends are registered
import src.trackers.mediapipe_tracker  # noqa: F401

logger = logging.getLogger("aegis.pipeline")


class Pipeline:
    """
    Main AEGIS orchestrator — call run() to start the full lifecycle.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._config = load_config(config_path)
        setup_logger(config=self._config)

        # Sub-components (created during init phase)
        self._cap: Optional[cv2.VideoCapture] = None
        self._bin_detector: Optional[BinDetector] = None
        self._hand_tracker: Optional[BaseHandTracker] = None
        self._assignment_engine: Optional[BinAssignmentEngine] = None
        self._fsm: Optional[TripleGateFSM] = None
        self._overlay: Optional[OverlayUI] = None

    # ── Lifecycle ────────────────────────────────────────────

    def run(self) -> None:
        """Full lifecycle: initialize → loop → cleanup."""
        try:
            self._open_camera()
            self._initialize_bins()
            self._create_hand_tracker()
            self._create_engines()
            self._create_overlay()
            self._tracking_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._cleanup()

    # ── Stage 1: Initialization ──────────────────────────────

    def _open_camera(self) -> None:
        cam_cfg = self._config.get("camera", {})
        source = cam_cfg.get("source", 0)
        logger.info("Opening camera source: %s", source)

        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))
        self._cap.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")

    def _initialize_bins(self) -> None:
        """Snapshot → detect bin boundaries → lock coordinates."""
        vision_cfg = self._config.get("vision", {})
        model_path = vision_cfg.get("model_detect", "yolov8n.pt")
        conf = vision_cfg.get("confidence_threshold", 0.6)

        self._bin_detector = BinDetector(model_path=model_path, conf_threshold=conf)

        logger.info("Taking initialization snapshot for bin detection...")
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture initialization snapshot")

        self._geofences = self._bin_detector.detect_bins(frame)
        logger.info("Bin map locked: %d regions detected", len(self._geofences))

    def _create_hand_tracker(self) -> None:
        ht_cfg = self._config.get("hand_tracker", {})
        backend_name = ht_cfg.get("backend", "mediapipe")
        logger.info("Creating hand tracker: %s", backend_name)
        self._hand_tracker = TrackerRegistry.create(backend_name, ht_cfg)

    def _create_engines(self) -> None:
        # Bin assignment engine
        assign_cfg = self._config.get("bin_assignment", {})
        self._assignment_engine = BinAssignmentEngine(assign_cfg)
        self._assignment_engine.set_bin_map_from_geofences(self._geofences)

        # Triple-Gate FSM
        self._fsm = TripleGateFSM(self._config)

    def _create_overlay(self) -> None:
        ui_cfg = self._config.get("overlay", {})
        if ui_cfg.get("enabled", False):
            bins = [
                BinRegion(
                    bin_id=bid, label=bid,
                    x_min=c["x_min"], x_max=c["x_max"],
                    y_min=c["y_min"], y_max=c["y_max"],
                    confidence=c.get("confidence", 0.0),
                )
                for bid, c in self._geofences.items()
            ]
            self._overlay = OverlayUI(ui_cfg, bins)

    # ── Stage 2: Real-time Sense-Analyse-Act loop ────────────

    def _tracking_loop(self) -> None:
        logger.info("Entering Sense-Analyse-Act loop (press 'q' to quit)...")
        frame_count = 0
        t0 = time.time()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame — retrying")
                continue

            # SENSE: Detect hands
            hands = self._hand_tracker.detect(frame)

            # ANALYSE: Assign hands to bins
            events = self._assignment_engine.assign(hands)

            # ANALYSE: Feed into FSM for each active bin event
            for ev in events:
                if ev.bin_id is not None:
                    reading = SensorReading(
                        timestamp=time.time(),
                        hand_in_geofence=True,
                        closed_fist_detected=False,   # TODO: gesture detection
                        weight_delta=0.0,              # TODO: Modbus integration
                        bin_id=ev.bin_id,
                    )
                    self._fsm.update(reading)

            # ACT: Render overlay (if enabled)
            if self._overlay:
                display = self._overlay.render(frame, hands, events)
                cv2.imshow("AEGIS — Bin Tracker", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed")
                    break

            frame_count += 1
            if frame_count % 300 == 0:
                fps = frame_count / (time.time() - t0)
                logger.info("FPS: %.1f", fps)

    # ── Cleanup ──────────────────────────────────────────────

    def _cleanup(self) -> None:
        logger.info("Shutting down AEGIS pipeline...")
        if self._hand_tracker:
            self._hand_tracker.release()
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        logger.info("Done.")
