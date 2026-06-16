"""
AEGIS v2 — Integration Pipeline
=================================
The main orchestrator that wires together:
  1. CV Model   → Bin boundary detection (two-snapshot OBB grid flow)
  2. Hand Model → Real-time hand tracking (any registered backend)
  3. Engine     → Bin assignment (which bin a hand is hovering over)
  4. UI         → OpenCV camera overlay + FastAPI web dashboard

The dashboard signals passively: a hand hovering a bin lights that bin as the
right bin (part of the work order) or the wrong bin (not part of it).

Lifecycle:
  INIT  — Open camera, load OBB model, load hand tracker, build overlay,
          launch dashboard
  LOOP  — Sense (detect hands) → Analyse (assign bins) → Act (UI highlight)
  STOP  — Cleanup resources

Usage:
    from integration.src.pipeline import Pipeline
    Pipeline("config/settings.yaml").run()
"""

from __future__ import annotations

import logging
import os
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

from integration.src.engine import BinAssignmentEngine, BinRegion
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
    through the bin assignment engine, with both an OpenCV camera overlay
    and a web dashboard for the operator.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG):
        self._config = _load_config(config_path)
        self._setup_logging()

        self._cap: Optional[cv2.VideoCapture] = None
        self._obb_model = None                     # ultralytics YOLO (OBB), loaded once
        self._grid_session = GridSession()         # snapshot-1 grid + snapshot-2 kit
        self._hand_tracker: Optional[BaseHandTracker] = None
        self._assignment: Optional[BinAssignmentEngine] = None
        self._overlay: Optional[OverlayUI] = None
        self._loadcells: Optional[LoadCellReader] = None
        self._geofences: dict = {}
        self._rotate_180: bool = True

        # Shared state for the web dashboard
        self._state = PipelineState()

    def run(self) -> None:
        """Full lifecycle: init → loop → cleanup.

        Bins are no longer locked at startup. The operator drives the two-snapshot
        flow live: key ``1`` calibrates the workstation grid, key ``2`` initialises
        the kit. (When ``bin_detector.manual_layout`` is set, the grid is built once
        at startup and the snapshot keys are not needed.)
        """
        try:
            self._open_camera()
            self._load_obb_model()
            self._init_loadcells()
            self._load_hand_tracker()
            self._create_engines()
            self._create_overlay()
            self._maybe_apply_manual_layout()
            self._start_dashboard()
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._cleanup()

    # ── Stage 1: Initialization ──────────────────────────────

    def _open_camera(self) -> None:
        cam = self._config.get("camera", {})
        self._rotate_180 = bool(cam.get("rotate_180", True))
        source = cam.get("source", 0)
        # YAML may provide "0" as a string; treat digit strings as camera indices.
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        logger.info("Opening camera: %s", source)

        # On Windows the default backend (MSMF) often fails to open USB webcams
        # (e.g. Logitech). DirectShow is reliable for integer camera indices; fall
        # back to the default backend if DirectShow can't open it.
        if isinstance(source, int):
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                logger.warning("DirectShow could not open camera %s; trying default backend...", source)
                self._cap.release()
                self._cap = cv2.VideoCapture(source)
        else:
            self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {source}")

        # Request MJPG. Most USB webcams only deliver 720p/1080p at 30 fps when the
        # stream is MJPG-compressed; the default (uncompressed YUY2) saturates USB
        # bandwidth and the driver silently drops to ~10 fps. Only meaningful for
        # real capture devices, not files.
        mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("width", 1280))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 720))
        self._cap.set(cv2.CAP_PROP_FPS, cam.get("fps", 30))
        # Re-assert MJPG AFTER the resolution. On Windows/DirectShow the first FOURCC
        # request is silently reverted to YUY2 once the resolution is set afterwards,
        # which is what caps 720p at ~10 fps. Setting it again here makes MJPG stick
        # (verified: re-assert -> 1280x720 @ 30 fps MJPG; without it -> YUY2 @ 10 fps).
        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        # Keep only the newest frame so a slow loop reads fresh frames instead of
        # draining a backlog of stale buffered ones (the source of growing lag).
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # The requests above are only hints — log what the driver actually
        # negotiated. This is the evidence for whether the camera (vs frame
        # processing) is the FPS bottleneck.
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()
        logger.info("Camera negotiated: %dx%d @ %.1f fps, FOURCC=%r",
                    actual_w, actual_h, actual_fps, fourcc)

    def _load_obb_model(self) -> None:
        """Load the native OBB bin model once. Skipped when manual_layout is set."""
        det_cfg = self._config.get("bin_detector", {})
        if det_cfg.get("manual_layout"):
            logger.info("Manual layout configured — OBB model not loaded")
            return

        model_path = det_cfg.get("model_path") or ""
        candidate = None
        if model_path:
            p = Path(model_path)
            if not p.is_absolute():
                p = _ROOT / model_path          # config paths are relative to aegis-v2/
            if p.exists():
                candidate = str(p)
        # load_model falls back to the bundled weights when candidate is None.
        self._obb_model = load_model(candidate)
        if self._obb_model is None:
            logger.warning("OBB model unavailable — snapshot keys 1/2 will not detect bins.")

    # ── Snapshot handlers (keys 1 and 2) ─────────────────────

    def _on_calibrate(self, frame) -> None:
        """Key 1 — workstation grid calibration (needs all 9 bins in view)."""
        if self._obb_model is None:
            logger.warning("No OBB model loaded — cannot calibrate.")
            return
        dets = detect_bins(frame, self._obb_model, expected_count=9)
        try:
            self._grid_session.calibrate(dets)
        except ValueError as e:
            logger.warning("Calibration failed (need 6 top + 3 bottom): %s", e)
            return
        self._apply_bins()
        logger.info("Workstation grid locked (key 2 to initialise the kit).")

    def _on_kit_init(self, frame) -> None:
        """Key 2 — kit initialisation: which calibrated slots hold a bin."""
        if not self._grid_session.calibrated:
            logger.warning("Press 1 to calibrate the workstation grid before kit init (2).")
            return
        dets = detect_bins(frame, self._obb_model)
        self._grid_session.init_kit(dets)
        self._apply_bins()
        self._apply_work_order()
        logger.info("Kit initialised — present slots %s, missing %s",
                    self._grid_session.present_slots, self._grid_session.missing_slots)

    def _apply_bins(self) -> None:
        """Push the session's current bins to state, engine, and overlay.

        The dashboard gets all 9 slots (missing ones flagged ``detected=False`` →
        grey); the assignment engine gets present bins only, so a hand can't be
        assigned to an empty slot.
        """
        self._geofences = self._grid_session.to_geofences()
        self._state.update_bins(self._geofences)
        if self._assignment is not None:
            self._assignment.set_bin_map_from_geofences(
                self._grid_session.to_geofences(present_only=True))
        if self._hold is not None:
            self._hold.set_eligible_bins(self._eligible_hold_bins())
        self._rebuild_overlay()

    def _maybe_apply_manual_layout(self) -> None:
        """Build and apply an even grid from per-layer counts (no camera/model)."""
        manual_layout = self._config.get("bin_detector", {}).get("manual_layout")
        if not manual_layout:
            return
        cam = self._config.get("camera", {})
        w, h = cam.get("width", 1280), cam.get("height", 720)
        self._geofences = self._build_manual_geofences(manual_layout, w, h)
        self._state.update_bins(self._geofences)
        if self._assignment is not None:
            self._assignment.set_bin_map_from_geofences(self._geofences)
        if self._hold is not None:
            self._hold.set_eligible_bins(self._eligible_hold_bins())
        self._rebuild_overlay()
        self._apply_work_order()
        logger.info("Manual bin layout applied: %d layer(s), %d bins total",
                    len(manual_layout), len(self._geofences))

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
        # Bin assignment — maps each hand to the bin it is hovering over.
        assign_cfg = self._config.get("bin_assignment", {})
        self._assignment = BinAssignmentEngine(assign_cfg)
        self._assignment.set_bin_map_from_geofences(self._geofences)

    def _create_overlay(self) -> None:
        """Build the OpenCV overlay renderer from the current geofences.

        Defaults to the bottom layer (largest row index present). When
        ``occlusion_hold.eligible_rows`` is set, those explicit row indices are
        used instead. Empty until a bin map exists.
        """
        rows = [PipelineState._parse_bin_id(b)[0] for b in self._geofences]
        if not rows:
            return set()
        explicit = self._hold.eligible_rows if self._hold else None
        wanted = set(explicit) if explicit else {max(rows)}
        return {b for b in self._geofences
                if PipelineState._parse_bin_id(b)[0] in wanted}

    def _create_overlay(self) -> None:
        """Build the OpenCV overlay renderer (starts empty; rebuilt on each snapshot)."""
        self._rebuild_overlay()

    def _rebuild_overlay(self) -> None:
        """(Re)build the overlay from the current geofences."""
        ui_cfg = self._config.get("ui", {})
        if not ui_cfg.get("enabled", True):
            self._overlay = None
            return

        bins = [
            BinRegion(
                bin_id=bid, label=bid,
                x_min=c["x_min"], x_max=c["x_max"],
                y_min=c["y_min"], y_max=c["y_max"],
                confidence=c.get("confidence", 0.0),
                polygon=c.get("polygon"),
            )
            for bid, c in geofences.items()
        ]
        self._overlay = OverlayUI(ui_cfg, bins)
        logger.info("OpenCV overlay built with %d bins", len(bins))

    def _start_dashboard(self) -> None:
        """Launch the FastAPI web dashboard in a background thread."""
        dash_cfg = self._config.get("dashboard", {})
        if not dash_cfg.get("enabled", True):
            logger.info("Web dashboard disabled in config")
            return

        port = dash_cfg.get("port", 8080)
        open_browser = dash_cfg.get("open_browser", True)
        try:
            from integration.src.ui.dashboard import start_dashboard
            start_dashboard(self._state, port=port)
            url = f"http://localhost:{port}"
            logger.info("Web dashboard available at %s", url)
            if dash_cfg.get("open_browser", True):
                self._open_browser(url)
        except ImportError as e:
            logger.warning("Could not start dashboard (missing deps): %s", e)
        except Exception as e:
            logger.warning("Dashboard failed to start: %s", e)

    @staticmethod
    def _open_browser(url: str) -> None:
        """Open the dashboard in the default browser, briefly delayed so the
        server is accepting connections by the time the tab loads."""
        import threading
        import webbrowser

        def _launch() -> None:
            time.sleep(1.0)
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.warning("Could not open browser automatically: %s", e)

        threading.Thread(target=_launch, daemon=True).start()

    # ── Stage 2: Sense → Analyse → Act loop ──────────────────

    def _main_loop(self) -> None:
        logger.info("Entering Sense-Analyse-Act loop "
                    "(1 = calibrate grid, 2 = init kit, q = quit)...")
        frame_count = 0
        t0 = time.time()
        show_overlay = self._overlay is not None

        # Per-stage time accumulators (seconds), reset each periodic log window.
        # These pinpoint the FPS bottleneck: camera read vs hand detection vs render.
        t_read = t_detect = t_render = 0.0

        if show_overlay:
            cv2.namedWindow("AEGIS v2 — Bin Tracker", cv2.WINDOW_NORMAL)

        while True:
            _ts = time.perf_counter()
            ret, frame = self._cap.read()
            t_read += time.perf_counter() - _ts
            if not ret:
                continue
            if self._rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            _ts = time.perf_counter()
            hands = self._hand_tracker.detect(frame)
            t_detect += time.perf_counter() - _ts
            events = self._assignment.assign(hands, frame.shape)
            events = self._hold.apply(events, hands)

            self._state.update_hands(hands, events)
            active_ids = {ev.bin_id for ev in events if ev.bin_id is not None}
            self._state.update_bins(self._geofences, active_ids)

            if show_overlay:
                display = self._overlay.render(frame, hands, events)
                cv2.imshow("AEGIS v2 — Bin Tracker", display)
                key = cv2.waitKey(1) & 0xFF
                t_render += time.perf_counter() - _ts
                if key == ord("q"):
                    break
                elif key == ord("1"):
                    self._on_calibrate(frame)
                elif key == ord("2"):
                    self._on_kit_init(frame)

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
                logger.info("FPS: %.1f | Hands: %d | Active bins: %s",
                            fps, len(hands),
                            ", ".join(active_ids) or "none")

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
