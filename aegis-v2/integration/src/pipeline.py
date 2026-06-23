"""
AEGIS v2 — Integration Pipeline
=================================
The main orchestrator that wires together:
  1. CV Model   → Bin boundary detection (two-snapshot OBB grid flow)
  2. Hand Model → Real-time hand tracking (any registered backend)
  3. Engine     → Bin assignment (which bin a hand is hovering over)
  4. UI         → OpenCV camera overlay + FastAPI web dashboard

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

from integration.src.engine import BinAssignmentEngine, BinRegion, OcclusionHold
from integration.src.detectors.foreground import ForegroundModel
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


def derive_pick_counts(weights: dict, tracker, connected: bool) -> dict:
    """Pick counts to push to state from load-cell weights.

    Returns ``{bin_id: count}`` for inventory-mapped bins when the cell is
    connected, else ``{}`` (the connected-guard: a dropped link must not
    overwrite counts). ``tracker.items_taken`` already clamps at >= 0 and
    only includes bins present in inventory.yaml.
    """
    if not connected:
        return {}
    return tracker.items_taken(weights)


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
        self._obb = None                          # initialize_bins_obb module (lazy)
        self._obb_model = None                    # loaded YOLO-OBB model (or None)
        self._grid_session = None                 # GridSession for the two-snapshot flow
        self._manual: bool = False                # manual-layout fallback active?
        self._hand_tracker: Optional[BaseHandTracker] = None
        self._assignment: Optional[BinAssignmentEngine] = None
        self._foreground: Optional[ForegroundModel] = None  # occlusion-gate oracle
        self._fg_present_ratio: float = 0.10                 # for the tuning overlay
        self._hold: Optional[OcclusionHold] = None           # occlusion-hold layer
        self._occlusion_ratio: float = 0.05                  # fingertip floor for hold
        self._occupancy_ratio: float = 0.05                  # bottom-bin "forearm present" floor
        self._overlay: Optional[OverlayUI] = None
        self._loadcells: Optional[LoadCellReader] = None
        self._inventory = None                    # InventoryTracker (built with load cells)
        self._geofences: dict = {}
        self._rotate_180: bool = bool(
            self._config.get("camera", {}).get("rotate_180", False))

        # Shared state for the web dashboard
        self._state = PipelineState()

    def run(self) -> None:
        """Full lifecycle: init → loop → cleanup."""
        try:
            self._open_camera()
            self._init_bins()
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

    def _grab_warm_frame(self, warmup: int = 30):
        """Read and discard frames so exposure/focus settle, then return the last frame.

        External USB webcams (e.g. Logitech via DirectShow) deliver dark/blurry
        frames for the first ~1s. Grabbing the init snapshot cold makes the bin
        detector see a black frame and find nothing. Returns None if no frame reads.
        """
        frame = None
        for _ in range(max(1, warmup)):
            ret, f = self._cap.read()
            if ret and f is not None:
                frame = f
        return frame

    def _init_bins(self) -> None:
        """Set up bin detection for the session.

        Two paths:
          * ``manual_layout`` set → build an even grid from per-layer counts at
            startup (no model needed beyond a frame for its size) and apply the
            work order immediately. The headless-friendly fallback.
          * otherwise → load the native OBB model and arm the two-snapshot
            operator flow. No bins are locked at INIT; the operator presses ``1``
            to calibrate the 6+3 workstation grid and ``2`` to initialise the kit.
        """
        det_cfg = self._config.get("bin_detector", {})
        manual_layout = det_cfg.get("manual_layout")

        if manual_layout:
            self._manual = True
            logger.info("Warming up camera for manual-layout frame size...")
            frame = self._grab_warm_frame(
                self._config.get("camera", {}).get("warmup_frames", 30))
            if frame is None:
                raise RuntimeError("Failed to capture initialization snapshot")
            if self._rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            h, w = frame.shape[:2]
            self._geofences = self._build_manual_geofences(manual_layout, w, h)
            logger.info("Manual bin layout: %d layer(s), %d bins total",
                        len(manual_layout), len(self._geofences))
            self._state.update_bins(self._geofences)
            self._apply_work_order()
            return

        # OBB two-snapshot flow — bins are locked interactively, not at INIT.
        self._load_obb_model()
        from integration.src.detectors.grid_session import GridSession
        self._grid_session = GridSession()
        self._geofences = {}
        logger.info("OBB two-snapshot flow ready — press '1' to calibrate the "
                    "workstation grid, '2' to initialise the kit")

    def _load_obb_model(self) -> None:
        """Load the native OBB bin model once at startup (fails soft to ``None``)."""
        from integration.src.detectors import initialize_bins_obb as obb
        self._obb = obb
        model_path = self._config.get("bin_detector", {}).get("model_path")
        if model_path:
            p = Path(model_path)
            if not p.is_absolute():
                p = _ROOT / model_path        # _ROOT = aegis-v2/
            model_path = str(p)
        self._obb_model = obb.load_model(model_path)
        if self._obb_model is None:
            logger.warning("OBB model unavailable — calibration ('1') will find no "
                           "bins until weights/ultralytics are present")

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
        from integration.src.sensing import InventoryTracker
        self._inventory = InventoryTracker()        # loads config/inventory.yaml
        layout = self._loadcells.get_layout()
        self._state.update_loadcells(layout, self._loadcells.get_weights())
        self._apply_loadcell_counts()
        if self._loadcells.is_connected():
            logger.info("Load cells connected: %d layer(s)", layout.num_layers)
        else:
            logger.info("Load cells not connected (stub) — layout from CV only")

    def _apply_loadcell_counts(self) -> None:
        """Drive mapped bins' pick counts from the latest load-cell weights.

        Authoritative for inventory-mapped bins while the cell is connected;
        the connected-guard in ``derive_pick_counts`` leaves counts alone when
        the link drops. Bins absent from inventory.yaml are never touched.
        """
        if self._loadcells is None or self._inventory is None:
            return
        counts = derive_pick_counts(
            self._loadcells.get_weights(),
            self._inventory,
            self._loadcells.is_connected(),
        )
        for bin_id, count in counts.items():
            self._state.set_pick_count(bin_id, count)

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

        # Foreground-evidence oracle for the occlusion gate. Built only when the
        # gate's foreground mode is configured; otherwise the gate uses its
        # landmark heuristic. See foreground.py / the 2026-06-22 design doc.
        gate_cfg = (assign_cfg.get("occlusion_gate", {}) or {})
        fg_cfg = gate_cfg.get("foreground")
        if gate_cfg.get("enabled", True) and fg_cfg is not None:
            self._fg_present_ratio = fg_cfg.get("present_ratio", 0.10)
            self._foreground = ForegroundModel(
                patch_size=fg_cfg.get("patch_size", 41),
                warmup_frames=fg_cfg.get("warmup_frames", 30),
                history=fg_cfg.get("history", 500),
                var_threshold=fg_cfg.get("var_threshold", 16.0),
            )
            logger.info("Occlusion gate: foreground-evidence mode enabled "
                        "(patch=%d, warmup=%d)",
                        self._foreground._patch, self._foreground._warmup)

        # Occlusion hold — keeps the bottom bin lit while a hand is hidden under
        # the shelf (visual continuity). Eligible bins are the detected bottom row.
        hold_cfg = (assign_cfg.get("occlusion_hold", {}) or {})
        if hold_cfg.get("enabled", True):
            self._occlusion_ratio = hold_cfg.get("occlusion_ratio", 0.05)
            self._occupancy_ratio = hold_cfg.get("occupancy_ratio", 0.05)
            self._hold = OcclusionHold(hold_cfg)
            self._hold.set_eligible_bins(self._assignment.bottom_bin_ids())
            logger.info("Occlusion hold enabled (occlusion_ratio=%.2f, eligible=%s)",
                        self._occlusion_ratio,
                        sorted(self._assignment.bottom_bin_ids()) or "none")

    def _create_overlay(self) -> None:
        """Build the OpenCV overlay renderer from the current geofences.

        In the OBB flow the grid is empty until the operator calibrates, so this
        may start with zero bins; ``_rebuild_overlay`` refreshes it after each
        snapshot.
        """
        ui_cfg = self._config.get("ui", {})
        if not ui_cfg.get("enabled", True):
            return
        self._rebuild_overlay(self._geofences)
        logger.info("OpenCV overlay created with %d bins", len(self._geofences))

    def _rebuild_overlay(self, geofences: dict) -> None:
        """(Re)build the overlay from a geofence dict. No-op when the UI is off."""
        ui_cfg = self._config.get("ui", {})
        if not ui_cfg.get("enabled", True):
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
            start_dashboard(self._state, port=port, open_browser=open_browser)
            logger.info("Web dashboard available at http://localhost:%d", port)
        except ImportError as e:
            logger.warning("Could not start dashboard (missing deps): %s", e)
        except Exception as e:
            logger.warning("Dashboard failed to start: %s", e)

    # ── Two-snapshot operator flow (OBB) ─────────────────────

    def _calibrate_grid(self, frame) -> None:
        """Key '1': snapshot → OBB detect (9 bins) → lock the 6+3 grid."""
        if self._grid_session is None:
            logger.info("Calibration is only available in the OBB flow "
                        "(manual_layout is set)")
            return
        dets = self._obb.detect_bins(frame, self._obb_model, expected_count=9)
        try:
            self._grid_session.calibrate(dets)
        except ValueError as e:
            logger.warning("Grid calibration failed: %s", e)
            return
        logger.info("Workstation grid calibrated — 9 slots locked")
        self._apply_bins(self._grid_session.to_geofences())

    def _init_kit(self, frame) -> None:
        """Key '2': snapshot → match to grid → present / missing bins."""
        if self._grid_session is None:
            logger.info("Kit init is only available in the OBB flow "
                        "(manual_layout is set)")
            return
        if not self._grid_session.calibrated:
            logger.warning("Press '1' to calibrate the workstation grid before "
                           "initialising the kit")
            return
        dets = self._obb.detect_bins(frame, self._obb_model)
        self._grid_session.init_kit(dets)
        logger.info("Kit initialised — present slots %s, missing slots %s",
                    self._grid_session.present_slots or "none",
                    self._grid_session.missing_slots or "none")
        self._apply_bins(self._grid_session.to_geofences(), apply_work_order=True)

    def _apply_bins(self, geofences: dict, apply_work_order: bool = False) -> None:
        """Push a new geofence set everywhere: state, assignment engine, overlay.

        The assignment engine only gets **present** bins (``detected=True``) so a
        hand can't be assigned to an empty slot; the dashboard and overlay get all
        slots (missing ones render greyed via ``detected=False``).
        """
        self._geofences = geofences
        self._state.update_bins(geofences)
        present = {bid: c for bid, c in geofences.items() if c.get("detected", True)}
        if self._assignment is not None:
            self._assignment.set_bin_map_from_geofences(present)
            if self._hold is not None:
                self._hold.set_eligible_bins(self._assignment.bottom_bin_ids())
        self._rebuild_overlay(geofences)
        if apply_work_order:
            self._apply_work_order()

    # ── Stage 2: Sense → Analyse → Act loop ──────────────────

    def _main_loop(self) -> None:
        logger.info("Entering Sense-Analyse-Act loop (press 'q' to quit, "
                    "'m' to toggle the foreground tuning view)...")
        frame_count = 0
        t0 = time.time()
        show_overlay = self._overlay is not None
        show_fg = False  # 'm' toggles the foreground "model's-eye" tuning view

        if show_overlay:
            cv2.namedWindow("AEGIS v2 — Bin Tracker", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self._cap.read()
            if not ret:
                continue
            if self._rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            hands = self._hand_tracker.detect(frame)

            # Feed the foreground model every frame; once warmed, hand it to the
            # occlusion gate as a presence oracle over the current frame's mask.
            presence_fn = None
            if self._foreground is not None:
                mask = self._foreground.update(frame)
                if self._foreground.ready:
                    presence_fn = lambda px, py: self._foreground.patch_ratio(mask, px, py)

            events = self._assignment.assign(hands, frame.shape, presence_fn)

            # Occlusion hold: keep a bottom bin lit while its hand is hidden under
            # the shelf. A handedness is "occluded" when its fingertip has almost
            # no foreground (below occlusion_ratio) — a genuine occlusion, not a
            # weakly-visible hand. Without a presence oracle the set is empty and
            # the hold degrades to absence-only latching.
            if self._hold is not None:
                occluded_ids = set()
                occupied_bins = None
                if presence_fn is not None:  # foreground model is warmed up
                    for ev in events:
                        if ev.hand_point is None:
                            continue
                        if presence_fn(ev.hand_point[0], ev.hand_point[1]) < self._occlusion_ratio:
                            occluded_ids.add(ev.hand_id)
                    # Which bottom bins still have a forearm in them (real foreground)?
                    occupied_bins = set()
                    for bid in self._assignment.bottom_bin_ids():
                        c = self._geofences.get(bid)
                        if c is None:
                            continue
                        if self._foreground.region_ratio(
                            mask, c["x_min"], c["y_min"], c["x_max"], c["y_max"]
                        ) >= self._occupancy_ratio:
                            occupied_bins.add(bid)
                    # If a visible fingertip is above a bottom bin (hand in a top
                    # bin OR emerged above the rack), the forearm is only transiting
                    # that bin — don't let it be held. Occluded tips are excluded so
                    # genuine under-shelf reaches still hold.
                    occupied_bins -= self._assignment.bottom_bins_with_hand_above(
                        events, occluded_ids)
                events = self._hold.apply(events, hands, occluded_ids, occupied_bins)

            self._state.update_hands(hands, events)
            active_ids = {ev.bin_id for ev in events if ev.bin_id is not None}
            self._state.update_bins(self._geofences, active_ids)

            if show_overlay:
                if show_fg and self._foreground is not None:
                    samples = [
                        (ev.hand_point[0], ev.hand_point[1],
                         self._foreground.patch_ratio(
                             mask, ev.hand_point[0], ev.hand_point[1]))
                        for ev in events
                        if ev.hand_point is not None and ev.method != "occlusion_hold"
                    ]
                    display = self._overlay.render_foreground_debug(
                        mask, samples, self._fg_present_ratio,
                        self._foreground.patch_size, self._foreground.ready)
                else:
                    display = self._overlay.render(frame, hands, events)
                cv2.imshow("AEGIS v2 — Bin Tracker", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("1"):
                    self._calibrate_grid(frame)
                elif key == ord("2"):
                    self._init_kit(frame)
                elif key == ord("m"):
                    show_fg = not show_fg
                    logger.info("Foreground tuning view %s",
                                "ON" if show_fg else "OFF")

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
                    self._apply_loadcell_counts()

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
