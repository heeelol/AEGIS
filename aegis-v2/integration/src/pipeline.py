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
import os
import select
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
from integration.src.sensing import LoadCellReader, PlacementTracker
from integration.src.actuators import Buzzer
from integration.src.engine.cycle import CycleManager
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


class _TerminalKeys:
    """Non-blocking single-key reader for the controlling terminal (stdin).

    The OpenCV window renders on the HMI's display (:0), so ``cv2.waitKey`` only
    sees keys typed at the MIC itself. This lets an operator on a remote SSH
    session drive the pipeline ('1'/'2' calibration, 'q' quit, 'm' tuning view,
    and F11 fullscreen) by typing in the terminal that launched it. Returns the
    logical key name, or None when nothing is pending. Recognises the F11 escape
    sequence (CSI ``23~``). On a non-TTY stdin it degrades to a no-op.
    """

    # Multi-byte terminal escape sequences we care about → logical key.
    _SEQS = {"\x1b[23~": "F11"}
    _SINGLE = set("12qm")

    def __init__(self) -> None:
        self._fd = None
        self._old = None
        self._buf = ""
        try:
            import termios  # noqa: F401  (POSIX-only; absent on Windows)
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
        except (ImportError, ValueError, OSError):
            self._fd = None

    def start(self) -> "_TerminalKeys":
        if self._fd is not None:
            import termios
            import tty
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # read keys without waiting for Enter
        return self

    def __enter__(self) -> "_TerminalKeys":
        return self.start()

    def get(self) -> Optional[str]:
        """Return one pending logical key ('1','2','q','m','F11'), or None."""
        if self._fd is None:
            return None
        # Drain everything currently available so multi-byte sequences arrive whole.
        while select.select([sys.stdin], [], [], 0)[0]:
            chunk = os.read(self._fd, 64)
            if not chunk:
                break
            self._buf += chunk.decode("latin-1", "ignore")
        if not self._buf:
            return None
        # Complete known escape sequence?
        for seq, name in self._SEQS.items():
            if self._buf.startswith(seq):
                self._buf = self._buf[len(seq):]
                return name
        # Partial prefix of a known sequence → wait for the rest.
        if any(seq.startswith(self._buf) for seq in self._SEQS):
            return None
        # Unknown escape sequence (arrows, etc.) → drop it and resync.
        if self._buf.startswith("\x1b"):
            self._buf = ""
            return None
        # Plain single character.
        c, self._buf = self._buf[0], self._buf[1:]
        return c if c in self._SINGLE else None

    def stop(self) -> None:
        if self._fd is not None and self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._old = None

    def __exit__(self, *exc) -> None:
        self.stop()


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
        self._buzzer: Optional[Buzzer] = None     # fault buzzer on the MIC DIO
        self._prev_fault: bool = False            # edge-detect FAULT for buzzer cues
        self._buzzer_sustain: bool = True         # keep pulsing until corrected
        self._inventory = None                    # InventoryTracker (built with load cells)
        self._placement: Optional[PlacementTracker] = None  # kitting-box placement counting
        self._cycle: Optional[CycleManager] = None           # work-order cycle of sets
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

        # Jetson hardware-accelerated capture: decode MJPG on the NVDEC/nvjpeg
        # block instead of the CPU (frees the CPU and unlocks a full 30 fps).
        # Gated by config so the Windows dev laptop keeps the normal path below.
        if cam.get("gstreamer_hw") and isinstance(source, int):
            w = cam.get("width", 1280); h = cam.get("height", 720); fps = cam.get("fps", 30)
            pipeline = (
                f"v4l2src device=/dev/video{source} io-mode=2 ! "
                f"image/jpeg,width={w},height={h},framerate={fps}/1 ! "
                "nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=1 max-buffers=1 sync=false"
            )
            logger.info("Opening camera via GStreamer HW decode (/dev/video%s)", source)
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self._cap.isOpened():
                raise RuntimeError(f"GStreamer HW pipeline failed for /dev/video{source}")
            logger.info("Camera (GStreamer HW): %dx%d @ %d fps MJPG->BGR", w, h, fps)
            return

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
        """Load the bin detection model once at startup (fails soft to ``None``).

        ``bin_detector.task`` selects the detector: ``"obb"`` (default, oriented
        boxes) or ``"detect"`` (axis-aligned YOLOv8 detect, e.g. FSV3). Both
        return the same ``[{id, corners, center, area, conf}]`` shape, so the rest
        of the flow is identical.
        """
        det_cfg = self._config.get("bin_detector", {})
        task = str(det_cfg.get("task", "obb")).lower()
        if task == "detect":
            from integration.src.detectors import initialize_bins_detect as det
            self._obb = det
        else:
            from integration.src.detectors import initialize_bins_obb as obb
            self._obb = obb
        model_path = det_cfg.get("model_path")
        if model_path:
            p = Path(model_path)
            if not p.is_absolute():
                p = _ROOT / model_path        # _ROOT = aegis-v2/
            model_path = str(p)
        self._obb_model = self._obb.load_model(model_path)
        if self._obb_model is None:
            logger.warning("Bin model unavailable (task=%s) — calibration ('1') will "
                           "find no bins until weights/ultralytics are present", task)
        else:
            logger.info("Bin detection task: %s", task)

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
        """Push the CURRENT set's target pick counts to the dashboard.

        The work order is a cycle of sets (``work_order.sets``); each set is a
        ``{bin_id: quantity}`` pick list (canonical grid ids). Bins absent from
        the current set get target 0 — marked not-in-use (using=False) by
        set_work_order, so a hand entering one trips the wrong-bin cross.
        Advancing or restarting the cycle re-applies this for the new set.
        """
        self._ensure_cycle()
        cur = self._cycle.current_targets()
        bin_targets = {bin_id: int(cur.get(bin_id, 0)) for bin_id in self._geofences}
        self._state.set_work_order(bin_targets)
        self._state.update_cycle(self._cycle.snapshot())
        in_use = sum(1 for t in bin_targets.values() if t > 0)
        logger.info("Set %d/%d applied: %d bins in use, %d items",
                    self._cycle.set_number, self._cycle.total_sets,
                    in_use, sum(bin_targets.values()))

    def _init_loadcells(self) -> None:
        """Initialise the load-cell reader and merge its layout into shared state.

        The driver is a stub today (returns nothing), so the dashboard layout
        falls back to the CV-detected grid. Once hardware is wired up, the same
        path supplies real layer counts and per-bin weights.
        """
        sensing_cfg = self._config.get("sensing", {})
        lc_cfg = sensing_cfg.get("loadcells", {})
        self._loadcells = LoadCellReader(lc_cfg)

        # Fault buzzer on the MIC's DIO. Started here (before any fault can
        # occur) so it's driven silent the instant the pipeline comes up,
        # regardless of the DO line's power-on default.
        buzzer_cfg = sensing_cfg.get("buzzer", {})
        self._buzzer = Buzzer(buzzer_cfg).start()
        # True (default): after the error cue, keep pulsing until the fault is
        # corrected. False: announce the error once, then stay silent.
        self._buzzer_sustain = bool(buzzer_cfg.get("sustain_error", True))

        from integration.src.sensing import InventoryTracker
        self._inventory = InventoryTracker()        # loads config/inventory.yaml

        # Kitting-box placement tracker (3-load-receptor demo). Counts items as
        # they land in the box; the BOM source bins + their unit weights come
        # from inventory.yaml, targets from the work order, box id/tolerance from
        # the loadcells.kit_box config block.
        units = self._inventory.units()
        box_cfg = lc_cfg.get("kit_box", {}) or {}
        bom_targets = {b: self._target_for(b) for b in units}
        # Friendly bin names ("BIN 1"…"BIN 9"): one shared map so the grid tiles
        # (via PipelineState) and the fault messages (via PlacementTracker) always
        # agree. Numbered top row first, then bottom, left→right.
        self._bin_labels = self._make_bin_labels(units)
        self._state.set_labels(self._bin_labels)
        self._placement = PlacementTracker(
            units, bom_targets,
            box_cfg.get("box_id", "kit_box"),
            box_cfg.get("tolerance_g"),
            ema_alpha=box_cfg.get("ema_alpha", 0.4),
            hysteresis=box_cfg.get("hysteresis", 0.25),
            box_hysteresis=box_cfg.get("box_hysteresis"),
            box_tolerance_g=box_cfg.get("box_tolerance_g"),
            box_step_tolerance_g=box_cfg.get("box_step_tolerance_g"),
            activation_frac=box_cfg.get("activation_frac", 0.5),
            wrong_bin_frac=box_cfg.get("wrong_bin_frac", 0.5),
            count_tolerance_g=box_cfg.get("count_tolerance_g"),
            fault_settle_s=box_cfg.get("fault_settle_s", 2.5),
            activation_confirm_s=box_cfg.get("activation_confirm_s", 0.4),
            labels=self._bin_labels,
        )
        boot_weights = self._wait_for_loadcell_data()
        self._placement.tare(boot_weights)  # software zero at boot
        self._empty_box_raw = float(boot_weights.get("kit_box", 0.0))

        layout = self._loadcells.get_layout()
        self._state.update_loadcells(layout, self._loadcells.get_weights())
        self._apply_loadcell_counts()
        if self._loadcells.is_connected():
            logger.info("Load cells connected: %d layer(s); kitting box=%s, "
                        "BOM bins=%s, expected box total=%.1f g",
                        layout.num_layers, self._placement.box_id,
                        sorted(units), self._placement.expected_grams)
        else:
            logger.info("Load cells not connected (stub) — layout from CV only")

    def _wait_for_loadcell_data(
        self, timeout_s: float = 30.0, poll_interval: float = 0.2, settle_samples: int = 10
    ) -> dict:
        """Block for the reader's first real reading before the boot tare, then
        average a few more samples so the tare isn't riding on one noisy instant.

        ``LoadCellReader`` starts its read thread and returns immediately — a
        naive next call grabs whatever was cached right then (often still ``{}``,
        since a fresh ESP32 boot can take several seconds of its own tare routine
        before it ever streams a line); taring on ``{}`` sets no offsets at all,
        so every later reading is raw/un-zeroed for the rest of the run. Beyond
        that first-data wait, this also collects ``settle_samples`` readings
        ``poll_interval`` apart (matching the firmware's own ~200ms line cadence)
        and averages them per key — the same idea as the firmware's own 20-sample
        hardware tare on EN-reset, so a good boot tare doesn't depend on the
        operator remembering to press EN first. Only waits when load cells are
        configured on; falls back to whatever's available (possibly still empty)
        if the timeout elapses, same as the pre-existing not-connected fallback.
        """
        if not self._loadcells.is_enabled:
            return self._loadcells.get_weights()
        deadline = time.time() + timeout_s
        weights = self._loadcells.get_weights()
        while not weights and time.time() < deadline:
            time.sleep(poll_interval)
            weights = self._loadcells.get_weights()
        if not weights:
            logger.warning("No load-cell data after %.0fs — booting with an empty "
                           "tare (readings will be un-zeroed until the next tare "
                           "event: empty-box confirm or cycle restart)", timeout_s)
            return weights

        samples = [weights]
        for _ in range(settle_samples - 1):
            time.sleep(poll_interval)
            w = self._loadcells.get_weights()
            if w:
                samples.append(w)
        keys = set().union(*(s.keys() for s in samples))
        averaged = {k: sum(s.get(k, 0.0) for s in samples) / len(samples) for k in keys}
        logger.info("Load-cell boot tare averaged over %d sample(s)", len(samples))
        return averaged

    def _target_for(self, bin_id: str) -> int:
        """Target for ``bin_id`` under the current set of the work-order cycle."""
        self._ensure_cycle()
        return int(self._cycle.current_targets().get(bin_id, 0))

    @staticmethod
    def _make_bin_labels(bin_ids) -> dict[str, str]:
        """Map canonical ids ('bin_{row}_{col}') to sequential 'BIN N' names,
        numbered by (row, col): top row 1..6, bottom row 7..9."""
        def key(bid):
            parts = bid.split("_")
            try:
                return (int(parts[-2]), int(parts[-1]))
            except (ValueError, IndexError):
                return (0, 0)
        return {bid: f"BIN {i + 1}" for i, bid in enumerate(sorted(bin_ids, key=key))}

    # ── Cycle / set sequencing ───────────────────────────────
    def _ensure_cycle(self) -> None:
        if self._cycle is None:
            sets = (self._config.get("work_order", {}) or {}).get("sets")
            self._cycle = CycleManager(sets)

    def _apply_set_to_placement(self) -> None:
        if self._placement is not None:
            self._placement.set_targets(self._cycle.current_targets())

    def _restart_cycle(self, weights: dict) -> None:
        """Operator started a new cycle from the cycle-complete popup."""
        self._ensure_cycle()
        self._cycle.restart()
        self._apply_work_order()
        self._apply_set_to_placement()
        self._waiting_to_empty = False
        if self._placement is not None:
            self._placement.tare(weights)
            self._empty_box_raw = float(weights.get("kit_box", 0.0))
        logger.info("Cycle restarted -> set 1/%d", self._cycle.total_sets)

    def _confirm_box_emptied(self, weights: dict) -> None:
        """Operator confirmed (via the popup that appears automatically once all
        bins are green) that the kitting box is emptied: advance to the next set,
        re-target, and tare — all in one step. No automatic weight check; the
        operator verifies by hand. (This replaces the old two-step Complete →
        Confirm-empty flow now that the completion button is gone.)"""
        self._ensure_cycle()
        just_done = self._cycle.advance()
        self._apply_work_order()            # push next set's targets + cycle snapshot
        self._apply_set_to_placement()
        if self._placement is not None:
            self._placement.tare(weights)
            self._empty_box_raw = float(weights.get("kit_box", 0.0))
        self._waiting_to_empty = False
        logger.info("Box emptied + confirmed -> set %d/%d%s (tared, next set active)",
                    self._cycle.set_number, self._cycle.total_sets,
                    "  (CYCLE COMPLETE)" if just_done else "")

    def _service_cycle_requests(self) -> None:
        """Handle operator cycle-restart / empty-confirmed each loop (with or
        without load cells). Set completion no longer has its own request: the
        empty-box confirmation advances the set (see _confirm_box_emptied)."""
        # Drain any stale complete request (the Complete button was removed);
        # completion now flows through the empty-box confirmation instead.
        self._state.consume_complete_request()
        if self._state.consume_restart_request():
            w = self._loadcells.get_weights() if self._loadcells is not None else {}
            self._restart_cycle(w)
        if self._state.consume_empty_confirmed_request():
            w = self._loadcells.get_weights() if self._loadcells is not None else {}
            self._confirm_box_emptied(w)

    def _apply_loadcell_counts(self) -> None:
        """Drive BOM-bin pick counts from the kitting box (placement-driven).

        A bin's count is how many of its items the box currently holds, not how
        many left the bin — the counter only moves on placement. Connected-guard:
        a dropped link leaves counts alone. When the UI requests "complete kit",
        the pipeline waits for the operator's "box emptied" confirmation (a UI
        popup, not an automatic weight check) before re-taring all receptors and
        starting the next set.
        """
        if self._loadcells is None or self._placement is None:
            self._silence_buzzer()
            return
        if not self._loadcells.is_connected():
            # A dropped link produces no new faults to announce; never leave
            # the buzzer stuck on when data stops.
            self._silence_buzzer()
            return
        weights = self._loadcells.get_weights()

        # NOTE: the tracker keeps running through the "all green, awaiting empty"
        # phase — no short-circuit. The empty-box popup is driven purely by
        # kit.complete in the UI, so a mistake made before emptying (e.g. one
        # more item placed) still faults, hides the popup, and beeps; correcting
        # it brings the popup back. The empty confirmation (confirm-empty)
        # advances the set and re-tares in one step.

        kit = self._placement.update(weights)
        # Buzzer cues for exactly the three FSM faults (overpack /
        # pick-from-wrong-bin / return-to-wrong-bin): an error cue on entering
        # FAULT, a 'resolved' cue the moment the operator corrects it.
        self._update_fault_buzzer(kit.state)
        for bin_id, count in kit.placed.items():
            self._state.set_pick_count(bin_id, count)
        self._state.update_kit({
            "placed": kit.placed,
            "removed": kit.removed,
            "targets": kit.targets,
            "active": kit.active,
            "done": kit.done,
            "box_grams": kit.box_grams,
            "complete": kit.complete,
            "state": kit.state,
            "overpick": kit.overpick,
            "alert": kit.alert,
            "box_id": self._placement.box_id,
        })

    def _silence_buzzer(self) -> None:
        """Stop the buzzer with no 'resolved' cue — for non-fault silence paths
        (link dropped, empty-box prompt, load cells absent). Resets the fault
        edge so a still-present fault re-announces when live updates resume."""
        self._prev_fault = False
        if self._buzzer is not None:
            self._buzzer.set_alarm(False)

    def _update_fault_buzzer(self, kit_state: str) -> None:
        """Edge-driven fault cues from a live kit update.

        Entering FAULT triggers the error sound; leaving FAULT triggers the
        'resolved' sound. Two independent buzzers are supported and either may
        be absent:
          * ESP32 buzzer — firmware ``err`` / ``ring`` tunes sent over the
            load-cell serial link (the buzzer wired to the ESP32).
          * DIO buzzer — an active buzzer on the MIC's DIO (``buzzer.py``).
        Cues fire only on the transition, so a persisting fault doesn't
        re-trigger every frame.
        """
        fault = (kit_state == "FAULT")
        if fault and not self._prev_fault:
            self._buzz_esp("err")                  # ESP32 firmware error tune
            if self._buzzer is not None:
                self._buzzer.play_error()          # DIO buzzer, if wired
                if self._buzzer_sustain:
                    self._buzzer.set_alarm(True)
        elif not fault and self._prev_fault:
            self._buzz_esp("ring")                 # ESP32 firmware correction tune
            if self._buzzer is not None:
                self._buzzer.set_alarm(False)
                self._buzzer.play_ok()
        self._prev_fault = fault

    def _buzz_esp(self, cmd: str) -> None:
        """Trigger the ESP32-side buzzer tune ('err'/'ring') over the serial link."""
        if self._loadcells is not None:
            self._loadcells.send_command(cmd)

    def _load_hand_tracker(self) -> None:
        ht_cfg = self._config.get("hand_tracker", {})
        backend = ht_cfg.get("backend", "mediapipe")
        logger.info("Loading hand tracker: %s", backend)
        logger.info("Available backends: %s", TrackerRegistry.available())
        self._hand_tracker = TrackerRegistry.create(backend, ht_cfg)
        # Run detection every Nth loop and reuse the last result between (hands
        # move at human speed). 1 = every frame (default; no change on the laptop).
        self._detect_every = max(1, int(ht_cfg.get("detect_every_n", 1)))

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
                use_cuda=fg_cfg.get("cuda", False),
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
            self._save_calib_debug(frame, dets)
            return
        logger.info("Workstation grid calibrated — 9 slots locked")
        self._apply_bins(self._grid_session.to_geofences())

    def _save_calib_debug(self, frame, dets) -> None:
        """On a failed calibration, save the frame with detections drawn so the
        bin count/positions can be inspected (e.g. a double-box vs a missed bin)."""
        try:
            import numpy as np
            dbg = frame.copy()
            ys = [d["center"][1] for d in dets]
            mid = (min(ys) + max(ys)) / 2 if ys else 0
            for d in dets:
                pts = np.asarray(d["corners"], dtype=np.int32).reshape(-1, 1, 2)
                row = "T" if d["center"][1] < mid else "B"      # rough top/bottom split
                color = (0, 255, 0) if row == "T" else (0, 180, 255)
                cv2.polylines(dbg, [pts], True, color, 2)
                cx, cy = int(d["center"][0]), int(d["center"][1])
                cv2.circle(dbg, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(dbg, f"{row} {d.get('conf', 0):.2f}", (cx - 20, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            n_top = sum(1 for d in dets if d["center"][1] < mid)
            cv2.putText(dbg, f"{len(dets)} bins  top={n_top} bottom={len(dets)-n_top}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out = _ROOT / "calib_debug.jpg"
            cv2.imwrite(str(out), dbg)
            logger.warning("Saved calibration debug image: %s (inspect bin boxes)", out)
        except Exception as e:  # debug aid must never crash the pipeline
            logger.debug("Could not save calib debug image: %s", e)

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

    # Full key codes reported by cv2.waitKeyEx at the HMI window. Normal ASCII
    # keys come back as their ordinal; F11 is a special code that varies by GUI
    # backend (GTK reports 65480). Terminal F11 is handled separately via CSI 23~.
    _WIN_KEYMAP = {
        ord("q"): "q", ord("1"): "1", ord("2"): "2", ord("m"): "m",
        65480: "F11", 0xFFC8: "F11",
    }

    def _read_command(self, show_overlay: bool) -> Optional[str]:
        """Return one logical command ('1','2','q','m','F11') from the HMI
        overlay window and/or the controlling terminal, or None.

        ``cv2.waitKeyEx`` must be called when the window is shown — it both pumps
        the GUI event loop (so ``imshow`` actually paints) and returns keys typed
        at the HMI. Terminal keys take priority so a remote SSH operator wins.
        """
        cmd = None
        if show_overlay:
            code = cv2.waitKeyEx(1)
            if code != -1:
                cmd = self._WIN_KEYMAP.get(code)
        term = getattr(self, "_term_keys", None)
        tcmd = term.get() if term is not None else None
        return tcmd or cmd

    def _main_loop(self) -> None:
        logger.info("Entering Sense-Analyse-Act loop. Keys: '1' calibrate grid, "
                    "'2' init kit, F11 fullscreen, 'm' tuning view, 'q' quit — "
                    "at the HMI window OR in this terminal.")
        win = "AEGIS v2 — Bin Tracker"
        frame_count = 0
        hands: list = []  # reused between detections when detect_every_n > 1
        t0 = time.time()
        show_overlay = self._overlay is not None
        show_fg = False  # 'm' toggles the foreground "model's-eye" tuning view
        fullscreen = False  # F11 toggles the overlay window fullscreen

        if show_overlay:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        # Accept keystrokes from the controlling terminal too (SSH), not just the
        # HMI overlay window. Restored in _cleanup().
        self._term_keys = _TerminalKeys().start()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                continue
            if self._rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Check for laptop snapshot request (file-based trigger)
            if frame_count % 10 == 0:
                trigger_path = _ROOT / "tools" / "Dev_test_data" / ".snap_trigger"
                if trigger_path.exists():
                    try:
                        out_path = _ROOT / "tools" / "Dev_test_data" / "latest_snap.jpg"
                        cv2.imwrite(str(out_path), frame)
                        trigger_path.unlink()
                        logger.info("Triggered snapshot saved to %s", out_path)
                    except Exception as e:
                        logger.error("Failed to save triggered snapshot: %s", e)

            if frame_count % self._detect_every == 0:
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
                cv2.imshow(win, display)

            # One command from the HMI window and/or this terminal (SSH).
            cmd = self._read_command(show_overlay)
            if cmd == "q":
                break
            elif cmd == "1":
                self._calibrate_grid(frame)
            elif cmd == "2":
                self._init_kit(frame)
            elif cmd == "F11" and show_overlay:
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    win, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                logger.info("Overlay fullscreen %s", "ON" if fullscreen else "OFF")
            elif cmd == "m":
                show_fg = not show_fg
                logger.info("Foreground tuning view %s",
                            "ON" if show_fg else "OFF")

            frame_count += 1

            # Load cells are cheap to poll (the reader caches in a bg thread), so
            # refresh counts frequently for a responsive, smooth counter — not the
            # old every-30-frames (~1.7 s) cadence that made it feel laggy.
            # Operator set-complete / cycle-restart — serviced every loop, with
            # or without load cells.
            self._service_cycle_requests()

            if self._loadcells is not None and frame_count % 3 == 0:
                self._state.update_loadcells(
                    self._loadcells.get_layout(),
                    self._loadcells.get_weights(),
                )
                self._apply_loadcell_counts()

            if frame_count % 30 == 0:
                elapsed = time.time() - t0
                fps = frame_count / max(elapsed, 1e-6)
                self._state.update_fps(fps)

            if frame_count % 300 == 0:
                fps = frame_count / (time.time() - t0)
                logger.info("FPS: %.1f | Hands: %d | Active bins: %s",
                            fps, len(hands),
                            ", ".join(active_ids) or "none")

    # ── Cleanup ──────────────────────────────────────────────

    def _cleanup(self) -> None:
        logger.info("Shutting down pipeline...")
        term = getattr(self, "_term_keys", None)
        if term is not None:
            term.stop()  # restore the terminal to normal (cooked) mode
        if self._hand_tracker:
            self._hand_tracker.release()
        if self._buzzer:
            self._buzzer.close()  # force silent before releasing the line
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
