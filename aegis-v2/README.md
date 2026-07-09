# AEGIS v2 — Bin Tracking System

Real-time computer vision system that detects bin boundaries, tracks hand position, and determines which bin a hand is reaching into for manual kitting workflows.

## How It Works

The system runs in three stages:

**1. Bin Detection (operator-driven)** — A YOLOv8 **OBB** (oriented bounding box) model locates the bins. The operator drives a two-snapshot flow at runtime: snapshot `1` calibrates the fixed workstation grid (6 top bins + 3 bottom bins) and snapshot `2` initialises the kit (which of those slots actually hold a bin). The resulting `bin_{row}_{col}` geofences are the bin map for the session. A `manual_layout` fallback can build an even grid with no model at all (headless-friendly).

**2. Hand Tracking (real-time)** — A hand recognition model (MediaPipe by default, swappable) detects hand position, landmarks, and grab gestures every frame.

**3. Integration (real-time)** — The bin assignment engine checks which bin each hand's keypoint falls into. An **occlusion gate** corrects a known failure mode: when a hand reaches into a bottom bin, the shelf lip hides the fingers and the tracker can extrapolate the fingertip up into a top bin — the gate detects this (proximal anchor at/below the bottom-bin rim) and reassigns to the bottom bin beneath. Results stream to both an OpenCV camera overlay and a web dashboard.

```
Camera Frame
    │
    ├──► Bin Detector (YOLOv8-OBB)  ──► GridSession (calibrate '1' / kit '2')
    │      [two operator snapshots]        │  → bin_{row}_{col} geofences
    │                                       ▼
    └──► Hand Tracker (MediaPipe/YOLO) ──► Bin Assignment Engine + Occlusion Gate
              [runs every frame]              │  → BinEvents
                                              ▼
                                  PipelineState (thread-safe)
                                       │              │
                                       ▼              ▼
                               OpenCV Overlay    Web Dashboard
```

## Project Structure

```
aegis-v2/
│
├── cv-models/                    # Bin boundary detection model + training data
│   ├── configs/
│   │   └── training_config.yaml        # Training hyperparameters
│   ├── data/                           # Raw / annotated / processed datasets
│   ├── models/weights/                 # Trained .pt weights
│   │   ├── project_9_yolov8_obb_1.pt   # Native OBB bin model (used by the pipeline)
│   │   └── best.pt                     # Legacy YOLOv8-seg model (retained)
│   └── requirements.txt
│
├── hand-models/                  # Hand recognition experiments (multiple backends)
│   ├── common/
│   │   ├── base_hand_tracker.py        # Abstract interface all backends implement
│   │   └── registry.py                 # Service locator — swap backends by name
│   ├── mediapipe/
│   │   └── tracker.py                  # MediaPipe Tasks Hand Landmarker (primary)
│   ├── yolo-hand/
│   │   ├── tracker.py                  # YOLOv8-pose hand tracker (experimental)
│   │   └── weights/                    # Place YOLO hand model weights here
│   ├── custom-model-template/
│   │   └── tracker.py                  # Copy this folder to add a new backend
│   ├── benchmarks/
│   │   └── run_benchmark.py            # Side-by-side latency/accuracy comparison
│   └── requirements.txt
│
├── integration/                  # Combined pipeline (wires cv-models + hand-models + UI)
│   ├── src/
│   │   ├── pipeline.py                 # Main orchestrator (Sense → Analyse → Act)
│   │   ├── detectors/
│   │   │   ├── initialize_bins_obb.py  # Native OBB bin detector (load + detect)
│   │   │   ├── grid_session.py         # Two-snapshot state → bin_{row}_{col} geofences
│   │   │   ├── grid_calibrator.py      # Calibrate 6+3 grid / match a kit snapshot to it
│   │   │   ├── grid_allocator.py       # Pure row-split + band allocator (library)
│   │   │   └── snapshot_obb.py         # CLI driver for the offline OBB handoff
│   │   ├── engine/
│   │   │   ├── bin_assignment.py       # Assigns hands to bins + occlusion gate
│   │   │   └── occlusion_hold.py       # Holds a bottom bin while the hand is occluded
│   │   ├── sensing/
│   │   │   ├── loadcell.py             # ESP32/HX711 per-bin weights over USB serial
│   │   │   └── inventory.py            # Converts weight deltas → item counts
│   │   └── ui/
│   │       ├── overlay.py              # OpenCV real-time camera overlay
│   │       ├── dashboard.py            # FastAPI web dashboard backend
│   │       ├── state.py                # Thread-safe shared state (pipeline ↔ dashboard)
│   │       └── static/                 # Dashboard frontend (HTML/CSS/JS)
│   │           ├── index.html
│   │           ├── style.css
│   │           └── app.js
│   ├── config/
│   │   ├── settings.yaml               # Master config (camera, models, UI, sensing)
│   │   └── inventory.yaml              # Item unit-weights + bin→item map (load cells)
│   ├── tests/                          # Pure-geometry unit tests (grid, occlusion)
│   └── requirements.txt
│
└── README.md                     # This file
```

## Quick Start

### 1. Set Up Environment

```bash
cd aegis-v2
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r integration/requirements.txt
```

### 2. Prepare the Bin Detection Model

Place the OBB bin model at `cv-models/models/weights/project_9_yolov8_obb_1.pt` (the path is configured in `settings.yaml` under `bin_detector.model_path`). The model is loaded once at startup and fails *soft* — if the weights or `ultralytics` are missing, the pipeline still boots and calibration simply finds no bins until they're present.

> `cv-models/` holds the model weights, the training config, and the datasets.

### 3. Set Up Hand Tracking

The MediaPipe backend works out of the box. Download the model file:

```bash
# Download hand_landmarker.task (float16, ~10MB)
curl -o hand-models/mediapipe/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### 4. Run the Pipeline

```bash
python -m integration.src.pipeline --config integration/config/settings.yaml
```

With the camera overlay focused:

- **`1`** — snapshot + calibrate the workstation grid (locks 9 slots: 6 top + 3 bottom).
- **`2`** — snapshot + initialise the kit (present slots fill in; missing slots grey out).
- **`q`** — quit.

The web dashboard launches automatically at `http://localhost:8080`.

> Using `manual_layout` in `settings.yaml` instead skips the model and locks an even grid at startup — keys `1`/`2` are not needed in that mode.

## Operator UI

AEGIS v2 has two UI layers that run simultaneously:

**OpenCV Camera Overlay** — drawn directly on the live camera feed, showing bin boundaries (polygon or box), hand skeletons (21 landmarks), a bottom status bar with each hand's current bin assignment, and an FPS counter. This is the low-latency view for debugging and development.

**Web Dashboard** — a FastAPI-served browser dashboard at `http://localhost:8080` for the operator. It shows a color-coded bin grid (grey = missing/not-in-job, white → orange → green as picks progress toward the work-order target, red = wrong bin), hand tracking data, system stats (FPS, uptime, frame count), per-bin weights when load cells are connected, and an error feed. Pick counts can be overridden via the API.

Both are enabled by default. To disable either, edit `integration/config/settings.yaml`:

```yaml
ui:
  enabled: true       # OpenCV overlay (set false for headless)

dashboard:
  enabled: true       # Web dashboard
  port: 8080
```

The dashboard reads from a thread-safe shared `PipelineState` object that the pipeline loop writes to every frame, so there's no performance coupling between the web server and the CV loop.

## Bin Assignment & Occlusion Handling

Hands are mapped to bins by the `BinAssignmentEngine` (`engine/bin_assignment.py`), called every frame. The method is config-selectable (`bin_assignment.method`):

| Method | Behavior |
|--------|----------|
| `point_in_polygon` (default) | Is the chosen hand keypoint (`index_tip`) inside a bin's box? |
| `nearest_centroid` | Assign to the closest bin center. |
| `area_overlap` | Overlap ratio between the hand bbox and bin, gated by `overlap_threshold`. |

Two occlusion mechanisms handle the rack lip hiding the hand:

- **Occlusion gate** (`bin_assignment.py`, on by default) — rewrites a single frame's assignment. When a fingertip is extrapolated under the shelf into a *top* bin but the hand's proximal anchor (wrist, else knuckle centroid) sits at/below the bottom-bin rim, it reassigns to the bottom bin beneath (or suppresses the event if no bottom bin is under the anchor).
- **Occlusion hold** (`occlusion_hold.py`) — a stateful layer that keeps a bottom bin marked active while the hand that was picking from it vanishes entirely under the lip, releasing when that hand is seen again. *Implemented and unit-tested but not yet wired into `pipeline.py`* (see Known gaps).

## Sensing (Load Cells)

Per-bin weights come from an **ESP32 + HX711** array that streams newline-delimited JSON over USB serial (`{"bins": {"bin_0_0": 123.4, ...}}` in grams). `LoadCellReader` (`sensing/loadcell.py`) reads on a background thread and caches the latest reading, so per-frame polling never blocks. `InventoryTracker` (`sensing/inventory.py`) turns weight deltas into item counts using `config/inventory.yaml` (item unit-weights + bin→item map).

Load cells are disabled by default (`sensing.loadcells.enabled: false`); while disabled or unconnected, the dashboard layout falls back to the CV-detected grid and no weights are shown.

## Plugging In Your Own Models

The repo is designed around two clean integration points: a bin (CV) model and a hand model.

### Plugging In Your Own Bin Model

The bin detector is selected by config path — no code changes needed if your model is a YOLOv8-OBB `.pt` file.

1. Drop the weights in `cv-models/models/weights/` (or anywhere — just remember the path).
2. Update `integration/config/settings.yaml`:
   ```yaml
   bin_detector:
     model_path: "./cv-models/models/weights/project_9_yolov8_obb_1.pt"
     confidence_threshold: 0.5
   ```
3. Run the pipeline and calibrate with keys `1`/`2`.

**Bin grid layout:** the system assumes the fixed rig of **6 single-width top bins + 3 double-width bottom bins**. Slot indices and the `bin_{row}_{col}` ids are assigned by `grid_calibrator`/`grid_session` from each row's left-to-right order — never from detection order. For a different layout, either use `manual_layout` (per-layer bin counts in `settings.yaml`) or adjust the slot mapping in `grid_session._slot_layout`. The ids flow through to the assignment engine, overlay, and dashboard.

### Switching Hand Models

If a backend is already registered, just edit `integration/config/settings.yaml`:

```yaml
hand_tracker:
  backend: "mediapipe"    # or "yolo-hand", or your custom backend name
```

All registered backends are listed at startup.

### Adding a New Hand Model

Hand models use a **registry pattern** — they don't just drop into a folder. You implement an interface, register the class, and flip a config value.

1. **Copy the template:**
   ```powershell
   Copy-Item -Recurse hand-models\custom-model-template hand-models\my-model
   ```
2. **Edit `hand-models/my-model/tracker.py`:**
   - Rename `CustomHandTracker` → `MyTracker`
   - Implement `load_model()` — load your weights (ONNX, PyTorch, TF, whatever)
   - Implement `detect(frame: np.ndarray) -> list[HandDetection]` — return `HandDetection` objects with at minimum `hand_id`, `handedness`, `landmarks` (must include `wrist` and `index_tip`), and `bounding_box`. Optionally set `is_grabbing` and `grab_score`.
   - Uncomment the bottom line: `TrackerRegistry.register("my-model", MyTracker)`
3. **Register the import in `integration/src/pipeline.py`** (in the auto-register block near the top):
   ```python
   try:
       import hand_models.my_model.tracker  # noqa: F401
   except ImportError:
       pass
   ```
   Note: hyphens in folder names become underscores in the Python import path (`my-model` → `my_model`).
4. **Activate it in config:**
   ```yaml
   hand_tracker:
     backend: "my-model"
   ```

The contract you must satisfy lives in `hand-models/common/base_hand_tracker.py` (the `HandDetection` / `HandLandmark` dataclasses). As long as your tracker returns those shapes, the bin assignment engine, overlay, and dashboard all work without modification.

## Benchmarking Hand Models

Compare all registered backends side-by-side:

```bash
python hand-models/benchmarks/run_benchmark.py --source 0 --frames 300
```

Output includes latency (ms), FPS, detection rate, and landmark jitter for each backend.

## Known gaps (worth flagging to peers)

- **Occlusion hold is not wired into the pipeline.** `engine/occlusion_hold.py` is implemented and unit-tested but `pipeline.py` does not yet apply it, so a bottom bin drops the instant the hand disappears under the lip.
- **Load-cell hardware is not connected.** `LoadCellReader` and `InventoryTracker` are implemented, but with `sensing.loadcells.enabled: false` the dashboard layout comes from CV only and no weights are shown.
- **Empty stub packages.** `integration/src/{logic, trackers, utils}/` exist with only a one-line `__init__.py`. They're placeholders — the actual logic lives in `engine/`, `hand-models/`, and `pipeline.py`.

## Hardware Requirements

- **Camera**: Any USB webcam (1280x720, MJPG at 30 fps)
- **GPU**: Optional — the OBB and hand models run on CPU by default (`device: "cpu"`)
- **Sensors**: ESP32 + HX711 load-cell array over USB serial (reader implemented; hardware not yet wired)
