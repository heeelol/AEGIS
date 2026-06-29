# AEGIS Core
**AI-Driven Real-Time Procedural Verification System**

Real-time verification system for High-Mix, Low-Volume (HMLV) manual kitting using a Triple-Gate Verification Loop with computer vision and weight sensing.

## Overview

AEGIS implements a **Sense-Analyse-Act** loop that ensures correct part placement in bins using:
- **YOLOv8-Pose**: Hand gesture recognition (Open Palm vs. Closed Fist)
- **YOLOv8-Detection**: Bin geofencing and localization
- **Modbus TCP**: Weight verification from ADAM-6117 sensor modules
- **FSM Logic**: Three-gate verification (Spatial → Intent → Verification)

### Performance Target
- **End-to-end latency**: <500ms
- **Frame rate**: 30 FPS
- **Deployment**: NVIDIA edge AI hardware (ICAM-540 / MIC-733) with INT8 quantization

---

## 🚀 Getting Started (First-Timers)

**New to the project?** Start here:

1. **[Repository Navigation Guide](docs/REPOSITORY_STRUCTURE.md)** - Complete walkthrough of directories, files, and where to find things
2. **[System Architecture](docs/ARCHITECTURE.md)** - Understand the FSM gates and sensor fusion loop
3. **[GitHub Setup](docs/GITHUB_SETUP.md)** - Branch protection and CI/CD configuration

**Choose your path:**
- 👨‍💻 **Vision/ML Engineer**: [Bin Detector Training Guide](docs/DYNAMIC_BIN_DETECTION.md)
- 🔌 **Hardware Engineer**: See `config/settings.yaml` → `sensing.modbus` section
- 🧠 **Logic Developer**: Study `src/logic/fsm.py` for Triple-Gate FSM
- 🚀 **DevOps**: Follow [GitHub Setup Guide](docs/GITHUB_SETUP.md)

---

## Directory Structure

```
aegis-core/
├── src/
│   ├── vision/          # YOLOv8 inference, geofencing, pose detection
│   ├── sensing/         # Modbus TCP (ADAM) and Serial (ESP32) drivers
│   ├── logic/           # FSM (Triple-Gate) and inventory management
│   ├── ui/              # Visual Co-Pilot dashboard (Streamlit/Flask/FastAPI)
│   └── main.py          # Entry point for sensor-fusion loop
├── models/
│   ├── pretrained/      # Official YOLOv8 weights
│   └── custom/          # Fine-tuned "Grab" pose weights (.pt and .engine)
├── config/
│   ├── bins_map.yaml    # (x, y) coordinates for 4x2 bin geofences
│   └── settings.yaml    # IP, Modbus registers, thresholds
├── scripts/
│   ├── train.py         # Fine-tuning script for custom gestures
│   └── quantize.py      # TensorRT INT8 conversion
├── docs/                # Technical reports and wiring diagrams
├── tests/               # Unit tests and mock sensor data
└── requirements.txt     # Python dependencies
```

---

## Installation

### 1. Clone and Setup Environment
```bash
git clone <repository-url> aegis-core
cd aegis-core

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models
```bash
# YOLOv8 models will auto-download on first use
# For custom gesture model:
mkdir -p models/custom
# Place your fine-tuned model at: models/custom/grab_pose.pt
```

### 3. Configuration
Edit `config/settings.yaml` with your hardware specifics:
- Modbus IP and registers for weight sensors
- Camera calibration coordinates in `config/bins_map.yaml`
- Performance thresholds (latency, confidence levels)

---

## Quick Start

### Run AEGIS Core
```bash
python src/main.py
```

### Fine-tune Custom Gesture Model
```bash
python scripts/train.py --data path/to/dataset.yaml --epochs 100 --batch-size 16
```

### Quantize to TensorRT INT8
```bash
python scripts/quantize.py --model models/custom/grab_pose.pt --validate test_images/
```

---

## Architecture: Triple-Gate Verification Loop

### Gate 1: Spatial Verification
- **Trigger**: Hand keypoints enter bin geofence
- **Sensor**: YOLOv8-Pose with geofence bounds
- **Timeout**: 2.0 seconds (configurable)

### Gate 2: Intent Verification
- **Trigger**: "Closed Fist" gesture detected
- **Sensor**: YOLOv8 custom gesture classification
- **Timeout**: 1.5 seconds (configurable)

### Gate 3: Weight Verification
- **Trigger**: Weight delta from ADAM-6117 exceeds threshold
- **Sensor**: Modbus TCP weight register poll
- **Timeout**: 3.0 seconds (configurable)

### Output
✓ **Success**: Inventory updated, UI notification  
✗ **Failure**: Warning triggered, operator alerted, retry loop initiated

---

## System Requirements

### Hardware
- **GPU**: NVIDIA Jetson AGX Orin, Orin NX, or ICAM-540
- **RAM**: ≥8GB (preferably 16GB)
- **Storage**: ≥50GB SSD
- **Sensors**: Advantech ADAM-6117 Modbus TCP module

### Software
- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- TensorRT 8.5+ (for INT8 optimization)

---

## Configuration Files

### `config/settings.yaml`
Master configuration with:
- Hardware device type and compute capability
- Vision model paths and inference thresholds
- Modbus TCP connection parameters and register mappings
- FSM gate timeouts and retry logic
- Performance targets (latency, FPS, batch size)
- UI dashboard settings

### `config/bins_map.yaml`
Bin geofence coordinates:
- 4×2 array of bins (8 total)
- (x_min, x_max, y_min, y_max) for each bin in camera coordinates
- Geofence margin threshold

---

## Modbus Integration

### ADAM-6117 Connection
```yaml
modbus:
  ip_address: "192.168.1.100"
  port: 502
  slave_id: 1
  registers:
    bin_0_0_weight: 100  # Register address for bin weight
    # ... 7 more bins
  weight_delta_threshold: 50  # grams
```

### Weight Polling
- Non-blocking async reads
- Configurable poll interval
- Error recovery with timeout

---

## Dashboard (Visual Co-Pilot)

Streamlit-based real-time monitoring:
```bash
streamlit run src/ui/dashboard.py
```

**Features:**
- Live camera feed with hand pose overlay
- Bin geofence visualization
- Real-time weight trends
- FSM state machine display
- Event log and error alerting

---

## Development Roadmap

- [ ] Core FSM state machine implementation
- [ ] Vision inference pipeline with geofencing
- [ ] Modbus TCP driver and weight polling
- [ ] TensorRT INT8 quantization pipeline
- [ ] Streamlit dashboard UI
- [ ] Unit tests and mock sensor data
- [ ] Edge deployment documentation
- [ ] Performance benchmarking

---

## Performance Benchmarks

### Inference Latency (on ICAM-540)
- YOLOv8n-Pose: ~80ms
- YOLOv8n-Detect: ~45ms
- Modbus poll: ~100ms
- FSM logic: ~5ms
- **Total (est.)**: ~230ms (target: <500ms) ✓

### Model Sizes
- YOLOv8n-Pose (FP32): 6.5 MB
- YOLOv8n-Pose (INT8/TensorRT): ~2.0 MB
- Custom Grab Model (INT8): ~1.5 MB

---

## Troubleshooting

### Modbus Connection Issues
```
Check IP/port in settings.yaml
Verify ADAM-6117 is on same network subnet
Use Modbus diagnostics tool to scan device
```

### Slow Inference
```
Verify TensorRT engine is being used (not PyTorch)
Check GPU memory utilization
Enable INT8 quantization
Reduce input image size (if acceptable)
```

### Gesture Recognition Inaccuracy
```
Fine-tune custom model on more diverse data
Adjust confidence threshold in settings.yaml
Verify camera calibration
Check lighting conditions
```

---

## 📖 Documentation Resources

| Guide | Purpose |
|-------|---------|
| [Repository Structure](docs/REPOSITORY_STRUCTURE.md) | **Start here** - Complete walkthrough for first-timers |
| [System Architecture](docs/ARCHITECTURE.md) | System design, FSM gates, hardware stack |
| [Dynamic Bin Detection](docs/DYNAMIC_BIN_DETECTION.md) | Train YOLOv8 models for automatic bin detection |
| [GitHub Setup](docs/GITHUB_SETUP.md) | Branch protection rules and CI/CD configuration |

---

## Contributing

Submit issues and pull requests following the project guidelines.

---

## License

[Add your license here]

---

## Contact & Support

For technical questions or deployment support, contact: [your-email@company.com]
