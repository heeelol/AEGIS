# AEGIS Repository Navigation Guide

**For first-time contributors and team members**

---

## **Quick Overview**

This is the **AEGIS** project - an AI-driven real-time procedural verification system for High-Mix, Low-Volume (HMLV) manual kitting using computer vision and weight sensors.

**Key concept:** Sense → Analyze → Act
- **Sense**: Detect hand gestures and bin locations with YOLOv8
- **Analyze**: Run Triple-Gate verification FSM
- **Act**: Update inventory or alert operators

---

## **Directory Structure Map**

```
aegis-core/                          # Root directory
│
├── src/                             # ⭐ MAIN APPLICATION CODE
│   ├── main.py                      # Entry point - Sense-Analyse-Act loop
│   ├── vision/                      # Computer vision pipeline
│   │   ├── bin_detector.py          # ← Dynamic bin detection (YOLOv8)
│   │   ├── geofencing.py            # Geofence boundary checking
│   │   └── pose_estimator.py        # (TODO) Hand pose keypoints
│   ├── sensing/                     # Hardware sensor drivers
│   │   └── modbus_client.py         # Weight polling from ADAM-6117
│   ├── logic/                       # FSM & business logic
│   │   ├── fsm.py                   # Triple-Gate state machine
│   │   └── inventory.py             # (TODO) Inventory database
│   └── ui/                          # Dashboard & visualization
│       └── dashboard.py             # (TODO) Streamlit real-time UI
│
├── models/                          # 📦 ML MODEL STORAGE
│   ├── pretrained/                  # Official YOLOv8 models (auto-download)
│   │   ├── yolov8n.pt               # Nano model
│   │   ├── yolov8n-pose.pt          # Pose detection
│   │   └── yolov8_bins_trained.pt   # Your trained bin detector (after training)
│   └── custom/                      # Custom fine-tuned models
│       ├── grab_pose.pt             # Custom gesture model
│       └── grab_pose.engine         # TensorRT INT8 optimized
│
├── config/                          # ⚙️ CONFIGURATION FILES
│   ├── settings.yaml                # Master config (hardware, Modbus, FSM)
│   └── bins_map.yaml                # (DEPRECATED) Static geofence fallback
│
├── scripts/                         # 🛠️ UTILITIES & TRAINING
│   ├── train_bin_detector.py        # Train YOLOv8 on bin images
│   │   └── Commands: prepare / train / eval / test
│   ├── train.py                     # Fine-tune gesture model
│   ├── quantize.py                  # Convert to TensorRT INT8
│   └── example_real_time_pipeline.py # Full pipeline example
│
├── tests/                           # ✅ UNIT TESTS
│   ├── __init__.py
│   └── test_core.py                 # FSM, geofencing tests
│
├── docs/                            # 📚 DOCUMENTATION
│   ├── ARCHITECTURE.md              # System design & FSM flow
│   ├── DYNAMIC_BIN_DETECTION.md     # Complete bin detector workflow
│   ├── GITHUB_SETUP.md              # Branch protection & CI/CD setup
│   └── REPOSITORY_STRUCTURE.md      # This file
│
├── .github/                         # GitHub Actions CI/CD
│   └── workflows/
│       ├── tests.yml                # Pytest workflow
│       ├── lint.yml                 # Code quality checks
│       └── build.yml                # Project validation
│
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Build configuration
├── README.md                        # Project overview
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
└── LICENSE                          # Project license
```

---

## **Key Files Explained**

### **For Understanding the Project**

| File | Purpose | Read if... |
|------|---------|-----------|
| `README.md` | Project overview, installation, quick start | You're new to the project |
| `docs/ARCHITECTURE.md` | System design, FSM gates, hardware stack | You want to understand how it works |
| `docs/DYNAMIC_BIN_DETECTION.md` | Bin detector training workflow | You need to train models |
| `docs/GITHUB_SETUP.md` | GitHub CI/CD setup | You're setting up the repo |
| `config/settings.yaml` | All system configuration | You need to adjust hardware/thresholds |

### **For Development**

| File | Purpose | Edit if... |
|------|---------|-----------|
| `src/main.py` | Main entry point & event loop | You're implementing the core loop |
| `src/vision/bin_detector.py` | Bin detection logic | You're working on vision features |
| `src/logic/fsm.py` | State machine logic | You're modifying FSM gates/timeouts |
| `src/sensing/modbus_client.py` | Weight sensor driver | You're debugging sensor integration |
| `scripts/train_bin_detector.py` | Model training pipeline | You're training custom models |

### **For Testing & Deployment**

| File | Purpose | Check if... |
|------|---------|-----------|
| `tests/test_core.py` | Unit tests | You need to verify logic changes |
| `.github/workflows/tests.yml` | Automated testing | You want to verify PR quality |
| `pyproject.toml` | Build & dependencies | You're adding new packages |
| `.env.example` | Environment setup | You need to configure local env |

---

## **Getting Started: Common Paths**

### **Path 1: I'm a New Developer**
1. Start: [README.md](../README.md) - Project overview
2. Read: [docs/ARCHITECTURE.md](ARCHITECTURE.md) - Understand system design
3. Explore: `src/` directories - See how code is organized
4. Try: `python src/main.py` - Run the application (placeholder mode)
5. Study: `src/logic/fsm.py` - Understand Triple-Gate FSM

### **Path 2: I Need to Train Bin Detector**
1. Read: [docs/DYNAMIC_BIN_DETECTION.md](DYNAMIC_BIN_DETECTION.md)
2. Prepare: Collect & annotate bin images
3. Run: `python scripts/train_bin_detector.py prepare --dataset-path ./my_dataset`
4. Train: `python scripts/train_bin_detector.py train --dataset-yaml bin_dataset.yaml`
5. Deploy: Copy trained model to `models/pretrained/yolov8_bins_trained.pt`

### **Path 3: I'm Setting Up Hardware Integration**
1. Edit: `config/settings.yaml` → `sensing.modbus` section
2. Configure: ADAM-6117 IP address and register addresses
3. Test: Study `src/sensing/modbus_client.py` → `ModbusWeightClient` class
4. Debug: Create mock test in `tests/test_core.py`
5. Verify: Run `pytest tests/ -v`

### **Path 4: I'm Setting Up GitHub CI/CD**
1. Read: [docs/GITHUB_SETUP.md](GITHUB_SETUP.md)
2. Create: GitHub repo and push code
3. Verify: Check `.github/workflows/` files exist
4. Configure: Set branch protection rules on GitHub
5. Test: Create test PR and verify all checks pass

### **Path 5: I'm Contributing Code**
1. Create branch: `git checkout -b feature/my-feature`
2. Make changes: Edit files in `src/`, `tests/`, or `docs/`
3. Test locally: `pytest tests/ -v` and `black src/`
4. Push: `git push -u origin feature/my-feature`
5. Create PR: Open Pull Request on GitHub
6. Wait: All CI checks must pass + 1 approval needed
7. Merge: Once approved and passing, merge to `main`

---

## **Configuration Quick Reference**

### **Where to Change What**

| Configuration | File | Section |
|---------------|------|---------|
| Modbus IP address | `config/settings.yaml` | `sensing.modbus.ip_address` |
| Modbus registers | `config/settings.yaml` | `sensing.modbus.registers` |
| FSM gate timeouts | `config/settings.yaml` | `fsm.gate*_timeout` |
| Model paths | `config/settings.yaml` | `vision.model_*` |
| Hardware device | `config/settings.yaml` | `hardware.device` |
| Environment variables | `.env` (copy from `.env.example`) | Any key-value pair |

### **Common Edits**

**Change ADAM-6117 IP:**
```yaml
# config/settings.yaml
sensing:
  modbus:
    ip_address: "192.168.1.100"  # ← Change this
    port: 502
```

**Adjust FSM timeouts:**
```yaml
# config/settings.yaml
fsm:
  gate1_spatial_timeout: 2.0    # ← Increase for slower motions
  gate2_intent_timeout: 1.5
  gate3_verification_timeout: 3.0
```

**Use custom trained model:**
```yaml
# config/settings.yaml
vision:
  model_detect: "models/pretrained/yolov8_bins_trained.pt"  # ← Your trained model
```

---

## **File Navigation by Role**

### **Vision/ML Engineer** 👨‍💻
```
Start here:
src/vision/bin_detector.py          # Bin detection logic
  ├─ BinDetector class              # Core detector
  └─ DynamicGeofenceManager         # Real-time geofence extraction

Then explore:
src/vision/geofencing.py            # Geofence checking
scripts/train_bin_detector.py       # Training pipeline
docs/DYNAMIC_BIN_DETECTION.md       # Training guide
```

### **Hardware/Sensing Engineer** 🔌
```
Start here:
src/sensing/modbus_client.py        # Modbus TCP driver
  ├─ ModbusWeightClient class       # Weight sensor polling
  └─ ModbusPoller class             # Async polling loop

Then configure:
config/settings.yaml                # Modbus settings
  └─ sensing.modbus section

Then test:
tests/test_core.py                  # Modbus test fixtures
```

### **Logic/FSM Developer** 🧠
```
Start here:
src/logic/fsm.py                    # State machine
  ├─ FSMState enum                  # 6 states (idle, gate 1-3, success, error)
  └─ TripleGateFSM class            # Main FSM logic

Then understand:
docs/ARCHITECTURE.md                # FSM gates diagram
config/settings.yaml                # FSM timeouts
tests/test_core.py                  # FSM unit tests
```

### **DevOps/Deployment** 🚀
```
Start here:
.github/workflows/                  # CI/CD pipelines
  ├─ tests.yml                      # Pytest automation
  ├─ lint.yml                       # Code quality
  └─ build.yml                      # Project validation

Then configure:
docs/GITHUB_SETUP.md                # Branch protection guide
pyproject.toml                      # Build config
requirements.txt                    # Dependencies
```

---

## **Common Commands**

```powershell
# Setup
pip install -r requirements.txt
pytest tests/ -v

# Development
python src/main.py                          # Run main app
python scripts/train_bin_detector.py train  # Train models
python scripts/train_bin_detector.py test   # Test real-time

# Code quality
black src/ scripts/                         # Format code
flake8 src/ scripts/                        # Lint
mypy src/                                   # Type check
isort src/ scripts/                         # Sort imports

# Git workflow
git checkout -b feature/name                # Create branch
git add .
git commit -m "feat: description"
git push -u origin feature/name             # Create PR
```

---

## **Troubleshooting: "Where is...?"**

| Question | Answer |
|----------|--------|
| Where do I configure the system? | `config/settings.yaml` |
| Where is the main event loop? | `src/main.py` |
| Where is the FSM state machine? | `src/logic/fsm.py` |
| Where are my trained models? | `models/pretrained/` |
| Where do I add tests? | `tests/test_core.py` |
| Where do I document features? | `docs/` directory |
| Where is the bin detection logic? | `src/vision/bin_detector.py` |
| Where is Modbus integration? | `src/sensing/modbus_client.py` |
| Where do I train a bin detector? | `scripts/train_bin_detector.py` |
| Where is the CI/CD setup? | `.github/workflows/` |
| Where are environment variables? | `.env` (copy from `.env.example`) |
| Where is the architecture explained? | `docs/ARCHITECTURE.md` |

---

## **Next Steps**

1. **Clone the repo** (if not done):
   ```bash
   git clone https://github.com/your-org/aegis-core.git
   cd aegis-core
   ```

2. **Read the appropriate guide** based on your role (see "Common Paths" above)

3. **Set up your environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run tests to verify setup**:
   ```bash
   pytest tests/ -v
   ```

5. **Start contributing!**

---

## **Questions?**

- Architecture questions → Read `docs/ARCHITECTURE.md`
- Setup questions → Read `docs/GITHUB_SETUP.md`
- Model training → Read `docs/DYNAMIC_BIN_DETECTION.md`
- General questions → Check `README.md`

---

**Welcome to AEGIS! 🎯**
