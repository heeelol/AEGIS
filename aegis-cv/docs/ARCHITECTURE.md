# AEGIS Core - System Architecture

## System Overview

AEGIS implements a real-time sensor-fusion system for procedural verification in HMLV kitting. The system operates on a continuous **Sense-Analyse-Act** loop with <500ms end-to-end latency.

```
┌─────────────┐
│   SENSE     │ ← YOLOv8-Pose (hand gesture)
│             │ ← YOLOv8-Detect (geofencing)
│             │ ← Modbus TCP (weight)
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  ANALYSE    │ ← Triple-Gate FSM
│             │ ← Inventory logic
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    ACT      │ → Visual Co-Pilot dashboard
│             │ → Event logging
└─────────────┘
```

## Module Architecture

### Vision Module (`src/vision/`)
- **Responsibilities**: Hand pose detection, bin geofencing, gesture classification
- **Key Components**:
  - YOLOv8-Pose model for hand keypoint extraction
  - Geofence boundary checking (spatial gate)
  - Custom gesture classifier (intent gate)
  
### Sensing Module (`src/sensing/`)
- **Responsibilities**: Hardware communication (Modbus TCP, Serial)
- **Key Components**:
  - Modbus TCP client (ADAM-6117 weight registers)
  - Serial driver (ESP32 optional auxiliary sensors)
  - Non-blocking async polling

### Logic Module (`src/logic/`)
- **Responsibilities**: FSM state management, inventory tracking
- **Key Components**:
  - Triple-Gate FSM state machine
  - Inventory database connector
  - Error handling and retry logic

### UI Module (`src/ui/`)
- **Responsibilities**: Real-time monitoring dashboard
- **Key Components**:
  - Streamlit/FastAPI backend
  - WebSocket for live data streaming
  - Event log viewer

---

## Triple-Gate Verification Logic

```
Input Sensor Data
      │
      ↓
┌─────────────────────┐
│ Gate 1: SPATIAL     │ Triggered: Hand in geofence?
│ (YOLOv8-Pose)       │ Duration: ≥ gate1_spatial_timeout
│ Timeout: 2.0s       │
└──────┬──────────────┘
       │ PASS → Continue
       │ FAIL → Reset
       ↓
┌─────────────────────┐
│ Gate 2: INTENT      │ Triggered: Closed Fist?
│ (Custom Gesture)    │ Duration: ≥ gate2_intent_timeout
│ Timeout: 1.5s       │
└──────┬──────────────┘
       │ PASS → Continue
       │ FAIL → Reset
       ↓
┌─────────────────────┐
│ Gate 3: VERIFY      │ Triggered: Weight delta ≥ threshold?
│ (Modbus Weight)     │ Duration: ≥ gate3_verification_timeout
│ Timeout: 3.0s       │
└──────┬──────────────┘
       │ PASS → Inventory Update + Event Log
       │ FAIL → Error Alert + Retry Logic
       ↓
Output (Success/Failure)
```

---

## Performance Optimization Strategy

### 1. Quantization (TensorRT INT8)
- Convert YOLOv8 models to INT8 for 3-4x speedup
- Minimal accuracy loss (<2% typical)
- Deploy `.engine` files on NVIDIA GPU

### 2. Batch Processing
- Group vision inferences where possible
- Async Modbus polling (non-blocking)

### 3. Latency Budget (Target: 500ms)
- Vision inference: 120ms (YOLOv8 + geofencing)
- Modbus poll: 100ms
- FSM logic: 5ms
- Network/UI update: 50ms
- **Headroom**: 225ms for overhead/retry

---

## Hardware Deployment

### Target Devices
- NVIDIA Jetson AGX Orin (16GB RAM, 12-core)
- ICAM-540 (industrial edge AI appliance)
- MIC-733 (fanless embedded AI platform)

### Dependencies
- CUDA 11.8+
- cuDNN 8.5+
- TensorRT 8.5+
- PyTorch 2.0+ (precompiled wheels for Jetson)

---

## Configuration & Calibration

### Camera Calibration
1. Mount camera with fixed FOV
2. Capture reference images of 4x2 bin array
3. Measure (x, y) pixel coordinates for each bin corner
4. Update `config/bins_map.yaml` with geofence bounds

### Modbus Register Mapping
```
Bin 0_0 weight → Register 100
Bin 0_1 weight → Register 101
... (one register per bin)
```

See `config/settings.yaml` for ADAM-6117 configuration.

---

## Integration with Existing Systems

### Database Connection
- Link inventory database for real-time updates
- Log all pick events with timestamp
- Track error rate per bin location

### MES/ERP Integration
- REST API endpoint for downstream systems
- Event stream (Kafka/RabbitMQ optional)

---

## Future Enhancements

- Multi-camera support for larger bin arrays
- Advanced anomaly detection (unusual gesture patterns)
- Federated learning for gesture model improvement
- Mobile app for remote monitoring
