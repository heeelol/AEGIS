# Dynamic Bin Detection - Quick Start Guide

## Overview

The dynamic bin detection approach eliminates manual `bins_map.yaml` calibration by training YOLOv8 to automatically detect bin locations in real-time.

**Workflow:**
```
Real-time Camera Feed
        ↓
YOLOv8 Bin Detector (detects bin bounding boxes)
        ↓
Automatic Geofence Extraction (pixel coordinates)
        ↓
Hand Pose + Geofence Check (spatial gate)
        ↓
FSM Verification (intent + weight)
```

---

## Step 1: Prepare Your Dataset

### Collect Training Images
- Mount your camera at the operational angle
- Capture 100-200 images of your bin array from different lighting/angles
- Include partial views and different hand gestures

### Annotate with Bounding Boxes
Use one of these tools:
- **LabelImg** (desktop app) - https://github.com/heartexlabs/labelImg
- **Roboflow** (web-based) - https://roboflow.com
- **CVAT** (advanced) - https://github.com/openvinotoolkit/cvat

**Requirements for each image:**
- One bounding box per bin
- Label: `bin`
- Save in YOLO format (`.txt` files with normalized coordinates)

### Directory Structure
```
my_bin_dataset/
├── images/
│   ├── train/      (70% of images)
│   ├── val/        (15% of images)
│   └── test/       (15% of images)
└── labels/
    ├── train/      (corresponding .txt files)
    ├── val/
    └── test/
```

**YOLO format example** (`image001.txt`):
```
0 0.45 0.30 0.35 0.40
0 0.80 0.30 0.35 0.40
```
(class_id center_x center_y width height - all normalized 0-1)

---

## Step 2: Prepare Dataset Configuration

```bash
cd aegis-core

python scripts/train_bin_detector.py prepare \
  --dataset-path ./my_bin_dataset \
  --output-yaml bin_dataset.yaml
```

This creates `bin_dataset.yaml`:
```yaml
path: /full/path/to/my_bin_dataset
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: bin
```

---

## Step 3: Train Bin Detector

```bash
python scripts/train_bin_detector.py train \
  --dataset-yaml bin_dataset.yaml \
  --model-size n \
  --epochs 100 \
  --batch-size 16 \
  --device 0
```

**Training options:**
- `--model-size`: `n` (nano, ~3MB) → `x` (xlarge, ~168MB)
- Nano is recommended for edge devices (ICAM-540/Jetson)
- Training typically takes 30-60 minutes on GPU

**Output:**
- Best model: `runs/bin_detection/yolov8_bins/weights/best.pt`
- Metrics: `runs/bin_detection/yolov8_bins/results.csv`

---

## Step 4: Evaluate Model

```bash
python scripts/train_bin_detector.py eval \
  --model runs/bin_detection/yolov8_bins/weights/best.pt \
  --dataset-yaml bin_dataset.yaml
```

Check for:
- **mAP50 > 0.85** - Good bin detection
- **mAP50-95 > 0.70** - Strong robustness

---

## Step 5: Test Real-Time Detection

```bash
# Test on live camera
python scripts/train_bin_detector.py test \
  --model runs/bin_detection/yolov8_bins/weights/best.pt \
  --camera 0
```

**Controls:**
- `q` - Quit
- `s` - Save frame with detections

**What to verify:**
- All 8 bins detected consistently
- Bounding boxes stable (minimal jitter)
- Correct bin ordering (left-to-right, top-to-bottom)

---

## Step 6: Deploy in AEGIS Pipeline

### Copy trained model to project:
```bash
cp runs/bin_detection/yolov8_bins/weights/best.pt \
   models/pretrained/yolov8_bins_trained.pt
```

### Update settings.yaml:
```yaml
vision:
  model_detect: "models/pretrained/yolov8_bins_trained.pt"
```

### Use in code:
```python
from src.vision.bin_detector import DynamicGeofenceManager

# Initialize dynamic geofences (replaces static bins_map.yaml)
geofence_mgr = DynamicGeofenceManager(
    model_path="models/pretrained/yolov8_bins_trained.pt",
    smoothing_window=5  # Temporal smoothing over 5 frames
)

# In main loop:
while True:
    ret, frame = cap.read()
    
    # Automatically extract geofences from frame
    geofences = geofence_mgr.update(frame)
    
    # Check hand in geofence
    hand_in_geofence, bin_id = geofence_mgr.check_hand_in_geofence(
        hand_keypoints, geofences
    )
    
    # Use in FSM as Gate 1 (spatial)
    fsm_state = fsm.update(sensor_reading)
```

---

## Troubleshooting

### Issue: Low mAP scores
- **Solution 1**: Add more training images (target 200+)
- **Solution 2**: Improve annotations (ensure boxes tightly fit bins)
- **Solution 3**: Use larger model (`--model-size s` or `m`)

### Issue: Bins not detected in real-time
- **Solution 1**: Test with `train_bin_detector.py test` to debug
- **Solution 2**: Lower confidence threshold in `DynamicGeofenceManager` init
- **Solution 3**: Verify lighting similar to training conditions

### Issue: Inconsistent bin ordering
- The `_sort_boxes_spatially()` function assumes:
  - Roughly rectangular 4×2 grid layout
  - Bins separated by visible gaps
- If bins are touching, implement custom sorting logic

### Issue: High latency
- Use smaller model: `yolov8n` instead of `yolov8m`
- Reduce input size: `imgsz=480` instead of `640`
- Enable TensorRT INT8 quantization (see `scripts/quantize.py`)

---

## Advantages Over Static Calibration

| Feature | Static (bins_map.yaml) | Dynamic (YOLOv8) |
|---------|-----|------|
| Setup | Manual pixel measurements | Train once, reuse |
| Adaptability | Fixed to one camera setup | Works with camera/bin changes |
| Robustness | Brittle to lighting changes | Learns robust features |
| Scalability | Manual per camera | One model for all cameras |
| Accuracy | Depends on calibration skill | Can achieve >85% mAP |

---

## Advanced: Quantize to TensorRT INT8

For faster inference on NVIDIA hardware:

```bash
python scripts/quantize.py \
  --model models/pretrained/yolov8_bins_trained.pt \
  --validate test_images/
```

This creates `yolov8_bins_trained.engine` (3-4x faster, ~25% smaller).

---

## Next: Hand Pose Detection

Once bins are detected, implement pose estimation to:
1. Extract hand keypoints (wrist, fingers)
2. Detect "Closed Fist" gesture (Gate 2)
3. Track hand trajectory into bin

See: `src/vision/pose_estimator.py` (to be implemented)

---

## Example Complete Pipeline

See `scripts/example_real_time_pipeline.py` for full integration:
- Dynamic bin detection
- Hand pose estimation
- FSM verification
- Real-time visualization
