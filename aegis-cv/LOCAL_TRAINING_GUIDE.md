"""
LOCAL GPU TRAINING SETUP GUIDE
==============================

Your bin boundary dataset is ready for local training!
Location: models/data/CDE3301.yolov8-obb/

QUICK START:
============

Step 1: Verify GPU Setup
------------------------
Run this to check if your NVIDIA GPU is detected:

    python scripts/verify_gpu.py

Expected output:
    ✓ CUDA Available: True
    ✓ GPU Name: NVIDIA GeForce RTX 4050 (or similar)
    ✓ GPU Memory: 8.00 GB
    ✓ Your GPU is ready for training!


Step 2: Install/Update PyTorch with CUDA Support
--------------------------------------------------
If verify_gpu.py shows "CUDA not available", run:

    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Then verify again with Step 1.


Step 3: Start Training
-----------------------
Run the training script:

    python scripts/train_bin_boundaries_local.py

What happens:
    - Loads YOLOv8 nano model (optimized for laptop GPU)
    - Trains on your bin boundary dataset (50 epochs)
    - Uses GPU acceleration (much faster than CPU)
    - Saves best model to: runs/bin_detector/weights/best.pt
    - Estimated time: 10-30 minutes depending on dataset size

Training settings (laptop-friendly):
    - Batch size: 16 (prevents out-of-memory)
    - Image size: 640x640
    - Workers: 4 (multi-core CPU for data loading)
    - AMP: Enabled (Automatic Mixed Precision for speed)


Step 4: Monitor Training
-------------------------
During training, you'll see:
    - Epoch progress and loss metrics
    - Validation results
    - GPU memory usage (check Task Manager to see GPU @ 90%+ = good)
    - Estimated time remaining

Keep an eye on your laptop fans - this is normal during GPU training!


Step 5: Use Trained Model
---------------------------
After training completes, your model is at:
    runs/bin_detector/weights/best.pt

Use it for inference:

    from ultralytics import YOLO
    model = YOLO("runs/bin_detector/weights/best.pt")
    results = model.predict(source="image.jpg", conf=0.5)


DATASET STRUCTURE
=================

Current structure:
    models/data/CDE3301.yolov8-obb/
    ├── data.yaml              (original - paths point to non-existent dirs)
    ├── data_local.yaml        (NEW - paths work for local training)
    ├── README.roboflow.txt
    └── train/
        ├── images/           (all annotated images)
        └── labels/           (YOLO OBB annotations)

For better model generalization, split your data:
    - Create: valid/images, valid/labels (15% of data)
    - Create: test/images, test/labels (15% of data)
    - Keep: train/images, train/labels (70% of data)

Then update data_local.yaml paths accordingly.


TROUBLESHOOTING
===============

Q: "CUDA not available"
A: Run: pip install torch --index-url https://download.pytorch.org/whl/cu121

Q: "Out of memory" error
A: Reduce batch size from 16 to 8 in train_bin_boundaries_local.py

Q: "ModuleNotFoundError: ultralytics"
A: Run: pip install ultralytics

Q: Slow training / GPU not being used
A: Check GPU utilization in Task Manager > Performance > GPU
   If <20%, run verify_gpu.py to check CUDA setup

Q: Model is overfitting (high train loss, low val loss)
A: The current dataset only uses train/ for validation.
   Create proper val/ and test/ splits for better results.


NEXT STEPS
==========

After training:
1. Copy best.pt to src/models/custom/
2. Update hand_detector.py to use local model instead of Roboflow API
3. Integrate with FSM pipeline (src/main.py)
4. Test end-to-end on live video

For questions, check: docs/ARCHITECTURE.md
"""
