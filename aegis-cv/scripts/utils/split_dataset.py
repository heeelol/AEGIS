import os
import shutil
from pathlib import Path
import random

dataset_dir = Path(r'c:\Users\yapor\OneDrive\Desktop\CDE3301\TEnterns\aegis-core\models\data\CDE3301.yolov8-obb')
train_img_dir = dataset_dir / 'train' / 'images'
train_lbl_dir = dataset_dir / 'train' / 'labels'

# Create validation and test directories
for split in ['valid', 'test']:
    (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
    (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

# Get all images
images = sorted([f.name for f in train_img_dir.glob('*') if f.is_file()])
print(f"Total images: {len(images)}")

# Split: 70% train, 15% val, 15% test
random.seed(42)
random.shuffle(images)

train_count = int(len(images) * 0.70)
val_count = int(len(images) * 0.15)

train_imgs = images[:train_count]
val_imgs = images[train_count:train_count + val_count]
test_imgs = images[train_count + val_count:]

print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

# Move validation images and labels
for img in val_imgs:
    lbl = img.rsplit('.', 1)[0] + '.txt'
    shutil.move(str(train_img_dir / img), str(dataset_dir / 'valid' / 'images' / img))
    if (train_lbl_dir / lbl).exists():
        shutil.move(str(train_lbl_dir / lbl), str(dataset_dir / 'valid' / 'labels' / lbl))

# Move test images and labels
for img in test_imgs:
    lbl = img.rsplit('.', 1)[0] + '.txt'
    shutil.move(str(train_img_dir / img), str(dataset_dir / 'test' / 'images' / img))
    if (train_lbl_dir / lbl).exists():
        shutil.move(str(train_lbl_dir / lbl), str(dataset_dir / 'test' / 'labels' / lbl))

# Verify
train_final = len(list((dataset_dir / 'train' / 'images').glob('*')))
val_final = len(list((dataset_dir / 'valid' / 'images').glob('*')))
test_final = len(list((dataset_dir / 'test' / 'images').glob('*')))

print(f"\nFinal split:")
print(f"  Train: {train_final} images")
print(f"  Val:   {val_final} images")
print(f"  Test:  {test_final} images")
print(f"  Total: {train_final + val_final + test_final} images")
