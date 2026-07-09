"""Retrain Mask R-CNN with proper dataset validation and training monitoring."""

import logging
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class COCODatasetFixed(torch.utils.data.Dataset):
    """Fixed COCO dataset with proper mask extraction and validation."""
    
    def __init__(self, img_dir, annotation_file):
        self.img_dir = Path(img_dir)
        
        with open(annotation_file) as f:
            self.coco_data = json.load(f)
        
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations_by_img = {}
        
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations_by_img:
                self.annotations_by_img[img_id] = []
            self.annotations_by_img[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())
        logger.info(f"Dataset loaded: {len(self.image_ids)} images, {len(self.coco_data['annotations'])} annotations")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        fname = img_info["file_name"]
        if fname.startswith("PNG image "):
            fname = fname[9:]
        
        img_path = self.img_dir / fname
        
        # Try alternative paths
        if not img_path.exists():
            img_path = self.img_dir / img_info["file_name"]
        
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}, trying alternatives...")
            # Try all jpg/png files
            for alt_path in self.img_dir.glob("*.[jp][pn]g"):
                if alt_path.stem == Path(fname).stem:
                    img_path = alt_path
                    break
        
        import PIL.Image
        image = PIL.Image.open(img_path).convert("RGB")
        width, height = image.size
        
        # Convert to tensor
        import torchvision.transforms as T
        image = T.ToTensor()(image)
        
        # Process annotations - FIXED with proper mask creation
        masks = []
        boxes = []
        labels = []
        
        anns = self.annotations_by_img.get(img_id, [])
        
        for ann in anns:
            if "segmentation" not in ann or not ann["segmentation"]:
                continue
            
            # Create mask from polygon using OpenCV
            mask = np.zeros((height, width), dtype=np.uint8)
            
            for seg in ann["segmentation"]:
                if len(seg) < 6:  # Need at least 3 points
                    continue
                
                # Convert normalized/absolute coordinates to pixel coordinates
                points = []
                for i in range(0, len(seg), 2):
                    x = float(seg[i])
                    y = float(seg[i + 1])
                    
                    # Handle normalized coordinates (0-1 range)
                    if x <= 1.0 and y <= 1.0:
                        x = int(x * width)
                        y = int(y * height)
                    else:
                        x = int(x)
                        y = int(y)
                    
                    points.append([x, y])
                
                if len(points) >= 3:
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 1)
            
            # Only add if mask has substantial content
            if mask.sum() > 100:  # At least 100 pixels
                masks.append(mask)
                
                # Get bounding box from mask
                coords = np.where(mask > 0)
                if len(coords[0]) > 0:
                    y_min = max(0, coords[0].min() - 5)
                    y_max = min(height, coords[0].max() + 5)
                    x_min = max(0, coords[1].min() - 5)
                    x_max = min(width, coords[1].max() + 5)
                    
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(1)  # Bin class
        
        # Return even if empty (model needs to learn negative examples)
        if len(masks) == 0:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = torch.from_numpy(np.stack(masks)).to(torch.uint8)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id])
        }
        
        return image, target


def collate_fn(batch):
    """Custom collate function."""
    return tuple(zip(*batch))


def train_one_epoch(model, dataloader, optimizer, device, epoch, num_epochs):
    """Train for one epoch with detailed logging."""
    model.train()
    total_loss = 0
    losses_dict = {"classifier": [], "bbox": [], "objectness": [], "rpn_box": [], "mask": []}
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += losses.item()
        
        # Track individual losses
        for key, val in loss_dict.items():
            if key in losses_dict:
                losses_dict[key].append(val.item())
        
        if (batch_idx + 1) % 2 == 0:
            progress = int(((batch_idx + 1) / len(dataloader)) * 20)
            bar = "█" * progress + "░" * (20 - progress)
            logger.info(f"Epoch {epoch}/{num_epochs} [{bar}] {((batch_idx+1)/len(dataloader))*100:.0f}%")
            logger.info(f"  Total Loss: {losses.item():.4f}")
            
            # Log individual losses
            for key, vals in losses_dict.items():
                if vals:
                    logger.info(f"    {key}: {np.mean(vals[-5:]):.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    """Retrain Mask R-CNN with better data handling."""
    
    base_dir = Path(__file__).parent.parent.parent
    
    logger.info("="*60)
    logger.info("MASK R-CNN RETRAINING (Fixed Dataset)")
    logger.info("="*60 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}\n")
    
    # Setup paths
    coco_dir = base_dir / "models" / "data" / "Training_5.coco-segmentation" / "train"
    annotation_file = coco_dir / "_annotations.coco.json"
    
    if not annotation_file.exists():
        logger.error(f"Annotations not found: {annotation_file}")
        return
    
    logger.info("Loading Training_5 COCO dataset (FIXED)...")
    
    # Create dataset with validation
    dataset = COCODatasetFixed(img_dir=coco_dir, annotation_file=str(annotation_file))
    
    # Validate dataset
    logger.info(f"\nValidating dataset...")
    mask_count = 0
    image_count = 0
    for i in range(min(5, len(dataset))):
        img, target = dataset[i]
        if target["masks"].shape[0] > 0:
            mask_count += 1
            logger.info(f"  Sample {i}: {target['masks'].shape[0]} masks, {img.shape}")
        image_count += 1
    
    logger.info(f"✓ Checked {image_count} samples, {mask_count} had masks\n")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    logger.info(f"Dataloader: {len(dataloader)} batches\n")
    
    # Load model
    logger.info("Loading Mask R-CNN R-50-FPN...")
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Modify for binary classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, 2)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, 2 * 4)
    
    model.to(device)
    logger.info("✓ Model ready\n")
    
    # Setup optimizer with lower learning rate
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Training
    num_epochs = 30
    output_dir = base_dir / "runs" / "maskrcnn" / "training_5_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Training configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch Size: 2")
    logger.info(f"  Learning Rate: 0.001")
    logger.info(f"  Output Dir: {output_dir}\n")
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch, num_epochs)
        scheduler.step()
        
        logger.info(f"Epoch {epoch} Complete - Avg Loss: {avg_loss:.4f}\n")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"✓ Checkpoint saved: {ckpt_path}\n")
    
    elapsed = time.time() - start_time
    
    # Save final model
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    model_path = base_dir / "models" / "custom" / "bin_segmentation_training_5_maskrcnn.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_path)
    logger.info(f"✓ Model saved: {model_path}")
    logger.info(f"Training Time: {elapsed/60:.1f} minutes")
    logger.info(f"Results Dir: {output_dir}")
    logger.info("="*60)
    
    logger.info("\n✓ Retraining complete! Run debug_maskrcnn.py to verify.")


if __name__ == "__main__":
    main()
