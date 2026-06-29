"""Train Mask R-CNN on Training_5 dataset (COCO format) - GPU Accelerated with torchvision."""

import logging
import json
import time
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class COCODataset(torch.utils.data.Dataset):
    """COCO dataset for torchvision Mask R-CNN."""
    
    def __init__(self, img_dir, annotation_file, transforms=None):
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        
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
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.img_dir / img_info["file_name"]
        if not img_path.exists():
            # Try without "PNG image " prefix
            fname = img_info["file_name"]
            if fname.startswith("PNG image "):
                img_path = self.img_dir / fname[9:]
        
        import PIL.Image
        image = PIL.Image.open(img_path).convert("RGB")
        width, height = image.size
        
        # Convert to tensor
        import torchvision.transforms as T
        image = T.ToTensor()(image)
        
        # Process annotations
        masks = []
        boxes = []
        labels = []
        
        for ann in self.annotations_by_img.get(img_id, []):
            if "segmentation" not in ann or not ann["segmentation"]:
                continue
            
            # Create mask from segmentation polygon
            mask = torch.zeros((height, width), dtype=torch.uint8)
            
            for seg in ann["segmentation"]:
                if len(seg) < 6:
                    continue
                
                # Convert polygon to mask
                points = []
                for i in range(0, len(seg), 2):
                    x = int(seg[i])
                    y = int(seg[i+1])
                    points.append([x, y])
                
                try:
                    import cv2
                    import numpy as np
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask.numpy(), [pts], 1)
                except:
                    pass
            
            if mask.sum() > 0:
                masks.append(mask)
                
                # Get bbox from mask
                coords = torch.where(mask)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min().item(), coords[0].max().item()
                    x_min, x_max = coords[1].min().item(), coords[1].max().item()
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)  # Class 1 = bin
        
        # If no valid masks, return empty
        if len(masks) == 0:
            masks = torch.zeros((1, height, width), dtype=torch.uint8)
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)
        else:
            masks = torch.stack(masks)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return tuple(zip(*batch))


def train_one_epoch(model, dataloader, optimizer, device, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if (batch_idx + 1) % 5 == 0:
            progress = int(((batch_idx + 1) / len(dataloader)) * 20)
            bar = "█" * progress + "░" * (20 - progress)
            logger.info(f"Epoch {epoch}/{num_epochs} [{bar}] {((batch_idx+1)/len(dataloader))*100:.0f}%")
            logger.info(f"  Loss: {losses.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    """Train Mask R-CNN on Training_5 dataset."""
    
    base_dir = Path(__file__).parent.parent.parent
    
    logger.info("="*60)
    logger.info("MASK R-CNN TRAINING - Training_5 (Positional Occlusion)")
    logger.info("="*60 + "\n")
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"✓ CUDA Version: {torch.version.cuda}\n")
    else:
        logger.warning("⚠ GPU not available - training will be slow\n")
    
    # Setup paths
    coco_dir = base_dir / "models" / "data" / "Training_5.coco-segmentation" / "train"
    annotation_file = coco_dir / "_annotations.coco.json"
    
    if not annotation_file.exists():
        logger.error(f"Annotations not found: {annotation_file}")
        return
    
    logger.info("Loading Training_5 COCO dataset...")
    
    # Create dataset
    dataset = COCODataset(
        img_dir=coco_dir,
        annotation_file=str(annotation_file),
        transforms=None
    )
    
    logger.info(f"✓ Dataset loaded: {len(dataset)} images")
    logger.info(f"✓ Annotations: {annotation_file}\n")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Small batch size for RTX 5060
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Load model
    logger.info("Loading Mask R-CNN R-50-FPN (pre-trained)...")
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Modify head for binary classification (bin vs background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, 2)  # Binary: bg + bin
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, 2 * 4)
    
    model.to(device)
    logger.info("✓ Model loaded and configured for binary segmentation\n")
    
    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    num_epochs = 20
    output_dir = base_dir / "runs" / "maskrcnn" / "training_5"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Task: Instance segmentation with mask prediction")
    logger.info("Dataset: Training_5 (100x Positional Occlusion samples)")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch Size: 2")
    logger.info(f"Learning Rate: 0.005\n")
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch, num_epochs)
        scheduler.step()
        
        logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}\n")
        
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
    logger.info(f"Dataset: Training_5 (100x Positional Occlusion)")
    logger.info(f"Model: Mask R-CNN R-50-FPN (torchvision)")
    logger.info(f"Results Directory: {output_dir}")
    logger.info("="*60)
    
    logger.info("\n✓ Training pipeline complete!")


if __name__ == "__main__":
    main()
