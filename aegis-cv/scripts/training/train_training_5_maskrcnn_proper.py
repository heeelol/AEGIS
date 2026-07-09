"""Properly fine-tune Mask R-CNN with transfer learning best practices."""

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
    """Fixed COCO dataset with proper mask extraction."""
    
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
        logger.info(f"Dataset: {len(self.image_ids)} images, {len(self.coco_data['annotations'])} annotations")
    
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
        if not img_path.exists():
            img_path = self.img_dir / img_info["file_name"]
        
        import PIL.Image
        image = PIL.Image.open(img_path).convert("RGB")
        width, height = image.size
        
        import torchvision.transforms as T
        image = T.ToTensor()(image)
        
        # Process annotations
        masks = []
        boxes = []
        labels = []
        
        anns = self.annotations_by_img.get(img_id, [])
        
        for ann in anns:
            if "segmentation" not in ann or not ann["segmentation"]:
                continue
            
            mask = np.zeros((height, width), dtype=np.uint8)
            
            for seg in ann["segmentation"]:
                if len(seg) < 6:
                    continue
                
                points = []
                for i in range(0, len(seg), 2):
                    x = float(seg[i])
                    y = float(seg[i + 1])
                    
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
            
            if mask.sum() > 100:
                masks.append(mask)
                
                coords = np.where(mask > 0)
                if len(coords[0]) > 0:
                    y_min = max(0, coords[0].min() - 5)
                    y_max = min(height, coords[0].max() + 5)
                    x_min = max(0, coords[1].min() - 5)
                    x_max = min(width, coords[1].max() + 5)
                    
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(1)
        
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


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += losses.item()
        
        if (batch_idx + 1) % 5 == 0:
            progress = int(((batch_idx + 1) / num_batches) * 20)
            bar = "█" * progress + "░" * (20 - progress)
            pct = ((batch_idx + 1) / num_batches) * 100
            logger.info(f"Epoch {epoch}/{num_epochs} [{bar}] {pct:.0f}%")
            logger.info(f"  Batch Loss: {losses.item():.4f} | Avg: {total_loss/(batch_idx+1):.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Proper transfer learning for Mask R-CNN."""
    
    base_dir = Path(__file__).parent.parent.parent
    
    logger.info("="*60)
    logger.info("MASK R-CNN - PROPER TRANSFER LEARNING")
    logger.info("="*60 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}\n")
    
    # Load dataset
    coco_dir = base_dir / "models" / "data" / "Training_5.coco-segmentation" / "train"
    annotation_file = coco_dir / "_annotations.coco.json"
    
    if not annotation_file.exists():
        logger.error(f"Annotations not found: {annotation_file}")
        return
    
    dataset = COCODatasetFixed(img_dir=coco_dir, annotation_file=str(annotation_file))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    logger.info(f"Dataloader: {len(dataloader)} batches\n")
    
    # Load model
    logger.info("Loading Mask R-CNN R-50-FPN (pre-trained on COCO)...")
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Modify classifier for binary classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, 2)  # bg + bin
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, 2 * 4)
    
    logger.info("✓ Model modified for binary classification\n")
    
    # Key fix: Create optimizer with ALL parameters (not just head)
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Total trainable parameters: {sum(p.numel() for p in params):,}\n")
    
    # Use different learning rates for backbone vs head
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "roi_heads" in name or "rpn" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    logger.info(f"Backbone parameters: {sum(p.numel() for p in backbone_params):,}")
    logger.info(f"Head parameters: {sum(p.numel() for p in head_params):,}\n")
    
    # Optimizer with layered learning rates
    optimizer = optim.SGD([
        {"params": backbone_params, "lr": 0.0001},  # Low LR for backbone
        {"params": head_params, "lr": 0.001}        # Higher LR for head
    ], momentum=0.9, weight_decay=0.0005)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    model.to(device)
    logger.info("✓ Model ready for training\n")
    
    # Training
    num_epochs = 50
    output_dir = base_dir / "runs" / "maskrcnn" / "training_5_proper"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Training configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Backbone LR: 0.0001")
    logger.info(f"  Head LR: 0.001")
    logger.info(f"  Batch Size: 2")
    logger.info(f"  Output: {output_dir}\n")
    
    start_time = time.time()
    best_loss = float("inf")
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, num_epochs)
        scheduler.step()
        
        logger.info(f"Epoch {epoch} Complete - Avg Loss: {avg_loss:.4f}\n")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = output_dir / "checkpoint_best.pth"
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"✓ New best - saved to {best_ckpt.name}\n")
        
        if epoch % 10 == 0:
            ckpt = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            logger.info(f"✓ Checkpoint saved: {ckpt.name}\n")
    
    elapsed = time.time() - start_time
    
    # Save final model
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    model_path = base_dir / "models" / "custom" / "bin_segmentation_training_5_maskrcnn.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_path)
    
    # Verify save
    state = torch.load(model_path, map_location=device)
    saved_params = sum(p.numel() for k, p in state.items() if p.dim() > 0)
    logger.info(f"✓ Model saved: {model_path}")
    logger.info(f"✓ Saved parameters: {saved_params:,}")
    logger.info(f"Training Time: {elapsed/60:.1f} minutes")
    logger.info(f"Best Loss: {best_loss:.4f}")
    logger.info("="*60)
    
    logger.info("\n✓ Next: Run debug_maskrcnn.py and evaluate_models.py")


if __name__ == "__main__":
    main()
