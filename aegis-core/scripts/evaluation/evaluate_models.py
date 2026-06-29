"""Evaluate all Training_5 models (YOLOv8 small/medium/large + Mask R-CNN) and compare accuracy."""

import json
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_iou(pred_mask, gt_mask):
    """Compute Intersection over Union (IoU) between predicted and ground truth masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0 if intersection == 0 else 1.0
    
    return intersection / union


def compute_dice(pred_mask, gt_mask):
    """Compute Dice Score (F1 for segmentation)."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    
    if pred_mask.sum() + gt_mask.sum() == 0:
        return 1.0
    
    return 2 * intersection / (pred_mask.sum() + gt_mask.sum())


def compute_boundary_accuracy(pred_mask, gt_mask, distance_threshold=5):
    """Compute boundary accuracy - how well edges match (important for occlusion)."""
    from scipy.ndimage import distance_transform_edt
    
    # Get boundary pixels
    pred_boundary = cv2.Canny((pred_mask * 255).astype(np.uint8), 50, 150)
    gt_boundary = cv2.Canny((gt_mask * 255).astype(np.uint8), 50, 150)
    
    if gt_boundary.sum() == 0:
        return 1.0 if pred_boundary.sum() == 0 else 0.0
    
    # Distance transform from gt boundary
    dist_transform = distance_transform_edt(gt_boundary == 0)
    
    # Check how many predicted boundary pixels are within threshold of gt boundary
    boundary_matches = (dist_transform[pred_boundary > 0] <= distance_threshold).sum()
    boundary_accuracy = boundary_matches / (pred_boundary.sum() + 1)
    
    return min(1.0, boundary_accuracy)


def load_yolov8_model(model_name):
    """Load YOLOv8 model."""
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "models" / "custom" / f"bin_segmentation_training_5_{model_name}.pt"
    
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None
    
    model = YOLO(str(model_path))
    return model


def load_maskrcnn_model():
    """Load Mask R-CNN model."""
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "models" / "custom" / "bin_segmentation_training_5_maskrcnn.pth"
    
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, 2)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, 2 * 4)
    
    state_dict = torch.load(model_path, map_location=device)
    total_params = sum(p.numel() for k, p in state_dict.items() if isinstance(p, torch.Tensor))
    
    if total_params < 1000000:
        logger.warning(f"⚠️ Model only has {total_params:,} parameters - may not be trained properly")
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model


def test_yolov8_model(model, test_images, test_labels, model_name):
    """Test YOLOv8 model and compute metrics."""
    
    logger.info(f"\nTesting YOLOv8 {model_name.upper()} model...")
    
    ious = []
    dice_scores = []
    boundary_accs = []
    
    for img_path, label_file in zip(test_images, test_labels):
        # Load image and ground truth
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        height, width = image.shape[:2]
        
        # Load ground truth mask
        gt_mask = np.zeros((height, width), dtype=np.uint8)
        
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    coords = list(map(float, line.strip().split()[1:]))
                    if len(coords) >= 6:
                        points = []
                        for i in range(0, len(coords), 2):
                            x = int(coords[i] * width)
                            y = int(coords[i+1] * height)
                            points.append([x, y])
                        
                        pts = np.array(points, dtype=np.int32)
                        cv2.fillPoly(gt_mask, [pts], 1)
        
        if gt_mask.sum() == 0:
            continue
        
        # Predict
        results = model.predict(image, verbose=False, conf=0.5)
        
        # Extract predicted mask
        pred_mask = np.zeros((height, width), dtype=np.uint8)
        
        if results[0].masks is not None:
            for mask in results[0].masks.data:
                pred_mask_i = (mask.cpu().numpy() > 0.5).astype(np.uint8)
                if pred_mask_i.shape != (height, width):
                    pred_mask_i = cv2.resize(pred_mask_i, (width, height), interpolation=cv2.INTER_NEAREST)
                pred_mask = np.logical_or(pred_mask, pred_mask_i).astype(np.uint8)
        
        # Compute metrics
        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)
        boundary_acc = compute_boundary_accuracy(pred_mask, gt_mask)
        
        ious.append(iou)
        dice_scores.append(dice)
        boundary_accs.append(boundary_acc)
    
    if not ious:
        logger.warning(f"No valid results for {model_name}")
        return None
    
    results = {
        "model": f"YOLOv8 {model_name.upper()}",
        "mean_iou": np.mean(ious),
        "std_iou": np.std(ious),
        "mean_dice": np.mean(dice_scores),
        "std_dice": np.std(dice_scores),
        "mean_boundary_acc": np.mean(boundary_accs),
        "std_boundary_acc": np.std(boundary_accs),
        "num_samples": len(ious)
    }
    
    return results


def test_maskrcnn_model(model, test_images, test_labels):
    """Test Mask R-CNN model and compute metrics."""
    
    logger.info(f"\nTesting Mask R-CNN model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ious = []
    dice_scores = []
    boundary_accs = []
    
    for img_path, label_file in zip(test_images, test_labels):
        # Load image and ground truth
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        gt_mask = np.zeros((height, width), dtype=np.uint8)
        
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    coords = list(map(float, line.strip().split()[1:]))
                    if len(coords) >= 6:
                        points = []
                        for i in range(0, len(coords), 2):
                            x = int(coords[i] * width)
                            y = int(coords[i+1] * height)
                            points.append([x, y])
                        
                        pts = np.array(points, dtype=np.int32)
                        cv2.fillPoly(gt_mask, [pts], 1)
        
        if gt_mask.sum() == 0:
            continue
        
        # Predict
        # Scale to [0,1] to match T.ToTensor() used in training; torchvision's
        # Mask R-CNN normalizes with ImageNet mean/std assuming [0,1] input.
        # Feeding [0,255] yields out-of-distribution activations and 0 detections.
        image_tensor = (torch.from_numpy(image_rgb).float() / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model([image_tensor.squeeze(0)])
        
        # Extract predicted mask
        pred_mask = np.zeros((height, width), dtype=np.uint8)
        
        if predictions[0]["masks"] is not None and len(predictions[0]["masks"]) > 0:
            for mask in predictions[0]["masks"]:
                pred_mask_i = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
                if pred_mask_i.shape != (height, width):
                    pred_mask_i = cv2.resize(pred_mask_i, (width, height), interpolation=cv2.INTER_NEAREST)
                pred_mask = np.logical_or(pred_mask, pred_mask_i).astype(np.uint8)
        
        # Compute metrics
        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)
        boundary_acc = compute_boundary_accuracy(pred_mask, gt_mask)
        
        ious.append(iou)
        dice_scores.append(dice)
        boundary_accs.append(boundary_acc)
    
    if not ious:
        logger.warning("No valid results for Mask R-CNN")
        return None
    
    results = {
        "model": "Mask R-CNN R-50-FPN",
        "mean_iou": np.mean(ious),
        "std_iou": np.std(ious),
        "mean_dice": np.mean(dice_scores),
        "std_dice": np.std(dice_scores),
        "mean_boundary_acc": np.mean(boundary_accs),
        "std_boundary_acc": np.std(boundary_accs),
        "num_samples": len(ious)
    }
    
    return results


def main():
    """Evaluate all models."""
    
    base_dir = Path(__file__).parent.parent.parent
    
    logger.info("="*70)
    logger.info("MODEL EVALUATION - Training_5 Test Set")
    logger.info("="*70)
    
    # Get test images and labels
    test_dir = base_dir / "models" / "data" / "Training_5.yolov8-seg" / "images" / "test"
    label_dir = base_dir / "models" / "data" / "Training_5.yolov8-seg" / "labels" / "test"
    
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    test_images = sorted(test_dir.glob("*.[jp][pn]g"))
    test_labels = [label_dir / (img.stem + ".txt") for img in test_images]
    
    logger.info(f"\nTest Set: {len(test_images)} images")
    logger.info(f"Test Directory: {test_dir}\n")
    
    # Test all models
    all_results = []
    
    # YOLOv8 models
    for size in ["small", "medium", "large"]:
        model = load_yolov8_model(size)
        if model:
            results = test_yolov8_model(model, test_images, test_labels, size)
            if results:
                all_results.append(results)
    
    # Mask R-CNN
    model = load_maskrcnn_model()
    if model:
        results = test_maskrcnn_model(model, test_images, test_labels)
        if results:
            all_results.append(results)
    
    # Print results
    logger.info("\n" + "="*70)
    logger.info("EVALUATION RESULTS")
    logger.info("="*70)
    
    if not all_results:
        logger.error("No models evaluated successfully")
        return
    
    logger.info(f"\n{'Model':<25} {'IoU':<12} {'Dice':<12} {'Boundary Acc':<15} {'Samples':<10}")
    logger.info("-" * 70)
    
    for result in all_results:
        logger.info(
            f"{result['model']:<25} "
            f"{result['mean_iou']:.4f}±{result['std_iou']:.4f}  "
            f"{result['mean_dice']:.4f}±{result['std_dice']:.4f}  "
            f"{result['mean_boundary_acc']:.4f}±{result['std_boundary_acc']:.4f}  "
            f"{result['num_samples']:<10}"
        )
    
    # Find best model
    best_result = max(all_results, key=lambda x: x["mean_iou"])
    logger.info("\n" + "="*70)
    logger.info(f"✓ BEST MODEL (by IoU): {best_result['model']}")
    logger.info(f"  Mean IoU: {best_result['mean_iou']:.4f}")
    logger.info(f"  Mean Dice: {best_result['mean_dice']:.4f}")
    logger.info(f"  Mean Boundary Accuracy: {best_result['mean_boundary_acc']:.4f}")
    logger.info("="*70)
    
    # Save results to file
    results_file = base_dir / "runs" / "evaluation_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
