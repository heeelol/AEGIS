"""Debug Mask R-CNN model - diagnose why it's getting 0% accuracy."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Debug Mask R-CNN model on single test image."""
    
    base_dir = Path(__file__).parent.parent.parent
    
    logger.info("="*70)
    logger.info("MASK R-CNN DEBUGGING")
    logger.info("="*70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nDevice: {device}\n")
    
    # Load model
    logger.info("Loading pre-trained Mask R-CNN R-50-FPN...")
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Modify for binary classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, 2)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, 2 * 4)
    
    logger.info("✓ Model created\n")
    
    # Load trained weights
    model_path = base_dir / "models" / "custom" / "bin_segmentation_training_5_maskrcnn.pth"
    
    logger.info(f"Loading model weights from: {model_path}")
    
    if not model_path.exists():
        logger.error(f"❌ Model file not found: {model_path}")
        logger.info("\nTip: Make sure to run train_training_5_maskrcnn.py first!")
        return
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Count actual parameters, not just dict keys
        total_params = sum(p.numel() for k, p in state_dict.items() if isinstance(p, torch.Tensor))
        logger.info(f"✓ Loaded state dict with {total_params:,} parameters ({len(state_dict)} keys)")
        
        model.load_state_dict(state_dict)
        logger.info("✓ State dict loaded successfully\n")
    except Exception as e:
        logger.error(f"❌ Failed to load state dict: {e}\n")
        return
    
    model.eval()
    model.to(device)
    logger.info("✓ Model set to eval mode\n")
    
    # Get a test image
    test_dir = base_dir / "models" / "data" / "Training_5.yolov8-seg" / "images" / "test"
    test_images = sorted(test_dir.glob("*.[jp][pn]g"))
    
    if not test_images:
        logger.error(f"❌ No test images found in {test_dir}")
        return
    
    test_img_path = test_images[0]
    logger.info(f"Testing on: {test_img_path.name}")
    
    # Load image
    image = cv2.imread(str(test_img_path))
    if image is None:
        logger.error(f"❌ Failed to load image")
        return
    
    height, width = image.shape[:2]
    logger.info(f"Image size: {width}x{height}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor. Scale to [0,1] to match T.ToTensor() used in training;
    # torchvision's Mask R-CNN normalizes with ImageNet mean/std assuming [0,1]
    # input. Feeding [0,255] yields 0 detections.
    image_tensor = (torch.from_numpy(image_rgb).float() / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    logger.info(f"Image tensor shape: {image_tensor.shape}\n")
    
    # Predict
    logger.info("Running inference...")
    
    with torch.no_grad():
        predictions = model([image_tensor.squeeze(0)])
    
    logger.info(f"✓ Inference complete\n")
    
    # Analyze predictions
    logger.info("Analyzing predictions:")
    logger.info(f"  Number of predictions: {len(predictions)}")
    
    pred = predictions[0]
    
    logger.info(f"  Keys in prediction dict: {list(pred.keys())}")
    
    if "boxes" in pred:
        logger.info(f"  Number of boxes: {len(pred['boxes'])}")
        if len(pred['boxes']) > 0:
            logger.info(f"  Box scores: {pred['scores'][:5]}")  # First 5
    
    if "masks" in pred:
        masks = pred["masks"]
        logger.info(f"  Masks shape: {masks.shape}")
        logger.info(f"  Number of masks: {len(masks)}")
        
        if len(masks) > 0:
            logger.info(f"  First mask min/max: {masks[0].min():.4f} / {masks[0].max():.4f}")
            logger.info(f"  First mask sum: {masks[0].sum():.0f} pixels\n")
            
            # Try to extract mask
            pred_mask = np.zeros((height, width), dtype=np.uint8)
            
            for i, mask in enumerate(masks):
                mask_np = mask[0].cpu().numpy()
                if mask_np.shape != (height, width):
                    logger.info(f"  ⚠️ Mask {i} shape mismatch: {mask_np.shape} vs expected ({height}, {width})")
                    mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
                    mask_np = mask_resized
                
                pred_mask_i = (mask_np > 0.5).astype(np.uint8)
                pred_mask = np.logical_or(pred_mask, pred_mask_i).astype(np.uint8)
                logger.info(f"  Mask {i}: {pred_mask_i.sum()} pixels after threshold")
            
            logger.info(f"\nFinal extracted mask: {pred_mask.sum()} pixels")
            
            if pred_mask.sum() == 0:
                logger.warning("  ⚠️ WARNING: Final mask is empty!")
                logger.info("  This is why Boundary Accuracy = 0")
                logger.info("\n  Debugging threshold:")
                for thresh in [0.3, 0.5, 0.7, 0.9]:
                    mask_test = (masks[0].cpu().numpy() > thresh).astype(np.uint8)
                    logger.info(f"    Threshold {thresh}: {mask_test.sum()} pixels")
            else:
                logger.info("  ✓ Mask extracted successfully")
        else:
            logger.warning("  ⚠️ WARNING: No masks in predictions!")
    else:
        logger.warning("  ⚠️ WARNING: No 'masks' key in predictions!")
    
    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS:")
    logger.info("="*70)
    
    if "masks" not in pred or len(pred.get("masks", [])) == 0:
        logger.info("1. The model is not generating masks")
        logger.info("2. Check if training actually used the mask head")
        logger.info("3. May need to retrain with proper mask supervision")
    elif pred_mask.sum() == 0:
        logger.info("1. Masks are being generated but all zeros or very low confidence")
        logger.info("2. Try lowering threshold below 0.5")
        logger.info("3. Check if model was properly trained with good data")
    else:
        logger.info("1. Model is working! Check evaluation script for other bugs")


if __name__ == "__main__":
    main()
