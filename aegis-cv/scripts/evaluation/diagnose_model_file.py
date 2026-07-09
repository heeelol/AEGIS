"""Verify model file and diagnose save/load issues."""

import logging
from pathlib import Path

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "models" / "custom" / "bin_segmentation_training_5_maskrcnn.pth"
    
    logger.info("="*70)
    logger.info("MODEL FILE DIAGNOSTICS")
    logger.info("="*70 + "\n")
    
    # Check file exists and size
    if not model_path.exists():
        logger.error(f"❌ Model file not found: {model_path}")
        return
    
    file_size = model_path.stat().st_size
    logger.info(f"✓ Model file found: {model_path}")
    logger.info(f"✓ File size: {file_size / 1024 / 1024:.1f} MB\n")
    
    # Load raw state dict
    logger.info("Loading raw state dict...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    
    logger.info(f"✓ State dict loaded")
    logger.info(f"  Type: {type(state_dict)}")
    logger.info(f"  Keys: {len(state_dict)}\n")
    
    # Analyze state dict
    logger.info("State dict analysis:")
    
    total_params = 0
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            num_params = param.numel()
            total_params += num_params
            
            if num_params > 100000:  # Only show large layers
                logger.info(f"  {name}: {num_params:,} params")
    
    logger.info(f"\nTotal parameters in saved file: {total_params:,}\n")
    
    if total_params < 1000000:
        logger.error("❌ ERROR: File only contains ~{:,} parameters!".format(total_params))
        logger.error("   This is NOT the properly trained model")
        logger.error("   Something went wrong during saving")
        return
    
    # Try loading into model
    logger.info("Attempting to load into fresh model...")
    
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Modify for binary classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, 2)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, 2 * 4)
    
    try:
        model.load_state_dict(state_dict)
        logger.info("✓ State dict loaded successfully\n")
    except Exception as e:
        logger.error(f"❌ Failed to load: {e}\n")
        return
    
    # Count model parameters
    model_total = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model total parameters: {model_total:,}")
    logger.info(f"✓ Saved state dict parameters: {total_params:,}\n")
    
    if total_params > 40000000:
        logger.info("="*70)
        logger.info("✓ SUCCESS: Model file is correct!")
        logger.info("  The properly trained model was saved.")
        logger.info("  Try running evaluate_models.py again")
        logger.info("="*70)
    else:
        logger.error("="*70)
        logger.error("❌ PROBLEM: Model file only has ~{:,} params".format(total_params))
        logger.error("  Need to retrain with train_training_5_maskrcnn_proper.py")
        logger.error("="*70)


if __name__ == "__main__":
    main()
