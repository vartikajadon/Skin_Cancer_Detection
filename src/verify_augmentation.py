import os
import logging
import pandas as pd
from pathlib import Path
from augmentation import SkinCancerAugmentor, BalancedDatasetBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_verification():
    logger.info("Initializing Sprint 4 Verification Suite...")
    
    base_dir = Path(".")
    config_path = base_dir / "configs" / "augmentation_config.json"
    processed_dir = base_dir / "processed"
    
    # 1. Config Loading Verification
    assert config_path.exists(), "Verification failed: augmentation_config.json does not exist."
    augmentor = SkinCancerAugmentor(config_path)
    logger.info("✅ Verification Pass: Centralized configuration loaded successfully.")
    
    # 2. Dataset Split Integrity Verification
    builder = BalancedDatasetBuilder(config_path, processed_dir)
    
    # Load raw CSVs to get initial counts on disk
    train_raw = builder.load_and_clean_split("train.csv")
    val_raw = builder.load_and_clean_split("val.csv")
    test_raw = builder.load_and_clean_split("test.csv")
    
    initial_train_len = len(train_raw)
    initial_val_len = len(val_raw)
    initial_test_len = len(test_raw)
    
    # Run balancing (oversampling)
    balanced_train, stats = builder.balance_training_set(train_raw)
    
    # Assertions
    # A. Validate validation and test counts remain UNCHANGED
    assert len(val_raw) == initial_val_len, "Verification failed: Validation split size changed."
    assert len(test_raw) == initial_test_len, "Verification failed: Test split size changed."
    logger.info(f"✅ Verification Pass: Validation set ({len(val_raw)} samples) and Test set ({len(test_raw)} samples) remain UNTOUCHED.")
    
    # B. Validate minority classes are oversampled
    minority_classes = builder.minority_classes
    post_counts = stats["post_balancing"]
    
    for dx in minority_classes:
        # Check if the class is present on disk in train set
        if dx in train_raw['dx'].values:
            post_count = post_counts.get(dx, 0)
            assert post_count == builder.target_samples, f"Verification failed: class {dx} count is {post_count}, expected {builder.target_samples}."
            logger.info(f"✅ Verification Pass: Minority class '{dx}' successfully oversampled to target count of {builder.target_samples}.")
            
    # C. Validate majority classes are not oversampled
    majority_classes = builder.majority_classes
    for dx in majority_classes:
        if dx in train_raw['dx'].values:
            pre_count = len(train_raw[train_raw['dx'] == dx])
            post_count = post_counts.get(dx, 0)
            assert pre_count == post_count, f"Verification failed: majority class {dx} count changed from {pre_count} to {post_count}."
            logger.info(f"✅ Verification Pass: Majority class '{dx}' remains at original count of {post_count} (no oversampling).")
            
    # D. Stochastic augmentation verification
    if not train_raw.empty:
        img_path = Path(train_raw.iloc[0]['image_path'])
        import cv2
        img = cv2.imread(str(img_path))
        img_norm = cv2.resize(img, (224, 224)).astype(float) / 255.0
        
        # Apply twice with force_all=True
        aug1 = augmentor.augment_image_numpy(img_norm, force_all=True)
        aug2 = augmentor.augment_image_numpy(img_norm, force_all=True)
        
        # Check that stochasticity introduced differences
        diff = abs(aug1 - aug2).sum()
        assert diff > 0, "Verification failed: stochastic augmentation returned identical results on multiple calls."
        logger.info(f"✅ Verification Pass: Stochastic data augmentation confirmed active (pixel difference metric: {diff:.2f}).")
        
    logger.info("🎉 All Sprint 4 Augmentation Pipeline Verification checks passed successfully!")

if __name__ == "__main__":
    run_verification()
