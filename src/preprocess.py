import os
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Import modules from src
from dataset_loader import HAM10000Loader
from label_encoder import LesionLabelEncoder
from splitter import GroupSplitter
from image_processor import ImageProcessor
from class_weights import ClassWeightCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Coordinates metadata cleaning, category encoding, dataset splitting,
    class weight calculations, and audit logging.
    """
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.processed_dir = self.base_dir / "processed"
        self.reports_dir = self.base_dir / "reports"
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def execute(self) -> None:
        logger.info("Initializing Sprint 3 Preprocessing Pipeline...")
        
        # 1. Dataset Loading
        metadata_csv = self.data_dir / "HAM10000_metadata.csv"
        image_dirs = [self.data_dir / "HAM10000_images_part_1", self.data_dir / "HAM10000_images_part_2"]
        
        # Verify CSV path
        if not metadata_csv.exists():
            error_msg = f"Cannot start preprocessing: metadata CSV not found at {metadata_csv.resolve()}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        loader = HAM10000Loader(metadata_csv, image_dirs)
        df, missing_imgs, corrupted_imgs = loader.verify_and_map()
        
        # 2. Metadata Cleaning
        cleaned_df = df.copy()
        
        # Fill missing age with median age dynamically
        median_age = cleaned_df['age'].median()
        if pd.isna(median_age):
            median_age = 50.0  # Fallback
            
        missing_age_count = cleaned_df['age'].isnull().sum()
        cleaned_df['age'] = cleaned_df['age'].fillna(median_age)
        logger.info(f"Imputed {missing_age_count} missing age values using median age: {median_age:.1f}")
        
        # Validate critical fields
        cleaned_df = cleaned_df.dropna(subset=['lesion_id', 'image_id', 'dx'])
        dropped_count = len(df) - len(cleaned_df)
        if dropped_count > 0:
            logger.warning(f"Removed {dropped_count} invalid records due to null lesion_id, image_id, or dx.")
            
        # 3. Label Encoding
        encoder = LesionLabelEncoder()
        cleaned_df['label'] = encoder.transform(cleaned_df['dx'])
        
        # Save Label Encoder Map
        encoder.save_mapping(self.processed_dir / "label_encoder.json")
        
        # 4. Image Preprocessing Audit
        processor = ImageProcessor()
        # Audit only valid paths currently mapped on disk
        valid_paths = [Path(p) for p in cleaned_df['image_path'].dropna()]
        image_audit_stats = processor.audit_images(valid_paths)
        
        # 5. Group Splitting (No Leakage)
        splitter = GroupSplitter()
        train_df, val_df, test_df = splitter.split(cleaned_df)
        
        # Validate splits for leakage
        valid_split, split_msg = splitter.validate_splits(train_df, val_df, test_df)
        
        # 6. Class Imbalance (Weights calculations on training set only)
        weight_calculator = ClassWeightCalculator()
        class_weights = weight_calculator.calculate_weights(train_df['label'])
        weight_calculator.save_weights(class_weights, self.processed_dir / "class_weights.json")
        
        # 7. Save split data CSVs
        # Keep only clean fields for deep learning training
        cols_to_save = ['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization', 'image_path', 'label']
        train_df[cols_to_save].to_csv(self.processed_dir / "train.csv", index=False)
        val_df[cols_to_save].to_csv(self.processed_dir / "val.csv", index=False)
        test_df[cols_to_save].to_csv(self.processed_dir / "test.csv", index=False)
        logger.info("Saved train.csv, val.csv, and test.csv to processed directory.")
        
        # 8. Report Generation
        self._write_report(
            total_records=len(df),
            cleaned_records=len(cleaned_df),
            missing_age_count=missing_age_count,
            median_age=median_age,
            missing_imgs_count=len(missing_imgs),
            corrupted_imgs_count=len(corrupted_imgs),
            image_audit=image_audit_stats,
            train_count=len(train_df),
            val_count=len(val_df),
            test_count=len(test_df),
            class_weights=class_weights,
            encoder_mapping=encoder.mapping,
            valid_split=valid_split,
            split_msg=split_msg,
            train_class_counts=train_df['dx'].value_counts().to_dict(),
            train_class_pcts=train_df['dx'].value_counts(normalize=True).to_dict()
        )
        
    def _write_report(
        self, total_records: int, cleaned_records: int, missing_age_count: int, median_age: float,
        missing_imgs_count: int, corrupted_imgs_count: int, image_audit: dict,
        train_count: int, val_count: int, test_count: int, class_weights: dict,
        encoder_mapping: dict, valid_split: bool, split_msg: str,
        train_class_counts: dict, train_class_pcts: dict
    ) -> None:
        
        report_path = self.reports_dir / "preprocessing_report.md"
        logger.info(f"Writing preprocessing audit report to {report_path.resolve()}...")
        
        # Map encoder mapping to labels for table display
        encoder_rows = ""
        for name, label_val in encoder_mapping.items():
            encoder_rows += f"| `{name}` | {label_val} |\n"
            
        # Class distribution and weights on Train Set
        weights_rows = ""
        for code, label_val in encoder_mapping.items():
            weight_val = class_weights.get(str(label_val), 0.0)
            count = train_class_counts.get(code, 0)
            pct = train_class_pcts.get(code, 0.0) * 100
            weights_rows += f"| `{code}` | {label_val} | {count} | {pct:.2f}% | {weight_val:.5f} |\n"
            
        content = f"""# HAM10000 Data Preprocessing Report (Sprint 3)

This report logs the cleaning, encoding, splitting, and weights computation audit for the HAM10000 dataset preprocessing.

---

## 1. Metadata Cleaning & Imputation
- **Total Input Records**: {total_records}
- **Successfully Mapped Records**: {cleaned_records}
- **Missing Age Fields Imputed**: {missing_age_count} values filled with median age of **{median_age:.1f} years**.
- **Records Removed (Invalid Fields)**: {total_records - cleaned_records}

---

## 2. Image Quality & Processing Audit
- **Images Located & Mapped**: {image_audit['total_audited']} files
- **Image Integrity Checks**:
  - **Valid RGB Images**: {image_audit['valid_count']} files
  - **Corrupted / Blank Files**: {image_audit['corrupted_count']} files
  - **Resolution Range Checked**: All read images resized to **224x224** pixels.
  - **Pixel Intensity Normalization**: Mapped to **[0.0, 1.0]** scale.
  - **Missing Files Registered**: {missing_imgs_count} records currently missing corresponding images on disk.

---

## 3. Label Encoding
Skin lesion diagnostic string keys mapped to target classification integers:

| Lesion Key | Numerical Label |
| :---: | :---: |
{encoder_rows}
*Note: Config saved to [label_encoder.json](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/processed/label_encoder.json).*

---

## 4. Patient Leakage & Split Verification
Split ratios targeted: **70% Train / 15% Val / 15% Test** using a nested `GroupShuffleSplit` on `lesion_id`.

### Split Results
- **Train Set**: {train_count} records ({train_count/cleaned_records*100:.1f}%)
- **Validation Set**: {val_count} records ({val_count/cleaned_records*100:.1f}%)
- **Test Set**: {test_count} records ({test_count/cleaned_records*100:.1f}%)

### Patient Leakage Assessment
- **Status**: { '✅ Passed' if valid_split else '❌ Failed' }
- **Verification Details**: {split_msg}
*This prevents identical patient lesions from appearing in both train and validation sets, ensuring split reliability.*

---

## 5. Class Distributions & Penalization Weights (Training Set)
Weights calculated using standard inverse-frequency balancing on the training set to prevent model bias towards Melanocytic Nevi (`nv`).

| Class Code | Label | Train Count | Proportion (%) | Balanced Weight |
| :---: | :---: | :---: | :---: | :---: |
{weights_rows}
*Note: Config saved to [class_weights.json](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/processed/class_weights.json).*
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Preprocessing report generated successfully.")

def main():
    base_dir = Path(".")
    pipeline = PreprocessingPipeline(base_dir)
    pipeline.execute()
    logger.info("Sprint 3 Preprocessing Pipeline executed successfully!")

if __name__ == "__main__":
    main()
