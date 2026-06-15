import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

# Import modules from src
from dataset_loader import HAM10000Loader
from visualization import plot_class_distribution, plot_metadata_analysis, plot_sample_images
from demo_setup import setup_demo_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CLASS_MAP = {
    'nv': 'Melanocytic Nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratoses',
    'vasc': 'Vascular Lesions',
    'df': 'Dermatofibroma'
}

class HAM10000Analyzer:
    """
    Orchestrates the statistical analysis, charts creation, 
    and detailed markdown report generation for the HAM10000 dataset.
    """
    def __init__(self, df: pd.DataFrame, missing_count: int, corrupted_count: int):
        self.df = df
        self.missing_count = missing_count
        self.corrupted_count = corrupted_count
        self.stats = {}

    def run_analysis(self):
        """
        Executes various numerical and categorical statistics.
        """
        logger.info("Starting dataset statistical analysis...")
        
        # 1. Basic Dimensions
        total_records = len(self.df)
        valid_paths = self.df['image_path'].dropna()
        total_images = len(valid_paths)
        
        # 2. Duplicate Detection
        # Lesion IDs can occur multiple times (multiple views of same mole)
        unique_lesions = self.df['lesion_id'].nunique() if 'lesion_id' in self.df.columns else 0
        duplicate_lesion_records = total_records - unique_lesions
        
        # 3. Class breakdown
        class_counts = self.df['dx'].value_counts()
        class_pcts = self.df['dx'].value_counts(normalize=True) * 100
        
        # 4. Missing metadata
        missing_metadata = self.df.isnull().sum().to_dict()
        
        # 5. Image Quality (Resolution, Aspect Ratio, Channels)
        widths, heights, channels = [], [], []
        
        logger.info("Analyzing image resolutions and channels (sampling image paths)...")
        # Load actual dimensions
        for path in valid_paths:
            try:
                # Read metadata without loading full pixels for speed if possible,
                # but cv2.imread is reliable for mock and real sizes
                img = cv2.imread(path)
                if img is not None:
                    h, w, c = img.shape
                    widths.append(w)
                    heights.append(h)
                    channels.append(c)
            except Exception as e:
                logger.error(f"Failed to read image at {path}: {str(e)}")
                
        # Aggregate image metrics
        if widths:
            min_w, max_w, mean_w = min(widths), max(widths), np.mean(widths)
            min_h, max_h, mean_h = min(heights), max(heights), np.mean(heights)
            unique_resolutions = set(zip(widths, heights))
            channel_counts = pd.Series(channels).value_counts().to_dict()
        else:
            min_w, max_w, mean_w = 0, 0, 0.0
            min_h, max_h, mean_h = 0, 0, 0.0
            unique_resolutions = set()
            channel_counts = {}

        # 6. Demographics
        mean_age = self.df['age'].mean() if 'age' in self.df.columns else 0.0
        median_age = self.df['age'].median() if 'age' in self.df.columns else 0.0
        gender_counts = self.df['sex'].value_counts().to_dict() if 'sex' in self.df.columns else {}
        site_counts = self.df['localization'].value_counts().to_dict() if 'localization' in self.df.columns else {}
        
        # Compile all stats
        self.stats = {
            'total_records': total_records,
            'total_images': total_images,
            'unique_lesions': unique_lesions,
            'duplicate_lesions': duplicate_lesion_records,
            'class_counts': class_counts,
            'class_pcts': class_pcts,
            'missing_metadata': missing_metadata,
            'image_dims': {
                'min_width': min_w,
                'max_width': max_w,
                'mean_width': mean_w,
                'min_height': min_h,
                'max_height': max_h,
                'mean_height': mean_h,
                'unique_resolutions': unique_resolutions,
                'channels': channel_counts
            },
            'demographics': {
                'mean_age': mean_age,
                'median_age': median_age,
                'gender_counts': gender_counts,
                'site_counts': site_counts
            }
        }
        logger.info("Analysis complete.")
        return self.stats

    def generate_dataset_summary(self, filepath: Path):
        """
        Writes the dataset_summary.md file.
        """
        logger.info(f"Writing dataset summary report to {filepath}...")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Identify majority / minority classes
        class_counts = self.stats['class_counts']
        majority_class = class_counts.index[0]
        minority_class = class_counts.index[-1]
        
        content = f"""# HAM10000 Skin Cancer Dataset Summary

This document provides a high-level administrative and structural overview of the HAM10000 dataset for Sprint 2.

## 📋 General File Status Checklist

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Total CSV Metadata Rows** | {self.stats['total_records']} | Verified |
| **Total Images Located on Disk** | {self.stats['total_images'] + self.corrupted_count} | Mapped |
| **Successfully Loaded Images** | {self.stats['total_images']} | Healthy |
| **Missing Image Files** | {self.missing_count} | { '⚠️ Warning: Missing files found' if self.missing_count > 0 else 'Passed' } |
| **Corrupted Image Files** | {self.corrupted_count} | { '❌ Error: Corrupted files found' if self.corrupted_count > 0 else 'Passed' } |

---

## 🔬 Dataset Core Metrics

- **Unique Lesions (Patient Cases)**: {self.stats['unique_lesions']}
- **Duplicate Lesion Records**: {self.stats['duplicate_lesions']} *(Multiple dermoscopic captures of the same lesion)*
- **Lesion Classes Identified**: {len(class_counts)} classes
- **Image Resolution Range**: {self.stats['image_dims']['min_width']}x{self.stats['image_dims']['min_height']} to {self.stats['image_dims']['max_width']}x{self.stats['image_dims']['max_height']}
- **Average Dimensions**: {self.stats['image_dims']['mean_width']:.1f} x {self.stats['image_dims']['mean_height']:.1f}
- **Image Color Channels**: {', '.join(f"{ch}-channel (RGB)" if ch==3 else f"{ch}-channel" for ch in self.stats['image_dims']['channels'].keys())}

---

## ⚖️ Imbalance Highlight

- **Majority Class**: `{majority_class}` ({CLASS_MAP.get(majority_class, majority_class)}) &mdash; **{self.stats['class_counts'].iloc[0]} images** ({self.stats['class_pcts'].iloc[0]:.1f}%)
- **Minority Class**: `{minority_class}` ({CLASS_MAP.get(minority_class, minority_class)}) &mdash; **{self.stats['class_counts'].iloc[-1]} images** ({self.stats['class_pcts'].iloc[-1]:.1f}%)
- **Imbalance Ratio**: 1 : { (class_counts.iloc[0] / class_counts.iloc[-1]):.1f}

### Potential Impact on Deep Learning Models
The severe class imbalance presents a significant risk for deep neural network training. Models will easily overfit to the majority class (**{CLASS_MAP.get(majority_class, majority_class)}**) and may achieve high overall accuracy while failing completely to recognize rarer, high-risk malignancies like **Melanoma (`mel`)**. Mitigation strategies such as class-weighted loss, SMOTE, or focal loss will be required in Sprint 3.
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Dataset summary report generated successfully.")

    def generate_eda_report(self, filepath: Path):
        """
        Writes the comprehensive eda_report.md file.
        """
        logger.info(f"Writing EDA report to {filepath}...")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata null values markdown
        missing_md = self.stats['missing_metadata']
        missing_md_rows = ""
        for field, count in missing_md.items():
            pct = (count / self.stats['total_records'] * 100) if self.stats['total_records'] > 0 else 0.0
            missing_md_rows += f"| `{field}` | {count} | {pct:.1f}% |\n"

        # Prepare class imbalance markdown table
        class_table_rows = ""
        for idx in range(len(self.stats['class_counts'])):
            cls = self.stats['class_counts'].index[idx]
            count = self.stats['class_counts'].values[idx]
            pct = self.stats['class_pcts'].values[idx]
            name = CLASS_MAP.get(cls, cls)
            class_table_rows += f"| `{cls}` | {name} | {count} | {pct:.2f}% |\n"

        # Demographic summary calculations
        mean_age = self.stats['demographics']['mean_age']
        median_age = self.stats['demographics']['median_age']
        gender_counts = self.stats['demographics']['gender_counts']
        top_site = list(self.stats['demographics']['site_counts'].keys())[0]
        top_site_count = list(self.stats['demographics']['site_counts'].values())[0]

        content = f"""# HAM10000 Exploratory Data Analysis (EDA) Report

This comprehensive report details the distribution of lesion categories, patient demographics, image quality metrics, and pre-processing recommendations.

---

## 1. Metadata Quality & Completeness
An analysis of the metadata values was conducted to flag duplicate cases, patient re-occurrences, and missing attributes.

### Missing Fields Report
| Metadata Column | Missing Records | Percentage |
| :--- | :--- | :--- |
{missing_md_rows}
*Note: Missing patient age values must be handled during preprocessing (e.g., mean/median imputation).*

### Patient Duplicate Records
- **Total Lesion Records**: {self.stats['total_records']}
- **Unique Mapped Lesions**: {self.stats['unique_lesions']}
- **Duplicate lesion occurrences**: {self.stats['duplicate_lesions']}
- **Implication**: Some lesions have multiple dermoscopic images. In splitting the training/validation/testing sets, **grouped splits based on `lesion_id` are mandatory**. Failing to do so will leak identical patient lesions into both training and validation sets, inflating validation accuracy artificially.

---

## 2. Class Distribution Analysis
The dataset contains a severe class imbalance across 7 structural categories.

| Class Code | Disease Category | Count | Proportion (%) |
| :---: | :--- | :---: | :---: |
{class_table_rows}
### Risk Category Assessment
- **Malignant / High-Risk**: Melanoma (`mel`, {self.stats['class_pcts'].get('mel', 0.0):.1f}%) and Basal Cell Carcinoma (`bcc`, {self.stats['class_pcts'].get('bcc', 0.0):.1f}%) represent critical cases where false negatives are highly dangerous.
- **Benign**: Melanocytic Nevi (`nv`, {self.stats['class_pcts'].get('nv', 0.0):.1f}%) dominates the dataset, meaning the model's natural default baseline would guess "Nevus" for all inputs.

---

## 3. Demographic & Localization Analysis
A summary of patient age, gender, and anatomical distributions:

- **Patient Age**: Mean of **{mean_age:.1f} years** (Median: {median_age:.1f}). Shows typical skewed distribution matching adult clinical dermatological visits.
- **Gender Balance**: {', '.join(f"{g.capitalize()}: {c}" for g, c in gender_counts.items())}
- **Primary Anatomical Location**: **{top_site}** represents the most common location ({top_site_count} lesions).

---

## 4. Image Quality & Resolution Verification
- **Unique Resolutions found**: {', '.join(f"{w}x{h}" for w, h in self.stats['image_dims']['unique_resolutions'])}
- **Channel Check**: Verified {self.stats['total_images']} images possess **{list(self.stats['image_dims']['channels'].keys())[0]} channels (RGB)**.
- **Corrupted Files Isolated**: {self.corrupted_count} files failed safety verification.

---

## 5. Recommended Preprocessing Pipeline (Sprint 3)
Based on these findings, we recommend implementing the following preprocessing steps prior to model training:

1. **Grouped Train/Val/Test Split**: Split data at the `lesion_id` level (using `GroupKFold` or `GroupShuffleSplit`) to prevent patient-level leakage.
2. **Missing Age Imputation**: Impute missing `age` fields with the dataset median value.
3. **Resolution Standardization**: Resize all images to a uniform square size (e.g., `224x224` or `299x299` pixels) to match standard deep neural networks (ResNet, EfficientNet) input layers.
4. **Contrast Normalization**: Apply color normalization (e.g., mean subtraction, scale normalization) to handle lighting variation across dermoscopic imaging devices.
5. **Class Imbalance Mitigation**:
   - Apply random rotations, shifts, zooms, and horizontal/vertical flips to minority classes during training.
   - Use a weighted loss function (e.g., `Weighted Cross Entropy`) or `Focal Loss` to penalize minority misclassifications more heavily.
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("EDA report generated successfully.")

def main():
    """
    Main runner script for Sprint 2 EDA pipeline.
    """
    base_dir = Path(".")
    data_dir = base_dir / "data"
    metadata_csv = data_dir / "HAM10000_metadata.csv"
    
    # Check if dataset files exist, otherwise trigger the synthetic demo setup
    if not metadata_csv.exists():
        logger.warning(f"Metadata CSV not found at: {metadata_csv.resolve()}")
        logger.warning("Initializing synthetic demo dataset for pipeline execution verification...")
        setup_demo_data(base_dir)
        
    # Set up directories
    image_dirs = [data_dir / "HAM10000_images_part_1", data_dir / "HAM10000_images_part_2"]
    reports_dir = base_dir / "reports"
    vis_dir = base_dir / "visualizations"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    loader = HAM10000Loader(metadata_csv, image_dirs)
    df, missing_imgs, corrupted_imgs = loader.verify_and_map()
    
    # Check for unmapped images
    loader.get_unmapped_images()
    
    # 2. Analyze
    analyzer = HAM10000Analyzer(df, len(missing_imgs), len(corrupted_imgs))
    analyzer.run_analysis()
    
    # 3. Generate Reports
    analyzer.generate_dataset_summary(reports_dir / "dataset_summary.md")
    analyzer.generate_eda_report(reports_dir / "eda_report.md")
    
    # 4. Generate Visualizations
    plot_class_distribution(df, CLASS_MAP, vis_dir / "class_distribution.png")
    plot_metadata_analysis(df, vis_dir / "metadata_analysis.png")
    plot_sample_images(df, CLASS_MAP, vis_dir / "sample_images.png")
    
    logger.info("Sprint 2 dataset analysis and EDA pipeline executed successfully!")

if __name__ == "__main__":
    main()
