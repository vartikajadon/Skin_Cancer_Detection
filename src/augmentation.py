import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Try to import TensorFlow for integration verification
try:
    import tensorflow as tf
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    layers = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SkinCancerAugmentor:
    """
    Standard TensorFlow/Keras-compatible image augmentation pipeline.
    Includes a robust NumPy/OpenCV fallback for local execution on systems 
    without TensorFlow installed.
    """
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Load parameters
        self.rotation_range = self.config.get("rotation_range", 0.0556) # fraction of 360 (0.0556 ≈ 20°)
        self.zoom_range = self.config.get("zoom_range", 0.15)
        self.brightness_range = self.config.get("brightness_range", 0.15)
        self.contrast_range = self.config.get("contrast_range", 0.15)
        self.width_shift_range = self.config.get("width_shift_range", 0.1)
        self.height_shift_range = self.config.get("height_shift_range", 0.1)
        self.fill_mode = self.config.get("fill_mode", "reflect")
        self.crop_padding = self.config.get("crop_padding", 16)
        self.apply_probability = self.config.get("apply_probability", 0.5)
        
        # Log status
        if TENSORFLOW_AVAILABLE:
            logger.info("TensorFlow/Keras is available. Augmentation layers initialized.")
            self.tf_augmentor = self._build_tf_augmentor()
        else:
            logger.warning("TensorFlow/Keras is NOT available in this environment. "
                           "Using NumPy/OpenCV fallback for local processing.")
            self.tf_augmentor = None
            
    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Augmentation config file not found at: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def _build_tf_augmentor(self):
        """
        Builds a Keras Sequential Model for image augmentation.
        Runs fully on GPU/CPU and handles border pixels using reflection.
        """
        # Map JSON fill mode to Keras fill mode
        # Keras supports: 'reflect', 'wrap', 'nearest', 'constant'
        keras_fill_mode = self.fill_mode if self.fill_mode in ["reflect", "wrap", "nearest", "constant"] else "reflect"
        
        # Convert translation factors for Keras RandomTranslation
        # Keras expects height/width factors as fraction of height/width
        model = tf.keras.Sequential([
            # 1. Flips (Horizontal & Vertical)
            layers.RandomFlip("horizontal_and_vertical"),
            
            # 2. Rotation (-20 to +20 degrees)
            layers.RandomRotation(factor=(-self.rotation_range, self.rotation_range), fill_mode=keras_fill_mode),
            
            # 3. Translation / Shifts
            layers.RandomTranslation(
                height_factor=(-self.height_shift_range, self.height_shift_range),
                width_factor=(-self.width_shift_range, self.width_shift_range),
                fill_mode=keras_fill_mode
            ),
            
            # 4. Zoom (In and Out)
            layers.RandomZoom(
                height_factor=(-self.zoom_range, self.zoom_range),
                width_factor=(-self.zoom_range, self.zoom_range),
                fill_mode=keras_fill_mode
            ),
            
            # 5. Brightness Adjustment (Lambda layer using tf.image to prevent version mismatch)
            layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=self.brightness_range)),
            
            # 6. Contrast Adjustment
            layers.Lambda(lambda x: tf.image.random_contrast(x, lower=1.0 - self.contrast_range, upper=1.0 + self.contrast_range)),
            
            # 7. Random Crop (Pad first with reflection to maintain size, then crop to target 224x224)
            layers.Lambda(lambda x: tf.pad(
                x, 
                [[0, 0], [self.crop_padding, self.crop_padding], [self.crop_padding, self.crop_padding], [0, 0]], 
                mode='REFLECT'
            )),
            layers.RandomCrop(224, 224)
        ], name="skin_cancer_augmentor")
        
        return model

    # ==========================================
    # NumPy/OpenCV Fallback Implementations
    # ==========================================
    
    def augment_image_numpy(self, img: np.ndarray, force_all: bool = False) -> np.ndarray:
        """
        Stochastically applies data augmentation to a single NumPy image (shape: 224x224x3, range [0, 1]).
        Used when TensorFlow is missing, or for explicit visualization.
        """
        augmented = img.copy()
        
        # Define stochastic application probability
        p = 1.0 if force_all else self.apply_probability
        
        # 1. Horizontal Flip
        if np.random.random() < p:
            augmented = cv2.flip(augmented, 1) # 1 = horizontal
            
        # 2. Vertical Flip
        if np.random.random() < p:
            augmented = cv2.flip(augmented, 0) # 0 = vertical
            
        # 3. Rotation
        if np.random.random() < p:
            angle = np.random.uniform(-self.rotation_range * 360, self.rotation_range * 360)
            h, w = augmented.shape[:2]
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            border_mode = cv2.BORDER_REFLECT if self.fill_mode == "reflect" else cv2.BORDER_CONSTANT
            augmented = cv2.warpAffine(augmented, rot_matrix, (w, h), borderMode=border_mode)
            
        # 4. Shifts (Width & Height)
        if np.random.random() < p:
            h, w = augmented.shape[:2]
            tx = np.random.uniform(-self.width_shift_range * w, self.width_shift_range * w)
            ty = np.random.uniform(-self.height_shift_range * h, self.height_shift_range * h)
            shift_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            border_mode = cv2.BORDER_REFLECT if self.fill_mode == "reflect" else cv2.BORDER_CONSTANT
            augmented = cv2.warpAffine(augmented, shift_matrix, (w, h), borderMode=border_mode)
            
        # 5. Zoom In/Out
        if np.random.random() < p:
            h, w = augmented.shape[:2]
            zoom_factor = np.random.uniform(1.0 - self.zoom_range, 1.0 + self.zoom_range)
            # Resize image
            nh, nw = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(augmented, (nw, nh), interpolation=cv2.INTER_LINEAR)
            
            # Crop or pad back to 224x224
            if zoom_factor > 1.0:
                # Crop center
                start_y = (nh - h) // 2
                start_x = (nw - w) // 2
                augmented = resized[start_y:start_y+h, start_x:start_x+w]
            else:
                # Pad border
                pad_y = (h - nh) // 2
                pad_x = (w - nw) // 2
                border_mode = cv2.BORDER_REFLECT if self.fill_mode == "reflect" else cv2.BORDER_CONSTANT
                augmented = cv2.copyMakeBorder(resized, pad_y, h - nh - pad_y, pad_x, w - nw - pad_x, borderType=border_mode)
                
        # 6. Brightness Adjustment
        if np.random.random() < p:
            # Shift pixel values
            delta = np.random.uniform(-self.brightness_range, self.brightness_range)
            augmented = np.clip(augmented + delta, 0.0, 1.0)
            
        # 7. Contrast Adjustment
        if np.random.random() < p:
            factor = np.random.uniform(1.0 - self.contrast_range, 1.0 + self.contrast_range)
            mean = np.mean(augmented, axis=(0, 1), keepdims=True)
            augmented = np.clip(mean + factor * (augmented - mean), 0.0, 1.0)
            
        # 8. Random Crop
        if np.random.random() < p:
            h, w = augmented.shape[:2]
            # Pad first
            border_mode = cv2.BORDER_REFLECT if self.fill_mode == "reflect" else cv2.BORDER_CONSTANT
            padded = cv2.copyMakeBorder(
                augmented, 
                self.crop_padding, self.crop_padding, self.crop_padding, self.crop_padding, 
                borderType=border_mode
            )
            # Take random crop of original size
            max_y = padded.shape[0] - h
            max_x = padded.shape[1] - w
            start_y = np.random.randint(0, max_y + 1)
            start_x = np.random.randint(0, max_x + 1)
            augmented = padded[start_y:start_y+h, start_x:start_x+w]
            
        return augmented


class BalancedDatasetBuilder:
    """
    Handles CSV loading, checks for missing files, performs minority class 
    oversampling to balance distributions, and builds TF datasets.
    """
    def __init__(self, config_path: Path, processed_dir: Path):
        self.config_path = Path(config_path)
        self.processed_dir = Path(processed_dir)
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.minority_classes = self.config.get("minority_classes", ["df", "vasc", "akiec", "bcc", "mel"])
        self.majority_classes = self.config.get("majority_classes", ["nv", "bkl"])
        
        balancing_config = self.config.get("class_balancing", {})
        self.balancing_enabled = balancing_config.get("enabled", True)
        self.target_samples = balancing_config.get("target_samples_per_minority_class", 2000)
        
    def load_and_clean_split(self, file_name: str) -> pd.DataFrame:
        """
        Loads split CSV (train/val/test) and filters out records missing physical image files.
        This allows clean execution on both demo and full scale datasets.
        """
        csv_path = self.processed_dir / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Split data CSV not found at: {csv_path}")
            
        df = pd.read_csv(csv_path)
        initial_len = len(df)
        
        # Filter for rows where image_path exists and is not null
        df = df[df['image_path'].notna()]
        df = df[df['image_path'].apply(lambda p: os.path.exists(p))]
        
        logger.info(f"Loaded {file_name}: Kept {len(df)} / {initial_len} files present on disk.")
        return df.reset_index(drop=True)
        
    def balance_training_set(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Performs minority class oversampling to balance class counts.
        Returns the balanced DataFrame and statistics dictionary.
        """
        stats = {
            "pre_balancing": train_df['dx'].value_counts().to_dict(),
            "pre_total": len(train_df),
            "strategy": "Oversampling to target threshold",
            "target_samples": self.target_samples
        }
        
        if not self.balancing_enabled or len(train_df) == 0:
            logger.info("Class balancing is disabled or training set is empty. Skipping oversampling.")
            stats["post_balancing"] = stats["pre_balancing"]
            stats["post_total"] = len(train_df)
            return train_df, stats
            
        balanced_dfs = []
        
        # 1. Group by category
        for dx, group in train_df.groupby("dx"):
            current_count = len(group)
            
            if dx in self.minority_classes and current_count < self.target_samples:
                # Replicate minority class
                reps = int(np.ceil(self.target_samples / current_count))
                oversampled_group = pd.concat([group] * reps, ignore_index=True)
                # Slice down to exactly target_samples
                oversampled_group = oversampled_group.sample(n=self.target_samples, random_state=42, replace=True)
                balanced_dfs.append(oversampled_group)
                logger.info(f"Oversampled minority class '{dx}' from {current_count} to {self.target_samples} samples.")
            else:
                # Keep majority classes (or larger classes) as-is
                balanced_dfs.append(group)
                logger.info(f"Retained class '{dx}' as-is with {current_count} samples.")
                
        balanced_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        stats["post_balancing"] = balanced_df['dx'].value_counts().to_dict()
        stats["post_total"] = len(balanced_df)
        
        return balanced_df, stats
        
    def build_tf_dataset(
        self, 
        df: pd.DataFrame, 
        augmentor: Optional[SkinCancerAugmentor] = None,
        is_training: bool = False,
        batch_size: int = 32
    ) -> Optional[Any]:
        """
        Constructs a TensorFlow tf.data.Dataset pipeline.
        Only applies augmentations if is_training=True and augmentor model is provided.
        """
        if not TENSORFLOW_AVAILABLE:
            return None
            
        paths = df['image_path'].tolist()
        labels = df['label'].tolist()
        
        # Create dataset of slices
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        
        # Load and preprocess parsing function
        def _parse_fn(path, label):
            # Read file
            file_content = tf.io.read_file(path)
            # Decode JPEG image
            img = tf.image.decode_jpeg(file_content, channels=3)
            # Resize
            img = tf.image.resize(img, [224, 224])
            # Scale range to [0.0, 1.0]
            img = tf.cast(img, tf.float32) / 255.0
            return img, label
            
        dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Apply data augmentations if training
        if is_training and augmentor and augmentor.tf_augmentor:
            # Batch first because Keras preprocessing layers expect batch dimension (N, H, W, C)
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(
                lambda x, y: (augmentor.tf_augmentor(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            # Unbatch if we want to shuffle individually, but usually we just keep it batched
            # Prefetch for performance
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        else:
            # Validation / Test remain completely unaugmented
            dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
        return dataset


def generate_visualization(
    img_path: Path, 
    augmentor: SkinCancerAugmentor, 
    output_path: Path
):
    """
    Loads a single skin cancer lesion image, generates 4 specific augmented 
    variants (Rotation, Flip, Zoom, Brightness), and creates a premium 
    comparison grid visualization.
    """
    logger.info("Generating premium data augmentation visualization...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load and scale original image
    bgr_img = cv2.imread(str(img_path))
    if bgr_img is None:
        raise FileNotFoundError(f"Failed to load sample image for visualization at: {img_path}")
        
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img_resized = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_AREA)
    norm_img = rgb_img_resized.astype(np.float32) / 255.0
    
    # Generate individual variants using NumPy/CV2 for precise visual isolating
    # 1. Rotation (apply only rotation)
    rot_angle = 15.0 # degrees
    h, w = norm_img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
    rot_variant = cv2.warpAffine(norm_img, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 2. Horizontal Flip (apply only flip)
    flip_variant = cv2.flip(norm_img, 1)
    
    # 3. Zoom In (apply only zoom)
    zoom_factor = 1.2
    nh, nw = int(h * zoom_factor), int(w * zoom_factor)
    resized_z = cv2.resize(norm_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    start_y = (nh - h) // 2
    start_x = (nw - w) // 2
    zoom_variant = resized_z[start_y:start_y+h, start_x:start_x+w]
    
    # 4. Brightness (apply only brightness boost)
    brightness_variant = np.clip(norm_img + 0.15, 0.0, 1.0)
    
    # Plotting layout with premium dark clinical theme
    fig, axes = plt.subplots(1, 5, figsize=(18, 5.5), facecolor='#111b24')
    
    titles = [
        "Original Image\n(Normalized [0, 1])", 
        "Rotation Variant\n(Angle: +15° | Reflect)", 
        "Flip Variant\n(Horizontal Mirror)", 
        "Zoom Variant\n(Factor: 1.2x Center)", 
        "Brightness Variant\n(Delta: +0.15)"
    ]
    
    images = [norm_img, rot_variant, flip_variant, zoom_variant, brightness_variant]
    
    for idx, (ax, img_arr, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(img_arr)
        ax.axis('off')
        
        # Text label with subtle card framing
        ax.set_title(title, fontsize=11, fontweight='bold', color='#ffffff', pad=12)
        
        # Colored borders to delineate original vs augmented
        border_color = '#0f62fe' if idx == 0 else '#00d8f6'
        rect = plt.Rectangle(
            (0, 0), 223, 223, fill=False, color=border_color, 
            linewidth=2.5, transform=ax.transData
        )
        ax.add_patch(rect)
        
    # Main Title
    plt.suptitle("Skin Lesion Augmentation Pipeline Samples (Sprint 4)", 
                 fontsize=15, fontweight='bold', color='#ffffff', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    logger.info(f"Augmentation visualization saved successfully to: {output_path.resolve()}")


def generate_report(
    config: dict, 
    stats: dict, 
    val_count: int,
    test_count: int,
    report_path: Path
):
    """
    Compiles the Markdown report logging the Sprint 4 Augmentation Pipeline specs.
    """
    logger.info("Compiling data augmentation pipeline report...")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compile parameters table
    params_rows = ""
    for k, v in config.items():
        if k not in ["minority_classes", "majority_classes", "class_balancing"]:
            params_rows += f"| `{k}` | {v} |\n"
            
    # Compile class balancing table
    class_map = {
        "akiec": "Actinic Keratosis",
        "bcc": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic Nevi",
        "vasc": "Vascular Lesion"
    }
    
    pre_counts = stats["pre_balancing"]
    post_counts = stats["post_balancing"]
    pre_total = stats["pre_total"]
    post_total = stats["post_total"]
    
    table_rows = ""
    for code, full_name in class_map.items():
        pre_c = pre_counts.get(code, 0)
        post_c = post_counts.get(code, 0)
        pre_pct = (pre_c / pre_total * 100) if pre_total > 0 else 0.0
        post_pct = (post_c / post_total * 100) if post_total > 0 else 0.0
        
        is_minority = "Yes (Oversampled)" if code in config.get("minority_classes", []) else "No"
        
        table_rows += f"| `{code}` | {full_name} | {pre_c} | {pre_pct:.2f}% | {post_c} | {post_pct:.2f}% | {is_minority} |\n"
        
    # Write file content
    content = f"""# HAM10000 Data Augmentation & Balancing Report (Sprint 4)

This report details the implementation, parameters, and distribution statistics for the Sprint 4 Image Augmentation Pipeline.

---

## 1. Augmentation Techniques & Configuration Parameters

Centralized pipeline parameters are configured via [augmentation_config.json](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/configs/augmentation_config.json).

| Augmentation Parameter | Configured Value | Description |
| :--- | :--- | :--- |
{params_rows}
* **Fill Border Mode**: `{config.get('fill_mode', 'reflect')}` is used. Pixel values near borders generated by shift, zoom, or rotation are dynamically filled using reflection (`REFLECT`) to maintain biological realism and prevent artificial black outlines.

---

## 2. Minority Class Balancing Strategy

Skin cancer classification tasks are heavily skewed towards Melanocytic Nevi (`nv`). To mitigate model bias without discarding majority data, we implement in-memory replication (oversampling) of minority classes in the training set:

- **Target Sample Count**: All minority classes (`df`, `vasc`, `akiec`, `bcc`, `mel`) are oversampled to reach **{config.get('class_balancing', {}).get('target_samples_per_minority_class', 2000)} samples** each.
- **Stochastic Variation**: Online augmentation applies random rotation, translation, crop, and brightness adjustments dynamically to these oversampled inputs during batch loading. Because the same image path is loaded multiple times, it receives a different stochastic transformation each time.

### Training Set Class Distributions (Disk Audited)
*Note: Below counts represent images physically found on disk during the script run.*

| Class | Lesion Name | Initial Count | Initial Proportion | Balanced Count | Balanced Proportion | Oversampled Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
{table_rows}

- **Initial Train Split Size**: {pre_total} images
- **Balanced Train Split Size**: {post_total} images (augmented online)
- **Validation Split Size**: {val_count} images (unaugmented)
- **Test Split Size**: {test_count} images (unaugmented)

---

## 3. TensorFlow / Keras Integration Pipeline

The `augmentation.py` module exposes a `BalancedDatasetBuilder` class which integrates with `tf.data.Dataset`:
1. **Validation & Test Integrity**: The pipeline ensures that validation and test dataset loaders perform strictly scaling and resizing. **No augmentations are applied to val/test sets** to keep evaluation benchmarks clean and unbiased.
2. **Performance Optimizations**:
   * Uses native TensorFlow file read APIs `tf.io.read_file` and decoders.
   * Leverages `tf.data.AUTOTUNE` for multi-threaded parallel mapping and asynchronous prefetching.
   * Augmentation is applied in batches to utilize GPU acceleration.

---

## 4. Recommendations for Sprint 5 (Baseline CNN Training)

1. **Loss Function**: When training the model, verify that class weights (`class_weights.json`) are disabled if training on the oversampled balanced dataset, as the class frequencies are now equalized.
2. **Preprocessing Check**: EfficientNetB0 has its own internal rescaling layer. Ensure the data pipeline output range (`[0, 1]` or `[0, 255]`) matches the selected input layer configuration of the CNN.
3. **Stochasticity**: Keep `apply_probability` set to `0.5` or higher to prevent overfitting on the oversampled minority classes.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Report compiled successfully.")


def main():
    base_dir = Path(".")
    config_path = base_dir / "configs" / "augmentation_config.json"
    processed_dir = base_dir / "processed"
    visualizations_dir = base_dir / "visualizations"
    reports_dir = base_dir / "reports"
    
    logger.info("Running Sprint 4 Data Augmentation Pipeline...")
    
    # 1. Initialize Augmentor
    augmentor = SkinCancerAugmentor(config_path)
    
    # 2. Initialize Dataset Builder
    builder = BalancedDatasetBuilder(config_path, processed_dir)
    
    # 3. Load Splits and Filter Missing Files on Disk
    train_df = builder.load_and_clean_split("train.csv")
    val_df = builder.load_and_clean_split("val.csv")
    test_df = builder.load_and_clean_split("test.csv")
    
    # 4. Perform Class Balancing on Training Set
    balanced_train_df, stats = builder.balance_training_set(train_df)
    
    # 5. Generate Augmentation Samples Visualization
    # Select a sample image from the training set (e.g. first row with valid path)
    if not train_df.empty:
        sample_path = Path(train_df.iloc[0]['image_path'])
        vis_output = visualizations_dir / "augmentation_samples.png"
        generate_visualization(sample_path, augmentor, vis_output)
    else:
        logger.error("No valid training images found to generate visualization samples.")
        
    # 6. Compile Report
    report_output = reports_dir / "augmentation_report.md"
    generate_report(
        config=augmentor.config,
        stats=stats,
        val_count=len(val_df),
        test_count=len(test_df),
        report_path=report_output
    )
    
    logger.info("Sprint 4 Data Augmentation execution completed successfully!")

if __name__ == "__main__":
    main()
