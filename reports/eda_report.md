# HAM10000 Exploratory Data Analysis (EDA) Report

This comprehensive report details the distribution of lesion categories, patient demographics, image quality metrics, and pre-processing recommendations.

---

## 1. Metadata Quality & Completeness
An analysis of the metadata values was conducted to flag duplicate cases, patient re-occurrences, and missing attributes.

### Missing Fields Report
| Metadata Column | Missing Records | Percentage |
| :--- | :--- | :--- |
| `lesion_id` | 0 | 0.0% |
| `image_id` | 0 | 0.0% |
| `dx` | 0 | 0.0% |
| `dx_type` | 0 | 0.0% |
| `age` | 57 | 0.6% |
| `sex` | 0 | 0.0% |
| `localization` | 0 | 0.0% |
| `image_path` | 9865 | 98.5% |

*Note: Missing patient age values must be handled during preprocessing (e.g., mean/median imputation).*

### Patient Duplicate Records
- **Total Lesion Records**: 10015
- **Unique Mapped Lesions**: 7470
- **Duplicate lesion occurrences**: 2545
- **Implication**: Some lesions have multiple dermoscopic images. In splitting the training/validation/testing sets, **grouped splits based on `lesion_id` are mandatory**. Failing to do so will leak identical patient lesions into both training and validation sets, inflating validation accuracy artificially.

---

## 2. Class Distribution Analysis
The dataset contains a severe class imbalance across 7 structural categories.

| Class Code | Disease Category | Count | Proportion (%) |
| :---: | :--- | :---: | :---: |
| `nv` | Melanocytic Nevi | 6705 | 66.95% |
| `mel` | Melanoma | 1113 | 11.11% |
| `bkl` | Benign Keratosis | 1099 | 10.97% |
| `bcc` | Basal Cell Carcinoma | 514 | 5.13% |
| `akiec` | Actinic Keratoses | 327 | 3.27% |
| `vasc` | Vascular Lesions | 142 | 1.42% |
| `df` | Dermatofibroma | 115 | 1.15% |

### Risk Category Assessment
- **Malignant / High-Risk**: Melanoma (`mel`, 11.1%) and Basal Cell Carcinoma (`bcc`, 5.1%) represent critical cases where false negatives are highly dangerous.
- **Benign**: Melanocytic Nevi (`nv`, 66.9%) dominates the dataset, meaning the model's natural default baseline would guess "Nevus" for all inputs.

---

## 3. Demographic & Localization Analysis
A summary of patient age, gender, and anatomical distributions:

- **Patient Age**: Mean of **51.9 years** (Median: 50.0). Shows typical skewed distribution matching adult clinical dermatological visits.
- **Gender Balance**: Male: 5406, Female: 4552, Unknown: 57
- **Primary Anatomical Location**: **back** represents the most common location (2192 lesions).

---

## 4. Image Quality & Resolution Verification
- **Unique Resolutions found**: 128x128, 64x64
- **Channel Check**: Verified 150 images possess **3 channels (RGB)**.
- **Corrupted Files Isolated**: 0 files failed safety verification.

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
