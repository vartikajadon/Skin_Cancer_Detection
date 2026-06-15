# HAM10000 Data Preprocessing Report (Sprint 3)

This report logs the cleaning, encoding, splitting, and weights computation audit for the HAM10000 dataset preprocessing.

---

## 1. Metadata Cleaning & Imputation
- **Total Input Records**: 10015
- **Successfully Mapped Records**: 10015
- **Missing Age Fields Imputed**: 57 values filled with median age of **50.0 years**.
- **Records Removed (Invalid Fields)**: 0

---

## 2. Image Quality & Processing Audit
- **Images Located & Mapped**: 150 files
- **Image Integrity Checks**:
  - **Valid RGB Images**: 150 files
  - **Corrupted / Blank Files**: 0 files
  - **Resolution Range Checked**: All read images resized to **224x224** pixels.
  - **Pixel Intensity Normalization**: Mapped to **[0.0, 1.0]** scale.
  - **Missing Files Registered**: 9865 records currently missing corresponding images on disk.

---

## 3. Label Encoding
Skin lesion diagnostic string keys mapped to target classification integers:

| Lesion Key | Numerical Label |
| :---: | :---: |
| `akiec` | 0 |
| `bcc` | 1 |
| `bkl` | 2 |
| `df` | 3 |
| `mel` | 4 |
| `nv` | 5 |
| `vasc` | 6 |

*Note: Config saved to [label_encoder.json](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/processed/label_encoder.json).*

---

## 4. Patient Leakage & Split Verification
Split ratios targeted: **70% Train / 15% Val / 15% Test** using a nested `GroupShuffleSplit` on `lesion_id`.

### Split Results
- **Train Set**: 6959 records (69.5%)
- **Validation Set**: 1529 records (15.3%)
- **Test Set**: 1527 records (15.2%)

### Patient Leakage Assessment
- **Status**: ✅ Passed
- **Verification Details**: ✅ Split Validation Passed. Zero overlap of lesion_id groups between Train, Val, and Test sets.
*This prevents identical patient lesions from appearing in both train and validation sets, ensuring split reliability.*

---

## 5. Class Distributions & Penalization Weights (Training Set)
Weights calculated using standard inverse-frequency balancing on the training set to prevent model bias towards Melanocytic Nevi (`nv`).

| Class Code | Label | Train Count | Proportion (%) | Balanced Weight |
| :---: | :---: | :---: | :---: | :---: |
| `akiec` | 0 | 222 | 3.19% | 0.99149 |
| `bcc` | 1 | 349 | 5.02% | 0.63069 |
| `bkl` | 2 | 770 | 11.06% | 0.28586 |
| `df` | 3 | 92 | 1.32% | 2.39250 |
| `mel` | 4 | 771 | 11.08% | 0.28549 |
| `nv` | 5 | 4662 | 66.99% | 0.04721 |
| `vasc` | 6 | 93 | 1.34% | 2.36677 |

*Note: Config saved to [class_weights.json](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/processed/class_weights.json).*
