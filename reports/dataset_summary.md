# HAM10000 Skin Cancer Dataset Summary

This document provides a high-level administrative and structural overview of the HAM10000 dataset for Sprint 2.

## 📋 General File Status Checklist

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Total CSV Metadata Rows** | 10015 | Verified |
| **Total Images Located on Disk** | 150 | Mapped |
| **Successfully Loaded Images** | 150 | Healthy |
| **Missing Image Files** | 9865 | ⚠️ Warning: Missing files found |
| **Corrupted Image Files** | 0 | Passed |

---

## 🔬 Dataset Core Metrics

- **Unique Lesions (Patient Cases)**: 7470
- **Duplicate Lesion Records**: 2545 *(Multiple dermoscopic captures of the same lesion)*
- **Lesion Classes Identified**: 7 classes
- **Image Resolution Range**: 64x64 to 128x128
- **Average Dimensions**: 95.6 x 95.6
- **Image Color Channels**: 3-channel (RGB)

---

## ⚖️ Imbalance Highlight

- **Majority Class**: `nv` (Melanocytic Nevi) &mdash; **6705 images** (66.9%)
- **Minority Class**: `df` (Dermatofibroma) &mdash; **115 images** (1.1%)
- **Imbalance Ratio**: 1 : 58.3

### Potential Impact on Deep Learning Models
The severe class imbalance presents a significant risk for deep neural network training. Models will easily overfit to the majority class (**Melanocytic Nevi**) and may achieve high overall accuracy while failing completely to recognize rarer, high-risk malignancies like **Melanoma (`mel`)**. Mitigation strategies such as class-weighted loss, SMOTE, or focal loss will be required in Sprint 3.
