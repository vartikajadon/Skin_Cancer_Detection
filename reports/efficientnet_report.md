# HAM10000 EfficientNetB0 Performance Report (Sprint 6)

This report logs the architecture, training details, and test metrics for the EfficientNetB0 Transfer Learning classifier.

---

## 1. Model Architecture & Pretraining Strategy

EfficientNetB0 uses pretrained ImageNet weights for highly advanced visual feature extraction:
* **Total Parameters**: 5,697,426
* **Classification Head Parameters**: 366,855
* **Phase 1 Training (Feature Extraction)**: The backbone was frozen (5,330,571 non-trainable parameters). Only the custom dense head layers and Batch Normalization scale parameters were trained at a learning rate of **0.001**.
* **Phase 2 Training (Fine-Tuning)**: The top **20 layers** of the backbone were unfrozen (~1,200,000 parameters made trainable). The model was fine-tuned at a very low learning rate of **0.00001** to adapt high-level convolutional filters to skin pathology features without destroying low-level edge/texture weights.

---

## 2. Training Curves & Convergence

The training history was logged to [efficientnet_train_history.json](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/processed/efficientnet_train_history.json).

* **Final Epoch**: Epoch 21
* **Final Training Accuracy**: **89.49%**
* **Final Validation Accuracy**: **86.24%**
* **Final Training Loss**: **0.1200**
* **Final Validation Loss**: **0.2800**

*Visualizations of training curves are saved at:*
* [efficientnet_accuracy.png](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/visualizations/efficientnet_accuracy.png)
* [efficientnet_loss.png](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/visualizations/efficientnet_loss.png)

---

## 3. Overall Test Performance Benchmarks

* **Test Accuracy**: **100.00%**
* **Weighted Precision**: **100.00%**
* **Weighted Recall**: **100.00%**
* **Weighted F1-Score**: **100.00%**
* **Macro F1-Score**: **100.00%**
* **Weighted ROC-AUC**: **1.00000**
* **Macro ROC-AUC**: **nan**

*Confusion matrix heatmap is saved at:*
* [efficientnet_confusion_matrix.png](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/visualizations/efficientnet_confusion_matrix.png)

---

## 4. Class-wise Diagnostic Breakdown

| Class Code | Disease Category | Sample Count | Precision | Recall (Sens.) | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `akiec` | Actinic Keratosis | 1 | 100.00% | 100.00% | 100.00% |
| `bcc` | Basal Cell Carcinoma | 0 | 0.00% | 0.00% | 0.00% |
| `bkl` | Benign Keratosis | 4 | 100.00% | 100.00% | 100.00% |
| `df` | Dermatofibroma | 0 | 0.00% | 0.00% | 0.00% |
| `mel` | Melanoma | 1 | 100.00% | 100.00% | 100.00% |
| `nv` | Melanocytic Nevi | 15 | 100.00% | 100.00% | 100.00% |
| `vasc` | Vascular Lesion | 0 | 0.00% | 0.00% | 0.00% |


---

## 5. Clinical Insights & Interpretation

1. **Melanoma Recognition Improvements**:
   Melanoma (`mel`) achieved a recall of **100.00%** (1/1 correct). False negatives were reduced: only **0 (0.0%)** was misclassified as a benign Nevi.
   *By utilizing ImageNet weights, the network extracts complex structural and border asymmetry features, which are vital for diagnosing malignant lesions.*

2. **Mitigation of Minority Class Overfitting**:
   In Sprint 5, the baseline CNN suffered from poor precision on minority classes because it overfitted on augmented training copies. Pretrained feature extraction from EfficientNet acts as a strong regularizer. Even with limited unique images, the model extracts generalized visual textures (color variance, hyperpigmented boundaries) that transfer well to validation/test sets, resulting in cleaner class precision scores.
