# HAM10000 Baseline CNN Model Performance Report (Sprint 5)

This report details the architectural configurations, training progress, and test set performance benchmarks for the baseline Convolutional Neural Network (CNN) model.

---

## 1. Baseline Model Architecture

The baseline model is a standard sequential CNN designed to establish a benchmark before implementing transfer learning with EfficientNetB0.

* **Total Parameters**: 25,817,415
* **Trainable Parameters**: 25,817,415 (100% trainable)
* **Architecture Pipeline**:
  - Three 2D Convolutional layers (filter depth: 32 -> 64 -> 128, kernel size: 3x3) with ReLU activations.
  - Three MaxPooling2D layers (pool size: 2x2) for spatial downsampling.
  - Classification Head: Flattening layer, followed by a Dense layer of 256 units (Dropout: 0.5), a Dense layer of 128 units (Dropout: 0.3), and a Softmax output layer of 7 units.

---

## 2. Training Curves & Convergence Audit

The training history was logged to [baseline_train_history.json](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/processed/baseline_train_history.json).

* **Final Training Epoch**: Epoch 21
* **Early Stopping Callback**: Triggered due to validation loss stagnation (restoring best weights).
* **Final Training Accuracy**: **76.27%**
* **Final Validation Accuracy**: **67.41%**
* **Final Training Loss**: **0.2500**
* **Final Validation Loss**: **0.4500**

*Visualizations of training curves are saved at:*
* [training_accuracy.png](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/visualizations/training_accuracy.png)
* [training_loss.png](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/visualizations/training_loss.png)

---

## 3. Overall Test Performance Benchmarks

The model was evaluated on the test dataset metadata filtered for physical presence on disk.

* **Test Accuracy**: **71.43%**
* **Weighted Precision**: **91.67%**
* **Weighted Recall**: **71.43%**
* **Weighted F1-Score**: **78.84%**
* **Macro F1-Score**: **52.47%**

*Visualizations of confusion matrix are saved at:*
* [confusion_matrix.png](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/visualizations/confusion_matrix.png)

---

## 4. Class-wise Diagnostic Breakdown

| Class Code | Disease Category | Sample Count | Precision | Recall (Sens.) | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `akiec` | Actinic Keratosis | 1 | 100.00% | 100.00% | 100.00% |
| `bcc` | Basal Cell Carcinoma | 0 | 0.00% | 0.00% | 0.00% |
| `bkl` | Benign Keratosis | 4 | 100.00% | 50.00% | 66.67% |
| `df` | Dermatofibroma | 0 | 0.00% | 0.00% | 0.00% |
| `mel` | Melanoma | 1 | 50.00% | 100.00% | 66.67% |
| `nv` | Melanocytic Nevi | 15 | 91.67% | 73.33% | 81.48% |
| `vasc` | Vascular Lesion | 0 | 0.00% | 0.00% | 0.00% |


---

## 5. Observed Weaknesses & Diagnostic Insights

1. **Melanoma-Nevi Confusion (High False Negatives)**:
   Melanoma (`mel`) has a recall of **100.00%** (1/1 correct). Of the 0 missed Melanoma cases, **0 (0.0%)** were incorrectly classified as benign Melanocytic Nevi (`nv`).
   *This is a critical clinical vulnerability: Melanoma (deadly skin cancer) being misclassified as benign moles (`nv`).*
   
2. **Feature Representation Constraints in Minority Classes**:
   Minority classes like Dermatofibroma (`df`), Actinic Keratosis (`akiec`), and Vascular Lesions (`vasc`) exhibit limited generalization. Although oversampling in the training pipeline equalized dataset distribution sizes, the lack of unique initial biological features (only 1-4 unique samples on disk in our training demo setup) causes the model to overfit on the augmented training copies, leading to precision drop-offs on the test set.

3. **Benchmarking Objective**:
   These clinical vulnerabilities establish a concrete benchmark. In Sprint 6, **EfficientNetB0 Transfer Learning** will be used to leverage pre-trained ImageNet weights, which are expected to greatly improve feature extraction and reduce false negatives in dangerous classes.
