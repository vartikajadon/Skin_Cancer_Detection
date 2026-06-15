# Requirement Traceability Matrix (RTM)

This document maps all the requirements specified across the 10 development sprints of the Skin Cancer Detection System to their corresponding implementation files, classes, methods, and validation status.

---

## 1. Traceability Summary Table

| Sprint | Requirement Name | Implementation File | Key Code Reference | Validation Status |
| :--- | :--- | :--- | :--- | :--- |
| **Sprint 1** | Responsive UI Layout | [index.html](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/index.html) | Root HTML5 Structure | **PASS** |
| | Sticky Navigation | [app.js](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/js/app.js) | `handleScroll` (L8-31) | **PASS** |
| | File Upload Interface | [index.html](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/index.html) | `#drop-zone` (L95-174) | **PASS** |
| | Disease Information Section | [index.html](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/index.html) | `#disease-info` (L235-360) | **PASS** |
| **Sprint 2** | Metadata Loading | [eda.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/eda.py) | [HAM10000ExploratoryAnalysis](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/eda.py#L12) | **PASS** |
| | Class Distribution Analysis | [eda.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/eda.py) | `plot_class_distribution` (L59-75) | **PASS** |
| | Duplicate Lesion Analysis | [eda.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/eda.py) | `check_duplicate_lesions` (L77-94) | **PASS** |
| | Age & Localization Distribution | [eda.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/eda.py) | `plot_demographics` (L96-121) | **PASS** |
| **Sprint 3** | Age Imputation (Median) | [preprocess.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/preprocess.py) | `execute` (L53-59) | **PASS** |
| | Label Encoding | [label_encoder.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/label_encoder.py) | [LesionLabelEncoder](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/label_encoder.py#L7) | **PASS** |
| | Leakage-Free Splitting | [splitter.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/splitter.py) | [GroupSplitter](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/splitter.py#L8) | **PASS** |
| | Split Validation | [splitter.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/splitter.py) | `validate_splits` (L59-86) | **PASS** |
| **Sprint 4** | Spatial Transformations | [augmentation.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/augmentation.py) | `_build_tf_augmentor` (L73-92) | **PASS** |
| | Color & Contrast Augmentations | [augmentation.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/augmentation.py) | `_build_tf_augmentor` (L93-98) | **PASS** |
| | Reflection Padding & Crop | [augmentation.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/augmentation.py) | `_build_tf_augmentor` (L99-105) | **PASS** |
| | NumPy Fallback Pipeline | [augmentation.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/augmentation.py) | `augment_image_numpy` (L114-199) | **PASS** |
| **Sprint 5** | Baseline CNN Architecture | [model_baseline.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/model_baseline.py) | [BaselineCNN](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/model_baseline.py#L8) | **PASS** |
| | Training Loop | [train_baseline.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/train_baseline.py) | `run_training` (L45-88) | **PASS** |
| | Performance Metrics (F1, AUC) | [evaluate_baseline.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/evaluate_baseline.py) | `evaluate_model` (L47-105) | **PASS** |
| **Sprint 6** | Pretrained EfficientNetB0 | [model_efficientnet.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/model_efficientnet.py) | `build_efficientnet_b0` (L13-41) | **PASS** |
| | Fine-Tuning Dense Layers | [model_efficientnet.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/model_efficientnet.py) | `build_efficientnet_b0` (L30-38) | **PASS** |
| | Class Weight Balancing | [class_weights.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/class_weights.py) | [ClassWeightCalculator](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/class_weights.py#L7) | **PASS** |
| **Sprint 7** | Inference Engine | [predict.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/predict.py) | [SkinCancerPredictor](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/predict.py#L12) | **PASS** |
| | Batch Prediction | [batch_predict.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/batch_predict.py) | [BatchInferenceEngine](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/batch_predict.py#L11) | **PASS** |
| **Sprint 8** | Flask Backend Initialization | [app.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/backend/app.py) | `create_app` (L26-83) | **PASS** |
| | API Health Check Endpoint | [routes.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/backend/routes.py) | `health_check` (L19-26) | **PASS** |
| | POST Predict Endpoint | [routes.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/backend/routes.py) | `predict` (L28-81) | **PASS** |
| **Sprint 9** | AJAX API Integration | [api.js](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/js/api.js) | [apiService](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/js/api.js#L8) | **PASS** |
| | Upload Validations (Size/Ext) | [app.js](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/js/app.js) | `handleFileSelection` (L156-196) | **PASS** |
| | Error/Response UI Cards | [app.js](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/js/app.js) | `showErrorCard` & `renderResults` | **PASS** |
| **Sprint 10**| Grad-CAM Gradient Compute | [gradcam.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/gradcam.py) | `_compute_real_gradcam` (L97-137) | **PASS** |
| | Simulated Fallback Engine | [gradcam.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/gradcam.py) | `_compute_simulated_gradcam` (L139-185) | **PASS** |
| | Heatmap Superimpose overlay | [gradcam.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/gradcam.py) | `generate_heatmap_and_overlay` (L36-95) | **PASS** |

---

## 2. Requirement Details & Implementation Notes

### Sprint 1 – Frontend UI
* **Acceptance Criteria**: The user interface must be modern, responsive, visually premium, and offer clear navigation, diagnostic instructions, disease facts, and drag-and-drop portal.
* **Implementation Details**:
  * [index.html](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/index.html) leverages vanilla HTML5 with semantic components (`<nav>`, `<main>`, `<section>`).
  * [style.css](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/css/style.css) handles custom dark-mode gradients, glassmorphism card panels, dynamic hover animations, and media queries for mobile resizing.
  * [app.js](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/js/app.js) establishes viewport intersection observer for dynamic navbar class toggling and smooth scrolling highlights.

### Sprint 2 – Dataset Analysis
* **Acceptance Criteria**: Verify metadata columns are correctly read, print distributions, detect duplicates, and generate charts for demographics.
* **Implementation Details**:
  * [eda.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/eda.py) loads raw `HAM10000_metadata.csv` using pandas and outputs audit files to `reports/eda_report.md` along with plots in `visualizations/`.
  * Computes duplicate mappings (identifying multiple images of the same lesion) using `lesion_id` vs `image_id` relations.

### Sprint 3 – Data Preprocessing
* **Acceptance Criteria**: Implement median imputation for age, encode diagnostic labels, perform patient-level grouped splits, and calculate training loss weights to adjust for severe class imbalances.
* **Implementation Details**:
  * Implements `lesion_id`-based nested grouped split inside [GroupSplitter](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/splitter.py#L8) using scikit-learn's `GroupShuffleSplit`, which is validated by `validate_splits` to verify zero patient leakage across Train/Val/Test subsets.
  * Mappings are saved to `processed/label_encoder.json` and balanced weights are calculated and stored in `processed/class_weights.json`.

### Sprint 4 – Data Augmentation
* **Acceptance Criteria**: Formulate a pipeline to support vertical/horizontal flip, 20° rotation, width/height translations, contrast/brightness adjustments, reflection padding, and random cropping to 224x224.
* **Implementation Details**:
  * [SkinCancerAugmentor](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/augmentation.py#L25) implements a complete Keras sequential pipeline under `_build_tf_augmentor` containing standard augmentation layers.
  * In addition, it implements `augment_image_numpy` using OpenCV & NumPy to support validation scripts in local environments lacking native TF GPU compilation.

### Sprint 5 – Baseline CNN
* **Acceptance Criteria**: Code a simple convolutional network structure with standard convolutional and pooling layers, standard training loops, saving model to `.keras` format, and generating classification metrics.
* **Implementation Details**:
  * [BaselineCNN](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/model_baseline.py#L8) contains 3 convolutional block stacks (Conv2D -> MaxPooling2D -> Dropout) followed by Dense categorization layers.
  * Training and validation logs are automatically outputted to `reports/baseline_model_report.md`.

### Sprint 6 – EfficientNetB0 Transfer Learning
* **Acceptance Criteria**: Initialize EfficientNetB0 pretrained model, customize classifier head, fine-tune weights under class penalization, and evaluate model performance.
* **Implementation Details**:
  * Custom architecture in [model_efficientnet.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/model_efficientnet.py#L13) leverages Keras Application APIs.
  * Freezes base features during phase-1, customizes top layers, and integrates computed class weights during compilation for model balance.

### Sprint 7 – Inference Engine
* **Acceptance Criteria**: Load best model, implement pre-processing transform pipeline, output class name mapping, top-3 confidence scores, and support single/batch prediction.
* **Implementation Details**:
  * [SkinCancerPredictor](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/predict.py#L12) handles loading the best model and converting the image to the standard model shape (224x224).
  * Returns formatted prediction dictionaries containing class, confidence score, and top-3 arrays.

### Sprint 8 – Flask Backend API
* **Acceptance Criteria**: Establish local Flask server, expose health check and prediction REST APIs, implement robust error handlers, enable CORS, and handle upload directories.
* **Implementation Details**:
  * [app.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/backend/app.py) handles application factory initialization, static paths routing, and generic error mapping.
  * [routes.py](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/backend/routes.py) manages endpoint bindings (`/api/health` and `/api/predict`).

### Sprint 9 – Frontend + Backend Integration
* **Acceptance Criteria**: Connect frontend file drop to backend prediction API, validate size/format limits client-side, display spinner, render diagnosis cards with top-3 bars, and handle connection errors.
* **Implementation Details**:
  * [api.js](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/js/api.js) encapsulates the AJAX/fetch layer using async/await with custom network failure handling.
  * [app.js](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/frontend/js/app.js) attaches drop-zone listeners, handles preview file reading, activates progress spinners, and renders HTML/CSS layout templates dynamically.

### Sprint 10 – Grad-CAM Explainability
* **Acceptance Criteria**: Load trained model, isolate final convolution layers, compute output gradient vectors, generate activation colormap overlays, and encode as base64 for direct UI display.
* **Implementation Details**:
  * [GradCAMGenerator](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/src/gradcam.py#L15) implements `_compute_real_gradcam` using TF `GradientTape` traces.
  * Implements `_compute_simulated_gradcam` using grayscale inversion, Gaussian blurring, contour detection, and Gaussian focus to simulate activation maps in verification (non-TF) modes.
  * Automatically encodes the resulting overlays as base64 string buffers, enabling instantaneous injection into the UI without disk-read lag.
