# Skin Cancer Detection System — AI Dermatology Portal

An interactive clinical-grade screening prototype that combines state-of-the-art convolutional neural network algorithms with clinical guidelines to assist in dermatological lesion screening. The system classifies skin lesions into 7 key categories and exposes a premium, patient-friendly dashboard equipped with Grad-CAM explainability maps.

---

## 🌟 Key Features

1. **AI-Powered Lesion Classification:**
   - Detects and classifies dermoscopic skin lesions into **7 distinct categories**:
     - *Nevus* (Benign Mole)
     - *Melanoma* (Malignant Lesion)
     - *Benign Keratosis-like Lesion*
     - *Basal Cell Carcinoma* (Suspected)
     - *Actinic Keratosis / Bowen's Disease* (Pre-cancerous)
     - *Vascular Lesion*
     - *Dermatofibroma*
   - Employs **EfficientNetB0** backend with Custom Focal Loss regularizations and **Test Time Augmentation (TTA)** to maximize classification safety.

2. **Interactive Before/After Grad-CAM Slider:**
   - Swipe interactively between raw close-up photos and AI attention heatmaps to identify focus zones showing border variance, shape asymmetry, or color shifts.

3. **Patient & Lesion Context Capture:**
   - Integrated input forms collect context: Patient Name, ID/Number, Age, Sex, and Anatomic Site of the lesion.

4. **Patient-ID Keyed Case Vault (`localStorage`):**
   - High privacy local database cache. Case records are stored under `localStorage` indexed directly by the patient's ID.

5. **Patient-Friendly PDF Report Generation:**
   - Custom `@media print` overrides isolate a clean summary template omitting technical prediction variables and neural heatmap overlays. Creates printable, layperson-friendly clinical briefs containing clear recommended next actions.

6. **Interactive Documentation Modals:**
   - Clean, custom popups explain:
     - Clinical observational trial parameters.
     - REST API payload specifications.
     - HIPAA anonymization regulations.
     - Terms of medical use.

---

## 📁 Repository Structure

```text
├── backend/                  # Flask server and router endpoints
│   ├── app.py                # Application entrypoint & static serving
│   ├── routes.py             # Prediction & health check routers
│   ├── services/             # Predictor singletons loading the model
│   └── utils/                # Base64 image codecs and validation checks
│
├── frontend/                 # Client UI
│   ├── assets/               # Video files, icons, and image directories
│   ├── css/                  # Custom design token css frameworks
│   │   └── style.css         # Theme styles & print media overrides
│   ├── js/                   # Event handling & localStorage syncing
│   │   ├── api.js            # Fetch calls to backend endpoints
│   │   └── app.js            # Slider, modals, and print handlers
│   └── index.html            # Main dashboard structural document
│
├── models/                   # Best trained Keras weight files
│   └── efficientnet_b0_best.keras
│
├── src/                      # ML core modules
│   ├── focal_loss.py         # Focal loss compiling layers
│   ├── tta_predict.py        # TTA inference wrapper
│   └── predict.py            # Custom image preprocessing functions
│
├── processed/                # Datasets splits & label encoder json configurations
└── requirements.txt          # Python dependencies
```

---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.11.x (Installed and added to your system `PATH`)

### Setup Instructions

1. **Clone/Open the project directory:**
   ```powershell
   cd "C:\Users\varti\OneDrive\Desktop\deep learning final"
   ```

2. **Activate the Virtual Environment:**
   - **On PowerShell:**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **On Command Prompt (CMD):**
     ```cmd
     .\.venv\Scripts\activate.bat
     ```

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```powershell
   python -m backend.app
   ```

5. **Access the Portal:**
   Open your browser and navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

---

## ⚠️ Medical Disclaimer

This application is an educational prototype and a clinical study screening helper. It is **not approved by the FDA** (or equivalent bodies) as an autonomous diagnostic tool and **does not replace professional medical evaluations**. Moles and lesions should be monitored regularly and checked visually by a certified dermatologist.
