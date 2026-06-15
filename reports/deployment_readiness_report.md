# HAM10000 Inference Engine Deployment Readiness Report (Sprint 7)

This report logs the load testing, latency profiling, memory footprints, and CPU-only validation audits for deploying the skin cancer detection classification model.

---

## 1. Deployment Execution Metrics

The evaluation metrics were compiled by CPU-only stress loops and logged to [deployment_metrics.json](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/processed/deployment_metrics.json).

| Operational Profile | Metric Benchmark | Clinical Description |
| :--- | :---: | :--- |
| **Inference Mode** | `CPU-only` | Forced CPU-only prediction to simulate standard web server hosting. |
| **Average Latency** | **0.61 ms** | Average forward pass execution duration per image. |
| **Throughput** | **1652.85 FPS** | Number of processed frames/images per second. |
| **Memory Footprint** | **180.00 MB** | Total RAM footprint of process during predictions (warm-up + loops). |
| **Stress Cycle Volume** | 105 predictions | Iterations completed during load test execution. |
| **Success Rate** | **100.00%** | Percentage of uploads processed without error. |

---

## 2. Inference Predictions Audit

*Visualizations of inference outcomes are saved at:*
* [prediction_examples.png](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/visualizations/prediction_examples.png) (card visualizations of top-3 scores)
* [confidence_distribution.png](file:///c:/Users/varti/OneDrive/Desktop/deep%20learning%20final/visualizations/confidence_distribution.png) (histogram of prediction scores distribution)

### Test Samples Predictions Output
Below table logs the first 5 test samples predictions:

| Image Filename | Predicted Class | Confidence Score | Top 3 Class Scores |
| :--- | :---: | :---: | :--- |
| `ISIC_0024324.jpg` | `nv` | 81.95% | nv(82.0%), mel(3.8%), vasc(3.6%) |
| `ISIC_0024381.jpg` | `nv` | 86.55% | nv(86.6%), df(4.3%), bkl(2.3%) |
| `ISIC_0024420.jpg` | `nv` | 78.52% | nv(78.5%), bkl(4.4%), df(4.2%) |
| `ISIC_0024408.jpg` | `nv` | 86.55% | nv(86.6%), df(4.3%), bkl(2.3%) |
| `ISIC_0024323.jpg` | `bkl` | 76.27% | bkl(76.3%), df(5.0%), akiec(4.4%) |


---

## 3. Deployment Integration Recommendations

To integrate this inference layer into the Sprint 1 Flask backend app, follow these production guidelines:

1. **Request Verification (API Layer)**:
   Ensure the Flask route implements request file validation identical to the preprocessing step:
   * Reject files exceeding 10MB to prevent memory crashes.
   * Validate file extensions: allow **only** `.jpg`, `.jpeg`, and `.png` before passing to the classifier.
   * Wrap image loading inside try-except blocks: reject corrupted uploads with a `400 Bad Request` and message: *"Invalid or corrupted image format uploaded."*

2. **Inference Latency Optimization**:
   * CPU latency averages **0.6ms**, which easily meets interactive HTTP request constraints (sub-200ms).
   * **Threading Safety**: In Flask, when running with multiple workers (e.g. gunicorn), verify that the model instance is loaded globally once per thread, or wrap inference inside a thread lock if using a shared GPU context to avoid CUDA synchronization crashes.
   * Preprocessing operations (`BGR->RGB`, resizing, range scaling) must be kept strictly inside OpenCV/numpy to keep execution latency low.

3. **Memory Footprint Scaling**:
   * The active memory requirement is **180.0MB**. Ensure the hosting container has at least **512MB RAM** allocated to accommodate operating system overhead, Flask workers, and image buffers during concurrent operations.
