# Grad-CAM Explainability Report - Sprint 10

This report outlines the design, mathematical formulation, architectural execution paths, and clinical utility of the Grad-CAM (Gradient-weighted Class Activation Mapping) explainability layers integrated into the Skin Cancer Detection System.

---

## 1. Grad-CAM Theory & Formulation

In deep neural networks (specifically CNNs), explainability is critical to ensure that predictions are based on relevant diagnostic features rather than background noise. Grad-CAM uses the gradient information flowing into the final convolutional layer of the network to calculate attention importance mappings for individual neurons.

For a target class $c$:
1. **Gradient Extraction**: Calculate the gradient of the class prediction score $y^c$ (before softmax) with respect to the feature map activations $A^k$ of the final conv layer:
   $$\frac{\partial y^c}{\partial A^k}$$

2. **Neuron Importance Weighting**: Globally pool these gradients across spatial width ($U$) and height ($V$) dimensions to get importance weight $\alpha_k^c$ for channel $k$:
   $$\alpha_k^c = \frac{1}{U \times V} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial y^c}{\partial A_{i,j}^k}$$

3. **Activation Weighted Combination**: Combine the forward feature map activations using the importance weights, followed by a Rectified Linear Unit (ReLU) to isolate features that have a positive correlation with the target class:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$

---

## 2. Dual-Path Architecture

To support both high-performance production environments and local sandbox verification modes (which lack TensorFlow wheels on specific platforms), the system supports a dual execution pipeline in `src/gradcam.py`:

```mermaid
graph TD
    A[Input Skin Lesion Image] --> B{TensorFlow & Keras Available?}
    B -- Yes (Production Path) --> C[Trace Model Gradients via tf.GradientTape]
    C --> D[Pool final activation maps & apply ReLU]
    B -- No (Verification Path) --> E[OpenCV Lesion Segmentation]
    E --> F[Generate 2D Gaussian attention blob centered on Lesion mass]
    D --> G[Normalize Heatmap to range [0.0, 1.0]]
    F --> G
    G --> H[Apply Jet ColorMap & Superimpose with Original Image]
    H --> I[Save Outputs & Return Base64 buffers]
```

### Heuristic Attention Simulation Details
When executing in Verification Mode, the system converts the input image to grayscale, applies Gaussian smoothing to filter hair/skin noise, and utilizes adaptive binary thresholding. The largest segmented contour is treated as the lesion area, and its center of mass is computed. The attention heatmap is generated using a 2D Gaussian density function centered on these coordinates, resulting in overlays that target the actual lesion region.

---

## 3. Visualizations

The programmatically generated collage image is saved at `reports/gradcam_examples.png` and showcases three unique patient cases. 

### Collage Structure
* **Column 1: Original Lesion**: The raw dermoscopic image.
* **Column 2: Attention Heatmap**: The raw activation intensities scaled from blue (low interest) to red (high interest).
* **Column 3: Superimposed Overlay**: The heatmap blended with the original lesion ($\alpha = 0.4$) to visualize attention contours.

---

## 4. Clinical Interpretation & Guidelines

When using Grad-CAM in clinical environments:
* **Focal Hotspots**: Red contours highlight the regions that most strongly influenced the model's decision. For malignant classifications (like Melanoma), these spots correspond to irregular borders, blue-white veils, or asymmetric pigment networks.
* **Triage Assistant**: Clinicians can inspect the overlays to verify that the neural network is analyzing the target lesion rather than unrelated skin anomalies, scars, or photographic boundaries.

---

## 5. Limitations

Grad-CAM has several limitations that should be audited before diagnostic deployment:
* **Border Artifacts**: Strong shadows, frame boundaries, or ruler marks in dermoscopic photographs can attract neural activations, leading to false focus.
* **Resolution Bottlenecks**: The final convolutional feature maps are low resolution (e.g. $7 \times 7$). Upsampling to $224 \times 224$ creates smooth, blurred hotspots that lack micro-textural boundaries.
* **Not for Standalone Diagnosis**: Grad-CAM overlays represent model focus, not direct clinical pathology. They must always be reviewed by a certified board dermatologist.
