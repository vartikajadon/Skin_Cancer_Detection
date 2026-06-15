"""Ground‑truth validation suite for the skin‑cancer model.
Generates per‑image predictions, overall metrics, per‑class performance,
confusion matrix, top‑3 accuracy, confidence analysis, and error reports.
"""

import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from backend.services.prediction_service import predict_lesion, initialize_prediction_service
from backend.confidence_config import CONFIDENCE_THRESHOLD

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "efficientnetb0.keras"
ENCODER_PATH = ROOT / "processed" / "label_encoder.json"
TEST_CSV = ROOT / "processed" / "test.csv"
OUTPUT_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
ERROR_EXAMPLES_DIR = REPORTS_DIR / "error_examples"

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
ERROR_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_test_data():
    df = pd.read_csv(TEST_CSV)
    # Expected columns: image_id, actual_diagnosis, optional image_path
    if "image_path" not in df.columns:
        df["image_path"] = df["image_id"].apply(
            lambda x: ROOT / "processed" / "images" / f"{x}.jpg"
        )
    return df

def run_inference(df):
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running inference"):
        # Resolve image path – fall back to constructing from image_id if missing or NaN
        img_path_val = row["image_path"]
        if pd.isna(img_path_val) or not str(img_path_val).strip():
            img_path = pathlib.Path(ROOT / "processed" / "images" / f"{row['image_id']}.jpg").resolve()
        else:
            img_path = pathlib.Path(img_path_val).resolve()
        if not img_path.is_file():
            # Skip missing image files with a warning
            print(f"Warning: Image not found for ID {row['image_id']}: {img_path}")
            continue
        pred = predict_lesion(img_path)
        probs = pred.get("probabilities", {})
        max_conf = pred.get("max_confidence", 0.0)
        rows.append({
            "image_id": row["image_id"],
            "actual_label": row["dx"],
            "predicted_label": pred.get("predicted_class"),
            "confidence": max_conf,
            "correct_prediction": int(pred.get("predicted_class") == row["dx"]),
            "probabilities": probs,
        })
    return pd.DataFrame(rows)

def save_validation_results(df):
    out_path = OUTPUT_DIR / "validation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Validation results saved → {out_path}")

def compute_overall_metrics(df):
    y_true = df["actual_label"]
    y_pred = df["predicted_label"]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return metrics, report

def write_classification_report(report):
    md_path = REPORTS_DIR / "classification_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Classification Report\n\n")
        f.write("| Class | Precision | Recall | F1‑Score | Support |\n")
        f.write("|------|-----------|--------|----------|---------|\n")
        for cls, vals in report.items():
            if cls in ["accuracy", "macro avg", "weighted avg"]:
                continue
            f.write(
                f"| {cls} | {vals['precision']:.3f} | {vals['recall']:.3f} | {vals['f1-score']:.3f} | {int(vals['support'])} |\n"
            )
        f.write("\n*Overall metrics also stored in `outputs/metrics.json`*\n")
    print(f"Classification report written → {md_path}")

def plot_confusion_matrix(df):
    labels = sorted(df["actual_label"].unique())
    cm = confusion_matrix(df["actual_label"], df["predicted_label"], labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {cm_path}")
    return cm

def per_class_metrics(df):
    per = []
    labels = sorted(df["actual_label"].unique())
    for cls in labels:
        cls_df = df[df["actual_label"] == cls]
        precision = precision_score(cls_df["actual_label"], cls_df["predicted_label"], labels=[cls], average="macro", zero_division=0)
        recall = recall_score(cls_df["actual_label"], cls_df["predicted_label"], labels=[cls], average="macro", zero_division=0)
        f1 = f1_score(cls_df["actual_label"], cls_df["predicted_label"], labels=[cls], average="macro", zero_division=0)
        per.append({"Class": cls, "Precision": precision, "Recall": recall, "F1": f1})
    per_df = pd.DataFrame(per)
    out_path = OUTPUT_DIR / "per_class_performance.csv"
    per_df.to_csv(out_path, index=False)
    print(f"Per‑class performance saved → {out_path}")
    return per_df

def top3_accuracy(df):
    top1 = df["correct_prediction"].mean()
    def in_top3(row):
        probs = row["probabilities"]
        if not isinstance(probs, dict) or len(probs) == 0:
            return False
        top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3_labels = [lbl for lbl, _ in top3]
        return row["actual_label"] in top3_labels
    df["top3_correct"] = df.apply(in_top3, axis=1)
    top3 = df["top3_correct"].mean()
    md_path = REPORTS_DIR / "top3_accuracy_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Top‑3 Accuracy Report\n\n")
        f.write(f"* Top‑1 Accuracy: {top1:.3%}\n")
        f.write(f"* Top‑3 Accuracy: {top3:.3%}\n")
    print(f"Top‑3 accuracy report written → {md_path}")
    return top1, top3

def confidence_analysis(df):
    bins = [(0.0, 0.70), (0.70, 0.90), (0.90, 1.0)]
    labels = ["Low (<70%)", "Medium (70‑90%)", "High (>90%)"]
    rows = []
    for (low, high), label in zip(bins, labels):
        subset = df[(df["confidence"] > low) & (df["confidence"] <= high)]
        acc = subset["correct_prediction"].mean() if len(subset) > 0 else None
        rows.append({"Range": label, "Count": len(subset), "Accuracy": acc})
    md_path = REPORTS_DIR / "confidence_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Confidence Analysis\n\n| Range | Count | Accuracy |\n|------|------|----------|\n")
        for r in rows:
            acc_str = f"{r['Accuracy']:.2%}" if r['Accuracy'] is not None else "N/A"
            f.write(f"| {r['Range']} | {r['Count']} | {acc_str} |\n")
    print(f"Confidence analysis report written → {md_path}")
    return rows

def wrong_predictions(df):
    errors = df[df["correct_prediction"] == 0].copy()
    top_errors = errors.sort_values(by="confidence").head(50)
    csv_path = OUTPUT_DIR / "wrong_predictions.csv"
    top_errors.to_csv(csv_path, index=False)
    print(f"Top‑50 wrong predictions saved → {csv_path}")
    # Copy image samples for manual inspection
    for _, row in top_errors.iterrows():
        src = ROOT / "processed" / "images" / f"{row['image_id']}.jpg"
        dst = ERROR_EXAMPLES_DIR / f"{row['image_id']}.jpg"
        if src.exists():
            dst.write_bytes(src.read_bytes())
    # Placeholder markdown for Grad‑CAM analysis of errors
    md_path = REPORTS_DIR / "gradcam_error_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Grad‑CAM Error Analysis\n\n*(Grad‑CAM visualisations for the above error samples can be generated manually or via a dedicated script.)*\n")
    print(f"Grad‑CAM placeholder report written → {md_path}")
    return top_errors

def main():
    # Initialise model and encoder
    initialize_prediction_service(MODEL_PATH, ENCODER_PATH)
    # Load test metadata
    test_df = load_test_data()
    # Run inference on every image
    results_df = run_inference(test_df)
    # Persist per‑image results
    save_validation_results(results_df)
    # Overall metrics & classification report
    metrics, class_report = compute_overall_metrics(results_df)
    write_classification_report(class_report)
    # Confusion matrix
    plot_confusion_matrix(results_df)
    # Per‑class analysis
    per_class_metrics(results_df)
    # Top‑3 accuracy
    top1, top3 = top3_accuracy(results_df)
    # Confidence analysis
    confidence_analysis(results_df)
    # Wrong‑prediction investigation
    wrong_predictions(results_df)
    # Final aggregated markdown
    final_md = REPORTS_DIR / "final_validation_report.md"
    with open(final_md, "w", encoding="utf-8") as f:
        f.write("# Final Validation Report\n\n")
        f.write(f"* Overall Accuracy: {metrics['accuracy']:.3%}\n")
        f.write(f"* Balanced Accuracy: {metrics['balanced_accuracy']:.3%}\n")
        f.write(f"* Top‑1 Accuracy: {top1:.3%}\n")
        f.write(f"* Top‑3 Accuracy: {top3:.3%}\n")
        f.write("\nSee the detailed markdown reports in the `reports/` folder for further breakdowns.\n")
    print(f"Final validation report written → {final_md}")

if __name__ == "__main__":
    main()
