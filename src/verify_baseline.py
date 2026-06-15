import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_verification():
    logger.info("Initializing Sprint 5 Verification Suite...")
    
    base_dir = Path(".")
    model_path = base_dir / "models" / "baseline_cnn.keras"
    history_path = base_dir / "processed" / "baseline_train_history.json"
    acc_curve = base_dir / "visualizations" / "training_accuracy.png"
    loss_curve = base_dir / "visualizations" / "training_loss.png"
    cm_heatmap = base_dir / "visualizations" / "confusion_matrix.png"
    report_path = base_dir / "reports" / "baseline_model_report.md"
    
    # 1. Check Model Checkpoint Existence
    assert model_path.exists(), f"Verification failed: model checkpoint not found at {model_path}"
    logger.info("✅ Verification Pass: Baseline CNN model checkpoint saved successfully.")
    
    # 2. Check Training History
    assert history_path.exists(), f"Verification failed: history logs not found at {history_path}"
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    assert "loss" in history and "accuracy" in history, "Verification failed: history does not contain expected fields."
    logger.info(f"✅ Verification Pass: Training history logs present with {len(history['loss'])} epochs recorded.")
    
    # 3. Check Visualizations
    assert acc_curve.exists(), f"Verification failed: accuracy plot not found at {acc_curve}"
    assert loss_curve.exists(), f"Verification failed: loss plot not found at {loss_curve}"
    assert cm_heatmap.exists(), f"Verification failed: confusion matrix heatmap not found at {cm_heatmap}"
    logger.info("✅ Verification Pass: All 3 visualization graphics generated and saved successfully.")
    
    # 4. Check Report
    assert report_path.exists(), f"Verification failed: model report not found at {report_path}"
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()
    assert "# HAM10000 Baseline CNN Model Performance Report" in report_text, "Verification failed: invalid report header."
    assert "Weighted F1-Score" in report_text, "Verification failed: missing overall f1 score metric."
    assert "Class-wise Diagnostic Breakdown" in report_text, "Verification failed: missing class metrics table."
    logger.info("✅ Verification Pass: Baseline CNN model performance report compiled successfully.")
    
    logger.info("🎉 All Sprint 5 Baseline CNN Training and Evaluation Verification checks passed successfully!")

if __name__ == "__main__":
    run_verification()
