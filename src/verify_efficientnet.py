import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_verification():
    logger.info("Initializing Sprint 6 Verification Suite...")
    
    base_dir = Path(".")
    model_path = base_dir / "models" / "efficientnet_b0_best.keras"
    history_path = base_dir / "processed" / "efficientnet_train_history.json"
    stats_path = base_dir / "processed" / "efficientnet_eval_stats.json"
    
    acc_curve = base_dir / "visualizations" / "efficientnet_accuracy.png"
    loss_curve = base_dir / "visualizations" / "efficientnet_loss.png"
    cm_heatmap = base_dir / "visualizations" / "efficientnet_confusion_matrix.png"
    comparison_plot = base_dir / "visualizations" / "model_comparison.png"
    
    report_path = base_dir / "reports" / "efficientnet_report.md"
    comparison_report_path = base_dir / "reports" / "model_comparison_report.md"
    
    # 1. Check Model Checkpoint Existence
    assert model_path.exists(), f"Verification failed: model checkpoint not found at {model_path}"
    logger.info("✅ Verification Pass: EfficientNet model checkpoint saved successfully.")
    
    # 2. Check Training History
    assert history_path.exists(), f"Verification failed: history logs not found at {history_path}"
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    assert "loss" in history and "accuracy" in history, "Verification failed: history does not contain expected fields."
    logger.info(f"✅ Verification Pass: Training history logs present with {len(history['loss'])} epochs recorded.")
    
    # 3. Check Statistics File
    assert stats_path.exists(), f"Verification failed: stats JSON not found at {stats_path}"
    logger.info("✅ Verification Pass: Evaluation statistics compiled and saved to processed folder.")
    
    # 4. Check Visualizations
    assert acc_curve.exists(), f"Verification failed: accuracy plot not found at {acc_curve}"
    assert loss_curve.exists(), f"Verification failed: loss plot not found at {loss_curve}"
    assert cm_heatmap.exists(), f"Verification failed: confusion matrix heatmap not found at {cm_heatmap}"
    assert comparison_plot.exists(), f"Verification failed: comparison plot not found at {comparison_plot}"
    logger.info("✅ Verification Pass: All 4 visualizations generated and saved successfully.")
    
    # 5. Check Reports
    assert report_path.exists(), f"Verification failed: model report not found at {report_path}"
    assert comparison_report_path.exists(), f"Verification failed: comparison report not found at {comparison_report_path}"
    logger.info("✅ Verification Pass: Both reports compiled successfully.")
    
    logger.info("🎉 All Sprint 6 EfficientNet Transfer Learning Verification checks passed successfully!")

if __name__ == "__main__":
    run_verification()
