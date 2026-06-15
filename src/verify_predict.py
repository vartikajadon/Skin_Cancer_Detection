import os
import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_verification():
    logger.info("Initializing Sprint 7 Verification Suite...")
    
    base_dir = Path(".")
    predictions_csv = base_dir / "processed" / "predictions.csv"
    metrics_json = base_dir / "processed" / "deployment_metrics.json"
    vis_examples = base_dir / "visualizations" / "prediction_examples.png"
    vis_dist = base_dir / "visualizations" / "confidence_distribution.png"
    report_path = base_dir / "reports" / "deployment_readiness_report.md"
    
    # 1. Check predictions.csv
    assert predictions_csv.exists(), f"Verification failed: predictions.csv not found at {predictions_csv}"
    df_preds = pd.read_csv(predictions_csv)
    expected_cols = {"image_name", "image_id", "predicted_class", "confidence", "error"}
    assert expected_cols.issubset(df_preds.columns), f"Verification failed: predictions.csv missing columns. Found {df_preds.columns}"
    assert len(df_preds) == 24, f"Verification failed: expected 24 rows, found {len(df_preds)}"
    logger.info("✅ Verification Pass: Batch predictions CSV exported successfully with valid columns and rows.")
    
    # 2. Check deployment_metrics.json
    assert metrics_json.exists(), f"Verification failed: deployment_metrics.json not found at {metrics_json}"
    with open(metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "average_latency_ms" in metrics and "throughput_fps" in metrics, "Verification failed: deployment_metrics.json missing key stats."
    logger.info("✅ Verification Pass: Deployment metrics JSON compiled and saved successfully.")
    
    # 3. Check Visualizations
    assert vis_examples.exists(), f"Verification failed: prediction_examples.png not found at {vis_examples}"
    assert vis_dist.exists(), f"Verification failed: confidence_distribution.png not found at {vis_dist}"
    logger.info("✅ Verification Pass: Both prediction visualization graphics generated and saved successfully.")
    
    # 4. Check Report
    assert report_path.exists(), f"Verification failed: deployment report not found at {report_path}"
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()
    assert "# HAM10000 Inference Engine Deployment Readiness Report" in report_text, "Verification failed: invalid report header."
    assert "Deployment Execution Metrics" in report_text, "Verification failed: missing metrics section."
    assert "Deployment Integration Recommendations" in report_text, "Verification failed: missing guidelines section."
    logger.info("✅ Verification Pass: Deployment readiness report compiled successfully.")
    
    logger.info("🎉 All Sprint 7 Inference Engine & Deployment Verification checks passed successfully!")

if __name__ == "__main__":
    run_verification()
