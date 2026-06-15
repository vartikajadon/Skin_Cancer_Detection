import os
import csv
import json
import shutil
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PredictionLogger:
    """
    Handles logging of real-time transactions from the prediction API.
    Saves a copy of the uploaded image to prevent temporary cleanup loss,
    and logs diagnosis class, confidence score, and top-3 probabilities.
    """
    def __init__(self, log_dir: Path = None):
        if log_dir is None:
            # Default to root directory / processed
            self.log_dir = Path(__file__).resolve().parent.parent.parent / "processed"
        else:
            self.log_dir = Path(log_dir)
            
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_images_dir = self.log_dir / "audit_uploads"
        self.audit_images_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "prediction_audit_log.csv"
        self._init_csv()
        
    def _init_csv(self):
        """Initializes the CSV log file with correct headers if it doesn't exist."""
        if not self.csv_path.exists():
            try:
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", 
                        "original_filename", 
                        "audit_image_path", 
                        "predicted_class", 
                        "confidence", 
                        "top_3_predictions"
                    ])
                logger.info(f"Initialized new prediction audit CSV log at: {self.csv_path}")
            except Exception as e:
                logger.error(f"Failed to initialize prediction CSV log: {str(e)}")
                
    def log_prediction(self, temp_image_path: Path, prediction_result: dict):
        """
        Saves a copy of the uploaded image and appends a structured prediction entry.
        """
        temp_image_path = Path(temp_image_path)
        if not temp_image_path.exists():
            logger.warning(f"Unable to audit prediction: temporary image file not found at {temp_image_path}")
            return
            
        try:
            # 1. Save uploaded image to audit folder
            original_name = temp_image_path.name
            unique_suffix = int(time.time() * 1000)
            audit_filename = f"audit_{unique_suffix}_{original_name}"
            audit_path = self.audit_images_dir / audit_filename
            
            shutil.copy(str(temp_image_path), str(audit_path))
            
            # 2. Extract prediction metadata
            predicted_class = prediction_result.get("predicted_class", "")
            confidence = prediction_result.get("confidence", 0.0)
            top_predictions = prediction_result.get("top_predictions", [])
            
            # Convert top_predictions list to JSON string for single-column serialization
            top_preds_json = json.dumps(top_predictions)
            
            # 3. Log to CSV
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    original_name,
                    str(audit_path.relative_to(self.log_dir.parent)),
                    predicted_class,
                    confidence,
                    top_preds_json
                ])
                
            logger.info(f"Successfully audited prediction for {original_name} -> {predicted_class} (conf: {confidence:.4f})")
            
        except Exception as e:
            logger.error(f"Failed to log prediction to audit: {str(e)}")
            raise e
