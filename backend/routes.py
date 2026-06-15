import os
import logging
import time
from pathlib import Path
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from services.prediction_service import predict_lesion, is_model_loaded
from utils.image_validator import validate_image, ImageValidationError

# Import OOD and Image Quality validators
from validation.image_quality import analyze_image_quality
from validation.ood_detector import OODDetector

logger = logging.getLogger(__name__)

# Define blueprint
api_bp = Blueprint('api', __name__)

# Configure uploads directory reference
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate the OOD detector
ood_detector = OODDetector()

CLASSIFICATION_THRESHOLD = 0.70

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Exposes health status and model load status."""
    loaded = is_model_loaded()
    return jsonify({
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded
    }), 200 if loaded else 500

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Exposes prediction endpoint accepting skin lesion images."""
    # 1. Verify file exists in request payload
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
        
    # 2. Save file temporarily
    raw_filename = secure_filename(file.filename)
    # Truncate filename to avoid exceeding MAX_PATH on Windows (BUG-01 remediation)
    base_name, ext = os.path.splitext(raw_filename)
    truncated_base = base_name[:50]
    unique_filename = f"{int(time.time() * 1000)}_{truncated_base}{ext}"
    temp_path = UPLOAD_DIR / unique_filename
    
    try:
        file.save(str(temp_path))
        
        # 3. Validate image using safety layer (format/corruption check)
        validate_image(temp_path)
        
        # 4. Run image quality checks
        quality_res = analyze_image_quality(temp_path)
        if not quality_res["valid"]:
            logger.warning(f"Image quality check failed for {raw_filename}: {quality_res['reason']}")
            if temp_path.exists():
                os.remove(temp_path)
            return jsonify({
                "status": "rejected",
                "message": f"Image quality check failed: {quality_res['reason']}. Please upload a clear dermoscopic image."
            }), 400

        # 5. Run skin lesion detection (OOD detector)
        lesion_prob = ood_detector.predict_lesion_probability(temp_path)
        if lesion_prob < 0.80:
            logger.warning(f"OOD detection rejected {raw_filename} (probability: {lesion_prob:.4f})")
            if temp_path.exists():
                os.remove(temp_path)
            return jsonify({
                "status": "rejected",
                "message": "Please upload a dermoscopic skin lesion image."
            }), 400
        
        # 6. Generate diagnosis predictions (EfficientNetB0)
        predictions = predict_lesion(temp_path)
        
        # Log prediction to Audit Module
        try:
            from utils.prediction_logger import PredictionLogger
            pred_logger = PredictionLogger()
            pred_logger.log_prediction(temp_path, predictions)
        except Exception as log_err:
            logger.error(f"Failed to log prediction to audit: {str(log_err)}")
            
        # 7. Clean up temporary upload immediately
        if temp_path.exists():
            os.remove(temp_path)
            
        # 8. Check classification confidence threshold
        conf = predictions.get("confidence", 0.0)
        if conf < CLASSIFICATION_THRESHOLD:
            logger.info(f"Low prediction confidence ({conf:.4f} < {CLASSIFICATION_THRESHOLD}) for {raw_filename}. Marking uncertain.")
            uncertain_res = {
                "status": "uncertain",
                "message": "Prediction confidence is too low.",
                "predicted_class": predictions.get("predicted_class"),
                "confidence": conf,
                "top_predictions": predictions.get("top_predictions"),
                "gradcam_image": predictions.get("gradcam_image", ""),
                "heatmap_image": predictions.get("heatmap_image", ""),
                "gradcam_image_base64": predictions.get("gradcam_image_base64", ""),
                "heatmap_image_base64": predictions.get("heatmap_image_base64", "")
            }
            return jsonify(uncertain_res), 200
            
        return jsonify(predictions), 200
        
    except ImageValidationError as e:
        # Client validation failures (bad format, corrupted image, size exceeded)
        logger.warning(f"Validation failed for upload {raw_filename}: {str(e)}")
        if temp_path.exists():
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 400
        
    except FileNotFoundError as e:
        # File missing error during process
        if temp_path.exists():
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 404
        
    except Exception as e:
        # Prediction/Server internal errors
        logger.error(f"Inference pipeline crash: {str(e)}")
        if temp_path.exists():
            os.remove(temp_path)
        return jsonify({"error": f"Prediction pipeline failure: {str(e)}"}), 500

