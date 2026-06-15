import sys
import logging
from pathlib import Path

# Configure path variables to import modules from src/
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Configure logging
logger = logging.getLogger(__name__)
from backend.confidence_config import CONFIDENCE_THRESHOLD

from predict import LesionPredictor, InferenceError

# Global predictor singleton reference
_predictor_instance = None

def initialize_prediction_service(model_path: Path, encoder_path: Path):
    """
    Initializes the LesionPredictor inference engine singleton at startup.
    """
    global _predictor_instance
    if _predictor_instance is not None:
        logger.warning("Prediction service is already initialized. Skipping reload.")
        return
        
    logger.info("Initializing Prediction Service (Loading EfficientNetB0 model)...")
    try:
        _predictor_instance = LesionPredictor(model_path, encoder_path)
        logger.info("Prediction Service initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Prediction Service: {str(e)}")
        raise e

# Global Grad-CAM generator reference
_gradcam_instance = None

def predict_lesion(image_path: Path) -> dict:
    """
    Generates class predictions and Grad-CAM explanations for a skin lesion image.
    """
    global _predictor_instance, _gradcam_instance
    if _predictor_instance is None:
        raise RuntimeError("Prediction Service is not initialized. Call initialize_prediction_service first.")
    
    # 1. Run predictions
    result = _predictor_instance.predict_image(Path(image_path))
    # Compute confidence metrics
    max_confidence = max(result.get("probabilities", {}).values()) if result.get("probabilities") else 0.0
    result["max_confidence"] = max_confidence
    result["is_uncertain"] = max_confidence < CONFIDENCE_THRESHOLD

    # 2. Initialize Grad-CAM singleton if not instantiated
    if _gradcam_instance is None:
        try:
            from gradcam import GradCAMGenerator
            _gradcam_instance = GradCAMGenerator(_predictor_instance.model)
        except Exception as e:
            logger.error(f"Failed to load Grad-CAM Generator dependency: {str(e)}")
            
    # 3. Generate Grad-CAM overlays
    if _gradcam_instance is not None:
        try:
            pred_class = result["predicted_class"]
            target_idx = _predictor_instance.encoder_map.get(pred_class)
            
            explain_res = _gradcam_instance.generate_heatmap_and_overlay(Path(image_path), target_idx)
            
            # Append outputs matching exact Sprint contracts and base64 tags
            result["gradcam_image"] = "overlay.png"
            result["heatmap_image"] = "heatmap.png"
            result["gradcam_image_base64"] = explain_res["overlay_base64"]
            result["heatmap_image_base64"] = explain_res["heatmap_base64"]
        except Exception as e:
            logger.error(f"Grad-CAM heatmap trace failed: {str(e)}")
            result["gradcam_image"] = ""
            result["heatmap_image"] = ""
            result["gradcam_image_base64"] = ""
            result["heatmap_image_base64"] = ""
    else:
        result["gradcam_image"] = ""
        result["heatmap_image"] = ""
        result["gradcam_image_base64"] = ""
        result["heatmap_image_base64"] = ""
        
    return result

def is_model_loaded() -> bool:
    """Checks if the model has been loaded successfully."""
    return _predictor_instance is not None
