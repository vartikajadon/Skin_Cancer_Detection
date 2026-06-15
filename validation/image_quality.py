import cv2
import numpy as np
from pathlib import Path

def analyze_image_quality(image_path: Path, min_res: int = 64, blur_threshold: float = 15.0) -> dict:
    """
    Performs image quality validation including checks for:
    - Minimum resolution
    - Blurry image detection
    - Empty (flat color block) image detection
    - Extremely dark image detection
    - Extremely bright image detection
    
    Returns:
        dict: {"valid": bool, "reason": str or None}
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return {"valid": False, "reason": "File does not exist"}

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return {"valid": False, "reason": "Corrupted image file. Failed to decode pixel arrays."}

    h, w, c = img.shape

    # 1. Minimum resolution check
    if h < min_res or w < min_res:
        return {
            "valid": False,
            "reason": f"Image resolution ({w}x{h}) is too low. Minimum required is {min_res}x{min_res} pixels."
        }

    # Convert to grayscale for statistical checks
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Empty / flat image check
    std_dev = np.std(gray)
    if std_dev < 2.0:
        return {
            "valid": False,
            "reason": "Image is empty or contains solid flat color blocks with no features."
        }

    # 3. Blurry image check (Laplacian variance method)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < blur_threshold:
        return {
            "valid": False,
            "reason": f"Image is too blurry (variance: {lap_var:.2f}). Please upload a sharp, in-focus image."
        }

    # 4. Extremely dark/bright image check
    mean_val = np.mean(gray)
    if mean_val < 20.0:
        return {
            "valid": False,
            "reason": f"Image is extremely dark (average intensity: {mean_val:.1f}). Please ensure adequate lighting."
        }
    if mean_val > 240.0:
        return {
            "valid": False,
            "reason": f"Image is extremely bright (average intensity: {mean_val:.1f}). Please avoid overexposure."
        }

    return {"valid": True, "reason": None}
