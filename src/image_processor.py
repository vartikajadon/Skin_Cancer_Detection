import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Handles loading, verification, color translation (BGR to RGB),
    resizing to 224x224, and range scaling ([0,1]) for skin cancer images.
    """
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
    def preprocess_image(self, filepath: Path) -> Optional[np.ndarray]:
        """
        Loads an image from disk, validates integrity, translates colors,
        resizes to target dimensions, and normalizes pixel arrays.
        
        Returns:
            np.ndarray: Preprocessed image of shape (target_height, target_width, 3), 
                        normalized to float range [0.0, 1.0].
            None: If the image is unreadable or malformed.
        """
        try:
            # 1. Load image (using cv2, returns BGR)
            img = cv2.imread(str(filepath))
            if img is None:
                logger.warning(f"Unable to read file as image: {filepath}")
                return None
                
            # 2. Verify image integrity
            if img.shape[0] == 0 or img.shape[1] == 0 or len(img.shape) != 3:
                logger.warning(f"Invalid dimensions or channel shape for image: {filepath}")
                return None
                
            # 3. Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 4. Resize image to 224x224
            img_resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_AREA)
            
            # 5. Normalize pixel values to range [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            return img_normalized
            
        except Exception as e:
            logger.error(f"Error processing image {filepath}: {str(e)}")
            return None
            
    def audit_images(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Processes a batch of images to check integrity, dimensions, and color profiles.
        Compiles an image quality report dictionary.
        """
        logger.info(f"Auditing image quality check on {len(file_paths)} mapped files...")
        
        corrupted_count = 0
        valid_count = 0
        min_pixels = []
        max_pixels = []
        shapes = []
        
        for path in file_paths:
            path = Path(path)
            processed = self.preprocess_image(path)
            
            if processed is None:
                corrupted_count += 1
            else:
                valid_count += 1
                min_pixels.append(processed.min())
                max_pixels.append(processed.max())
                shapes.append(processed.shape)
                
        # Aggregate statistics
        stats = {
            'total_audited': len(file_paths),
            'valid_count': valid_count,
            'corrupted_count': corrupted_count,
            'min_value_observed': float(np.min(min_pixels)) if min_pixels else 0.0,
            'max_value_observed': float(np.max(max_pixels)) if max_pixels else 0.0,
            'target_shapes_verified': all(s == (self.target_size[0], self.target_size[1], 3) for s in shapes) if shapes else False
        }
        
        logger.info(f"Audit complete. Valid: {valid_count}, Corrupted: {corrupted_count}")
        return stats
