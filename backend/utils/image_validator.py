import os
from pathlib import Path
import cv2

class ImageValidationError(Exception):
    """Exception raised when uploaded image fails safety or format audits."""
    pass

def validate_image(file_path: Path, max_size_bytes: int = 10 * 1024 * 1024) -> None:
    """
    Performs safety, format, and corruption audits on an uploaded image.
    Raises:
        ImageValidationError: If validation fails with specific description.
    """
    file_path = Path(file_path)
    
    # 1. Verify file exists
    if not file_path.exists() or not file_path.is_file():
        raise ImageValidationError("Uploaded file is missing or invalid.")
        
    # 2. Check file size limits
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ImageValidationError("Uploaded file is empty (0 bytes).")
    if file_size > max_size_bytes:
        raise ImageValidationError(f"File size exceeds maximum limit of {max_size_bytes / (1024 * 1024):.1f}MB.")
        
    # 3. Check format suffix (only allow .jpg, .jpeg, .png)
    valid_suffixes = {'.jpg', '.jpeg', '.png'}
    if file_path.suffix.lower() not in valid_suffixes:
        raise ImageValidationError(f"Unsupported file format '{file_path.suffix}'. Only .jpg, .jpeg, and .png are allowed.")
        
    # 4. Check for binary corruption using OpenCV image decoding
    try:
        img = cv2.imread(str(file_path))
        if img is None:
            raise ImageValidationError("Corrupted image file. Failed to decode pixel arrays.")
        if img.shape[0] == 0 or img.shape[1] == 0 or len(img.shape) != 3:
            raise ImageValidationError("Invalid image dimensions or channel profile.")
    except ImageValidationError:
        raise
    except Exception as e:
        raise ImageValidationError(f"Failed to read image structure: {str(e)}")
