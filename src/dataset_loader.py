import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HAM10000Loader:
    """
    Loader class for the HAM10000 skin cancer dataset.
    Loads metadata, scans directories, maps images to records, and validates image integrity.
    """
    def __init__(self, metadata_path: Path, image_dirs: List[Path]):
        self.metadata_path = Path(metadata_path)
        self.image_dirs = [Path(d) for d in image_dirs]
        self.metadata: Optional[pd.DataFrame] = None
        self.image_map: Dict[str, Path] = {}
        
    def load_metadata(self) -> pd.DataFrame:
        """
        Loads the HAM10000_metadata.csv file.
        """
        if not self.metadata_path.exists():
            error_msg = f"Metadata file not found at: {self.metadata_path.resolve()}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            self.metadata = pd.read_csv(self.metadata_path)
            logger.info(f"Successfully loaded metadata CSV with {len(self.metadata)} records.")
            return self.metadata
        except Exception as e:
            logger.error(f"Error reading metadata CSV: {str(e)}")
            raise e

    def scan_image_directories(self) -> Dict[str, Path]:
        """
        Scans all specified image directories and maps image IDs (filenames without extension) 
        to their absolute paths.
        """
        self.image_map = {}
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        for image_dir in self.image_dirs:
            if not image_dir.exists() or not image_dir.is_dir():
                logger.warning(f"Image directory does not exist: {image_dir.resolve()}. Skipping.")
                continue
                
            logger.info(f"Scanning directory: {image_dir.resolve()}")
            file_count = 0
            for item in image_dir.iterdir():
                if item.is_file() and item.suffix in valid_extensions:
                    image_id = item.stem  # Get filename without extension
                    
                    if image_id in self.image_map:
                        logger.warning(f"Duplicate image_id '{image_id}' found. Overwriting path: "
                                       f"{self.image_map[image_id]} with {item.resolve()}")
                        
                    self.image_map[image_id] = item.resolve()
                    file_count += 1
            logger.info(f"Scanned {file_count} images in {image_dir.name}.")
            
        logger.info(f"Total unique images scanned across all directories: {len(self.image_map)}")
        return self.image_map

    def verify_and_map(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Maps metadata rows to image paths. Detects missing and corrupted images.
        
        Returns:
            Tuple containing:
            - Mapped DataFrame with a new column 'image_path'
            - List of missing image IDs (present in metadata but missing on disk)
            - List of corrupted image IDs (present on disk but failing to open)
        """
        if self.metadata is None:
            self.load_metadata()
            
        if not self.image_map:
            self.scan_image_directories()
            
        df = self.metadata.copy()
        paths = []
        missing_images = []
        corrupted_images = []
        
        logger.info("Verifying image files and mapping metadata...")
        
        for idx, row in df.iterrows():
            image_id = str(row['image_id'])
            
            if image_id in self.image_map:
                img_path = self.image_map[image_id]
                
                # Check for file corruption
                if not self._is_image_valid(img_path):
                    corrupted_images.append(image_id)
                    paths.append(None)
                else:
                    paths.append(str(img_path))
            else:
                missing_images.append(image_id)
                paths.append(None)
                
        # Insert mapped paths into DataFrame
        df['image_path'] = paths
        
        # Log results
        logger.info(f"Mapping complete. Mapped: {df['image_path'].notna().sum()} / {len(df)} records.")
        if missing_images:
            logger.error(f"Found {len(missing_images)} missing images (present in CSV, missing on disk).")
        if corrupted_images:
            logger.error(f"Found {len(corrupted_images)} corrupted images (present on disk, failing to load).")
            
        return df, missing_images, corrupted_images

    @staticmethod
    def _is_image_valid(filepath: Path) -> bool:
        """
        Verifies if the image can be loaded successfully by OpenCV.
        """
        try:
            # Try to read the image file header using cv2
            # Use imread with IMREAD_UNCHANGED or similar. If it returns None, it is corrupted.
            img = cv2.imread(str(filepath))
            if img is None:
                return False
            # Check dimensions are valid
            if img.shape[0] == 0 or img.shape[1] == 0:
                return False
            return True
        except Exception:
            return False
            
    def get_unmapped_images(self) -> List[str]:
        """
        Finds images present in the directories but not listed in the metadata CSV.
        """
        if self.metadata is None:
            self.load_metadata()
            
        csv_image_ids = set(self.metadata['image_id'].astype(str))
        dir_image_ids = set(self.image_map.keys())
        
        unmapped = list(dir_image_ids - csv_image_ids)
        if unmapped:
            logger.warning(f"Found {len(unmapped)} files in folders that are not in the metadata CSV.")
        return unmapped
