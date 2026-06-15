import json
import logging
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd

logger = logging.getLogger(__name__)

class LesionLabelEncoder:
    """
    Handles encoding of categorical skin lesion diagnosis strings into integer labels,
    and serializes mapping states to JSON format for deployment.
    """
    # Fixed class mapping as required
    CLASS_MAPPING = {
        'akiec': 0,
        'bcc': 1,
        'bkl': 2,
        'df': 3,
        'mel': 4,
        'nv': 5,
        'vasc': 6
    }
    
    def __init__(self, mapping: Dict[str, int] = None):
        self.mapping = mapping if mapping is not None else self.CLASS_MAPPING
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        
    def transform(self, series: pd.Series) -> pd.Series:
        """
        Encodes categorical Series values using integer label mappings.
        """
        logger.info("Transforming lesion categories to numerical labels...")
        return series.map(self.mapping)
        
    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """
        Decodes integer label mappings back to standard categories.
        """
        logger.info("Reversing numerical labels back to text categories...")
        return series.map(self.inverse_mapping)
        
    def save_mapping(self, filepath: Path) -> None:
        """
        Saves the label encoder dictionary states to local JSON file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.mapping, f, indent=4)
            logger.info(f"Label encoder dictionary saved to: {filepath.resolve()}")
        except Exception as e:
            logger.error(f"Failed to save label mapping: {str(e)}")
            raise e
            
    @classmethod
    def load_mapping(cls, filepath: Path) -> 'LesionLabelEncoder':
        """
        Loads encoder mapping rules from local JSON configuration.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            error_msg = f"Mapping config not found at: {filepath.resolve()}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            logger.info(f"Loaded encoder mapping from: {filepath.resolve()}")
            return cls(mapping)
        except Exception as e:
            logger.error(f"Failed to load encoder mapping: {str(e)}")
            raise e
