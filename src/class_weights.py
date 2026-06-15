import json
import logging
from pathlib import Path
from typing import Dict, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ClassWeightCalculator:
    """
    Computes class weights to offset severe dataset class imbalances during training,
    saving the configuration to JSON.
    """
    
    @staticmethod
    def calculate_weights(labels: pd.Series) -> Dict[str, float]:
        """
        Calculates balanced class weights.
        Formula: weight = total_samples / (n_classes * class_samples)
        
        Args:
            labels: pd.Series containing encoded integer class labels.
            
        Returns:
            Dict[str, float]: Mappings from label key string to weight float value.
        """
        logger.info("Calculating balanced training class weights...")
        
        total_samples = len(labels)
        classes = np.unique(labels)
        n_classes = len(classes)
        
        # Calculate occurrences
        class_counts = labels.value_counts()
        
        weights = {}
        for cls in classes:
            count = class_counts.get(cls, 0)
            if count > 0:
                # Standard balanced class weights formula
                weight = total_samples / (n_classes * count)
                # Keep keys as strings for JSON serialization
                weights[str(cls)] = float(weight)
            else:
                weights[str(cls)] = 0.0
                
        # Normalize weights so that the average weight is 1.0 (recommended for stable loss gradients)
        mean_weight = np.mean(list(weights.values()))
        if mean_weight > 0:
            for k in weights:
                weights[k] = round(weights[k] / mean_weight, 5)
                
        logger.info(f"Class weights computed: {weights}")
        return weights

    @staticmethod
    def save_weights(weights: Dict[str, float], filepath: Path) -> None:
        """
        Saves computed class weights config to JSON file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(weights, f, indent=4)
            logger.info(f"Class weights saved successfully to: {filepath.resolve()}")
        except Exception as e:
            logger.error(f"Failed to save class weights: {str(e)}")
            raise e
