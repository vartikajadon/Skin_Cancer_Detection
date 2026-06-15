import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

class GroupSplitter:
    """
    Splits the HAM10000 metadata into Train (70%), Val (15%), and Test (15%) splits
    grouped by lesion_id to completely avoid patient-level data leakage.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs a group-based nested split of the dataframe:
        - Step 1: Separates Test (15%) from Train+Val (85%)
        - Step 2: Separates Val (15% of total, i.e., 15/85 of Train+Val) from Train (70% of total)
        """
        logger.info("Executing patient-group splits on lesion_id...")
        
        if 'lesion_id' not in df.columns:
            error_msg = "Cannot perform grouped splitting: 'lesion_id' column is missing from metadata."
            logger.error(error_msg)
            raise KeyError(error_msg)
            
        # Step 1: Group split to isolate Test (15%)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=self.random_state)
        try:
            train_val_indices, test_indices = next(gss_test.split(df, groups=df['lesion_id']))
        except Exception as e:
            logger.error(f"GSS Test split failed: {str(e)}")
            raise e
            
        train_val_df = df.iloc[train_val_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)
        
        # Step 2: Group split Train+Val into Train (70% total) and Val (15% total)
        # Test size ratio is 15 / (70 + 15) = 15 / 85 ≈ 0.17647
        gss_val = GroupShuffleSplit(n_splits=1, test_size=15/85, random_state=self.random_state)
        try:
            train_indices, val_indices = next(gss_val.split(train_val_df, groups=train_val_df['lesion_id']))
        except Exception as e:
            logger.error(f"GSS Val split failed: {str(e)}")
            raise e
            
        train_df = train_val_df.iloc[train_indices].reset_index(drop=True)
        val_df = train_val_df.iloc[val_indices].reset_index(drop=True)
        
        logger.info(f"Split complete. Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
                    f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
                    f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
                    
        return train_df, val_df, test_df
        
    @staticmethod
    def validate_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Asserts and validates that there is zero overlap of lesion_id groups between splits.
        """
        train_lesions = set(train_df['lesion_id'].dropna())
        val_lesions = set(val_df['lesion_id'].dropna())
        test_lesions = set(test_df['lesion_id'].dropna())
        
        overlap_train_val = train_lesions.intersection(val_lesions)
        overlap_train_test = train_lesions.intersection(test_lesions)
        overlap_val_test = val_lesions.intersection(test_lesions)
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            error_details = []
            if overlap_train_val:
                error_details.append(f"Train & Val overlap: {len(overlap_train_val)} lesion_ids")
            if overlap_train_test:
                error_details.append(f"Train & Test overlap: {len(overlap_train_test)} lesion_ids")
            if overlap_val_test:
                error_details.append(f"Val & Test overlap: {len(overlap_val_test)} lesion_ids")
            
            msg = f"❌ Patient Leakage Detected! Overlaps found: {', '.join(error_details)}"
            logger.error(msg)
            return False, msg
            
        msg = "✅ Split Validation Passed. Zero overlap of lesion_id groups between Train, Val, and Test sets."
        logger.info(msg)
        return True, msg
