# Copy of train_efficientnet_v2.py to src folder
from pathlib import Path
import sys

# Import or duplicate definition
try:
    from train_efficientnet_v2 import StratifiedGroupSplitter, load_class_weights, build_tf_dataset_v2
except ImportError:
    import json
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    class StratifiedGroupSplitter:
        def __init__(self, random_state: int = 42):
            self.random_state = random_state
        def split(self, df: pd.DataFrame):
            unique_lesions = df.groupby('lesion_id').first().reset_index()
            sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=self.random_state)
            train_val_idx, test_idx = next(sss_test.split(unique_lesions, unique_lesions['label']))
            train_val_lesions = unique_lesions.iloc[train_val_idx]
            test_lesions = unique_lesions.iloc[test_idx]
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=15/85, random_state=self.random_state)
            train_idx, val_idx = next(sss_val.split(train_val_lesions, train_val_lesions['label']))
            train_lesions = train_val_lesions.iloc[train_idx]
            val_lesions = train_val_lesions.iloc[val_idx]
            train_df = df[df['lesion_id'].isin(train_lesions['lesion_id'])].reset_index(drop=True)
            val_df = df[df['lesion_id'].isin(val_lesions['lesion_id'])].reset_index(drop=True)
            test_df = df[df['lesion_id'].isin(test_lesions['lesion_id'])].reset_index(drop=True)
            return train_df, val_df, test_df
