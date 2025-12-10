"""
FreqAI Data Processor
Handles data cleaning, normalization, and preparation for ML models.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger
import pickle
from pathlib import Path

class DataProcessor:
    """
    Handles data processing pipeline:
    - NaN handling
    - Outlier detection (basic)
    - Normalization/Scaling
    - Train/Test splitting
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataframe by handling NaNs and Infs.
        """
        # Replace infs with nan
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill first, then backward fill (or drop)
        df = df.ffill().fillna(0)
        
        return df
        
    def normalize(self, df: pd.DataFrame, train_mode: bool = False) -> pd.DataFrame:
        """
        Normalize features using StandardScaler.
        
        Args:
            df: Dataframe with features
            train_mode: If True, fit the scaler. If False, just transform.
        """
        # Filter only numeric columns that aren't labels or metadata
        feature_cols = [c for c in df.columns if not c.startswith('&') and not c in ['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
        
        if not feature_cols:
            logger.warning("No feature columns found to normalize")
            return df
            
        features = df[feature_cols]
        
        if train_mode:
            self.scaler.fit(features)
            self.is_fitted = True
            self.save_scaler()
        elif not self.is_fitted:
            self.load_scaler()
            if not self.is_fitted:
                logger.warning("Scaler not fitted, skipping normalization")
                return df
                
        normalized_features = self.scaler.transform(features)
        
        # Update dataframe with normalized values
        df_norm = df.copy()
        df_norm[feature_cols] = normalized_features
        
        return df_norm
        
    def save_scaler(self):
        """Save scaler to disk."""
        with open(self.data_path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
            
    def load_scaler(self):
        """Load scaler from disk."""
        scaler_path = self.data_path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True
