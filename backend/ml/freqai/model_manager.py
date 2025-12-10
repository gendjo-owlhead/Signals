"""
FreqAI Model Manager
Manages model lifecycle: training, saving, loading, and inference.
Supports XGBoost and Sklearn models.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import joblib
from loguru import logger
import json
from datetime import datetime

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

class ModelManager:
    """
    Manages the machine learning model.
    """
    
    def __init__(self, model_path: Path, model_type: str = "XGBoostRegressor"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.model_info = {}
        
        self.model_path.mkdir(parents=True, exist_ok=True)
        self._load_metadata()
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model.
        """
        logger.info(f"Training {self.model_type} on {len(X)} samples")
        
        if self.model_type == "XGBoostRegressor" and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                objective='reg:squarederror',
                n_jobs=-1
            )
        elif self.model_type == "RandomForestRegressor":
             self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        else:
            # Default to Sklearn GradientBoosting
            logger.info("Using Sklearn GradientBoostingRegressor")
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05)
            
        self.model.fit(X, y)
        
        # Save model info
        self.model_info = {
            "type": self.model_type,
            "trained_at": datetime.now().isoformat(),
            "samples": len(X),
            "features": list(X.columns)
        }
        self.save_model()
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")
                
        # Ensure columns match training
        if "features" in self.model_info:
            expected_cols = self.model_info["features"]
            # Add missing cols with 0
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0
            # Filter extra cols and reorder
            X = X[expected_cols]
            
        return self.model.predict(X)
        
    def save_model(self):
        """Save model and metadata."""
        if self.model is None:
            return
            
        # Save model binary
        joblib.dump(self.model, self.model_path / "model.joblib")
        
        # Save metadata
        with open(self.model_path / "model_info.json", "w") as f:
            json.dump(self.model_info, f, indent=2)
            
    def load_model(self) -> bool:
        """Load model from disk."""
        try:
            model_file = self.model_path / "model.joblib"
            if model_file.exists():
                self.model = joblib.load(model_file)
                self._load_metadata()
                return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            
        return False

    def _load_metadata(self):
        meta_file = self.model_path / "model_info.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                self.model_info = json.load(f)
