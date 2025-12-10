"""
FreqAI Engine
Main orchestrator for the FreqAI adaptive learning system.
Manages the training loop, data gathering, and model updating.
"""
import asyncio
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from config import settings
from .features import feature_extractor
from .data_processor import DataProcessor
from .model_manager import ModelManager
from data.historical import historical_fetcher

class FreqAIEngine:
    """
    Orchestrates the FreqAI adaptive learning process.
    """
    
    def __init__(self):
        self.data_path = Path(settings.ml_model_path) / "freqai"
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.data_processor = DataProcessor(self.data_path)
        self.model_manager = ModelManager(self.data_path, settings.freqai_model_type)
        
        self.running = False
        self.last_train_time = None
        self.training_interval = settings.freqai_retrain_interval_minutes * 60
        
        # State
        self.current_model_id = None
        
    async def start(self):
        """Start the FreqAI background service."""
        if not settings.freqai_enabled:
            logger.info("FreqAI disabled in settings")
            return
            
        self.running = True
        logger.info("FreqAI Engine started")
        asyncio.create_task(self._training_loop())
        
    async def stop(self):
        """Stop the service."""
        self.running = False
        logger.info("FreqAI Engine stopped")
        
    async def predict(self, df: pd.DataFrame) -> float:
        """
        Predict target for the current candle.
        
        Args:
            df: OHLCV dataframe (recent)
            
        Returns:
            Predicted value (e.g. price change)
        """
        if not self.model_manager.model:
            return 0.0
            
        # 1. Compute features (on a copy/recent slice)
        # We assume df has enough history for indicators
        df_features = feature_extractor.compute_features(df)
        
        # 2. Get the last row (current candle)
        last_row = df_features.iloc[[-1]] 
        
        # 3. Normalize
        normalized = self.data_processor.normalize(last_row, train_mode=False)
        
        # 4. Predict
        prediction = self.model_manager.predict(normalized)
        return float(prediction[0])
        
    async def _training_loop(self):
        """Background loop for periodic retraining."""
        while self.running:
            try:
                should_train = False
                if self.last_train_time is None:
                    should_train = True
                else:
                    elapsed = (datetime.now() - self.last_train_time).total_seconds()
                    if elapsed >= self.training_interval:
                        should_train = True
                
                if should_train:
                    logger.info("Starting scheduled FreqAI retraining...")
                    await self.train_model()
                    self.last_train_time = datetime.now()
                    
                # Check status periodically
                await asyncio.sleep(60) 
                
            except Exception as e:
                logger.exception(f"Error in FreqAI training loop: {e}")
                await asyncio.sleep(300) # Wait longer on error
                
    async def train_model(self):
        """
        Execute full training pipeline.
        
        1. Fetch historical data (Sliding window)
        2. Generate features and labels
        3. Clean and Normalize
        4. Train Model
        """
        try:
            # 1. Fetch Data
            # For now, we only train on primary pair/timeframe as a proof of concept
            # Should expand to loop over pairs
            pair = settings.trading_pairs[0]
            tf = settings.primary_timeframe
            days = settings.freqai_train_period_days
            limit = 1440 * days // 5 # Approx candles
            
            # Fetch klines
            klines = await historical_fetcher.fetch_klines(
                symbol=pair,
                interval=tf,
                limit=limit
            )
            
            if not klines:
                logger.warning("No data fetched for FreqAI training")
                return
                
            # Convert to DataFrame
            df = historical_fetcher.klines_to_dataframe(klines)
            
            if df.empty or len(df) < settings.min_samples_for_training:
                logger.warning("Insufficient data for FreqAI training")
                return

            logger.info(f"FreqAI collected {len(df)} candles for training")

            # 2. Feature Engineering
            df = feature_extractor.compute_features(df)

            df = feature_extractor.compute_labels(df)
            
            # Drop NaN rows created by lags/shifts
            df = self.data_processor.clean_data(df)
            
            # Select target
            target_col = [c for c in df.columns if c.startswith('&')][0]
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # 3. Normalize (Train Mode)
            X_norm = self.data_processor.normalize(X, train_mode=True)
            
            # 4. Train
            self.model_manager.train(X_norm, y)
            
            logger.info("FreqAI model training completed successfully")
            
        except Exception as e:
            logger.error(f"FreqAI training failed: {e}")
            raise

# Global Instance
freqai_engine = FreqAIEngine()
