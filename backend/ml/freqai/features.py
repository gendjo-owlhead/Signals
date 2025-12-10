"""
FreqAI Feature Engineering Module
Centralizes feature generation for the adaptive learning system.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from loguru import logger

class FreqAIFeatureExtractor:
    """
    Centralized feature extractor for FreqAI.
    Generates technical indicators and custom features.
    """
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical features for the dataframe.
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Dataframe with added feature columns
        """
        df = df.copy()
        
        # 1. Basic Momentum
        df['irsi'] = self._rsi(df['close'], 14)
        df['imomentum'] = df['close'].diff(10)
        
        # 2. Volatility
        df['ibbands_upper'], df['ibbands_lower'] = self._bollinger_bands(df['close'], 20, 2.0)
        df['ibbands_width'] = (df['ibbands_upper'] - df['ibbands_lower']) / df['close']
        
        # 3. Volume
        # Fill NaN volume with 0 if missing
        vol = df['volume'].fillna(0)
        df['ivolume_mean_20'] = vol.rolling(20).mean()
        df['ivolume_ratio'] = vol / df['ivolume_mean_20'].replace(0, 1)
        
        # 4. Custom FreqAI specific features (Lagged features)
        # Add lags for closing price and rsi
        for lag in [1, 2, 3]:
            df[f'items_close_lag_{lag}'] = df['close'].shift(lag)
            df[f'items_rsi_lag_{lag}'] = df['irsi'].shift(lag)
            
        return df

    def compute_labels(self, df: pd.DataFrame, shift: int = 5) -> pd.DataFrame:
        """
        Compute training targets (labels).
        
        Args:
            df: Dataframe with features
            shift: How many candles into the future to predict
            
        Returns:
            Dataframe with added label columns (starting with &)
        """
        df = df.copy()
        
        # Label: Future price return (relative to current close)
        # &s_close_shift => (Future_Close - Current_Close) / Current_Close
        future_close = df['close'].shift(-shift)
        df[f'&s_close_shift_{shift}'] = (future_close - df['close']) / df['close']
        
        return df
        
    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))
        
    def _bollinger_bands(self, series: pd.Series, period: int, stds: float):
        ma = series.rolling(period).mean()
        std = series.rolling(period).std()
        return ma + (std * stds), ma - (std * stds)

feature_extractor = FreqAIFeatureExtractor()
