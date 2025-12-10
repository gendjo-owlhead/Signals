import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from ml.freqai.features import feature_extractor
from ml.freqai.data_processor import DataProcessor
from ml.freqai.model_manager import ModelManager

@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 105,
        'low': np.random.rand(100) * 95,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000,
        'symbol': ['BTCUSDT'] * 100
    })
    return df

class TestFreqAIFeatures:
    def test_compute_features(self, sample_ohlcv):
        df = feature_extractor.compute_features(sample_ohlcv)
        
        # Check if new columns exist
        assert 'irsi' in df.columns
        assert 'imomentum' in df.columns
        assert 'ibbands_width' in df.columns
        
        # Check simple logic
        assert len(df) == 100

    def test_compute_labels(self, sample_ohlcv):
        df = feature_extractor.compute_labels(sample_ohlcv, shift=2)
        assert '&s_close_shift_2' in df.columns
        # Last 2 rows should be NaN for the label
        assert pd.isna(df.iloc[-1]['&s_close_shift_2'])

class TestDataProcessor:
    def test_normalization(self, tmp_path, sample_ohlcv):
        processor = DataProcessor(tmp_path)
        
        # Add some features
        df = feature_extractor.compute_features(sample_ohlcv)
        df = processor.clean_data(df)
        
        # Train normalize
        df_norm = processor.normalize(df, train_mode=True)
        assert processor.is_fitted
        
        # Check if valid file created
        assert (tmp_path / "scaler.pkl").exists()
        
        # Inference normalize
        df_inf = processor.normalize(df, train_mode=False)
        # Should be roughly same range (standard scaled)
        assert abs(df_inf['irsi'].mean()) < 0.5

class TestModelManager:
    def test_train_predict(self, tmp_path):
        manager = ModelManager(tmp_path, model_type="Sklearn")
        
        # Create X, y
        X = pd.DataFrame({
            'f1': np.random.rand(50),
            'f2': np.random.rand(50)
        })
        y = pd.Series(np.random.rand(50))
        
        # Train
        manager.train(X, y)
        assert (tmp_path / "model.joblib").exists()
        assert (tmp_path / "model_info.json").exists()
        
        # Predict
        preds = manager.predict(X)
        assert len(preds) == 50
