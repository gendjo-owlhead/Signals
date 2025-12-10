import pytest
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

@pytest.fixture(autouse=True)
def mock_settings(tmp_path):
    """
    Automatically mock environment settings for all tests.
    Ensures tests don't write to production data folders.
    """
    # Create temp directory for test data
    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir()
    
    # Override path
    original_path = settings.ml_model_path
    settings.ml_model_path = str(test_data_dir)
    
    # Patch global instances that have already cached the path
    from trading.position_manager import position_manager
    from trading.risk_manager import risk_manager
    
    orig_pos_file = position_manager._positions_file
    orig_hist_file = position_manager._history_file
    orig_risk_stat = risk_manager._stats_file
    
    position_manager._positions_file = test_data_dir / "positions.json"
    position_manager._history_file = test_data_dir / "trade_history.json"
    risk_manager._stats_file = test_data_dir / "risk_stats.json"
    
    yield
    
    # Restore original value
    settings.ml_model_path = original_path
    
    position_manager._positions_file = orig_pos_file
    position_manager._history_file = orig_hist_file
    risk_manager._stats_file = orig_risk_stat
