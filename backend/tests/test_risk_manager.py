import pytest
import json
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.risk_manager import RiskManager, DailyStats

class TestRiskManager:
    
    @pytest.fixture
    def risk_manager(self):
        with patch('builtins.open', new_callable=MagicMock) as mock_open:
            # Mock json.load to return empty or default stats
            with patch('json.load') as mock_json:
                mock_json.return_value = {
                    'date': str(date.today()),
                    'starting_balance': 1000.0,
                    'realized_pnl': 0.0,
                    'trades_count': 0,
                    'wins': 0,
                    'losses': 0
                }
                rm = RiskManager()
                # Reset to known clean state
                rm.daily_stats = DailyStats(str(date.today()), 1000.0)
                return rm

    def test_daily_loss_limit(self, risk_manager):
        """Test that trading stops when daily loss limit is reached."""
        # Limit is 5% by default
        # Balance 1000, so 5% is $50.
        
        # Simulate loss
        risk_manager.record_trade_result(-51.0, is_win=False)
        
        can_trade, reason = risk_manager.can_trade(0.9, 0)
        assert can_trade is False
        assert "Daily loss limit" in reason

    def test_kill_switch(self, risk_manager):
        """Test manual kill switch."""
        risk_manager.kill_switch(False) # Enable kill switch (disable trading)
        can_trade, reason = risk_manager.can_trade(0.9, 0)
        assert can_trade is False
        assert "kill switch" in reason.lower()
        
        risk_manager.kill_switch(True) # Disable kill switch (enable trading)
        can_trade, _ = risk_manager.can_trade(0.9, 0)
        assert can_trade is True

    def test_cooldown_logic(self, risk_manager):
        """Test cooldown after consecutive losses."""
        # Consecutive losses limit is 3
        
        # 1st loss
        risk_manager.record_trade_result(-10, False)
        assert risk_manager.consecutive_losses == 1
        assert risk_manager.cooldown_until is None
        
        # 2nd loss
        risk_manager.record_trade_result(-10, False)
        assert risk_manager.consecutive_losses == 2
        
        # 3rd loss - should trigger cooldown
        risk_manager.record_trade_result(-10, False)
        assert risk_manager.consecutive_losses == 3
        assert risk_manager.cooldown_until is not None
        
        # Try to trade
        can_trade, reason = risk_manager.can_trade(0.9, 0)
        assert can_trade is False
        assert "Cooldown active" in reason
        
        # Simulate time passing (reset cooldown manually for test)
        risk_manager.cooldown_until = datetime.now() - timedelta(minutes=1)
        can_trade, _ = risk_manager.can_trade(0.9, 0)
        assert can_trade is True

    def test_win_resets_streak(self, risk_manager):
        """Test that a win resets the consecutive loss streak."""
        risk_manager.record_trade_result(-10, False)
        risk_manager.record_trade_result(-10, False)
        assert risk_manager.consecutive_losses == 2
        
        # Win
        risk_manager.record_trade_result(20, True)
        assert risk_manager.consecutive_losses == 0

    def test_calculate_position_size(self, risk_manager):
        """Test position sizing formula."""
        # Account $1000, Risk 1% = $10 risk amount
        # Entry $100, SL $90 (Distance $10) -> Size = 1 unit
        
        size = risk_manager.calculate_position_size(1000.0, 100.0, 90.0)
        assert size == 1.0
        
        # Test max position cap
        # Entry $100, SL $99 (Distance $1) -> Size = 10 units -> $1000 value
        # If max position is $110 (default in code), it should be capped
        
        risk_manager.limits.max_position_size_usdt = 110.0
        size_capped = risk_manager.calculate_position_size(1000.0, 100.0, 99.0)
        # 110 / 100 = 1.1 units
        assert size_capped == 1.1
