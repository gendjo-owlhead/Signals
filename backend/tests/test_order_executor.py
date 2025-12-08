"""
Order Executor Unit Tests.

Tests for signal validation and execution flow logic.
Run with: pytest tests/test_order_executor.py -v
"""
import pytest
import sys
import os
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class MockSignal:
    """Mock signal for testing validation logic."""
    symbol: str = "BTCUSDT"
    direction: str = "LONG"
    entry_price: float = 50000.0
    stop_loss: float = 49000.0
    take_profit: float = 52000.0
    confidence: float = 0.8
    risk_reward: float = 2.0


class TestSignalValidation:
    """Tests for OrderExecutor signal validation."""
    
    @pytest.fixture
    def executor(self):
        """Get the order executor instance."""
        from trading.order_executor import OrderExecutor
        return OrderExecutor()
    
    def test_valid_signal_passes(self, executor):
        """A valid signal should pass validation."""
        signal = MockSignal()
        valid, reason = executor.validate_signal(signal)
        
        assert valid is True
        assert reason == "OK"
    
    def test_missing_entry_price_fails(self, executor):
        """Signal without entry price should fail."""
        signal = MockSignal(entry_price=None)
        valid, reason = executor.validate_signal(signal)
        
        assert valid is False
        assert "entry" in reason.lower() or "missing" in reason.lower()
    
    def test_missing_stop_loss_fails(self, executor):
        """Signal without stop loss should fail."""
        signal = MockSignal(stop_loss=None)
        valid, reason = executor.validate_signal(signal)
        
        assert valid is False
        assert "missing" in reason.lower()
    
    def test_missing_take_profit_fails(self, executor):
        """Signal without take profit should fail."""
        signal = MockSignal(take_profit=None)
        valid, reason = executor.validate_signal(signal)
        
        assert valid is False
        assert "missing" in reason.lower()
    
    def test_invalid_direction_fails(self, executor):
        """Signal with invalid direction should fail."""
        signal = MockSignal(direction="SIDEWAYS")
        valid, reason = executor.validate_signal(signal)
        
        assert valid is False
        assert "direction" in reason.lower()
    
    def test_low_risk_reward_fails(self, executor):
        """Signal with R:R < 1.0 should fail."""
        signal = MockSignal(risk_reward=0.5)
        valid, reason = executor.validate_signal(signal)
        
        assert valid is False
        assert "risk" in reason.lower() or "reward" in reason.lower()


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""
    
    def test_result_to_dict(self):
        """ExecutionResult should serialize correctly."""
        from trading.order_executor import ExecutionResult
        
        result = ExecutionResult(
            success=True,
            message="Trade executed"
        )
        
        data = result.to_dict()
        assert data['success'] is True
        assert data['message'] == "Trade executed"
        assert data['position_id'] is None


class TestOrderExecutorStatus:
    """Tests for OrderExecutor status reporting."""
    
    def test_get_status_structure(self):
        """get_status should return properly structured data."""
        from trading.order_executor import order_executor
        
        status = order_executor.get_status()
        
        assert 'is_running' in status
        assert 'trading_enabled' in status
        assert 'positions' in status
        assert 'risk' in status
