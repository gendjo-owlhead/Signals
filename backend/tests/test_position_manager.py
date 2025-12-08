"""
Position Manager Unit Tests.

Tests for position lifecycle, P&L calculations, and state management.
Run with: pytest tests/test_position_manager.py -v
"""
import pytest
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestManagedPosition:
    """Tests for ManagedPosition dataclass."""
    
    def test_position_long_pnl_calculation(self):
        """Long position P&L should be (current - entry) * quantity."""
        from trading.position_manager import ManagedPosition
        
        position = ManagedPosition(
            id="test_1",
            symbol="BTCUSDT",
            direction="LONG",
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.now().isoformat(),
            stop_loss=49000.0,
            take_profit=52000.0,
            entry_order_id="order_123"
        )
        
        # Price went up to 51000 - should be profit
        pnl = position.calculate_pnl(51000.0)
        expected_pnl = (51000.0 - 50000.0) * 0.1  # $100
        assert pnl == pytest.approx(expected_pnl, rel=1e-5)
    
    def test_position_short_pnl_calculation(self):
        """Short position P&L should be (entry - current) * quantity."""
        from trading.position_manager import ManagedPosition
        
        position = ManagedPosition(
            id="test_2",
            symbol="BTCUSDT",
            direction="SHORT",
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.now().isoformat(),
            stop_loss=51000.0,
            take_profit=48000.0,
            entry_order_id="order_456"
        )
        
        # Price went down to 49000 - should be profit for SHORT
        pnl = position.calculate_pnl(49000.0)
        expected_pnl = (50000.0 - 49000.0) * 0.1  # $100
        assert pnl == pytest.approx(expected_pnl, rel=1e-5)
    
    def test_position_to_dict(self):
        """Position should serialize to dictionary correctly."""
        from trading.position_manager import ManagedPosition
        
        position = ManagedPosition(
            id="test_3",
            symbol="ETHUSDT",
            direction="LONG",
            quantity=1.0,
            entry_price=2000.0,
            entry_time="2024-01-01T12:00:00",
            stop_loss=1900.0,
            take_profit=2200.0,
            entry_order_id="order_789"
        )
        
        data = position.to_dict()
        assert data['id'] == "test_3"
        assert data['symbol'] == "ETHUSDT"
        assert data['direction'] == "LONG"
        assert data['entry_price'] == 2000.0


class TestPositionManager:
    """Tests for PositionManager class."""
    
    @pytest.fixture
    def fresh_manager(self, tmp_path, monkeypatch):
        """Create a PositionManager with temporary storage."""
        from trading.position_manager import position_manager
        
        # Save and reset positions for clean test
        original_positions = position_manager.positions.copy()
        position_manager.positions = {}
        
        yield position_manager
        
        # Restore original state
        position_manager.positions = original_positions
    
    def test_add_position_creates_entry(self, fresh_manager):
        """Adding a position should create it with correct fields."""
        position = fresh_manager.add_position(
            symbol="BTCUSDT",
            direction="LONG",
            quantity=0.05,
            entry_price=45000.0,
            stop_loss=44000.0,
            take_profit=47000.0,
            entry_order_id="test_entry_001"
        )
        
        assert position is not None
        assert position.symbol == "BTCUSDT"
        assert position.direction == "LONG"
        assert position.quantity == 0.05
        assert position.entry_price == 45000.0
        assert position.status == "OPEN"
    
    def test_has_position_check(self, fresh_manager):
        """has_position should return True after adding position."""
        assert fresh_manager.has_position("BTCUSDT") is False
        
        fresh_manager.add_position(
            symbol="BTCUSDT",
            direction="LONG",
            quantity=0.05,
            entry_price=45000.0,
            stop_loss=44000.0,
            take_profit=47000.0,
            entry_order_id="test_entry_002"
        )
        
        assert fresh_manager.has_position("BTCUSDT") is True
    
    def test_get_position_by_symbol(self, fresh_manager):
        """get_position_by_symbol should return the correct position."""
        fresh_manager.add_position(
            symbol="ETHUSDT",
            direction="SHORT",
            quantity=1.0,
            entry_price=2500.0,
            stop_loss=2600.0,
            take_profit=2300.0,
            entry_order_id="test_entry_003"
        )
        
        position = fresh_manager.get_position_by_symbol("ETHUSDT")
        assert position is not None
        assert position.symbol == "ETHUSDT"
        assert position.direction == "SHORT"
        
        # Non-existent symbol should return None
        assert fresh_manager.get_position_by_symbol("SOLUSDT") is None
    
    def test_close_position_calculates_pnl(self, fresh_manager):
        """Closing a position should calculate and return correct P&L."""
        position = fresh_manager.add_position(
            symbol="BTCUSDT",
            direction="LONG",
            quantity=0.1,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            entry_order_id="test_entry_004"
        )
        
        # Close with TP hit
        pnl = fresh_manager.close_position(position.id, exit_price=52000.0, reason="TP_HIT")
        
        expected_pnl = (52000.0 - 50000.0) * 0.1  # $200
        assert pnl == pytest.approx(expected_pnl, rel=1e-5)
        
        # Position should be removed from open positions
        assert fresh_manager.has_position("BTCUSDT") is False
    
    def test_positions_count(self, fresh_manager):
        """get_positions_count should return correct number."""
        assert fresh_manager.get_positions_count() == 0
        
        fresh_manager.add_position(
            symbol="BTCUSDT",
            direction="LONG",
            quantity=0.1,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            entry_order_id="test_entry_005"
        )
        
        assert fresh_manager.get_positions_count() == 1
        
        fresh_manager.add_position(
            symbol="ETHUSDT",
            direction="SHORT",
            quantity=1.0,
            entry_price=2500.0,
            stop_loss=2600.0,
            take_profit=2300.0,
            entry_order_id="test_entry_006"
        )
        
        assert fresh_manager.get_positions_count() == 2


class TestPositionStats:
    """Tests for position statistics tracking."""
    
    def test_get_stats_structure(self):
        """get_stats should return properly structured data."""
        from trading.position_manager import position_manager
        
        stats = position_manager.get_stats()
        
        assert 'open_positions' in stats
        assert 'total_trades' in stats
        assert 'wins' in stats
        assert 'losses' in stats
