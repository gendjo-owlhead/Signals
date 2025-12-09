"""
Sanity tests for Auction Market Signal Generator.

Quick checks that core components initialize correctly.
Run with: pytest tests/ -v
"""
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfigLoading:
    """Test configuration loads correctly."""
    
    def test_settings_import(self):
        """Config module should import without errors."""
        from config import settings
        assert settings is not None
    
    def test_required_settings_exist(self):
        """Core settings should have values (even defaults)."""
        from config import settings
        
        assert hasattr(settings, 'trading_pairs')
        assert hasattr(settings, 'timeframes')
        assert hasattr(settings, 'binance_testnet')
        assert isinstance(settings.trading_pairs, list)
        assert len(settings.trading_pairs) > 0
    
    def test_api_url_functions(self):
        """API URL helper functions should work."""
        from config import get_ws_url, get_api_url
        
        ws_url = get_ws_url()
        api_url = get_api_url()
        
        assert ws_url.startswith('wss://')
        assert api_url.startswith('https://')


class TestSignalModels:
    """Test signal generation models initialize correctly."""
    
    def test_scalper_generator_import(self):
        """ScalperGenerator should import."""
        from signals.scalper_strategy import ScalperGenerator
        assert ScalperGenerator is not None
    
    def test_scalper_generator_init(self):
        """ScalperGenerator should initialize with defaults."""
        from signals.scalper_strategy import ScalperGenerator
        
        generator = ScalperGenerator()
        assert generator is not None
        assert generator.rr_ratio >= 1.0
    
    def test_scalper_signal_import(self):
        """ScalperSignal should import."""
        from signals.scalper_strategy import ScalperSignal
        assert ScalperSignal is not None
    
    def test_global_scalper_generator(self):
        """Global scalper_generator instance should be available."""
        from signals.scalper_strategy import scalper_generator
        assert scalper_generator is not None
        assert scalper_generator.ema_fast > 0


class TestMLModels:
    """Test ML components initialize correctly."""
    
    def test_signal_accuracy_model_import(self):
        """SignalAccuracyModel should import."""
        from ml.signal_accuracy import SignalAccuracyModel
        assert SignalAccuracyModel is not None
    
    def test_signal_accuracy_model_init(self):
        """SignalAccuracyModel should initialize."""
        from ml.signal_accuracy import SignalAccuracyModel
        
        model = SignalAccuracyModel()
        assert model is not None
        assert hasattr(model, 'feature_weights')
    
    def test_lvn_pattern_recognizer_import(self):
        """LVNPatternRecognizer should import."""
        from ml.lvn_patterns import LVNPatternRecognizer
        assert LVNPatternRecognizer is not None
    
    def test_state_classifier_import(self):
        """MarketStateClassifier should import."""
        from ml.state_classifier import MarketStateClassifier
        assert MarketStateClassifier is not None


class TestAnalysisComponents:
    """Test analysis components."""
    
    def test_volume_profile_import(self):
        """VolumeProfileCalculator should import."""
        from analysis.volume_profile import VolumeProfileCalculator
        assert VolumeProfileCalculator is not None
    
    def test_order_flow_import(self):
        """OrderFlowAnalyzer should import."""
        from analysis.order_flow import OrderFlowAnalyzer
        assert OrderFlowAnalyzer is not None
    
    def test_market_analyzer_import(self):
        """MarketStateAnalyzer should import."""
        from analysis.market_state import MarketStateAnalyzer
        assert MarketStateAnalyzer is not None


class TestRiskCalculations:
    """Test risk management calculations."""
    
    def test_risk_manager_init(self):
        """RiskManager should initialize."""
        from trading.risk_manager import RiskManager
        
        rm = RiskManager()
        assert rm is not None
        assert hasattr(rm, 'limits')
    
    def test_position_size_calculation(self):
        """Position size calculation should work correctly."""
        from trading.risk_manager import RiskManager
        
        rm = RiskManager()
        
        # Test with $1000 balance, entry at 50000, SL at 49000 (2% away)
        account_balance = 1000.0
        entry_price = 50000.0
        stop_loss_price = 49000.0
        
        position_size = rm.calculate_position_size(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )
        
        assert position_size > 0
    
    def test_can_trade_check(self):
        """Can trade check should work."""
        from trading.risk_manager import RiskManager
        
        rm = RiskManager()
        
        # Check with high confidence and 0 positions
        can_trade, reason = rm.can_trade(confidence=0.9, current_positions=0)
        
        # Should return tuple
        assert isinstance(can_trade, bool)
        assert isinstance(reason, str)
    
    def test_kill_switch_blocks_trading(self):
        """Kill switch should block trading when is_enabled=False."""
        from trading.risk_manager import RiskManager
        
        rm = RiskManager()
        
        # Disable trading (is_enabled = False blocks trading)
        rm.is_enabled = False
        can_trade, reason = rm.can_trade(confidence=0.95, current_positions=0)
        
        assert can_trade is False
        assert "kill" in reason.lower() or "disabled" in reason.lower()
        
        # Re-enable trading
        rm.is_enabled = True
    
    def test_low_confidence_rejected(self):
        """Low confidence signals should be rejected."""
        from trading.risk_manager import RiskManager
        
        rm = RiskManager()
        rm.is_enabled = True  # Ensure trading is enabled
        
        # Very low confidence
        can_trade, reason = rm.can_trade(confidence=0.3, current_positions=0)
        
        assert can_trade is False
        assert "confidence" in reason.lower()
    
    def test_max_positions_respected(self):
        """Trading should be blocked when max positions reached."""
        from trading.risk_manager import RiskManager
        
        rm = RiskManager()
        rm.is_enabled = True  # Ensure trading is enabled
        
        # Try with max positions (default is usually 3)
        max_pos = rm.limits.max_concurrent_positions
        can_trade, reason = rm.can_trade(confidence=0.9, current_positions=max_pos)
        
        assert can_trade is False
        assert "position" in reason.lower() or "max" in reason.lower()
    
    def test_position_size_respects_max(self):
        """Position size should not exceed max limit."""
        from trading.risk_manager import RiskManager
        
        rm = RiskManager()
        
        # Large balance that would normally give huge position
        position_size = rm.calculate_position_size(
            account_balance=1000000.0,
            entry_price=50000.0,
            stop_loss_price=49900.0  # Very tight SL
        )
        
        # Should be capped at max_position_size_usdt
        max_usdt = rm.limits.max_position_size_usdt
        # position_size is in quote currency value
        assert position_size * 50000 <= max_usdt or position_size <= max_usdt / 50000
    
    def test_risk_status_structure(self):
        """get_status should return properly structured data."""
        from trading.risk_manager import RiskManager
        
        rm = RiskManager()
        status = rm.get_status()
        
        assert 'daily_pnl' in status
        assert 'trades_today' in status
        assert 'is_enabled' in status
        assert 'limits' in status

class TestDataModels:
    """Test data model definitions."""
    
    def test_kline_model(self):
        """Kline model should work."""
        from data.binance_ws import Kline
        
        kline = Kline(
            symbol='BTCUSDT',
            interval='5m',
            open_time=1000000000,
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            close_time=1000000300,
            quote_volume=5000000.0,
            trades=1000,
            taker_buy_volume=60.0,
            taker_buy_quote_volume=3000000.0,
            is_closed=True
        )
        
        assert kline.symbol == 'BTCUSDT'
        assert kline.close == 50050.0
    
    def test_trade_model(self):
        """Trade model should work."""
        from data.binance_ws import Trade
        
        trade = Trade(
            trade_id=12345,
            price=50000.0,
            quantity=0.1,
            timestamp=1000000000,
            is_buyer_maker=True
        )
        
        assert trade.price == 50000.0
        assert trade.is_buyer_maker is True
        assert trade.side == 'SELL'
