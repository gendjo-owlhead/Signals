"""
Backtest smoke test.

Runs a minimal backtest to verify the signal generation pipeline works.
Run with: pytest tests/test_backtest_smoke.py -v
"""
import pytest
import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBacktestSmoke:
    """Smoke test for backtest functionality."""
    
    @pytest.fixture
    def sample_klines(self):
        """Generate sample kline data for testing."""
        from data.binance_ws import Kline
        
        klines = []
        base_time = int(datetime(2024, 1, 1).timestamp() * 1000)
        base_price = 50000.0
        
        for i in range(100):
            # Simple oscillating price pattern
            price_offset = (i % 20 - 10) * 10  # Oscillate +-$100
            price = base_price + price_offset
            
            kline = Kline(
                symbol='BTCUSDT',
                interval='5m',
                open_time=base_time + (i * 300000),
                open=price - 5,
                high=price + 20,
                low=price - 20,
                close=price + 5,
                volume=100.0 + (i % 50),
                close_time=base_time + ((i + 1) * 300000) - 1,
                quote_volume=price * 100,
                trades=500,
                taker_buy_volume=60.0,
                taker_buy_quote_volume=price * 60,
                is_closed=True
            )
            klines.append(kline)
        
        return klines
    
    @pytest.fixture
    def sample_trades(self):
        """Generate sample trade data for testing."""
        from data.binance_ws import Trade
        
        trades = []
        base_time = int(datetime(2024, 1, 1).timestamp() * 1000)
        
        for i in range(200):
            trade = Trade(
                trade_id=i,
                price=50000.0 + (i % 100 - 50),
                quantity=0.01 + (i % 10) * 0.01,
                timestamp=base_time + (i * 1000),
                is_buyer_maker=(i % 3 == 0)
            )
            trades.append(trade)
        
        return trades
    
    def test_volume_profile_calculation(self, sample_klines):
        """Volume profile should calculate from klines."""
        from analysis.volume_profile import VolumeProfileCalculator
        
        calc = VolumeProfileCalculator()
        profile = calc.calculate_from_klines(sample_klines)
        
        assert profile is not None
        # VolumeProfile uses poc_price, vah, val
        assert hasattr(profile, 'levels')
        assert len(profile.levels) > 0
    
    def test_order_flow_analysis(self, sample_trades):
        """Order flow should analyze trades."""
        from analysis.order_flow import order_flow_analyzer
        
        aggression = order_flow_analyzer.analyze_aggression(
            trades=sample_trades,
            symbol='BTCUSDT',
            direction='LONG'
        )
        
        assert aggression is not None
        assert 0 <= aggression.strength <= 1
        assert aggression.description is not None
    
    def test_market_state_analysis(self, sample_klines):
        """Market state should analyze from klines."""
        from analysis.market_state import MarketStateAnalyzer
        from analysis.volume_profile import VolumeProfileCalculator
        
        # First calculate volume profile
        vp_calc = VolumeProfileCalculator()
        profile = vp_calc.calculate_from_klines(sample_klines)
        
        # Then analyze market state
        analyzer = MarketStateAnalyzer()
        state = analyzer.analyze(sample_klines, profile)
        
        assert state is not None
        # state.state is an enum, use .value for string comparison
        assert state.state.value in ['trending_up', 'trending_down', 'balanced', 'breakout', 'rotation']
        assert 0 <= state.confidence <= 1
    
    def test_trend_signal_generation(self, sample_klines, sample_trades):
        """Trend model should attempt signal generation without error."""
        from signals.trend_model import TrendModelGenerator
        
        generator = TrendModelGenerator()
        
        # This may return None (no signal), but should not error
        signal = generator.generate_signal(
            klines=sample_klines,
            trades=sample_trades,
            symbol='BTCUSDT',
            prior_poc=50000.0
        )
        
        # Signal can be None if conditions not met, that's OK
        assert signal is None or signal.symbol == 'BTCUSDT'
    
    def test_mean_reversion_signal_generation(self, sample_klines, sample_trades):
        """Mean reversion model should attempt signal generation without error."""
        from signals.mean_reversion import MeanReversionGenerator
        
        generator = MeanReversionGenerator()
        
        # This may return None (no signal), but should not error
        signal = generator.generate_signal(
            klines=sample_klines,
            trades=sample_trades,
            symbol='BTCUSDT'
        )
        
        # Signal can be None if conditions not met, that's OK
        assert signal is None or signal.symbol == 'BTCUSDT'
    
    def test_ml_confidence_adjustment(self):
        """ML model should adjust confidence scores."""
        from ml.signal_accuracy import SignalAccuracyModel
        
        model = SignalAccuracyModel()
        
        # Test confidence adjustment
        base_confidence = 0.7
        features = {
            'market_state_conf': 0.8,
            'aggression_strength': 0.6,
            'cvd_confirming': True,
            'risk_reward': 2.0
        }
        
        adjusted = model.adjust_confidence(base_confidence, features)
        
        # Should return a valid confidence score
        assert 0 <= adjusted <= 1


class TestBacktestIntegration:
    """Integration test for backtest module."""
    
    def test_backtest_module_imports(self):
        """Backtest module should import."""
        import backtest
        assert hasattr(backtest, 'BacktestResult')
    
    def test_backtest_result_init(self):
        """BacktestResult should initialize."""
        from backtest import BacktestResult
        
        result = BacktestResult(
            symbol='BTCUSDT',
            timeframe='5m',
            start_date='2024-01-01',
            end_date='2024-01-31',
            total_candles=1000
        )
        
        assert result.symbol == 'BTCUSDT'
        assert result.total_signals == 0
        assert result.wins == 0
