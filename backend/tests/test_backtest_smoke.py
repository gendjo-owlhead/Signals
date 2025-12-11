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


class TestBacktestOrchestrator:
    """Tests for BacktestOrchestrator."""
    
    def test_orchestrator_import(self):
        """BacktestOrchestrator should import."""
        from backtest_orchestrator import BacktestOrchestrator, BACKTEST_MATRIX
        assert BacktestOrchestrator is not None
        assert len(BACKTEST_MATRIX) > 0
    
    def test_config_matrix_completeness(self):
        """Verify all 8 symbol/timeframe combinations are configured."""
        from backtest_orchestrator import BACKTEST_MATRIX
        
        # Should have 2 symbols x 4 timeframes = 8 configs
        assert len(BACKTEST_MATRIX) == 8
        
        # Verify BTC across all timeframes
        btc_timeframes = [c.timeframe for c in BACKTEST_MATRIX if c.symbol == "BTCUSDT"]
        assert "1m" in btc_timeframes
        assert "5m" in btc_timeframes
        assert "15m" in btc_timeframes
        assert "1h" in btc_timeframes
        
        # Verify ETH across all timeframes
        eth_timeframes = [c.timeframe for c in BACKTEST_MATRIX if c.symbol == "ETHUSDT"]
        assert len(eth_timeframes) == 4
    
    def test_config_days_by_timeframe(self):
        """Verify correct number of days configured per timeframe."""
        from backtest_orchestrator import BACKTEST_MATRIX
        
        for config in BACKTEST_MATRIX:
            if config.timeframe == "1m":
                assert config.days == 365, f"1m should have 365 days, got {config.days}"
            elif config.timeframe in ["5m", "15m"]:
                assert config.days == 1095, f"{config.timeframe} should have 1095 days"
            elif config.timeframe == "1h":
                assert config.days == 1825, f"1h should have 1825 days"
    
    def test_orchestrator_init(self):
        """Orchestrator should initialize properly."""
        from backtest_orchestrator import BacktestOrchestrator
        
        orch = BacktestOrchestrator(parallel_limit=1)
        
        assert orch.parallel_limit == 1
        assert orch.running is False
        assert len(orch.current_runs) == 0
    
    def test_get_status(self):
        """get_status should return expected structure."""
        from backtest_orchestrator import BacktestOrchestrator
        
        orch = BacktestOrchestrator()
        status = orch.get_status()
        
        assert "running" in status
        assert "pending_backtests" in status
        assert "current_backtests" in status
        assert "aggregate_stats" in status
        assert "matrix_size" in status
        assert status["matrix_size"] == 8
    
    def test_get_performance_summary(self):
        """get_performance_summary should return expected structure."""
        from backtest_orchestrator import BacktestOrchestrator
        
        orch = BacktestOrchestrator()
        summary = orch.get_performance_summary()
        
        assert "by_symbol" in summary
        assert "by_timeframe" in summary
        assert "overall" in summary
    
    def test_backtest_config_id(self):
        """BacktestConfig should generate correct ID."""
        from backtest_orchestrator import BacktestConfig
        
        config = BacktestConfig("BTCUSDT", "5m", 365, 24)
        
        assert config.id == "BTCUSDT_5m"
        assert config.symbol == "BTCUSDT"
        assert config.timeframe == "5m"
