import pytest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.scalper_strategy import ScalperGenerator, ScalperSignal
from data.binance_ws import Kline

@pytest.fixture
def mock_klines():
    """Create a list of mock klines for testing."""
    klines = []
    base_price = 50000.0
    
    # Generate 100 candles
    for i in range(100):
        # Create a simple trend for testing
        price = base_price + (i * 10)
        
        kline = Kline(
            symbol="BTCUSDT",
            interval="1m",
            open_time=1000 + i * 60,
            open=price,
            high=price + 5,
            low=price - 5,
            close=price,
            volume=100.0,
            close_time=1000 + i * 60 + 59,
            quote_volume=price * 100,
            trades=50,
            taker_buy_volume=50.0,
            taker_buy_quote_volume=price * 50,
            is_closed=True
        )
        klines.append(kline)
    return klines

class TestScalperGenerator:
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default
        gen = ScalperGenerator()
        assert gen.ema_fast == 5
        assert gen.ema_mid == 8
        assert gen.ema_slow == 13
        assert gen.rr_ratio == 1.5
        
        # Custom
        gen_custom = ScalperGenerator(ema_fast=10, rr_ratio=2.0)
        assert gen_custom.ema_fast == 10
        assert gen_custom.rr_ratio == 2.0

    @patch('signals.scalper_strategy.calculate_ema')
    @patch('signals.scalper_strategy.calculate_stoch_rsi')
    @patch('signals.scalper_strategy.calculate_atr')
    def test_generate_long_signal(self, mock_atr, mock_stoch, mock_ema, mock_klines):
        """Test generation of a valid LONG signal."""
        gen = ScalperGenerator()
        
        # Mock indicator returns
        # Need enough data points for the length of klines (100)
        
        # 1. EMAs setup for Bullish Crossover and Alignment
        # EMA5 crosses above EMA8 at the last step
        # EMA5 > EMA8 > EMA13 at last step
        
        # Create arrays of length 100
        ema5 = np.full(100, 50000.0)
        ema8 = np.full(100, 50000.0)
        ema13 = np.full(100, 49000.0)
        
        # Last index: EMA5 > EMA8 (51000 > 50500)
        ema5[-1] = 51000.0
        ema8[-1] = 50500.0
        ema13[-1] = 50000.0
        
        # Second to last: EMA5 <= EMA8 (50000 <= 50000) for crossover
        ema5[-2] = 50000.0
        ema8[-2] = 50000.0
        
        # Mock side effects for calculate_ema called 3 times (fast, mid, slow)
        mock_ema.side_effect = [ema5, ema8, ema13]
        
        # 2. StochRSI setup usually needs K and D
        # Not overbought (e.g. 40 < 80)
        stoch_k = np.full(100, 40.0)
        stoch_d = np.full(100, 40.0)
        mock_stoch.return_value = (stoch_k, stoch_d)
        
        # 3. ATR setup
        atr = np.full(100, 100.0)
        mock_atr.return_value = atr
        
        # Ensure latest close > EMA13
        # In mock_klines close is ~ 50000 + 990 = 50990
        # EMA13 is 50000, so Price > EMA13 holds.
        # Wait, mock_klines last price is base + 99*10 = 50000 + 990 = 50990. 
        # EMA13[-1] is 50000. So 50990 > 50000. Correct.
        
        signal = gen.generate_signal(mock_klines, [], "BTCUSDT")
        
        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.symbol == "BTCUSDT"
        assert signal.confidence > 0
        assert bool(signal.ema_crossover) is True
        assert bool(signal.ema_aligned) is True

    @patch('signals.scalper_strategy.calculate_ema')
    @patch('signals.scalper_strategy.calculate_stoch_rsi')
    @patch('signals.scalper_strategy.calculate_atr')
    def test_generate_short_signal(self, mock_atr, mock_stoch, mock_ema, mock_klines):
        """Test generation of a valid SHORT signal."""
        gen = ScalperGenerator()
        
        # 1. EMAs setup for Bearish Crossover and Alignment
        # EMA5 < EMA8 < EMA13
        
        ema5 = np.full(100, 50000.0)
        ema8 = np.full(100, 50000.0)
        ema13 = np.full(100, 51000.0)
        
        # Last index: EMA5 < EMA8 (49000 < 49500)
        ema5[-1] = 49000.0
        ema8[-1] = 49500.0
        ema13[-1] = 50000.0
        
        # Second to last: EMA5 >= EMA8 (50000 >= 50000)
        ema5[-2] = 50000.0
        ema8[-2] = 50000.0
        
        # Note: Price needs to be < EMA13.
        # Last kline close is 50990 (from fixture). 
        # We need mock_klines to have lower price for SHORT test or adjust EMA13.
        # Let's adjust EMA13 to be HIGHER than price. Price is ~51000.
        ema13[-1] = 52000.0 
        ema5[-1] = 49000.0
        ema8[-1] = 49500.0
        
        mock_ema.side_effect = [ema5, ema8, ema13]
        
        # 2. StochRSI setup
        # Not oversold (e.g. 60 > 20)
        stoch_k = np.full(100, 60.0)
        stoch_d = np.full(100, 60.0)
        mock_stoch.return_value = (stoch_k, stoch_d)
        
        # 3. ATR
        atr = np.full(100, 100.0)
        mock_atr.return_value = atr
        
        signal = gen.generate_signal(mock_klines, [], "BTCUSDT")
        
        assert signal is not None
        assert signal.direction == "SHORT"
        assert signal.confidence > 0

    def test_insufficient_data(self):
        """Test that None is returned when not enough klines."""
        gen = ScalperGenerator()
        klines = [] # Empty
        result = gen.generate_signal(klines, [], "BTCUSDT")
        assert result is None

    @patch('signals.scalper_strategy.calculate_ema')
    def test_no_crossover(self, mock_ema, mock_klines):
        """Test that None is returned when there is no crossover."""
        gen = ScalperGenerator()
        
        # EMAs parallel, no cross
        ema5 = np.full(100, 50000.0)
        ema8 = np.full(100, 49000.0)
        ema13 = np.full(100, 48000.0)
        
        mock_ema.side_effect = [ema5, ema8, ema13]
        
        result = gen.generate_signal(mock_klines, [], "BTCUSDT")
        assert result is None
    
    def test_calculate_confidence(self):
        """Test confidence calculation logic."""
        gen = ScalperGenerator()
        
        # Case 1: All perfect
        conf = gen._calculate_confidence(
            ema_aligned=True,
            ema_crossover=True,
            stoch_k=40,
            stoch_confirming=True,
            direction="LONG"
        )
        # Base 0.1 + Cross 0.3 + Align 0.3 + Stoch 0.2 + Momentum 0.1 = 1.0
        assert conf >= 1.0 or conf == pytest.approx(1.0)
        
        # Case 2: No Stoch confirmation
        conf = gen._calculate_confidence(
            ema_aligned=True,
            ema_crossover=True,
            stoch_k=90, # Overbought, bad for LONG
            stoch_confirming=False,
            direction="LONG"
        )
        # Base 0.1 + Cross 0.3 + Align 0.3 = 0.7
        assert conf == 0.7

class TestScalperSignal:
    
    def test_is_valid(self):
        """Test signal validation logic."""
        sig = ScalperSignal(
            timestamp=1000,
            symbol="BTCUSDT",
            timeframe="1m",
            direction="LONG",
            entry_price=50000,
            stop_loss=49000,
            take_profit=51500, # 1.5 R:R
            atr_value=100,
            confidence=0.8,
            ema5=50000, ema8=49900, ema13=49800,
            ema_aligned=True, ema_crossover=True,
            stoch_k=50, stoch_d=50,
            risk_reward=1.5,
            risk_percent=1.0
        )
        assert sig.is_valid()
        
        # Invalid R:R
        sig.risk_reward = 0.5
        assert not sig.is_valid()
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        sig = ScalperSignal(
            timestamp=1000,
            symbol="BTCUSDT",
            timeframe="1m",
            direction="LONG",
            entry_price=50000,
            stop_loss=49000,
            take_profit=51500,
            atr_value=100,
            confidence=0.8,
            ema5=50000, ema8=49900, ema13=49800,
            ema_aligned=True, ema_crossover=True,
            stoch_k=50, stoch_d=50,
            risk_reward=1.5,
            risk_percent=1.0
        )
        data = sig.to_dict()
        assert data['symbol'] == "BTCUSDT"
        assert data['direction'] == "LONG"
        assert data['confidence'] == 0.8
