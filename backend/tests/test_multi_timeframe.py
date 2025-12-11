"""
Tests for Multi-Timeframe Analyzer.
"""
import pytest
from signals.multi_timeframe_analyzer import (
    MultiTimeframeAnalyzer,
    mtf_analyzer,
    TimeframeSignal,
    ConfluenceResult
)


class TestTimeframeWeights:
    """Test timeframe weight configuration."""
    
    def test_weights_sum_to_one(self):
        """Verify timeframe weights sum to 1.0."""
        total = sum(MultiTimeframeAnalyzer.TIMEFRAME_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"
    
    def test_all_timeframes_have_weights(self):
        """Verify all expected timeframes have weights."""
        expected_tfs = ["1m", "5m", "15m", "1h"]
        for tf in expected_tfs:
            assert tf in MultiTimeframeAnalyzer.TIMEFRAME_WEIGHTS, f"Missing weight for {tf}"
    
    def test_higher_timeframes_have_more_weight(self):
        """Verify 15m and 1h have greater weight than 1m."""
        weights = MultiTimeframeAnalyzer.TIMEFRAME_WEIGHTS
        assert weights["15m"] > weights["1m"]
        assert weights["1h"] > weights["1m"]


class TestConfluenceCalculation:
    """Test confluence score calculation."""
    
    def test_empty_signals_returns_neutral(self):
        """Empty signals should return NEUTRAL with 0 confluence."""
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.get_confluence("UNKNOWN_SYMBOL")
        
        assert result.overall_direction == "NEUTRAL"
        assert result.confluence_score == 0.0
        assert result.aligned_timeframes == 0
    
    def test_all_long_signals_high_confluence(self):
        """All timeframes showing LONG should have high confluence."""
        analyzer = MultiTimeframeAnalyzer()
        
        # Update all timeframes with LONG signals
        for tf in ["1m", "5m", "15m", "1h"]:
            analyzer.update_signal(
                symbol="BTCUSDT",
                timeframe=tf,
                direction="LONG",
                strength=0.8,
                ema_trend="UP",
                stoch_k=30.0,
                confidence=0.7
            )
        
        result = analyzer.get_confluence("BTCUSDT")
        
        assert result.overall_direction == "LONG"
        assert result.confluence_score > 0.5
        assert result.aligned_timeframes == 4
    
    def test_all_short_signals_high_confluence(self):
        """All timeframes showing SHORT should have high confluence."""
        analyzer = MultiTimeframeAnalyzer()
        
        for tf in ["1m", "5m", "15m", "1h"]:
            analyzer.update_signal(
                symbol="ETHUSDT",
                timeframe=tf,
                direction="SHORT",
                strength=0.8,
                ema_trend="DOWN",
                stoch_k=70.0,
                confidence=0.7
            )
        
        result = analyzer.get_confluence("ETHUSDT")
        
        assert result.overall_direction == "SHORT"
        assert result.confluence_score > 0.5
        assert result.aligned_timeframes == 4
    
    def test_mixed_signals_lower_confluence(self):
        """Mixed direction signals should have lower confluence."""
        analyzer = MultiTimeframeAnalyzer()
        
        # Mix of LONG and SHORT
        analyzer.update_signal("BTCUSDT", "1m", "LONG", 0.8, "UP", 30.0, 0.7)
        analyzer.update_signal("BTCUSDT", "5m", "LONG", 0.8, "UP", 35.0, 0.7)
        analyzer.update_signal("BTCUSDT", "15m", "SHORT", 0.8, "DOWN", 70.0, 0.7)
        analyzer.update_signal("BTCUSDT", "1h", "SHORT", 0.8, "DOWN", 75.0, 0.7)
        
        result = analyzer.get_confluence("BTCUSDT")
        
        # Should be either NEUTRAL or low confluence
        assert result.confluence_score < 0.7
        assert result.aligned_timeframes < 4


class TestTradeAlignment:
    """Test trade alignment validation."""
    
    def test_aligned_trade_approved(self):
        """Trade direction matching MTF should be approved."""
        analyzer = MultiTimeframeAnalyzer()
        
        # Set up strong LONG signals
        for tf in ["1m", "5m", "15m", "1h"]:
            analyzer.update_signal("BTCUSDT", tf, "LONG", 0.8, "UP", 25.0, 0.8)
        
        assert analyzer.is_trade_aligned("BTCUSDT", "LONG") is True
    
    def test_opposing_trade_rejected(self):
        """Trade direction opposing MTF should be rejected."""
        analyzer = MultiTimeframeAnalyzer()
        
        # Set up strong LONG signals
        for tf in ["1m", "5m", "15m", "1h"]:
            analyzer.update_signal("BTCUSDT", tf, "LONG", 0.8, "UP", 25.0, 0.8)
        
        # SHORT trade should be rejected
        assert analyzer.is_trade_aligned("BTCUSDT", "SHORT") is False
    
    def test_low_confluence_rejects_trade(self):
        """Low confluence should reject even aligned trades."""
        analyzer = MultiTimeframeAnalyzer()
        
        # Only 1m shows LONG, rest are neutral/weak
        analyzer.update_signal("BTCUSDT", "1m", "LONG", 0.3, "NEUTRAL", 50.0, 0.4)
        
        # Should reject due to low confluence
        assert analyzer.is_trade_aligned("BTCUSDT", "LONG") is False


class TestSignalUpdate:
    """Test signal update functionality."""
    
    def test_update_creates_signal(self):
        """Updating signal should create entry in current_signals."""
        analyzer = MultiTimeframeAnalyzer()
        
        analyzer.update_signal(
            symbol="BTCUSDT",
            timeframe="5m",
            direction="LONG",
            strength=0.7,
            ema_trend="UP",
            stoch_k=35.0,
            confidence=0.65
        )
        
        assert "BTCUSDT" in analyzer.current_signals
        assert "5m" in analyzer.current_signals["BTCUSDT"]
        
        signal = analyzer.current_signals["BTCUSDT"]["5m"]
        assert signal.direction == "LONG"
        assert signal.stoch_k == 35.0
    
    def test_get_all_timeframe_signals(self):
        """Should return all signals for a symbol as dicts."""
        analyzer = MultiTimeframeAnalyzer()
        
        analyzer.update_signal("BTCUSDT", "1m", "LONG", 0.5, "UP", 30.0, 0.6)
        analyzer.update_signal("BTCUSDT", "5m", "LONG", 0.6, "UP", 35.0, 0.7)
        
        signals = analyzer.get_all_timeframe_signals("BTCUSDT")
        
        assert "1m" in signals
        assert "5m" in signals
        assert signals["1m"]["direction"] == "LONG"
    
    def test_clear_symbol(self):
        """Clearing symbol should remove all signals."""
        analyzer = MultiTimeframeAnalyzer()
        
        analyzer.update_signal("BTCUSDT", "1m", "LONG", 0.5, "UP", 30.0, 0.6)
        analyzer.clear_symbol("BTCUSDT")
        
        assert "BTCUSDT" not in analyzer.current_signals


class TestConfluenceResult:
    """Test ConfluenceResult dataclass."""
    
    def test_to_dict(self):
        """ConfluenceResult should convert to dict properly."""
        result = ConfluenceResult(
            overall_direction="LONG",
            confluence_score=0.75,
            aligned_timeframes=3,
            total_timeframes=4,
            weighted_confidence=0.7,
            reasoning="1h=LONG | 15m=LONG | 5m=LONG | 1m=NEUTRAL"
        )
        
        d = result.to_dict()
        
        assert d["overall_direction"] == "LONG"
        assert d["confluence_score"] == 0.75
        assert d["aligned_timeframes"] == 3
        assert "reasoning" in d
