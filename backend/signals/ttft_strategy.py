"""
TTFT Strategy - Trend Trading with Filters.

Implements a trend-following strategy with multiple confirmation filters:
- Trend Detection: SMA of highs/lows ± ATR bands
- Filters: MACD Zero-Cross, Volume, StochRSI, Awesome Oscillator
- TP/SL: Based on Risk/Reward ratio from trend band
"""
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from config import settings
from data.binance_ws import Kline, Trade
from analysis.indicators import (
    calculate_macd, calculate_stoch_rsi, calculate_awesome_oscillator,
    calculate_atr, calculate_sma, calculate_volume_sma,
    detect_crossover, detect_crossunder, MACDResult, StochRSIResult
)


@dataclass
class TTFTSignal:
    """
    TTFT (Trend Trading with Filters) signal.
    
    Entry: On trend flip with filter confirmation
    Stop Loss: Beyond the trend band
    Take Profit: Based on R:R ratio
    """
    timestamp: int
    symbol: str
    timeframe: str
    direction: str  # "LONG" or "SHORT"
    
    # Entry parameters
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Trend info
    trend_band: float  # Current trend band level
    atr_value: float
    
    # Confidence and filters
    confidence: float
    filters_passed: Dict[str, bool]
    
    # Risk metrics
    risk_reward: float
    risk_percent: float
    
    @property
    def is_valid(self) -> bool:
        """Check if signal has positive risk/reward."""
        return self.risk_reward >= 1.5
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trend_band': self.trend_band,
            'atr_value': self.atr_value,
            'confidence': self.confidence,
            'risk_reward': self.risk_reward,
            'filters_passed': self.filters_passed,
            'model': 'TTFT'
        }


class TTFTGenerator:
    """
    TTFT Signal Generator.
    
    Strategy Logic:
    1. Calculate SMA of highs + ATR (upper band) and SMA of lows - ATR (lower band)
    2. Trend flips to UP when close > upper band
    3. Trend flips to DOWN when close < lower band
    4. Entry signal on trend flip with filter confirmation
    5. SL at opposite trend band, TP at R:R ratio
    """
    
    def __init__(
        self,
        trend_length: int = 20,
        atr_length: int = 200,
        atr_sma_length: int = 200,
        atr_multiplier: float = 0.8,
        sl_multiplier: float = 1.0,
        rr_ratio: float = 2.0,
        confidence_threshold: float = 0.7,
        # Filter toggles
        use_macd_cross: bool = True,
        use_volume: bool = True,
        use_stoch_rsi: bool = True,
        use_ao_color: bool = True,
        filter_lookback: int = 3
    ):
        """
        Initialize TTFT Generator.
        
        Args:
            trend_length: Period for SMA of highs/lows (default 20)
            atr_length: Period for ATR calculation (default 200)
            atr_sma_length: Period for smoothing ATR (default 200)
            atr_multiplier: Multiplier for entry bands (default 0.8)
            sl_multiplier: Multiplier for stop loss distance (default 1.0)
            rr_ratio: Risk/Reward ratio for take profit (default 2.0)
            confidence_threshold: Minimum confidence for signal (default 0.7)
            use_macd_cross: Enable MACD zero-cross filter
            use_volume: Enable volume above average filter
            use_stoch_rsi: Enable StochRSI filter
            use_ao_color: Enable Awesome Oscillator color filter
            filter_lookback: Number of bars to look back for filter confirmation
        """
        self.trend_length = trend_length
        self.atr_length = atr_length
        self.atr_sma_length = atr_sma_length
        self.atr_multiplier = atr_multiplier
        self.sl_multiplier = sl_multiplier
        self.rr_ratio = rr_ratio
        self.confidence_threshold = confidence_threshold
        
        # Filters
        self.use_macd_cross = use_macd_cross
        self.use_volume = use_volume
        self.use_stoch_rsi = use_stoch_rsi
        self.use_ao_color = use_ao_color
        self.filter_lookback = filter_lookback
        
        # Track trend state per symbol
        self.trend_state: Dict[str, bool] = {}  # True = uptrend, False = downtrend
    
    def generate_signal(
        self,
        klines: List[Kline],
        trades: List[Trade],
        symbol: str
    ) -> Optional[TTFTSignal]:
        """
        Generate TTFT signal from klines.
        
        Args:
            klines: Recent klines (need at least 200 for ATR)
            trades: Recent trades (unused but kept for interface compatibility)
            symbol: Trading pair
        
        Returns:
            TTFTSignal if conditions met, None otherwise
        """
        min_length = max(self.atr_length, self.atr_sma_length, self.trend_length) + 10
        
        if len(klines) < min_length:
            logger.debug(f"{symbol}: Not enough klines ({len(klines)} < {min_length})")
            return None
        
        # Extract price data
        opens = np.array([k.open for k in klines])
        highs = np.array([k.high for k in klines])
        lows = np.array([k.low for k in klines])
        closes = np.array([k.close for k in klines])
        volumes = np.array([k.volume for k in klines])
        
        timeframe = klines[0].interval
        current_price = closes[-1]
        
        # Calculate indicators
        atr = calculate_atr(highs, lows, closes, self.atr_length)
        atr_smoothed = calculate_sma(atr, self.atr_sma_length)
        atr_value = atr_smoothed[-1] * self.atr_multiplier
        
        sma_high = calculate_sma(highs, self.trend_length)
        sma_low = calculate_sma(lows, self.trend_length)
        
        upper_band = sma_high + atr_smoothed * self.atr_multiplier
        lower_band = sma_low - atr_smoothed * self.atr_multiplier
        
        # Detect trend
        prev_trend = self.trend_state.get(symbol)
        
        # Check for trend flip
        if closes[-1] > upper_band[-1]:
            current_trend = True  # Uptrend
        elif closes[-1] < lower_band[-1]:
            current_trend = False  # Downtrend
        else:
            current_trend = prev_trend
        
        # Update trend state
        self.trend_state[symbol] = current_trend
        
        # Check for signal (trend flip)
        signal_up = current_trend == True and prev_trend == False
        signal_down = current_trend == False and prev_trend == True
        
        if prev_trend is None:
            # First time - no signal, just establish trend
            logger.debug(f"{symbol}: Trend established - {'UP' if current_trend else 'DOWN'}")
            return None
        
        if not signal_up and not signal_down:
            return None
        
        direction = "LONG" if signal_up else "SHORT"
        logger.info(f"{symbol}: Trend flip detected - {direction}")
        
        # Calculate filters
        filters_passed = self._check_filters(
            closes, highs, lows, volumes, direction
        )
        
        # Check if all enabled filters pass
        all_filters_pass = all(filters_passed.values())
        
        if not all_filters_pass:
            failed = [k for k, v in filters_passed.items() if not v]
            logger.debug(f"{symbol}: Filters failed - {failed}")
            return None
        
        logger.info(f"{symbol}: All filters passed - {filters_passed}")
        
        # Calculate entry, SL, TP
        entry_price = current_price
        
        if direction == "LONG":
            trend_band = lower_band[-1]
            stop_distance = abs(entry_price - trend_band) * self.sl_multiplier
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * self.rr_ratio)
        else:
            trend_band = upper_band[-1]
            stop_distance = abs(entry_price - trend_band) * self.sl_multiplier
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * self.rr_ratio)
        
        # Calculate risk metrics
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward = reward / risk if risk > 0 else 0
        risk_percent = (risk / entry_price) * 100
        
        # Calculate confidence based on filter strength
        confidence = self._calculate_confidence(closes, highs, lows, volumes, direction)
        
        if confidence < self.confidence_threshold:
            logger.debug(f"{symbol}: Confidence too low ({confidence:.2f})")
            return None
        
        # Generate signal
        signal = TTFTSignal(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trend_band=trend_band,
            atr_value=atr_value,
            confidence=confidence,
            filters_passed=filters_passed,
            risk_reward=risk_reward,
            risk_percent=risk_percent
        )
        
        logger.info(
            f"{symbol}: TTFT SIGNAL - {direction} @ {entry_price:.2f} | "
            f"SL: {stop_loss:.2f} | TP: {take_profit:.2f} | RR: {risk_reward:.2f}"
        )
        
        return signal
    
    def _check_filters(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        direction: str
    ) -> Dict[str, bool]:
        """Check all filter conditions."""
        filters = {}
        
        # MACD Zero-Cross Filter
        if self.use_macd_cross:
            macd = calculate_macd(closes, 12, 26, 9)
            hist = macd.histogram
            
            # Check for crossover/under in lookback period
            if direction == "LONG":
                crosses = detect_crossover(hist, 0)
            else:
                crosses = detect_crossunder(hist, 0)
            
            filters['macd_cross'] = any(crosses[-self.filter_lookback:])
        
        # Volume Filter
        if self.use_volume:
            vol_sma = calculate_volume_sma(volumes, 30)
            filters['volume'] = any(volumes[-self.filter_lookback:] > vol_sma[-self.filter_lookback:])
        
        # StochRSI Filter
        if self.use_stoch_rsi:
            stoch = calculate_stoch_rsi(closes, 3, 3, 14, 14)
            k = stoch.k_line
            
            if direction == "LONG":
                # K crosses above oversold (20)
                crosses = detect_crossover(k, 20)
                above = k[-self.filter_lookback:] > 20
                filters['stoch_rsi'] = any(crosses[-self.filter_lookback:]) or any(above)
            else:
                # K crosses below overbought (80)
                crosses = detect_crossunder(k, 80)
                below = k[-self.filter_lookback:] < 80
                filters['stoch_rsi'] = any(crosses[-self.filter_lookback:]) or any(below)
        
        # Awesome Oscillator Color Filter
        if self.use_ao_color:
            ao = calculate_awesome_oscillator(highs, lows, 5, 34)
            ao_momentum = np.diff(ao)
            
            if direction == "LONG":
                filters['ao_color'] = any(ao_momentum[-self.filter_lookback:] > 0)
            else:
                filters['ao_color'] = any(ao_momentum[-self.filter_lookback:] < 0)
        
        return filters
    
    def _calculate_confidence(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        direction: str
    ) -> float:
        """Calculate signal confidence based on indicator strength."""
        scores = []
        
        # MACD strength
        macd = calculate_macd(closes, 12, 26, 9)
        hist = macd.histogram[-1]
        macd_score = min(abs(hist) / (closes[-1] * 0.001), 1.0)  # Normalize
        scores.append(macd_score * 0.25)
        
        # Volume strength
        vol_sma = calculate_volume_sma(volumes, 30)[-1]
        vol_ratio = volumes[-1] / vol_sma if vol_sma > 0 else 1
        vol_score = min(vol_ratio / 2, 1.0)  # Normalize (2x avg = max)
        scores.append(vol_score * 0.25)
        
        # StochRSI position
        stoch = calculate_stoch_rsi(closes, 3, 3, 14, 14)
        k = stoch.k_line[-1]
        if direction == "LONG":
            stoch_score = 1 - (k / 100)  # Lower K = better for longs (room to go up)
        else:
            stoch_score = k / 100  # Higher K = better for shorts (room to go down)
        scores.append(stoch_score * 0.25)
        
        # AO momentum
        ao = calculate_awesome_oscillator(highs, lows, 5, 34)
        ao_change = ao[-1] - ao[-2] if len(ao) > 1 else 0
        if direction == "LONG":
            ao_score = 1.0 if ao_change > 0 else 0.5
        else:
            ao_score = 1.0 if ao_change < 0 else 0.5
        scores.append(ao_score * 0.25)
        
        return sum(scores)
    
    def get_trend_state(self, symbol: str) -> Optional[str]:
        """Get current trend state for a symbol."""
        trend = self.trend_state.get(symbol)
        if trend is None:
            return None
        return "UPTREND" if trend else "DOWNTREND"


# Global generator instance - SCALPING configuration
ttft_generator = TTFTGenerator(
    trend_length=10,       # Faster trend detection (was 20)
    atr_length=14,         # Scalping ATR (was 200)
    atr_sma_length=14,     # Faster smoothing (was 200)
    atr_multiplier=0.4,    # Tighter bands (was 0.8)
    sl_multiplier=1.0,
    rr_ratio=2.0,
    confidence_threshold=0.5,  # Lower threshold for more signals
    use_macd_cross=True,
    use_volume=True,
    use_stoch_rsi=True,
    use_ao_color=True,
    filter_lookback=2      # Faster confirmation (was 3)
)
