"""
EMA 5-8-13 High-Frequency Scalping Strategy.

Fibonacci-based EMA crossover strategy designed for 1-minute scalping.
Generates 20-50+ trades per day for true scalping.

## Strategy Logic:
1. Use EMA 5, 8, 13 (Fibonacci sequence)
2. LONG: EMA5 crosses above EMA8, both above EMA13, price above all EMAs
3. SHORT: EMA5 crosses below EMA8, both below EMA13, price below all EMAs
4. StochRSI for additional confirmation (optional)
5. Quick exits on EMA recross

## Entry Conditions:

LONG:
- EMA5 crosses above EMA8 (current bar)
- EMA5 > EMA8 > EMA13 (aligned)
- Price > EMA13 (trend confirmation)
- StochRSI K > 20 (not deeply oversold - catching momentum)

SHORT:
- EMA5 crosses below EMA8 (current bar)
- EMA5 < EMA8 < EMA13 (aligned)
- Price < EMA13 (trend confirmation)
- StochRSI K < 80 (not deeply overbought - catching momentum)
"""
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from loguru import logger

from data.binance_ws import Kline, Trade
from analysis.indicators import calculate_ema, calculate_atr, calculate_rsi


def calculate_stoch_rsi(
    closes: np.ndarray,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic RSI.
    
    Returns:
        k_line: Fast stochastic of RSI
        d_line: Smoothed K line
    """
    rsi = calculate_rsi(closes, rsi_period)
    
    length = len(rsi)
    stoch_rsi = np.zeros(length)
    
    for i in range(stoch_period - 1, length):
        window = rsi[i - stoch_period + 1:i + 1]
        min_rsi = np.min(window)
        max_rsi = np.max(window)
        
        if max_rsi - min_rsi > 0:
            stoch_rsi[i] = ((rsi[i] - min_rsi) / (max_rsi - min_rsi)) * 100
        else:
            stoch_rsi[i] = 50
    
    # Smooth K
    k_line = np.zeros(length)
    for i in range(k_smooth - 1, length):
        k_line[i] = np.mean(stoch_rsi[i - k_smooth + 1:i + 1])
    
    # Smooth D
    d_line = np.zeros(length)
    for i in range(d_smooth - 1, length):
        d_line[i] = np.mean(k_line[i - d_smooth + 1:i + 1])
    
    return k_line, d_line


@dataclass
class ScalperSignal:
    """
    High-Frequency Scalping Signal.
    
    Entry: EMA crossover with StochRSI confirmation
    """
    timestamp: int
    symbol: str
    timeframe: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_value: float
    confidence: float
    
    # EMA values
    ema5: float
    ema8: float
    ema13: float
    ema_aligned: bool
    ema_crossover: bool
    
    # StochRSI
    stoch_k: float
    stoch_d: float
    
    risk_reward: float
    risk_percent: float
    filters_passed: Dict[str, bool] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if signal has positive risk/reward."""
        return self.risk_reward >= 1.0 and self.confidence > 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        # Convert numpy types to native Python types for JSON serialization
        return {
            'timestamp': int(self.timestamp),
            'symbol': str(self.symbol),
            'timeframe': str(self.timeframe),
            'direction': str(self.direction),
            'entry_price': float(self.entry_price),
            'stop_loss': float(self.stop_loss),
            'take_profit': float(self.take_profit),
            'atr_value': float(self.atr_value),
            'confidence': float(self.confidence),
            'ema5': float(self.ema5),
            'ema8': float(self.ema8),
            'ema13': float(self.ema13),
            'ema_aligned': bool(self.ema_aligned),
            'ema_crossover': bool(self.ema_crossover),
            'stoch_k': float(self.stoch_k),
            'stoch_d': float(self.stoch_d),
            'risk_reward': float(self.risk_reward),
            'risk_percent': float(self.risk_percent),
            'filters_passed': {str(k): bool(v) for k, v in self.filters_passed.items()}
        }


class ScalperGenerator:
    """
    EMA 5-8-13 High-Frequency Scalping Signal Generator.
    
    Designed for 1-minute timeframe with 20-50+ trades per day.
    Uses Fibonacci-based EMAs with optional StochRSI confirmation.
    """
    
    def __init__(
        self,
        # EMA Settings (Fibonacci)
        ema_fast: int = 5,
        ema_mid: int = 8,
        ema_slow: int = 13,
        
        # StochRSI Settings
        use_stoch_filter: bool = True,
        stoch_rsi_period: int = 14,
        stoch_period: int = 14,
        stoch_k_smooth: int = 3,
        stoch_oversold: float = 20.0,
        stoch_overbought: float = 80.0,
        
        # ATR for stops
        atr_period: int = 14,
        atr_mult_sl: float = 1.0,    # Tight stops for scalping
        
        # Risk Management
        rr_ratio: float = 1.5,       # Lower R:R but higher win rate expected
        
        # Signal Threshold
        confidence_threshold: float = 0.5
    ):
        """Initialize Scalper Generator."""
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        
        self.use_stoch_filter = use_stoch_filter
        self.stoch_rsi_period = stoch_rsi_period
        self.stoch_period = stoch_period
        self.stoch_k_smooth = stoch_k_smooth
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        
        self.atr_period = atr_period
        self.atr_mult_sl = atr_mult_sl
        
        self.rr_ratio = rr_ratio
        self.confidence_threshold = confidence_threshold
        
        logger.info(
            f"ScalperGenerator initialized: EMA {ema_fast}/{ema_mid}/{ema_slow}, "
            f"StochRSI={use_stoch_filter}, R:R={rr_ratio}"
        )
    
    def generate_signal(
        self,
        klines: List[Kline],
        trades: List[Trade],
        symbol: str
    ) -> Optional[ScalperSignal]:
        """
        Generate high-frequency scalping signal.
        
        Args:
            klines: Recent klines (need at least 50 for calculations)
            trades: Recent trades (not used but kept for interface)
            symbol: Trading pair
        
        Returns:
            ScalperSignal if conditions met, None otherwise
        """
        min_required = max(self.ema_slow + 20, 50)
        if len(klines) < min_required:
            logger.warning(f"Not enough klines for Scalper: {len(klines)} < {min_required}")
            return None
        
        # Convert to arrays
        closes = np.array([k.close for k in klines], dtype=float)
        highs = np.array([k.high for k in klines], dtype=float)
        lows = np.array([k.low for k in klines], dtype=float)
        
        current_close = closes[-1]
        
        # ════════════════════════════════════════════════════════════════
        # 1. CALCULATE EMAs
        # ════════════════════════════════════════════════════════════════
        ema5 = calculate_ema(closes, self.ema_fast)
        ema8 = calculate_ema(closes, self.ema_mid)
        ema13 = calculate_ema(closes, self.ema_slow)
        
        current_ema5 = ema5[-1]
        current_ema8 = ema8[-1]
        current_ema13 = ema13[-1]
        
        prev_ema5 = ema5[-2]
        prev_ema8 = ema8[-2]
        
        # ════════════════════════════════════════════════════════════════
        # 2. DETECT EMA CROSSOVER
        # ════════════════════════════════════════════════════════════════
        # Bullish crossover: EMA5 crosses above EMA8
        bullish_cross = prev_ema5 <= prev_ema8 and current_ema5 > current_ema8
        
        # Bearish crossover: EMA5 crosses below EMA8
        bearish_cross = prev_ema5 >= prev_ema8 and current_ema5 < current_ema8
        
        if not bullish_cross and not bearish_cross:
            # logger.debug(f"{symbol}: No EMA crossover (5/8)")
            return None
        
        direction = "LONG" if bullish_cross else "SHORT"
        ema_crossover = True
        
        # ════════════════════════════════════════════════════════════════
        # 3. CHECK EMA ALIGNMENT
        # ════════════════════════════════════════════════════════════════
        if direction == "LONG":
            # Bullish: EMA5 > EMA8 > EMA13 and price > EMA13
            ema_aligned = (
                current_ema5 > current_ema8 > current_ema13 and
                current_close > current_ema13
            )
        else:
            # Bearish: EMA5 < EMA8 < EMA13 and price < EMA13
            ema_aligned = (
                current_ema5 < current_ema8 < current_ema13 and
                current_close < current_ema13
            )
        
        if not ema_aligned:
            logger.debug(f"{symbol}: EMA crossover detected but alignment failed. Direction: {direction}, EMA5:{current_ema5:.2f}, EMA8:{current_ema8:.2f}, EMA13:{current_ema13:.2f}, Close:{current_close:.2f}")
            return None
        
        # ════════════════════════════════════════════════════════════════
        # 4. STOCHASTIC RSI FILTER (optional)
        # ════════════════════════════════════════════════════════════════
        stoch_k, stoch_d = calculate_stoch_rsi(
            closes, self.stoch_rsi_period, self.stoch_period, self.stoch_k_smooth
        )
        
        current_stoch_k = stoch_k[-1]
        current_stoch_d = stoch_d[-1]
        
        stoch_confirming = True
        if self.use_stoch_filter:
            if direction == "LONG":
                # For longs, avoid deeply overbought (catching pullback recovery)
                stoch_confirming = current_stoch_k < self.stoch_overbought
            else:
                # For shorts, avoid deeply oversold
                stoch_confirming = current_stoch_k > self.stoch_oversold
            
            if not stoch_confirming:
                logger.debug(f"{symbol}: EMA/Alignment OK but StochRSI failed. K:{current_stoch_k:.2f}, Limit:{self.stoch_overbought if direction == 'LONG' else self.stoch_oversold}")
                return None
        
        # ════════════════════════════════════════════════════════════════
        # 5. CALCULATE ATR FOR STOP LOSS
        # ════════════════════════════════════════════════════════════════
        atr = calculate_atr(highs, lows, closes, self.atr_period)
        current_atr = atr[-1]
        
        # ════════════════════════════════════════════════════════════════
        # 6. CALCULATE STOP LOSS & TAKE PROFIT (tight for scalping)
        # ════════════════════════════════════════════════════════════════
        entry_price = current_close
        
        if direction == "LONG":
            # Stop loss below EMA13 or ATR-based
            stop_loss = min(current_ema13, entry_price - (current_atr * self.atr_mult_sl))
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * self.rr_ratio)
        else:
            # Stop loss above EMA13 or ATR-based
            stop_loss = max(current_ema13, entry_price + (current_atr * self.atr_mult_sl))
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * self.rr_ratio)
        
        risk_percent = (abs(entry_price - stop_loss) / entry_price) * 100
        
        # ════════════════════════════════════════════════════════════════
        # 7. CALCULATE CONFIDENCE
        # ════════════════════════════════════════════════════════════════
        confidence = self._calculate_confidence(
            ema_aligned=ema_aligned,
            ema_crossover=ema_crossover,
            stoch_k=current_stoch_k,
            stoch_confirming=stoch_confirming,
            direction=direction
        )
        
        if confidence < self.confidence_threshold:
            logger.debug(f"{symbol}: Confidence too low ({confidence:.2f} < {self.confidence_threshold})")
            return None
        
        # ════════════════════════════════════════════════════════════════
        # 8. BUILD SIGNAL
        # ════════════════════════════════════════════════════════════════
        filters_passed = {
            'ema_crossover': ema_crossover,
            'ema_aligned': ema_aligned,
            'stoch_confirming': stoch_confirming
        }
        
        signal = ScalperSignal(
            timestamp=klines[-1].close_time,
            symbol=symbol,
            timeframe=klines[-1].interval if hasattr(klines[-1], 'interval') else "1m",
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_value=current_atr,
            confidence=confidence,
            ema5=current_ema5,
            ema8=current_ema8,
            ema13=current_ema13,
            ema_aligned=ema_aligned,
            ema_crossover=ema_crossover,
            stoch_k=current_stoch_k,
            stoch_d=current_stoch_d,
            risk_reward=self.rr_ratio,
            risk_percent=risk_percent,
            filters_passed=filters_passed
        )
        
        logger.info(
            f"⚡ SCALPER {direction} Signal: {symbol} @ {entry_price:.2f} | "
            f"SL: {stop_loss:.2f} | TP: {take_profit:.2f} | "
            f"StochK: {current_stoch_k:.1f} | Conf: {confidence:.2%}"
        )
        
        return signal
    
    def _calculate_confidence(
        self,
        ema_aligned: bool,
        ema_crossover: bool,
        stoch_k: float,
        stoch_confirming: bool,
        direction: str
    ) -> float:
        """
        Calculate signal confidence (0.0 - 1.0).
        
        Factors:
        - EMA crossover: 30%
        - EMA alignment: 30%
        - StochRSI position: 30%
        - Base score: 10%
        """
        base_score = 0.10
        
        # EMA crossover (30%)
        if ema_crossover:
            base_score += 0.30
        
        # EMA alignment (30%)
        if ema_aligned:
            base_score += 0.30
        
        # StochRSI bonus (30%)
        if stoch_confirming:
            base_score += 0.20
            
            # Extra bonus for momentum direction
            if direction == "LONG" and stoch_k < 50:
                base_score += 0.10  # Catching an oversold bounce
            elif direction == "SHORT" and stoch_k > 50:
                base_score += 0.10  # Catching overbought drop
        
        return min(base_score, 1.0)


# Global generator instance - High-frequency scalping configuration
scalper_generator = ScalperGenerator(
    ema_fast=5,
    ema_mid=8,
    ema_slow=13,
    use_stoch_filter=True,
    stoch_rsi_period=14,
    stoch_period=14,
    stoch_k_smooth=3,
    stoch_oversold=20.0,
    stoch_overbought=80.0,
    atr_period=14,
    atr_mult_sl=1.0,
    rr_ratio=1.5,
    confidence_threshold=0.5
)
