"""
EMA Crossover Scalping Strategy.

A fast 1-minute scalping strategy based on:
- EMA 9 & 21 crossover for trend direction
- RSI confirmation (above/below 50)
- Optional Bollinger Bands for range detection
- Tight stop losses with 1:1 to 1.5:1 R:R ratio
"""
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from data.binance_ws import Kline, Trade
from analysis.indicators import (
    calculate_ema, calculate_rsi, calculate_sma,
    detect_crossover, detect_crossunder
)


@dataclass 
class ScalpSignal:
    """
    EMA Crossover Scalping Signal.
    
    Entry: On EMA crossover with RSI confirmation
    Stop Loss: Beyond last swing high/low (tight)
    Take Profit: 1:1 to 1.5:1 R:R
    """
    timestamp: int
    symbol: str
    timeframe: str
    direction: str  # "LONG" or "SHORT"
    
    # Entry parameters
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Indicator values
    ema_9: float
    ema_21: float
    rsi: float
    
    # Confidence and confirmation
    confidence: float
    
    # Risk metrics
    risk_reward: float
    risk_percent: float
    
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
            'ema_9': self.ema_9,
            'ema_21': self.ema_21,
            'rsi': self.rsi,
            'confidence': self.confidence,
            'risk_reward': self.risk_reward,
            'model': 'EMA_SCALP'
        }


class EMAScalpGenerator:
    """
    EMA Crossover Scalping Signal Generator.
    
    Strategy Logic:
    1. Detect EMA 9/21 crossover
    2. Confirm with RSI > 50 (longs) or RSI < 50 (shorts)
    3. Confirm price closes above/below both EMAs
    4. Set tight SL at recent swing high/low
    5. TP at 1:1 to 1.5:1 R:R
    """
    
    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        rr_ratio: float = 1.5,  # 1:1.5 R:R
        atr_sl_mult: float = 1.0,  # SL at 1x ATR
        lookback_bars: int = 5,  # Bars to check for swing high/low
        min_rsi_distance: float = 5.0  # RSI must be 5+ points from 50
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rr_ratio = rr_ratio
        self.atr_sl_mult = atr_sl_mult
        self.lookback_bars = lookback_bars
        self.min_rsi_distance = min_rsi_distance
        
        # Track last EMA state for crossover detection
        self.last_ema_state: Dict[str, bool] = {}  # True = fast above slow
    
    def generate_signal(
        self,
        klines: List[Kline],
        trades: List[Trade],
        symbol: str
    ) -> Optional[ScalpSignal]:
        """
        Generate scalping signal.
        
        Args:
            klines: Recent klines (need at least 30)
            trades: Not used but kept for interface compatibility
            symbol: Trading pair
        
        Returns:
            ScalpSignal if conditions met, None otherwise
        """
        min_length = max(self.ema_slow, self.rsi_period) + 10
        
        if len(klines) < min_length:
            return None
        
        # Extract price data
        highs = np.array([k.high for k in klines])
        lows = np.array([k.low for k in klines])
        closes = np.array([k.close for k in klines])
        
        timeframe = klines[0].interval
        current_price = closes[-1]
        
        # Calculate indicators
        ema_9 = calculate_ema(closes, self.ema_fast)
        ema_21 = calculate_ema(closes, self.ema_slow)
        rsi = calculate_rsi(closes, self.rsi_period)
        
        current_ema_9 = ema_9[-1]
        current_ema_21 = ema_21[-1]
        current_rsi = rsi[-1]
        
        # Determine current EMA state
        fast_above_slow = current_ema_9 > current_ema_21
        prev_state = self.last_ema_state.get(symbol)
        
        # Update state
        self.last_ema_state[symbol] = fast_above_slow
        
        if prev_state is None:
            logger.debug(f"{symbol}: EMA state established")
            return None
        
        # Detect crossover
        bullish_cross = fast_above_slow and not prev_state
        bearish_cross = not fast_above_slow and prev_state
        
        if not bullish_cross and not bearish_cross:
            return None
        
        direction = "LONG" if bullish_cross else "SHORT"
        logger.info(f"{symbol}: EMA crossover detected - {direction}")
        
        # RSI confirmation
        if direction == "LONG":
            if current_rsi < 50 + self.min_rsi_distance:
                logger.debug(f"{symbol}: RSI not confirming long ({current_rsi:.1f})")
                return None
            # Price must be above both EMAs
            if current_price < current_ema_9 or current_price < current_ema_21:
                logger.debug(f"{symbol}: Price not above EMAs")
                return None
        else:
            if current_rsi > 50 - self.min_rsi_distance:
                logger.debug(f"{symbol}: RSI not confirming short ({current_rsi:.1f})")
                return None
            # Price must be below both EMAs
            if current_price > current_ema_9 or current_price > current_ema_21:
                logger.debug(f"{symbol}: Price not below EMAs")
                return None
        
        logger.info(f"{symbol}: RSI confirmed ({current_rsi:.1f})")
        
        # Calculate stop loss from recent swing high/low
        if direction == "LONG":
            recent_lows = lows[-self.lookback_bars:]
            swing_low = np.min(recent_lows)
            sl_buffer = (current_price - swing_low) * 0.1  # 10% buffer
            stop_loss = swing_low - sl_buffer
        else:
            recent_highs = highs[-self.lookback_bars:]
            swing_high = np.max(recent_highs)
            sl_buffer = (swing_high - current_price) * 0.1  # 10% buffer
            stop_loss = swing_high + sl_buffer
        
        # Calculate take profit with R:R ratio
        risk = abs(current_price - stop_loss)
        reward = risk * self.rr_ratio
        
        if direction == "LONG":
            take_profit = current_price + reward
        else:
            take_profit = current_price - reward
        
        # Risk metrics
        risk_percent = (risk / current_price) * 100
        
        # Calculate confidence based on indicator strength
        confidence = self._calculate_confidence(
            current_rsi, direction, current_ema_9, current_ema_21, current_price
        )
        
        # Generate signal
        signal = ScalpSignal(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ema_9=current_ema_9,
            ema_21=current_ema_21,
            rsi=current_rsi,
            confidence=confidence,
            risk_reward=self.rr_ratio,
            risk_percent=risk_percent
        )
        
        logger.info(
            f"{symbol}: SCALP SIGNAL - {direction} @ {current_price:.2f} | "
            f"SL: {stop_loss:.2f} | TP: {take_profit:.2f} | RSI: {current_rsi:.1f}"
        )
        
        return signal
    
    def _calculate_confidence(
        self,
        rsi: float,
        direction: str,
        ema_9: float,
        ema_21: float,
        price: float
    ) -> float:
        """Calculate signal confidence."""
        scores = []
        
        # RSI distance from 50 (stronger = better)
        rsi_distance = abs(rsi - 50)
        rsi_score = min(rsi_distance / 30, 1.0) * 0.4
        scores.append(rsi_score)
        
        # EMA separation (stronger = better)
        ema_separation = abs(ema_9 - ema_21) / price
        ema_score = min(ema_separation / 0.002, 1.0) * 0.3
        scores.append(ema_score)
        
        # Price distance from EMAs (confirms direction)
        if direction == "LONG":
            price_above = (price - max(ema_9, ema_21)) / price
            price_score = min(price_above / 0.001, 1.0) * 0.3 if price_above > 0 else 0
        else:
            price_below = (min(ema_9, ema_21) - price) / price
            price_score = min(price_below / 0.001, 1.0) * 0.3 if price_below > 0 else 0
        scores.append(price_score)
        
        return sum(scores)


# Global generator instance - 1:1 R:R for scalping profitability
ema_scalp_generator = EMAScalpGenerator(
    ema_fast=9,
    ema_slow=21,
    rsi_period=14,
    rr_ratio=1.0,  # 1:1 R:R (profitable in backtest)
    lookback_bars=5,
    min_rsi_distance=3.0  # RSI must be 53+ for longs, 47- for shorts
)
