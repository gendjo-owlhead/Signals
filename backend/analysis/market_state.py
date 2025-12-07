"""
Market State Analysis - Balance vs Imbalance detection.
Core component determining which trading model to use.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from config import settings
from data.binance_ws import Kline
from analysis.volume_profile import VolumeProfile, volume_profile_calculator


class MarketState(Enum):
    """Market state classification."""
    BALANCED = "balanced"           # Range-bound, rotating around POC
    TRENDING_UP = "trending_up"     # Out of balance, moving up
    TRENDING_DOWN = "trending_down" # Out of balance, moving down
    BREAKOUT_UP = "breakout_up"     # Attempting upside breakout
    BREAKOUT_DOWN = "breakout_down" # Attempting downside breakout
    CHOPPY = "choppy"               # Unclear, mixed signals


@dataclass
class MarketStateAnalysis:
    """Complete market state analysis result."""
    state: MarketState
    confidence: float  # 0-1 confidence in classification
    
    # Balance metrics
    is_balanced: bool
    balance_score: float  # 0-1, higher = more balanced
    
    # Range metrics
    range_high: float
    range_low: float
    current_price: float
    price_in_range_pct: float  # Position in range (0=low, 100=high)
    
    # Momentum metrics
    momentum: float  # -1 to 1
    momentum_strength: float  # 0-1
    
    # Volume Profile reference
    poc: float
    vah: float
    val: float
    
    # Trend metrics
    higher_highs: int
    higher_lows: int
    lower_highs: int
    lower_lows: int
    
    @property
    def is_out_of_balance(self) -> bool:
        """Market is out of balance (trending)."""
        return self.state in [MarketState.TRENDING_UP, MarketState.TRENDING_DOWN]
    
    @property
    def has_breakout_attempt(self) -> bool:
        """Market is attempting a breakout."""
        return self.state in [MarketState.BREAKOUT_UP, MarketState.BREAKOUT_DOWN]
    
    @property
    def direction(self) -> int:
        """Direction: 1 = bullish, -1 = bearish, 0 = neutral."""
        if self.state in [MarketState.TRENDING_UP, MarketState.BREAKOUT_UP]:
            return 1
        elif self.state in [MarketState.TRENDING_DOWN, MarketState.BREAKOUT_DOWN]:
            return -1
        return 0


class MarketStateAnalyzer:
    """
    Analyze market state to determine if market is balanced or out of balance.
    
    Per TradeZella Auction Market Playbook:
    "The market is an auction that moves between two conditions:
    - Balance – where price rotates around fair value
    - Imbalance – where one side is aggressive, pushing price away"
    
    This analyzer determines which condition the market is in,
    which then determines which trading model to use:
    - Balanced → Mean Reversion Model
    - Out of Balance → Trend Model
    """
    
    def __init__(
        self,
        lookback_periods: int = None,
        balance_threshold: float = 0.6,
        breakout_atr_mult: float = 1.5
    ):
        """
        Initialize market state analyzer.
        
        Args:
            lookback_periods: Periods for analysis (default from settings)
            balance_threshold: Score threshold for balanced classification
            breakout_atr_mult: ATR multiple for breakout detection
        """
        self.lookback_periods = lookback_periods or settings.vp_lookback_periods
        self.balance_threshold = balance_threshold
        self.breakout_atr_mult = breakout_atr_mult
    
    def analyze(
        self,
        klines: List[Kline],
        volume_profile: Optional[VolumeProfile] = None
    ) -> MarketStateAnalysis:
        """
        Analyze market state from klines and volume profile.
        
        Args:
            klines: Recent klines for analysis
            volume_profile: Optional pre-calculated volume profile
        
        Returns:
            MarketStateAnalysis with complete classification
        """
        if len(klines) < 10:
            return self._empty_analysis()
        
        # Calculate volume profile if not provided
        if volume_profile is None:
            volume_profile = volume_profile_calculator.calculate_from_klines(klines)
        
        # Extract key data
        closes = np.array([k.close for k in klines])
        highs = np.array([k.high for k in klines])
        lows = np.array([k.low for k in klines])
        volumes = np.array([k.volume for k in klines])
        
        current_price = closes[-1]
        
        # 1. Calculate balance score
        balance_score = self._calculate_balance_score(
            closes, highs, lows, volumes, volume_profile
        )
        
        is_balanced = balance_score >= self.balance_threshold
        
        # 2. Calculate momentum
        momentum, momentum_strength = self._calculate_momentum(closes)
        
        # 3. Detect swing structure
        higher_highs, higher_lows, lower_highs, lower_lows = self._analyze_swing_structure(highs, lows)
        
        # 4. Calculate ATR for breakout detection
        atr = self._calculate_atr(highs, lows, closes)
        
        # 5. Determine range
        range_high = volume_profile.vah
        range_low = volume_profile.val
        price_range = range_high - range_low
        
        if price_range > 0:
            price_in_range = (current_price - range_low) / price_range * 100
        else:
            price_in_range = 50
        
        # 6. Classify state
        state, confidence = self._classify_state(
            is_balanced=is_balanced,
            balance_score=balance_score,
            momentum=momentum,
            momentum_strength=momentum_strength,
            current_price=current_price,
            vah=volume_profile.vah,
            val=volume_profile.val,
            atr=atr,
            higher_highs=higher_highs,
            lower_lows=lower_lows
        )
        
        return MarketStateAnalysis(
            state=state,
            confidence=confidence,
            is_balanced=is_balanced,
            balance_score=balance_score,
            range_high=range_high,
            range_low=range_low,
            current_price=current_price,
            price_in_range_pct=price_in_range,
            momentum=momentum,
            momentum_strength=momentum_strength,
            poc=volume_profile.poc_price,
            vah=volume_profile.vah,
            val=volume_profile.val,
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            lower_lows=lower_lows
        )
    
    def _calculate_balance_score(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        vp: VolumeProfile
    ) -> float:
        """
        Calculate balance score (0-1).
        Higher score = more balanced/ranging.
        Lower score = more directional/trending.
        """
        scores = []
        
        # 1. Price rotation around POC
        if vp.poc_price > 0:
            poc_crosses = 0
            for i in range(1, len(closes)):
                if (closes[i-1] < vp.poc_price and closes[i] >= vp.poc_price) or \
                   (closes[i-1] > vp.poc_price and closes[i] <= vp.poc_price):
                    poc_crosses += 1
            
            # More crosses = more balanced
            rotation_score = min(poc_crosses / (len(closes) / 10), 1.0)
            scores.append(rotation_score)
        
        # 2. Price containment within value area
        in_va_count = sum(1 for c in closes if vp.val <= c <= vp.vah)
        containment_score = in_va_count / len(closes)
        scores.append(containment_score)
        
        # 3. Range-bound check (ATR relative to range)
        atr = self._calculate_atr(highs, lows, closes)
        range_size = max(highs) - min(lows)
        if range_size > 0:
            atr_ratio = atr / range_size
            range_score = 1 - min(atr_ratio * 2, 1.0)  # Higher ATR relative to range = less balanced
            scores.append(range_score)
        
        # 4. Volume distribution (balanced has more uniform distribution)
        if len(vp.levels) > 0:
            vol_values = [l.volume for l in vp.levels if l.volume > 0]
            if vol_values:
                vol_std = np.std(vol_values)
                vol_mean = np.mean(vol_values)
                if vol_mean > 0:
                    cv = vol_std / vol_mean  # Coefficient of variation
                    distribution_score = 1 - min(cv, 1.0)
                    scores.append(distribution_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_momentum(self, closes: np.ndarray) -> Tuple[float, float]:
        """
        Calculate momentum (-1 to 1) and strength (0 to 1).
        """
        if len(closes) < 5:
            return 0.0, 0.0
        
        # Calculate returns
        returns = np.diff(closes) / closes[:-1]
        
        # Recent momentum (last 5 periods weighted more)
        weights = np.exp(np.linspace(-1, 0, len(returns)))
        weighted_return = np.sum(returns * weights) / np.sum(weights)
        
        # Normalize to -1 to 1
        momentum = np.tanh(weighted_return * 100)
        
        # Strength is absolute momentum with consistency check
        return_signs = np.sign(returns[-5:])
        consistency = abs(np.mean(return_signs))
        strength = min(abs(momentum) * consistency, 1.0)
        
        return momentum, strength
    
    def _analyze_swing_structure(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        swing_period: int = 5
    ) -> Tuple[int, int, int, int]:
        """
        Analyze swing high/low structure.
        Returns (higher_highs, higher_lows, lower_highs, lower_lows) counts.
        """
        higher_highs = 0
        higher_lows = 0
        lower_highs = 0
        lower_lows = 0
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(swing_period, len(highs) - swing_period):
            # Swing high: highest in window
            if highs[i] == max(highs[i-swing_period:i+swing_period+1]):
                swing_highs.append((i, highs[i]))
            
            # Swing low: lowest in window
            if lows[i] == min(lows[i-swing_period:i+swing_period+1]):
                swing_lows.append((i, lows[i]))
        
        # Count higher highs / lower highs
        for i in range(1, len(swing_highs)):
            if swing_highs[i][1] > swing_highs[i-1][1]:
                higher_highs += 1
            else:
                lower_highs += 1
        
        # Count higher lows / lower lows
        for i in range(1, len(swing_lows)):
            if swing_lows[i][1] > swing_lows[i-1][1]:
                higher_lows += 1
            else:
                lower_lows += 1
        
        return higher_highs, higher_lows, lower_highs, lower_lows
    
    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate Average True Range."""
        if len(highs) < 2:
            return 0.0
        
        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        
        if not tr_list:
            return 0.0
        
        # Use EMA for ATR
        atr = tr_list[0]
        for tr in tr_list[1:]:
            atr = (atr * (period - 1) + tr) / period
        
        return atr
    
    def _classify_state(
        self,
        is_balanced: bool,
        balance_score: float,
        momentum: float,
        momentum_strength: float,
        current_price: float,
        vah: float,
        val: float,
        atr: float,
        higher_highs: int,
        lower_lows: int
    ) -> Tuple[MarketState, float]:
        """
        Classify market state based on all metrics.
        
        Returns (state, confidence)
        """
        # Check for breakout
        breakout_threshold = atr * self.breakout_atr_mult
        
        if current_price > vah + breakout_threshold:
            confidence = min(momentum_strength + 0.3, 1.0)
            return MarketState.BREAKOUT_UP, confidence
        
        if current_price < val - breakout_threshold:
            confidence = min(momentum_strength + 0.3, 1.0)
            return MarketState.BREAKOUT_DOWN, confidence
        
        # Check for trending
        if not is_balanced:
            if momentum > 0.3 and higher_highs > lower_lows:
                confidence = (1 - balance_score) * momentum_strength
                return MarketState.TRENDING_UP, confidence
            
            if momentum < -0.3 and lower_lows > higher_highs:
                confidence = (1 - balance_score) * momentum_strength
                return MarketState.TRENDING_DOWN, confidence
        
        # Check for balanced
        if is_balanced:
            confidence = balance_score
            return MarketState.BALANCED, confidence
        
        # Default to choppy
        return MarketState.CHOPPY, 0.4
    
    def _empty_analysis(self) -> MarketStateAnalysis:
        """Return empty analysis for insufficient data."""
        return MarketStateAnalysis(
            state=MarketState.CHOPPY,
            confidence=0.0,
            is_balanced=False,
            balance_score=0.0,
            range_high=0.0,
            range_low=0.0,
            current_price=0.0,
            price_in_range_pct=50.0,
            momentum=0.0,
            momentum_strength=0.0,
            poc=0.0,
            vah=0.0,
            val=0.0,
            higher_highs=0,
            higher_lows=0,
            lower_highs=0,
            lower_lows=0
        )
    
    def detect_impulse_leg(
        self,
        klines: List[Kline],
        min_atr_move: float = 2.0
    ) -> Optional[Tuple[int, int]]:
        """
        Detect an impulse leg (strong directional move) for Trend Model.
        
        Per TradeZella: "Take the impulse leg that broke the structure."
        
        Args:
            klines: List of klines
            min_atr_move: Minimum ATR multiple for impulse
        
        Returns:
            Tuple of (start_idx, end_idx) or None if no impulse found
        """
        if len(klines) < 10:
            return None
        
        highs = np.array([k.high for k in klines])
        lows = np.array([k.low for k in klines])
        closes = np.array([k.close for k in klines])
        
        atr = self._calculate_atr(highs, lows, closes)
        threshold = atr * min_atr_move
        
        # Look for rapid price movement
        for i in range(len(klines) - 5, 4, -1):  # Search from recent to past
            window = klines[i-5:i+1]
            window_high = max(k.high for k in window)
            window_low = min(k.low for k in window)
            window_range = window_high - window_low
            
            if window_range >= threshold:
                # Found impulse, determine direction and bounds
                start_price = window[0].open
                end_price = window[-1].close
                
                if end_price > start_price:  # Bullish impulse
                    return (i - 5, i)
                else:  # Bearish impulse
                    return (i - 5, i)
        
        return None
    
    def detect_failed_breakout(
        self,
        klines: List[Kline],
        vp: VolumeProfile
    ) -> Optional[Dict]:
        """
        Detect a failed breakout for Mean Reversion Model.
        
        Per TradeZella: "Watch for times when price breaks out but can't hold.
        When that happens, the most likely outcome is that the market returns to the POC."
        
        Args:
            klines: Recent klines
            vp: Volume profile
        
        Returns:
            Dict with breakout details or None
        """
        if len(klines) < 10:
            return None
        
        recent = klines[-10:]
        
        # Check for price that went above VAH then came back
        went_above_vah = any(k.high > vp.vah for k in recent[:-3])
        now_below_vah = recent[-1].close < vp.vah
        
        if went_above_vah and now_below_vah:
            return {
                'type': 'failed_upside_breakout',
                'breakout_high': max(k.high for k in recent),
                'current_price': recent[-1].close,
                'target': vp.poc_price
            }
        
        # Check for price that went below VAL then came back
        went_below_val = any(k.low < vp.val for k in recent[:-3])
        now_above_val = recent[-1].close > vp.val
        
        if went_below_val and now_above_val:
            return {
                'type': 'failed_downside_breakout',
                'breakout_low': min(k.low for k in recent),
                'current_price': recent[-1].close,
                'target': vp.poc_price
            }
        
        return None


# Global analyzer instance
market_state_analyzer = MarketStateAnalyzer()
