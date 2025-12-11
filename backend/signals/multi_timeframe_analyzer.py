"""
Multi-Timeframe Analyzer - Provides cross-timeframe confluence analysis.

Analyzes signals across 1m, 5m, 15m, and 1h timeframes to determine
the overall market direction and signal strength.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

from config import settings


@dataclass
class TimeframeSignal:
    """Signal state for a single timeframe."""
    timeframe: str
    direction: str  # LONG, SHORT, NEUTRAL
    strength: float  # 0.0 to 1.0
    ema_trend: str  # UP, DOWN, NEUTRAL
    stoch_k: float
    confidence: float
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    
    def to_dict(self) -> dict:
        return {
            "timeframe": self.timeframe,
            "direction": self.direction,
            "strength": self.strength,
            "ema_trend": self.ema_trend,
            "stoch_k": self.stoch_k,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }


@dataclass
class ConfluenceResult:
    """Result of multi-timeframe confluence analysis."""
    overall_direction: str  # LONG, SHORT, NEUTRAL
    confluence_score: float  # 0.0 to 1.0
    aligned_timeframes: int
    total_timeframes: int
    weighted_confidence: float
    signals_by_timeframe: Dict[str, TimeframeSignal] = field(default_factory=dict)
    reasoning: str = ""
    
    def to_dict(self) -> dict:
        return {
            "overall_direction": self.overall_direction,
            "confluence_score": self.confluence_score,
            "aligned_timeframes": self.aligned_timeframes,
            "total_timeframes": self.total_timeframes,
            "weighted_confidence": self.weighted_confidence,
            "signals_by_timeframe": {
                k: v.to_dict() for k, v in self.signals_by_timeframe.items()
            },
            "reasoning": self.reasoning
        }


class MultiTimeframeAnalyzer:
    """
    Analyzes signals across multiple timeframes to determine confluence.
    
    Timeframe Weights:
    - 1m:  0.15 (fast scalping, high noise)
    - 5m:  0.25 (short-term momentum)
    - 15m: 0.30 (medium-term trend)
    - 1h:  0.30 (primary trend direction)
    
    A trade is only approved when there's sufficient confluence across timeframes.
    """
    
    # Timeframe weights - must sum to 1.0
    TIMEFRAME_WEIGHTS = {
        "1m": 0.15,
        "5m": 0.25,
        "15m": 0.30,
        "1h": 0.30
    }
    
    # Minimum confluence score to approve a trade
    MIN_CONFLUENCE_THRESHOLD = 0.50
    
    def __init__(self):
        # Store current signals per symbol and timeframe
        # Structure: {symbol: {timeframe: TimeframeSignal}}
        self.current_signals: Dict[str, Dict[str, TimeframeSignal]] = {}
    
    def update_signal(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        strength: float,
        ema_trend: str,
        stoch_k: float,
        confidence: float
    ):
        """Update the current signal for a symbol/timeframe combination."""
        if symbol not in self.current_signals:
            self.current_signals[symbol] = {}
        
        signal = TimeframeSignal(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            ema_trend=ema_trend,
            stoch_k=stoch_k,
            confidence=confidence
        )
        
        self.current_signals[symbol][timeframe] = signal
        logger.debug(f"Updated MTF signal: {symbol} {timeframe} = {direction} ({strength:.2f})")
    
    def get_confluence(self, symbol: str) -> ConfluenceResult:
        """
        Calculate confluence score for a symbol across all timeframes.
        
        Returns:
            ConfluenceResult with overall direction and score
        """
        signals = self.current_signals.get(symbol, {})
        
        if not signals:
            return ConfluenceResult(
                overall_direction="NEUTRAL",
                confluence_score=0.0,
                aligned_timeframes=0,
                total_timeframes=0,
                weighted_confidence=0.0,
                reasoning="No signals available"
            )
        
        # Count directions with weights
        long_weight = 0.0
        short_weight = 0.0
        neutral_weight = 0.0
        total_weight = 0.0
        weighted_confidence_sum = 0.0
        
        aligned_long = 0
        aligned_short = 0
        
        for tf, signal in signals.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.0)
            total_weight += weight
            
            # Weight the direction
            if signal.direction == "LONG":
                long_weight += weight * signal.strength * signal.confidence
                aligned_long += 1
            elif signal.direction == "SHORT":
                short_weight += weight * signal.strength * signal.confidence
                aligned_short += 1
            else:
                neutral_weight += weight
            
            # Track weighted confidence
            weighted_confidence_sum += weight * signal.confidence
        
        # Normalize weights
        if total_weight == 0:
            total_weight = 1.0
        
        long_score = long_weight / total_weight
        short_score = short_weight / total_weight
        
        # Determine overall direction
        if long_score > short_score and long_score > 0.3:
            overall_direction = "LONG"
            confluence_score = long_score
            aligned_count = aligned_long
        elif short_score > long_score and short_score > 0.3:
            overall_direction = "SHORT"
            confluence_score = short_score
            aligned_count = aligned_short
        else:
            overall_direction = "NEUTRAL"
            confluence_score = max(long_score, short_score)
            aligned_count = 0
        
        # Calculate weighted confidence
        weighted_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0
        
        # Build reasoning
        reasoning_parts = []
        for tf in ["1h", "15m", "5m", "1m"]:
            if tf in signals:
                s = signals[tf]
                reasoning_parts.append(f"{tf}={s.direction}({s.confidence:.0%})")
        reasoning = " | ".join(reasoning_parts)
        
        return ConfluenceResult(
            overall_direction=overall_direction,
            confluence_score=confluence_score,
            aligned_timeframes=aligned_count,
            total_timeframes=len(signals),
            weighted_confidence=weighted_confidence,
            signals_by_timeframe=signals.copy(),
            reasoning=reasoning
        )
    
    def get_aligned_direction(self, symbol: str) -> str:
        """Get the aligned direction for a symbol (LONG/SHORT/NEUTRAL)."""
        result = self.get_confluence(symbol)
        return result.overall_direction
    
    def get_confluence_score(self, symbol: str) -> float:
        """Get the confluence score (0.0 to 1.0) for a symbol."""
        result = self.get_confluence(symbol)
        return result.confluence_score
    
    def is_trade_aligned(self, symbol: str, proposed_direction: str) -> bool:
        """
        Check if a proposed trade direction aligns with multi-timeframe analysis.
        
        Args:
            symbol: Trading pair
            proposed_direction: The direction of the proposed trade (LONG/SHORT)
        
        Returns:
            True if the trade aligns with MTF analysis and meets minimum confluence
        """
        result = self.get_confluence(symbol)
        
        # Must meet minimum confluence threshold
        if result.confluence_score < self.MIN_CONFLUENCE_THRESHOLD:
            logger.info(
                f"MTF: Trade {proposed_direction} rejected - "
                f"confluence too low ({result.confluence_score:.2f} < {self.MIN_CONFLUENCE_THRESHOLD})"
            )
            return False
        
        # Direction must align
        if result.overall_direction != proposed_direction:
            logger.info(
                f"MTF: Trade {proposed_direction} rejected - "
                f"MTF direction is {result.overall_direction}"
            )
            return False
        
        logger.info(
            f"MTF: Trade {proposed_direction} approved - "
            f"confluence {result.confluence_score:.2f}, {result.aligned_timeframes}/{result.total_timeframes} aligned"
        )
        return True
    
    def get_all_timeframe_signals(self, symbol: str) -> Dict[str, dict]:
        """Get signals for all timeframes as dicts."""
        signals = self.current_signals.get(symbol, {})
        return {tf: s.to_dict() for tf, s in signals.items()}
    
    def clear_symbol(self, symbol: str):
        """Clear all signals for a symbol."""
        if symbol in self.current_signals:
            del self.current_signals[symbol]
    
    def get_status(self) -> dict:
        """Get current MTF analyzer status."""
        status = {
            "weights": self.TIMEFRAME_WEIGHTS,
            "min_confluence_threshold": self.MIN_CONFLUENCE_THRESHOLD,
            "symbols": {}
        }
        
        for symbol in self.current_signals:
            result = self.get_confluence(symbol)
            status["symbols"][symbol] = result.to_dict()
        
        return status


# Global instance
mtf_analyzer = MultiTimeframeAnalyzer()
