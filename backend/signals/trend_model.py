"""
Trend Model Signal Generator.

Implements the TradeZella Auction Market Playbook Trend Model:
"Out-of-Balance → LVN Retracement → Order Flow Confirmation → POC Target"

This model is used when the market is trending (out of balance).
"""
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from config import settings
from data.binance_ws import Kline, Trade
from analysis.volume_profile import VolumeProfile, volume_profile_calculator
from analysis.order_flow import order_flow_analyzer, AggressionSignal
from analysis.market_state import market_state_analyzer, MarketState


@dataclass
class TrendSignal:
    """
    Trend Model trading signal.
    
    Per TradeZella:
    - Entry: At LVN retracement with order flow confirmation
    - Stop Loss: Just beyond the aggressive print (beyond LVN)
    - Take Profit: Prior balance POC
    """
    timestamp: int
    symbol: str
    timeframe: str
    direction: str  # "LONG" or "SHORT"
    
    # Entry parameters
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Key levels
    lvn_price: float
    poc_target: float
    impulse_start: float
    impulse_end: float
    
    # Confidence and confirmation
    confidence: float
    aggression: AggressionSignal
    market_state: str
    
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
            'lvn_price': self.lvn_price,
            'poc_target': self.poc_target,
            'confidence': self.confidence,
            'risk_reward': self.risk_reward,
            'market_state': self.market_state,
            'aggression_strength': self.aggression.strength,
            'aggression_description': self.aggression.description,
            'model': 'TREND'
        }


class TrendModelGenerator:
    """
    Generate Trend Model signals per TradeZella Auction Market Playbook.
    
    Strategy Steps:
    1. Confirm market is out of balance with momentum
    2. Calculate Volume Profile on the impulse leg
    3. Identify LVN retracement zones
    4. Wait for pullback to LVN with order flow confirmation
    5. Generate signal with SL beyond LVN, TP at prior POC
    """
    
    def __init__(
        self,
        confidence_threshold: float = None,
        min_risk_reward: float = 1.5,
        sl_buffer_pct: float = 0.1
    ):
        """
        Initialize Trend Model generator.
        
        Args:
            confidence_threshold: Minimum confidence for signal generation
            min_risk_reward: Minimum risk/reward ratio
            sl_buffer_pct: Stop loss buffer beyond LVN (percentage)
        """
        self.confidence_threshold = confidence_threshold or settings.trend_confidence_threshold
        self.min_risk_reward = min_risk_reward
        self.sl_buffer_pct = sl_buffer_pct
    
    def generate_signal(
        self,
        klines: List[Kline],
        trades: List[Trade],
        symbol: str,
        prior_poc: Optional[float] = None
    ) -> Optional[TrendSignal]:
        """
        Attempt to generate a Trend Model signal.
        
        Args:
            klines: Recent klines for analysis
            trades: Recent trades for order flow
            symbol: Trading pair
            prior_poc: POC from prior session (target)
        
        Returns:
            TrendSignal if conditions met, None otherwise
        """
        if len(klines) < 20:
            return None
        
        timeframe = klines[0].interval
        current_price = klines[-1].close
        
        # Step 1: Confirm market is out of balance
        market_analysis = market_state_analyzer.analyze(klines)
        
        if not market_analysis.is_out_of_balance:
            logger.debug(f"{symbol}: Market is balanced, Trend Model not applicable")
            return None
        
        # Determine trend direction
        is_bullish = market_analysis.state == MarketState.TRENDING_UP
        direction = "LONG" if is_bullish else "SHORT"
        
        logger.info(f"{symbol}: Market out of balance - {market_analysis.state.value}")
        
        # Step 2: Detect impulse leg and calculate Volume Profile on it
        impulse = market_state_analyzer.detect_impulse_leg(klines)
        
        if impulse is None:
            logger.debug(f"{symbol}: No clear impulse leg detected")
            return None
        
        impulse_start_idx, impulse_end_idx = impulse
        impulse_klines = klines[impulse_start_idx:impulse_end_idx + 1]
        
        impulse_vp = volume_profile_calculator.calculate_from_klines(impulse_klines)
        
        if not impulse_vp.lvn_zones:
            logger.debug(f"{symbol}: No LVN zones in impulse leg")
            return None
        
        impulse_start_price = impulse_klines[0].open
        impulse_end_price = impulse_klines[-1].close
        
        logger.info(f"{symbol}: Impulse leg found - {impulse_start_price:.2f} to {impulse_end_price:.2f}")
        
        # Step 3: Find relevant LVN for retracement
        # For bullish: LVN below current price (pullback zone)
        # For bearish: LVN above current price (retracement zone)
        
        if is_bullish:
            relevant_lvns = [(p, v) for p, v in impulse_vp.lvn_zones if p < current_price]
        else:
            relevant_lvns = [(p, v) for p, v in impulse_vp.lvn_zones if p > current_price]
        
        if not relevant_lvns:
            logger.debug(f"{symbol}: No relevant LVN zones for current price")
            return None
        
        # Get nearest LVN to current price
        nearest_lvn = min(relevant_lvns, key=lambda x: abs(x[0] - current_price))
        lvn_price = nearest_lvn[0]
        
        # Check if price is at or near LVN
        lvn_distance_pct = abs(current_price - lvn_price) / current_price * 100
        
        if lvn_distance_pct > 0.5:  # Not at LVN yet
            logger.debug(f"{symbol}: Price not at LVN (distance: {lvn_distance_pct:.2f}%)")
            return None
        
        logger.info(f"{symbol}: Price at LVN zone - {lvn_price:.2f}")
        
        # Step 4: Check order flow confirmation
        trade_direction = "BUY" if is_bullish else "SELL"
        aggression = order_flow_analyzer.analyze_aggression(trades, symbol, trade_direction)
        
        if aggression.strength < 0.4:  # Minimum aggression needed
            logger.debug(f"{symbol}: Insufficient aggression ({aggression.strength:.2f})")
            return None
        
        logger.info(f"{symbol}: Order flow confirmed - {aggression.description}")
        
        # Step 5: Calculate entry, stop loss, and take profit
        # Entry: Current price (at LVN)
        entry_price = current_price
        
        # Stop Loss: Just beyond LVN with buffer
        sl_buffer = entry_price * self.sl_buffer_pct / 100
        
        if is_bullish:
            stop_loss = lvn_price - sl_buffer
        else:
            stop_loss = lvn_price + sl_buffer
        
        # Take Profit: Prior POC or calculate from impulse
        if prior_poc:
            take_profit = prior_poc
        else:
            # Use POC from previous balance (before impulse)
            pre_impulse_klines = klines[:impulse_start_idx]
            if len(pre_impulse_klines) >= 10:
                prior_vp = volume_profile_calculator.calculate_from_klines(pre_impulse_klines)
                take_profit = prior_vp.poc_price
            else:
                # Fallback: Project based on impulse size
                impulse_size = abs(impulse_end_price - impulse_start_price)
                if is_bullish:
                    take_profit = entry_price + impulse_size * 0.5
                else:
                    take_profit = entry_price - impulse_size * 0.5
        
        # Calculate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward = reward / risk if risk > 0 else 0
        risk_percent = risk / entry_price * 100
        
        if risk_reward < self.min_risk_reward:
            logger.debug(f"{symbol}: Risk/reward too low ({risk_reward:.2f})")
            return None
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            market_analysis.confidence,
            aggression.strength,
            risk_reward,
            lvn_distance_pct
        )
        
        if confidence < self.confidence_threshold:
            logger.debug(f"{symbol}: Confidence below threshold ({confidence:.2f})")
            return None
        
        # Generate signal
        signal = TrendSignal(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lvn_price=lvn_price,
            poc_target=take_profit,
            impulse_start=impulse_start_price,
            impulse_end=impulse_end_price,
            confidence=confidence,
            aggression=aggression,
            market_state=market_analysis.state.value,
            risk_reward=risk_reward,
            risk_percent=risk_percent
        )
        
        logger.info(
            f"{symbol}: TREND SIGNAL - {direction} @ {entry_price:.2f} | "
            f"SL: {stop_loss:.2f} | TP: {take_profit:.2f} | RR: {risk_reward:.2f}"
        )
        
        return signal
    
    def _calculate_confidence(
        self,
        market_conf: float,
        aggression_strength: float,
        risk_reward: float,
        lvn_distance: float
    ) -> float:
        """Calculate overall signal confidence."""
        
        # Market state confidence (30%)
        market_score = market_conf * 0.3
        
        # Aggression confirmation (40%)
        aggression_score = aggression_strength * 0.4
        
        # Risk/reward quality (20%)
        rr_score = min(risk_reward / 3, 1.0) * 0.2
        
        # Precision to LVN (10%)
        precision_score = max(0, 1 - lvn_distance / 0.5) * 0.1
        
        return market_score + aggression_score + rr_score + precision_score


# Global generator instance
trend_model = TrendModelGenerator()
