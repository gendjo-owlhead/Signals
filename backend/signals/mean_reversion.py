"""
Mean Reversion Model Signal Generator.

Implements the TradeZella Auction Market Playbook Mean Reversion Model:
"Failed Breakout → Reclaim Balance → LVN Pullback → POC Target"

This model is used when the market attempts a breakout but fails.
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
class ReversionSignal:
    """
    Mean Reversion Model trading signal.
    
    Per TradeZella:
    - Entry: At LVN pullback after failed breakout with order flow confirmation
    - Stop Loss: Just beyond the aggressive print (beyond failed breakout)
    - Take Profit: Balance POC (center of value)
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
    failed_breakout_price: float
    value_area_high: float
    value_area_low: float
    
    # Confidence and confirmation
    confidence: float
    aggression: AggressionSignal
    breakout_type: str  # "failed_upside" or "failed_downside"
    
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
            'breakout_type': self.breakout_type,
            'aggression_strength': self.aggression.strength,
            'aggression_description': self.aggression.description,
            'model': 'MEAN_REVERSION'
        }


class MeanReversionGenerator:
    """
    Generate Mean Reversion Model signals per TradeZella Auction Market Playbook.
    
    Strategy Steps:
    1. Market must be in balance or attempting breakout
    2. Identify failed breakout (price pushed out then came back inside)
    3. Wait for clear reclaim inside balance
    4. On pullback into reclaim leg LVN, check order flow
    5. Enter with SL beyond failed breakout, TP at POC
    
    Per TradeZella:
    "This setup is the opposite of the Trend Model. Instead of following a strong 
    move away from balance, you watch for times when the price breaks out but 
    can't hold. When that happens, the most likely outcome is that the market 
    returns to the Point of Control (POC)."
    """
    
    def __init__(
        self,
        confidence_threshold: float = None,
        min_risk_reward: float = 1.5,
        sl_buffer_pct: float = 0.1
    ):
        """
        Initialize Mean Reversion Model generator.
        
        Args:
            confidence_threshold: Minimum confidence for signal generation
            min_risk_reward: Minimum risk/reward ratio
            sl_buffer_pct: Stop loss buffer beyond failed breakout (percentage)
        """
        self.confidence_threshold = confidence_threshold or settings.reversion_confidence_threshold
        self.min_risk_reward = min_risk_reward
        self.sl_buffer_pct = sl_buffer_pct
    
    def generate_signal(
        self,
        klines: List[Kline],
        trades: List[Trade],
        symbol: str
    ) -> Optional[ReversionSignal]:
        """
        Attempt to generate a Mean Reversion Model signal.
        
        Args:
            klines: Recent klines for analysis
            trades: Recent trades for order flow
            symbol: Trading pair
        
        Returns:
            ReversionSignal if conditions met, None otherwise
        """
        if len(klines) < 30:
            return None
        
        timeframe = klines[0].interval
        current_price = klines[-1].close
        
        # Step 1: Calculate Volume Profile on recent balance
        # Use previous session/day as reference for balance
        balance_klines = klines[:-10]  # Exclude recent for balance reference
        balance_vp = volume_profile_calculator.calculate_from_klines(balance_klines)
        
        poc = balance_vp.poc_price
        vah = balance_vp.vah
        val = balance_vp.val
        
        if poc == 0:
            return None
        
        # Step 2: Detect failed breakout
        failed_breakout = market_state_analyzer.detect_failed_breakout(klines, balance_vp)
        
        if failed_breakout is None:
            logger.debug(f"{symbol}: No failed breakout detected")
            return None
        
        breakout_type = failed_breakout['type']
        is_failed_upside = breakout_type == 'failed_upside_breakout'
        
        # Direction: Short for failed upside, Long for failed downside
        direction = "SHORT" if is_failed_upside else "LONG"
        
        logger.info(f"{symbol}: Failed breakout detected - {breakout_type}")
        
        if is_failed_upside:
            failed_breakout_price = failed_breakout['breakout_high']
        else:
            failed_breakout_price = failed_breakout['breakout_low']
        
        # Step 3: Check that price has reclaimed inside balance
        # For failed upside: price should be back below VAH
        # For failed downside: price should be back above VAL
        
        if is_failed_upside and current_price > vah:
            logger.debug(f"{symbol}: Price not yet reclaimed (above VAH)")
            return None
        
        if not is_failed_upside and current_price < val:
            logger.debug(f"{symbol}: Price not yet reclaimed (below VAL)")
            return None
        
        # Step 4: Calculate Volume Profile on reclaim leg and find LVN
        # Reclaim leg: from breakout peak to current
        reclaim_start = len(klines) - 10
        reclaim_klines = klines[reclaim_start:]
        
        reclaim_vp = volume_profile_calculator.calculate_from_klines(reclaim_klines)
        
        if not reclaim_vp.lvn_zones:
            logger.debug(f"{symbol}: No LVN zones in reclaim leg")
            return None
        
        # Find LVN for pullback entry
        if is_failed_upside:
            # For short: LVN above current price (pullback up)
            relevant_lvns = [(p, v) for p, v in reclaim_vp.lvn_zones if p > current_price]
        else:
            # For long: LVN below current price (pullback down)
            relevant_lvns = [(p, v) for p, v in reclaim_vp.lvn_zones if p < current_price]
        
        if not relevant_lvns:
            # Check if already at an LVN
            nearest_lvn = reclaim_vp.get_nearest_lvn(current_price)
            if nearest_lvn:
                lvn_price = nearest_lvn[0]
                lvn_distance_pct = abs(current_price - lvn_price) / current_price * 100
                if lvn_distance_pct <= 0.3:
                    relevant_lvns = [nearest_lvn]
        
        if not relevant_lvns:
            logger.debug(f"{symbol}: No relevant LVN for pullback")
            return None
        
        # Get nearest LVN
        nearest_lvn = min(relevant_lvns, key=lambda x: abs(x[0] - current_price))
        lvn_price = nearest_lvn[0]
        
        # Check if price is at LVN
        lvn_distance_pct = abs(current_price - lvn_price) / current_price * 100
        
        if lvn_distance_pct > 0.5:
            logger.debug(f"{symbol}: Price not at LVN (distance: {lvn_distance_pct:.2f}%)")
            return None
        
        logger.info(f"{symbol}: Price at reclaim LVN - {lvn_price:.2f}")
        
        # Step 5: Check order flow confirmation
        # Look for aggression in the snap-back direction
        trade_direction = "SELL" if is_failed_upside else "BUY"
        aggression = order_flow_analyzer.analyze_aggression(trades, symbol, trade_direction)
        
        if aggression.strength < 0.4:
            logger.debug(f"{symbol}: Insufficient aggression ({aggression.strength:.2f})")
            return None
        
        logger.info(f"{symbol}: Order flow confirmed - {aggression.description}")
        
        # Step 6: Calculate entry, stop loss, and take profit
        entry_price = current_price
        
        # Stop Loss: Just beyond the failed breakout
        sl_buffer = entry_price * self.sl_buffer_pct / 100
        
        if is_failed_upside:
            stop_loss = failed_breakout_price + sl_buffer
        else:
            stop_loss = failed_breakout_price - sl_buffer
        
        # Take Profit: Balance POC
        take_profit = poc
        
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
            aggression.strength,
            risk_reward,
            lvn_distance_pct,
            failed_breakout_price,
            vah if is_failed_upside else val,
            current_price
        )
        
        if confidence < self.confidence_threshold:
            logger.debug(f"{symbol}: Confidence below threshold ({confidence:.2f})")
            return None
        
        # Generate signal
        signal = ReversionSignal(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lvn_price=lvn_price,
            poc_target=poc,
            failed_breakout_price=failed_breakout_price,
            value_area_high=vah,
            value_area_low=val,
            confidence=confidence,
            aggression=aggression,
            breakout_type=breakout_type,
            risk_reward=risk_reward,
            risk_percent=risk_percent
        )
        
        logger.info(
            f"{symbol}: REVERSION SIGNAL - {direction} @ {entry_price:.2f} | "
            f"SL: {stop_loss:.2f} | TP: {take_profit:.2f} | RR: {risk_reward:.2f}"
        )
        
        return signal
    
    def _calculate_confidence(
        self,
        aggression_strength: float,
        risk_reward: float,
        lvn_distance: float,
        failed_breakout_price: float,
        value_area_edge: float,
        current_price: float
    ) -> float:
        """Calculate overall signal confidence."""
        
        # Breakout failure clarity (25%)
        # How far did price extend beyond value area?
        breakout_extension = abs(failed_breakout_price - value_area_edge)
        reclaim_progress = abs(current_price - value_area_edge)
        if breakout_extension > 0:
            reclaim_score = min(reclaim_progress / breakout_extension, 1.0) * 0.25
        else:
            reclaim_score = 0.1
        
        # Aggression confirmation (40%)
        aggression_score = aggression_strength * 0.4
        
        # Risk/reward quality (20%)
        rr_score = min(risk_reward / 3, 1.0) * 0.2
        
        # Precision to LVN (15%)
        precision_score = max(0, 1 - lvn_distance / 0.5) * 0.15
        
        return reclaim_score + aggression_score + rr_score + precision_score


# Global generator instance
mean_reversion_model = MeanReversionGenerator()
