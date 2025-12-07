"""
Order Flow Analysis - CVD, Footprint, and Aggression Detection.
Core component for confirming entries per the Auction Market Playbook.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from loguru import logger

from config import settings
from data.binance_ws import Trade, OrderBook


@dataclass
class CVDPoint:
    """Cumulative Volume Delta data point."""
    timestamp: int
    cvd: float  # Cumulative delta
    delta: float  # Period delta
    buy_volume: float
    sell_volume: float
    total_volume: float


@dataclass
class FootprintLevel:
    """Footprint data at a single price level."""
    price: float
    bid_volume: float = 0.0  # Volume hitting bid (sells)
    ask_volume: float = 0.0  # Volume hitting ask (buys)
    trade_count: int = 0
    
    @property
    def delta(self) -> float:
        """Volume delta (buys - sells)."""
        return self.ask_volume - self.bid_volume
    
    @property
    def total_volume(self) -> float:
        return self.bid_volume + self.ask_volume
    
    @property
    def imbalance(self) -> float:
        """Imbalance ratio. >1 = buyer dominant, <1 = seller dominant."""
        if self.bid_volume == 0:
            return float('inf') if self.ask_volume > 0 else 1.0
        return self.ask_volume / self.bid_volume
    
    @property
    def is_buy_imbalance(self) -> bool:
        """Check if significant buy imbalance (3:1 ratio per TradeZella)."""
        return self.imbalance >= 3.0
    
    @property
    def is_sell_imbalance(self) -> bool:
        """Check if significant sell imbalance (1:3 ratio)."""
        return self.imbalance <= 0.33


@dataclass
class FootprintBar:
    """Footprint chart bar containing all price levels."""
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    levels: Dict[float, FootprintLevel] = field(default_factory=dict)
    
    @property
    def total_delta(self) -> float:
        """Total bar delta."""
        return sum(l.delta for l in self.levels.values())
    
    @property
    def buy_imbalances(self) -> List[float]:
        """Prices with buy imbalances."""
        return [price for price, level in self.levels.items() if level.is_buy_imbalance]
    
    @property
    def sell_imbalances(self) -> List[float]:
        """Prices with sell imbalances."""
        return [price for price, level in self.levels.items() if level.is_sell_imbalance]
    
    @property
    def has_strong_buy_aggression(self) -> bool:
        """Check for strong buy aggression (multiple buy imbalances)."""
        return len(self.buy_imbalances) >= 2 and self.total_delta > 0
    
    @property
    def has_strong_sell_aggression(self) -> bool:
        """Check for strong sell aggression (multiple sell imbalances)."""
        return len(self.sell_imbalances) >= 2 and self.total_delta < 0


@dataclass
class AggressionSignal:
    """
    Aggression detection signal for trade confirmation.
    Per TradeZella: "Wait for clear order flow confirmation at those levels - 
    big prints, imbalances, or strong CVD pressure."
    """
    timestamp: int
    symbol: str
    direction: str  # "BUY" or "SELL"
    strength: float  # 0-1 score
    cvd_confirming: bool
    imbalance_count: int
    large_prints_count: int
    description: str


class OrderFlowAnalyzer:
    """
    Order Flow Analysis for trade execution confirmation.
    
    Implements:
    1. CVD (Cumulative Volume Delta) - measures buy/sell pressure over time
    2. Footprint analysis - volume at each price level with imbalances
    3. Large order detection - identifies significant "prints"
    4. Aggression scoring - combines all factors for entry confirmation
    """
    
    def __init__(
        self,
        cvd_lookback: int = None,
        aggression_threshold: float = None,
        large_order_multiplier: float = None,
        tick_size_pct: float = 0.01
    ):
        """
        Initialize order flow analyzer.
        
        Args:
            cvd_lookback: Number of periods for CVD calculation
            aggression_threshold: Threshold for aggression detection
            large_order_multiplier: Multiple of avg for large order detection
            tick_size_pct: Tick size as percentage for footprint levels
        """
        self.cvd_lookback = cvd_lookback or settings.cvd_lookback
        self.aggression_threshold = aggression_threshold or settings.aggression_threshold
        self.large_order_mult = large_order_multiplier or settings.large_order_multiplier
        self.tick_size_pct = tick_size_pct
        
        # Rolling storage
        self.cvd_history: Dict[str, deque] = {}
        self.avg_trade_size: Dict[str, float] = {}
    
    def calculate_cvd(self, trades: List[Trade], symbol: str) -> List[CVDPoint]:
        """
        Calculate Cumulative Volume Delta from trades.
        
        CVD shows the running total of buy volume minus sell volume.
        Rising CVD = buyers in control, Falling CVD = sellers in control.
        
        Args:
            trades: List of trades
            symbol: Trading pair symbol
        
        Returns:
            List of CVD data points
        """
        if not trades:
            return []
        
        cvd_points = []
        cumulative_delta = 0.0
        
        # Group trades by time bucket (e.g., 1 second)
        time_bucket_ms = 1000
        current_bucket = trades[0].timestamp // time_bucket_ms * time_bucket_ms
        
        bucket_buy_vol = 0.0
        bucket_sell_vol = 0.0
        
        for trade in trades:
            trade_bucket = trade.timestamp // time_bucket_ms * time_bucket_ms
            
            if trade_bucket != current_bucket:
                # Save previous bucket
                delta = bucket_buy_vol - bucket_sell_vol
                cumulative_delta += delta
                
                cvd_points.append(CVDPoint(
                    timestamp=current_bucket,
                    cvd=cumulative_delta,
                    delta=delta,
                    buy_volume=bucket_buy_vol,
                    sell_volume=bucket_sell_vol,
                    total_volume=bucket_buy_vol + bucket_sell_vol
                ))
                
                # Reset for new bucket
                current_bucket = trade_bucket
                bucket_buy_vol = 0.0
                bucket_sell_vol = 0.0
            
            # Accumulate volume
            if trade.side == "BUY":
                bucket_buy_vol += trade.quantity
            else:
                bucket_sell_vol += trade.quantity
        
        # Don't forget last bucket
        delta = bucket_buy_vol - bucket_sell_vol
        cumulative_delta += delta
        cvd_points.append(CVDPoint(
            timestamp=current_bucket,
            cvd=cumulative_delta,
            delta=delta,
            buy_volume=bucket_buy_vol,
            sell_volume=bucket_sell_vol,
            total_volume=bucket_buy_vol + bucket_sell_vol
        ))
        
        # Update history
        if symbol not in self.cvd_history:
            self.cvd_history[symbol] = deque(maxlen=1000)
        self.cvd_history[symbol].extend(cvd_points)
        
        return cvd_points
    
    def build_footprint(
        self,
        trades: List[Trade],
        symbol: str,
        start_time: int,
        end_time: int
    ) -> FootprintBar:
        """
        Build footprint bar from trades.
        
        Footprint shows buy/sell volume at each price level,
        revealing imbalances (areas of strong aggression).
        
        Args:
            trades: List of trades
            symbol: Trading pair
            start_time: Bar start timestamp
            end_time: Bar end timestamp
        
        Returns:
            FootprintBar with all levels
        """
        if not trades:
            return FootprintBar(symbol=symbol, timestamp=start_time, 
                              open=0, high=0, low=0, close=0)
        
        # Filter trades to time range
        bar_trades = [t for t in trades if start_time <= t.timestamp <= end_time]
        
        if not bar_trades:
            return FootprintBar(symbol=symbol, timestamp=start_time,
                              open=0, high=0, low=0, close=0)
        
        # Determine tick size
        mid_price = bar_trades[len(bar_trades)//2].price
        tick_size = mid_price * self.tick_size_pct / 100
        
        # Group trades by price level
        levels: Dict[float, FootprintLevel] = {}
        
        for trade in bar_trades:
            # Round to tick
            level_price = round(trade.price / tick_size) * tick_size
            level_price = round(level_price, 8)
            
            if level_price not in levels:
                levels[level_price] = FootprintLevel(price=level_price)
            
            level = levels[level_price]
            level.trade_count += 1
            
            if trade.side == "BUY":
                level.ask_volume += trade.quantity
            else:
                level.bid_volume += trade.quantity
        
        # Build bar
        prices = [t.price for t in bar_trades]
        
        return FootprintBar(
            symbol=symbol,
            timestamp=start_time,
            open=bar_trades[0].price,
            high=max(prices),
            low=min(prices),
            close=bar_trades[-1].price,
            levels=levels
        )
    
    def detect_large_orders(
        self,
        trades: List[Trade],
        symbol: str
    ) -> List[Trade]:
        """
        Detect large orders ("big prints") that indicate significant aggression.
        
        Per TradeZella: "Look for aggression in the direction of the trend:
        Big sell bubbles or footprint imbalance for shorts.
        Big buy bubbles or imbalance for longs."
        
        Args:
            trades: List of trades
            symbol: Trading pair
        
        Returns:
            List of large trades
        """
        if not trades:
            return []
        
        # Calculate average trade size
        quantities = [t.quantity for t in trades]
        avg_size = np.mean(quantities)
        
        # Update running average
        if symbol in self.avg_trade_size:
            self.avg_trade_size[symbol] = (self.avg_trade_size[symbol] * 0.9) + (avg_size * 0.1)
        else:
            self.avg_trade_size[symbol] = avg_size
        
        # Find large orders (X times average)
        threshold = self.avg_trade_size[symbol] * self.large_order_mult
        
        large_orders = [t for t in trades if t.quantity >= threshold]
        
        return large_orders
    
    def analyze_aggression(
        self,
        trades: List[Trade],
        symbol: str,
        direction: str
    ) -> AggressionSignal:
        """
        Analyze order flow for aggression in a specific direction.
        
        This is the confirmation step per TradeZella:
        "Only enter when you see aggression. No aggression = no trade."
        
        Args:
            trades: Recent trades to analyze
            symbol: Trading pair
            direction: Expected direction ("BUY" for longs, "SELL" for shorts)
        
        Returns:
            AggressionSignal with strength score
        """
        if not trades:
            return AggressionSignal(
                timestamp=int(datetime.now().timestamp() * 1000),
                symbol=symbol,
                direction=direction,
                strength=0.0,
                cvd_confirming=False,
                imbalance_count=0,
                large_prints_count=0,
                description="No trades to analyze"
            )
        
        # 1. Calculate CVD trend
        cvd_points = self.calculate_cvd(trades, symbol)
        cvd_confirming = False
        
        if len(cvd_points) >= 3:
            recent_cvd = [p.cvd for p in cvd_points[-5:]]
            cvd_trend = recent_cvd[-1] - recent_cvd[0]
            
            if direction == "BUY" and cvd_trend > 0:
                cvd_confirming = True
            elif direction == "SELL" and cvd_trend < 0:
                cvd_confirming = True
        
        # 2. Build footprint and count imbalances
        footprint = self.build_footprint(
            trades, symbol, 
            trades[0].timestamp, 
            trades[-1].timestamp
        )
        
        if direction == "BUY":
            imbalance_count = len(footprint.buy_imbalances)
        else:
            imbalance_count = len(footprint.sell_imbalances)
        
        # 3. Count large orders in direction
        large_orders = self.detect_large_orders(trades, symbol)
        large_prints_count = sum(1 for t in large_orders if t.side == direction)
        
        # 4. Calculate overall strength score
        score_components = []
        
        # CVD component (0-0.4)
        if cvd_confirming:
            score_components.append(0.4)
        
        # Imbalance component (0-0.3)
        imb_score = min(imbalance_count / 3, 1.0) * 0.3
        score_components.append(imb_score)
        
        # Large prints component (0-0.3)
        print_score = min(large_prints_count / 2, 1.0) * 0.3
        score_components.append(print_score)
        
        strength = sum(score_components)
        
        # Build description
        descriptions = []
        if cvd_confirming:
            descriptions.append("CVD confirming")
        if imbalance_count > 0:
            descriptions.append(f"{imbalance_count} imbalances")
        if large_prints_count > 0:
            descriptions.append(f"{large_prints_count} large prints")
        
        return AggressionSignal(
            timestamp=trades[-1].timestamp,
            symbol=symbol,
            direction=direction,
            strength=strength,
            cvd_confirming=cvd_confirming,
            imbalance_count=imbalance_count,
            large_prints_count=large_prints_count,
            description=", ".join(descriptions) if descriptions else "Weak aggression"
        )
    
    def get_cvd_pressure(self, symbol: str, lookback: int = None) -> Dict:
        """
        Get CVD pressure analysis for a symbol.
        
        Returns:
            Dict with CVD analysis (trend, strength, direction)
        """
        lookback = lookback or self.cvd_lookback
        history = list(self.cvd_history.get(symbol, []))
        
        if len(history) < 2:
            return {
                'cvd': 0,
                'trend': 'neutral',
                'strength': 0,
                'direction': 0
            }
        
        recent = history[-lookback:] if len(history) > lookback else history
        
        cvd_values = [p.cvd for p in recent]
        cvd_change = cvd_values[-1] - cvd_values[0]
        
        # Calculate trend strength
        if len(cvd_values) > 1:
            cvd_std = np.std(cvd_values)
            strength = abs(cvd_change) / (cvd_std + 0.0001)
        else:
            strength = 0
        
        if cvd_change > 0:
            trend = 'bullish'
            direction = 1
        elif cvd_change < 0:
            trend = 'bearish'
            direction = -1
        else:
            trend = 'neutral'
            direction = 0
        
        return {
            'cvd': cvd_values[-1],
            'cvd_change': cvd_change,
            'trend': trend,
            'strength': min(strength, 10) / 10,  # Normalize to 0-1
            'direction': direction
        }


# Global analyzer instance
order_flow_analyzer = OrderFlowAnalyzer()
