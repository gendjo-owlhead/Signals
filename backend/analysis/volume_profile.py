"""
Volume Profile calculation with LVN (Low Volume Node) and POC (Point of Control) detection.
Core component of the Auction Market Playbook strategy.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from config import settings
from data.binance_ws import Kline, Trade


@dataclass
class VolumeLevel:
    """Volume at a specific price level."""
    price: float
    volume: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trade_count: int = 0
    
    @property
    def delta(self) -> float:
        """Volume delta (buy - sell)."""
        return self.buy_volume - self.sell_volume
    
    @property
    def imbalance_ratio(self) -> float:
        """Ratio of dominant side."""
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0
        return max(self.buy_volume, self.sell_volume) / total


@dataclass
class VolumeProfile:
    """
    Volume Profile representing volume distribution across price levels.
    Used to identify:
    - POC (Point of Control) - price with highest volume
    - VAH (Value Area High) - upper bound of 70% volume
    - VAL (Value Area Low) - lower bound of 70% volume
    - LVN (Low Volume Nodes) - prices with significantly low volume
    - HVN (High Volume Nodes) - prices with significantly high volume
    """
    symbol: str
    timeframe: str
    start_time: int
    end_time: int
    levels: List[VolumeLevel] = field(default_factory=list)
    
    # Key levels
    poc_price: float = 0.0
    poc_volume: float = 0.0
    vah: float = 0.0  # Value Area High
    val: float = 0.0  # Value Area Low
    profile_high: float = 0.0
    profile_low: float = 0.0
    total_volume: float = 0.0
    
    # LVN and HVN zones
    lvn_zones: List[Tuple[float, float]] = field(default_factory=list)  # (price, volume)
    hvn_zones: List[Tuple[float, float]] = field(default_factory=list)
    
    @property
    def value_area_range(self) -> float:
        """Value area range as percentage of profile."""
        if self.profile_high == 0:
            return 0
        return (self.vah - self.val) / self.profile_high * 100
    
    @property
    def is_balanced(self) -> bool:
        """Check if profile suggests balanced market (tight value area)."""
        return self.value_area_range < 2.0  # Less than 2% range
    
    def get_nearest_lvn(self, price: float) -> Optional[Tuple[float, float]]:
        """Get the LVN nearest to a given price."""
        if not self.lvn_zones:
            return None
        
        nearest = min(self.lvn_zones, key=lambda x: abs(x[0] - price))
        return nearest
    
    def is_price_at_lvn(self, price: float, tolerance_pct: float = 0.1) -> bool:
        """Check if price is at or near an LVN."""
        for lvn_price, _ in self.lvn_zones:
            if abs(price - lvn_price) / price * 100 <= tolerance_pct:
                return True
        return False


class VolumeProfileCalculator:
    """
    Calculate Volume Profile from klines or trades.
    
    Implements the core Volume Profile logic needed for the Auction Market strategy:
    1. Build volume distribution across price levels
    2. Identify POC (Point of Control) - fair value / target
    3. Calculate Value Area (70% of volume)
    4. Detect LVN (Low Volume Nodes) - entry points
    """
    
    def __init__(
        self,
        tick_size: float = 0.01,
        value_area_pct: float = None,
        lvn_threshold_pct: float = None
    ):
        """
        Initialize calculator.
        
        Args:
            tick_size: Price granularity for volume buckets (as % of price)
            value_area_pct: Percentage of volume for value area (default from settings)
            lvn_threshold_pct: Percentile threshold for LVN detection (default from settings)
        """
        self.tick_size_pct = tick_size
        self.value_area_pct = value_area_pct or settings.value_area_percentage
        self.lvn_threshold_pct = lvn_threshold_pct or settings.lvn_threshold_percentile
    
    def calculate_from_klines(
        self,
        klines: List[Kline],
        num_levels: int = 50
    ) -> VolumeProfile:
        """
        Calculate Volume Profile from OHLCV klines.
        
        This method distributes each candle's volume across its price range,
        approximating where volume occurred within the candle.
        
        Args:
            klines: List of Kline objects
            num_levels: Number of price levels in the profile
        
        Returns:
            VolumeProfile object with all key levels identified
        """
        if not klines:
            return VolumeProfile(symbol="", timeframe="", start_time=0, end_time=0)
        
        symbol = klines[0].symbol
        timeframe = klines[0].interval
        start_time = klines[0].open_time
        end_time = klines[-1].close_time
        
        # Find price range
        all_highs = [k.high for k in klines]
        all_lows = [k.low for k in klines]
        profile_high = max(all_highs)
        profile_low = min(all_lows)
        
        # Create price levels
        price_step = (profile_high - profile_low) / num_levels
        if price_step == 0:
            price_step = profile_low * 0.001  # Fallback for flat price
        
        # Initialize volume buckets
        volume_by_level: Dict[float, VolumeLevel] = {}
        
        for i in range(num_levels + 1):
            price = profile_low + (i * price_step)
            volume_by_level[round(price, 8)] = VolumeLevel(
                price=round(price, 8),
                volume=0.0,
                buy_volume=0.0,
                sell_volume=0.0
            )
        
        # Distribute volume from each candle
        for kline in klines:
            candle_range = kline.high - kline.low
            if candle_range == 0:
                candle_range = 0.0001
            
            # Volume distribution within candle (simplified TPO)
            # More volume at close area, distributed along the range
            for level_price in volume_by_level.keys():
                if kline.low <= level_price <= kline.high:
                    # Calculate volume share based on position in candle
                    # Weight towards close price
                    close_dist = abs(level_price - kline.close)
                    weight = 1 - (close_dist / candle_range) * 0.5
                    
                    vol_share = (kline.volume / num_levels) * weight
                    
                    # Estimate buy/sell based on taker_buy_volume
                    buy_ratio = kline.taker_buy_volume / kline.volume if kline.volume > 0 else 0.5
                    
                    volume_by_level[level_price].volume += vol_share
                    volume_by_level[level_price].buy_volume += vol_share * buy_ratio
                    volume_by_level[level_price].sell_volume += vol_share * (1 - buy_ratio)
                    volume_by_level[level_price].trade_count += 1
        
        # Convert to sorted list
        levels = sorted(volume_by_level.values(), key=lambda x: x.price)
        
        # Build the profile
        profile = self._build_profile(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            levels=levels,
            profile_high=profile_high,
            profile_low=profile_low
        )
        
        return profile
    
    def calculate_from_trades(
        self,
        trades: List[Trade],
        symbol: str,
        timeframe: str,
        num_levels: int = 50
    ) -> VolumeProfile:
        """
        Calculate Volume Profile from individual trades.
        More accurate than kline-based calculation.
        
        Args:
            trades: List of Trade objects
            symbol: Trading pair symbol
            timeframe: Reference timeframe
            num_levels: Number of price levels
        
        Returns:
            VolumeProfile object
        """
        if not trades:
            return VolumeProfile(symbol=symbol, timeframe=timeframe, start_time=0, end_time=0)
        
        start_time = trades[0].timestamp
        end_time = trades[-1].timestamp
        
        # Find price range
        prices = [t.price for t in trades]
        profile_high = max(prices)
        profile_low = min(prices)
        
        # Create price buckets
        price_step = (profile_high - profile_low) / num_levels
        if price_step == 0:
            price_step = profile_low * 0.001
        
        # Aggregate volume by price level
        volume_by_level: Dict[int, VolumeLevel] = {}
        
        for trade in trades:
            # Find bucket index
            bucket_idx = int((trade.price - profile_low) / price_step)
            bucket_idx = min(bucket_idx, num_levels - 1)
            bucket_price = profile_low + (bucket_idx * price_step)
            bucket_price = round(bucket_price, 8)
            
            if bucket_idx not in volume_by_level:
                volume_by_level[bucket_idx] = VolumeLevel(
                    price=bucket_price,
                    volume=0.0,
                    buy_volume=0.0,
                    sell_volume=0.0
                )
            
            level = volume_by_level[bucket_idx]
            level.volume += trade.quantity
            level.trade_count += 1
            
            if trade.side == "BUY":
                level.buy_volume += trade.quantity
            else:
                level.sell_volume += trade.quantity
        
        # Fill missing levels
        for i in range(num_levels):
            if i not in volume_by_level:
                bucket_price = round(profile_low + (i * price_step), 8)
                volume_by_level[i] = VolumeLevel(
                    price=bucket_price,
                    volume=0.0
                )
        
        # Convert to sorted list
        levels = sorted(volume_by_level.values(), key=lambda x: x.price)
        
        return self._build_profile(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            levels=levels,
            profile_high=profile_high,
            profile_low=profile_low
        )
    
    def _build_profile(
        self,
        symbol: str,
        timeframe: str,
        start_time: int,
        end_time: int,
        levels: List[VolumeLevel],
        profile_high: float,
        profile_low: float
    ) -> VolumeProfile:
        """Build VolumeProfile with all derived values."""
        
        # Calculate total volume
        total_volume = sum(l.volume for l in levels)
        
        # Find POC (Point of Control) - highest volume level
        poc_level = max(levels, key=lambda x: x.volume) if levels else None
        poc_price = poc_level.price if poc_level else 0
        poc_volume = poc_level.volume if poc_level else 0
        
        # Calculate Value Area (levels containing X% of total volume)
        value_area_volume = total_volume * (self.value_area_pct / 100)
        
        # Start from POC and expand outward
        poc_idx = next((i for i, l in enumerate(levels) if l.price == poc_price), len(levels) // 2)
        
        included_volume = poc_volume
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        
        while included_volume < value_area_volume:
            # Compare volume above and below current value area
            vol_above = levels[va_high_idx + 1].volume if va_high_idx + 1 < len(levels) else 0
            vol_below = levels[va_low_idx - 1].volume if va_low_idx - 1 >= 0 else 0
            
            if vol_above == 0 and vol_below == 0:
                break
            
            if vol_above >= vol_below and va_high_idx + 1 < len(levels):
                va_high_idx += 1
                included_volume += levels[va_high_idx].volume
            elif va_low_idx - 1 >= 0:
                va_low_idx -= 1
                included_volume += levels[va_low_idx].volume
            else:
                break
        
        vah = levels[va_high_idx].price if va_high_idx < len(levels) else profile_high
        val = levels[va_low_idx].price if va_low_idx >= 0 else profile_low
        
        # Identify LVN (Low Volume Nodes)
        volumes = [l.volume for l in levels if l.volume > 0]
        if volumes:
            lvn_threshold = np.percentile(volumes, self.lvn_threshold_pct)
            hvn_threshold = np.percentile(volumes, 100 - self.lvn_threshold_pct)
            
            lvn_zones = [(l.price, l.volume) for l in levels if 0 < l.volume <= lvn_threshold]
            hvn_zones = [(l.price, l.volume) for l in levels if l.volume >= hvn_threshold]
        else:
            lvn_zones = []
            hvn_zones = []
        
        return VolumeProfile(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            levels=levels,
            poc_price=poc_price,
            poc_volume=poc_volume,
            vah=vah,
            val=val,
            profile_high=profile_high,
            profile_low=profile_low,
            total_volume=total_volume,
            lvn_zones=lvn_zones,
            hvn_zones=hvn_zones
        )
    
    def calculate_impulse_profile(
        self,
        klines: List[Kline],
        impulse_start_idx: int,
        impulse_end_idx: int
    ) -> VolumeProfile:
        """
        Calculate Volume Profile on an impulse leg for the Trend Model.
        
        Per TradeZella strategy:
        "Take the impulse leg that broke the structure. Apply a Volume Profile 
        to that leg. Identify Low-Volume Nodes (LVNs) inside that move."
        
        Args:
            klines: Full kline list
            impulse_start_idx: Index where impulse started
            impulse_end_idx: Index where impulse ended
        
        Returns:
            VolumeProfile for the impulse leg
        """
        impulse_klines = klines[impulse_start_idx:impulse_end_idx + 1]
        return self.calculate_from_klines(impulse_klines)
    
    def calculate_reclaim_profile(
        self,
        klines: List[Kline],
        reclaim_start_idx: int,
        reclaim_end_idx: int
    ) -> VolumeProfile:
        """
        Calculate Volume Profile on a reclaim leg for the Mean Reversion Model.
        
        Per TradeZella strategy:
        "Wait for a clear reclaim inside balance. A pullback into the reclaim leg.
        Apply volume profile on the reclaim leg and mark the LVNs."
        
        Args:
            klines: Full kline list
            reclaim_start_idx: Index where reclaim started
            reclaim_end_idx: Index where reclaim ended
        
        Returns:
            VolumeProfile for the reclaim leg
        """
        reclaim_klines = klines[reclaim_start_idx:reclaim_end_idx + 1]
        return self.calculate_from_klines(reclaim_klines)


# Global calculator instance
volume_profile_calculator = VolumeProfileCalculator()
