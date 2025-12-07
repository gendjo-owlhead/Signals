"""
Signal Manager - Aggregates and manages all signal generation.
"""
import asyncio
from typing import List, Dict, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from loguru import logger

from config import settings
from data.binance_ws import binance_ws, Kline, Trade
from data.storage import storage
from analysis.volume_profile import volume_profile_calculator
from analysis.market_state import market_state_analyzer
from signals.trend_model import trend_model, TrendSignal
from signals.mean_reversion import mean_reversion_model, ReversionSignal


@dataclass
class SignalUpdate:
    """Real-time signal update for WebSocket broadcast."""
    type: str  # "new_signal", "signal_update", "analysis_update"
    timestamp: int
    symbol: str
    data: dict


@dataclass
class AnalysisSnapshot:
    """Current analysis state for a symbol."""
    symbol: str
    timeframe: str
    timestamp: int
    
    # Price data
    current_price: float
    
    # Market state
    market_state: str
    market_state_confidence: float
    is_balanced: bool
    
    # Volume Profile
    poc: float
    vah: float
    val: float
    lvn_zones: List[tuple]
    
    # Order flow
    cvd: float
    cvd_trend: str
    aggression_direction: str
    aggression_strength: float
    
    # Active signals
    active_signals: List[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp,
            'current_price': self.current_price,
            'market_state': self.market_state,
            'market_state_confidence': self.market_state_confidence,
            'is_balanced': self.is_balanced,
            'poc': self.poc,
            'vah': self.vah,
            'val': self.val,
            'lvn_zones': [{'price': p, 'volume': v} for p, v in self.lvn_zones],
            'cvd': self.cvd,
            'cvd_trend': self.cvd_trend,
            'aggression_direction': self.aggression_direction,
            'aggression_strength': self.aggression_strength,
            'active_signals': self.active_signals
        }


class SignalManager:
    """
    Central signal management and coordination.
    
    Responsibilities:
    - Run continuous analysis on all configured pairs
    - Generate signals from both Trend and Mean Reversion models
    - Store signals in database
    - Broadcast updates to connected clients
    - Track signal outcomes for ML training
    """
    
    def __init__(self):
        self.running = False
        
        # Signal storage
        self.active_signals: Dict[str, List[Union[TrendSignal, ReversionSignal]]] = {}
        self.signal_history: Dict[str, deque] = {}
        
        # Analysis snapshots
        self.analysis_snapshots: Dict[str, AnalysisSnapshot] = {}
        
        # Callbacks for real-time updates
        self.update_callbacks: List[Callable] = []
        
        # Initialize for configured pairs
        for symbol in settings.trading_pairs:
            self.active_signals[symbol] = []
            self.signal_history[symbol] = deque(maxlen=100)
    
    def on_update(self, callback: Callable[[SignalUpdate], None]):
        """Register callback for signal updates."""
        self.update_callbacks.append(callback)
    
    async def _broadcast_update(self, update: SignalUpdate):
        """Broadcast update to all registered callbacks."""
        for callback in self.update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(update)
                else:
                    callback(update)
            except Exception as e:
                logger.error(f"Update callback error: {e}")
    
    async def start(self):
        """Start continuous signal generation."""
        self.running = True
        logger.info("Signal Manager started")
        
        # Start analysis loop for each symbol
        tasks = [self._analysis_loop(symbol) for symbol in settings.trading_pairs]
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop signal generation."""
        self.running = False
        logger.info("Signal Manager stopped")
    
    async def _analysis_loop(self, symbol: str):
        """
        Continuous analysis loop for a symbol.
        Runs analysis on each new candle close.
        """
        logger.info(f"Starting analysis loop for {symbol}")
        
        # Track last analyzed candle
        last_analyzed_time = 0
        
        while self.running:
            try:
                # Get latest klines
                klines = binance_ws.get_klines(symbol, settings.primary_timeframe, 100)
                
                if not klines:
                    await asyncio.sleep(1)
                    continue
                
                current_kline = klines[-1]
                
                # Check for new closed candle
                if current_kline.is_closed and current_kline.close_time > last_analyzed_time:
                    last_analyzed_time = current_kline.close_time
                    
                    # Run analysis
                    await self._analyze_and_generate(symbol, klines)
                
                # Update snapshot even without new candle (for live price)
                await self._update_snapshot(symbol, klines)
                
                # Wait before next check
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Analysis loop error for {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def _analyze_and_generate(self, symbol: str, klines: List[Kline]):
        """Run full analysis and attempt signal generation."""
        
        logger.debug(f"{symbol}: Running analysis...")
        
        # Get recent trades for order flow
        trades = binance_ws.get_recent_trades(symbol, 500)
        
        # Try Trend Model first
        trend_signal = trend_model.generate_signal(klines, trades, symbol)
        
        if trend_signal and trend_signal.confidence >= settings.trend_confidence_threshold:
            await self._handle_new_signal(symbol, trend_signal)
        
        # Try Mean Reversion Model
        reversion_signal = mean_reversion_model.generate_signal(klines, trades, symbol)
        
        if reversion_signal and reversion_signal.confidence >= settings.reversion_confidence_threshold:
            await self._handle_new_signal(symbol, reversion_signal)
    
    async def _handle_new_signal(
        self,
        symbol: str,
        signal: Union[TrendSignal, ReversionSignal]
    ):
        """Process and store a new signal."""
        
        # Check if we already have an active signal for this symbol
        if self.active_signals[symbol]:
            # For now, don't generate conflicting signals
            logger.debug(f"{symbol}: Signal skipped - active signal exists")
            return
        
        # Store in memory
        self.active_signals[symbol].append(signal)
        self.signal_history[symbol].append(signal)
        
        # Store in database
        signal_type = "TREND" if isinstance(signal, TrendSignal) else "MEAN_REVERSION"
        
        try:
            signal_id = await storage.save_signal(
                symbol=symbol,
                timeframe=signal.timeframe,
                signal_type=signal_type,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                confidence=signal.confidence,
                model_type=signal_type,
                market_state=getattr(signal, 'market_state', None),
                lvn_price=signal.lvn_price,
                poc_price=signal.poc_target,
                cvd_value=getattr(signal.aggression, 'cvd_confirming', None),
                aggression_score=signal.aggression.strength
            )
            logger.info(f"{symbol}: Signal saved to database (ID: {signal_id})")
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
        
        # Broadcast update
        update = SignalUpdate(
            type="new_signal",
            timestamp=signal.timestamp,
            symbol=symbol,
            data=signal.to_dict()
        )
        await self._broadcast_update(update)
        
        # Execute trade if trading is enabled
        if settings.trading_enabled:
            try:
                from trading import order_executor
                result = await order_executor.execute_signal(signal)
                if result.success:
                    logger.info(f"{symbol}: Trade executed - Position ID: {result.position.id}")
                else:
                    logger.warning(f"{symbol}: Trade not executed - {result.message}")
            except Exception as e:
                logger.error(f"{symbol}: Trade execution error - {e}")
    
    async def _update_snapshot(self, symbol: str, klines: List[Kline]):
        """Update analysis snapshot for real-time display."""
        
        if not klines:
            return
        
        trades = binance_ws.get_recent_trades(symbol, 200)
        
        # Calculate Volume Profile
        vp = volume_profile_calculator.calculate_from_klines(klines[-50:])
        
        # Analyze market state
        market_analysis = market_state_analyzer.analyze(klines)
        
        # Get CVD pressure
        from analysis.order_flow import order_flow_analyzer
        cvd_pressure = order_flow_analyzer.get_cvd_pressure(symbol)
        
        # Get aggression
        if trades:
            # Check both directions, use stronger
            buy_agg = order_flow_analyzer.analyze_aggression(trades, symbol, "BUY")
            sell_agg = order_flow_analyzer.analyze_aggression(trades, symbol, "SELL")
            
            if buy_agg.strength > sell_agg.strength:
                aggression_direction = "BUY"
                aggression_strength = buy_agg.strength
            else:
                aggression_direction = "SELL"
                aggression_strength = sell_agg.strength
        else:
            aggression_direction = "NEUTRAL"
            aggression_strength = 0
        
        # Build snapshot
        snapshot = AnalysisSnapshot(
            symbol=symbol,
            timeframe=settings.primary_timeframe,
            timestamp=int(datetime.now().timestamp() * 1000),
            current_price=klines[-1].close,
            market_state=market_analysis.state.value,
            market_state_confidence=market_analysis.confidence,
            is_balanced=market_analysis.is_balanced,
            poc=vp.poc_price,
            vah=vp.vah,
            val=vp.val,
            lvn_zones=vp.lvn_zones[:5],  # Top 5 LVNs
            cvd=cvd_pressure.get('cvd', 0),
            cvd_trend=cvd_pressure.get('trend', 'neutral'),
            aggression_direction=aggression_direction,
            aggression_strength=aggression_strength,
            active_signals=[s.to_dict() for s in self.active_signals[symbol]]
        )
        
        self.analysis_snapshots[symbol] = snapshot
        
        # Broadcast analysis update
        update = SignalUpdate(
            type="analysis_update",
            timestamp=snapshot.timestamp,
            symbol=symbol,
            data=snapshot.to_dict()
        )
        await self._broadcast_update(update)
    
    def get_snapshot(self, symbol: str) -> Optional[AnalysisSnapshot]:
        """Get current analysis snapshot for a symbol."""
        return self.analysis_snapshots.get(symbol)
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[dict]:
        """Get all active signals, optionally filtered by symbol."""
        if symbol:
            return [s.to_dict() for s in self.active_signals.get(symbol, [])]
        
        all_signals = []
        for signals in self.active_signals.values():
            all_signals.extend([s.to_dict() for s in signals])
        return all_signals
    
    def get_signal_history(self, symbol: str, limit: int = 50) -> List[dict]:
        """Get signal history for a symbol."""
        history = list(self.signal_history.get(symbol, []))
        return [s.to_dict() for s in history[-limit:]]
    
    async def clear_signal(self, symbol: str, signal_id: Optional[int] = None):
        """Clear active signal(s) for a symbol."""
        if signal_id:
            # Clear specific signal
            self.active_signals[symbol] = [
                s for s in self.active_signals[symbol] 
                if s.timestamp != signal_id
            ]
        else:
            # Clear all for symbol
            self.active_signals[symbol] = []
        
        logger.info(f"{symbol}: Active signals cleared")


# Global signal manager instance
signal_manager = SignalManager()
