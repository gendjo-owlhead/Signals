"""
Signal Manager - Aggregates and manages all signal generation across multiple timeframes.
"""
import asyncio
from typing import List, Dict, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from loguru import logger

from config import settings
from data.binance_ws import binance_ws, Kline, Trade
from data.storage import storage
from analysis.volume_profile import volume_profile_calculator
from analysis.market_state import market_state_analyzer
from signals.scalper_strategy import scalper_generator, ScalperSignal
from signals.multi_timeframe_analyzer import mtf_analyzer
from ml.trainer import online_trainer


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
    
    # Scalper info
    scalper_signal: str = "NONE"
    scalper_stoch: float = 50.0
    scalper_trend: str = "NEUTRAL"
    
    # Active signals
    active_signals: List[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp,
            'current_price': float(self.current_price),
            'market_state': self.market_state,
            'market_state_confidence': float(self.market_state_confidence),
            'is_balanced': bool(self.is_balanced),
            'poc': float(self.poc) if self.poc else 0.0,
            'vah': float(self.vah) if self.vah else 0.0,
            'val': float(self.val) if self.val else 0.0,
            'lvn_zones': [{'price': float(p), 'volume': float(v)} for p, v in self.lvn_zones],
            'cvd': float(self.cvd) if self.cvd else 0.0,
            'cvd_trend': self.cvd_trend,
            'aggression_direction': self.aggression_direction,
            'aggression_strength': float(self.aggression_strength),
            'scalper_signal': self.scalper_signal,
            'scalper_stoch': float(self.scalper_stoch),
            'scalper_trend': self.scalper_trend,
            'active_signals': self.active_signals
        }


class SignalManager:
    """
    Central signal management and coordination for multi-timeframe analysis.
    
    Responsibilities:
    - Run continuous analysis on all configured pairs across all timeframes
    - Generate signals using EMA 5-8-13 Scalping strategy
    - Validate signals against multi-timeframe confluence
    - Store signals in database
    - Broadcast updates to connected clients
    - Track signal outcomes for ML training
    """
    
    def __init__(self):
        self.running = False
        
        # All timeframes we monitor
        self.all_timeframes = settings.timeframes  # ["1m", "5m", "15m", "1h"]
        
        # Signal storage
        self.active_signals: Dict[str, List[ScalperSignal]] = {}
        self.signal_history: Dict[str, deque] = {}
        
        # Analysis snapshots - now keyed by symbol AND timeframe
        self.analysis_snapshots: Dict[str, AnalysisSnapshot] = {}  # Legacy: primary TF only
        self.timeframe_snapshots: Dict[str, Dict[str, AnalysisSnapshot]] = {}  # New: all TFs
        
        # Callbacks for real-time updates
        self.update_callbacks: List[Callable] = []
        
        # Initialize for configured pairs
        for symbol in settings.trading_pairs:
            self.active_signals[symbol] = []
            self.signal_history[symbol] = deque(maxlen=100)
            self.timeframe_snapshots[symbol] = {}
    
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
        """Start continuous signal generation for all timeframes."""
        self.running = True
        logger.info(f"Signal Manager started - analyzing {len(self.all_timeframes)} timeframes: {self.all_timeframes}")
        
        # Start analysis loop for each symbol (handles all timeframes internally)
        tasks = [self._analysis_loop(symbol) for symbol in settings.trading_pairs]
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop signal generation."""
        self.running = False
        logger.info("Signal Manager stopped")
    
    async def _analysis_loop(self, symbol: str):
        """
        Continuous analysis loop for a symbol across all timeframes.
        Runs analysis on each new candle close.
        """
        logger.info(f"Starting multi-timeframe analysis loop for {symbol}")
        
        # Track last analyzed candle per timeframe
        last_analyzed_times: Dict[str, int] = {tf: 0 for tf in self.all_timeframes}
        last_generated_time: int = 0
        
        while self.running:
            try:
                # Analyze ALL timeframes
                for timeframe in self.all_timeframes:
                    await self._analyze_timeframe(symbol, timeframe, last_analyzed_times)
                
                # Primary timeframe generates actual trades
                primary_klines = binance_ws.get_klines(symbol, settings.primary_timeframe, 150)
                if primary_klines:
                    current_kline = primary_klines[-1]
                    if current_kline.is_closed and current_kline.close_time > last_generated_time:
                        last_generated_time = current_kline.close_time
                        await self._analyze_and_generate(symbol, primary_klines)
                    
                    # Update primary snapshot for backward compatibility
                    await self._update_snapshot(symbol, primary_klines)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Analysis loop error for {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def _analyze_timeframe(self, symbol: str, timeframe: str, last_analyzed_times: Dict[str, int]):
        """
        Analyze a specific timeframe and update MTF analyzer.
        """
        try:
            klines = binance_ws.get_klines(symbol, timeframe, 100)
            if not klines:
                return
            
            current_kline = klines[-1]
            
            # Only update on new closed candle
            if not current_kline.is_closed or current_kline.close_time <= last_analyzed_times.get(timeframe, 0):
                return
            
            last_analyzed_times[timeframe] = current_kline.close_time
            
            # Calculate EMA trend and stochastic for this timeframe
            closes = [k.close for k in klines]
            
            # Simple EMA calculation for trend detection
            ema_fast = self._calculate_ema(closes, 5)
            ema_mid = self._calculate_ema(closes, 8)
            ema_slow = self._calculate_ema(closes, 13)
            
            # Determine trend from EMAs
            if ema_fast > ema_mid > ema_slow:
                ema_trend = "UP"
            elif ema_fast < ema_mid < ema_slow:
                ema_trend = "DOWN"
            else:
                ema_trend = "NEUTRAL"
            
            # Determine direction and strength
            price = current_kline.close
            if ema_trend == "UP" and price > ema_fast:
                direction = "LONG"
                strength = min(1.0, (price - ema_slow) / ema_slow * 20)
            elif ema_trend == "DOWN" and price < ema_fast:
                direction = "SHORT"
                strength = min(1.0, (ema_slow - price) / ema_slow * 20)
            else:
                direction = "NEUTRAL"
                strength = 0.3
            
            # Calculate a simple stochastic approximation
            highs = [k.high for k in klines[-14:]]
            lows = [k.low for k in klines[-14:]]
            highest = max(highs) if highs else price
            lowest = min(lows) if lows else price
            stoch_k = ((price - lowest) / (highest - lowest) * 100) if highest != lowest else 50.0
            
            # Confidence based on trend alignment
            confidence = 0.5
            if ema_trend != "NEUTRAL":
                confidence += 0.2
            if abs(stoch_k - 50) > 30:  # Strong stoch reading
                confidence += 0.15
            confidence = min(1.0, confidence)
            
            # Update MTF analyzer
            mtf_analyzer.update_signal(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                ema_trend=ema_trend,
                stoch_k=stoch_k,
                confidence=confidence
            )
            
            logger.debug(f"{symbol} {timeframe}: {direction} (strength={strength:.2f}, stoch={stoch_k:.0f})")
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
    
    def _calculate_ema(self, values: List[float], period: int) -> float:
        """Calculate EMA for the given values."""
        if len(values) < period:
            return values[-1] if values else 0.0
        
        multiplier = 2 / (period + 1)
        ema = sum(values[:period]) / period  # Start with SMA
        
        for value in values[period:]:
            ema = (value - ema) * multiplier + ema
        
        return ema
    
    async def _analyze_and_generate(self, symbol: str, klines: List[Kline]):
        """Run full analysis and attempt signal generation using Scalper strategy."""
        
        logger.debug(f"{symbol}: Running Scalper analysis...")
        
        # Get recent trades for order flow (kept for interface compatibility)
        trades = binance_ws.get_recent_trades(symbol, 500)
        
        # Use EMA 5-8-13 Scalping Strategy
        scalper_signal = scalper_generator.generate_signal(klines, trades, symbol)
        
        if scalper_signal:
            await self._handle_new_signal(symbol, scalper_signal)
    
    async def _handle_new_signal(
        self,
        symbol: str,
        signal: ScalperSignal
    ):
        """Process and store a new signal."""
        
        if self.active_signals[symbol]:
            # For now, don't generate conflicting signals
            logger.debug(f"{symbol}: Signal skipped - active signal exists")
            return

        # -------------------------------------------------------------------------
        # MULTI-TIMEFRAME CHECK: Validate signal against MTF confluence
        # -------------------------------------------------------------------------
        confluence = mtf_analyzer.get_confluence(symbol)
        
        if not mtf_analyzer.is_trade_aligned(symbol, signal.direction):
            logger.info(
                f"{symbol}: ❌ MTF REJECTED {signal.direction} - "
                f"confluence={confluence.confluence_score:.2f} | {confluence.reasoning}"
            )
            return
        
        logger.info(
            f"{symbol}: ✅ MTF ALIGNED {signal.direction} - "
            f"confluence={confluence.confluence_score:.2f} | "
            f"{confluence.aligned_timeframes}/{confluence.total_timeframes} TFs aligned"
        )

        # -------------------------------------------------------------------------
        # ML GATEKEEPER: Trade approver must approve before proceeding
        # -------------------------------------------------------------------------
        from ml.trade_approver import trade_approver, TradeContext
        from analysis.market_state import market_state_analyzer
        
        # Build context for ML approval
        try:
            # Get recent klines for market context
            klines = binance_ws.get_klines(symbol, settings.primary_timeframe, 50)
            market_analysis = market_state_analyzer.analyze(klines) if klines else None
            
            context = TradeContext(
                symbol=symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                original_confidence=signal.confidence,
                market_state=market_analysis.state.value if market_analysis else "unknown",
                market_state_confidence=market_analysis.confidence if market_analysis else 0.0,
                cvd_trend="up" if signal.stoch_k > 50 else "down",
                aggression_strength=signal.confidence,
                stoch_k=signal.stoch_k,
                ema_aligned=signal.ema_aligned,
                risk_reward=signal.risk_reward
            )
        except Exception as e:
            logger.warning(f"{symbol}: Failed to build ML context: {e}")
            # Use minimal context
            context = TradeContext(
                symbol=symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                original_confidence=signal.confidence
            )
        
        # Get ML approval decision
        approval = await trade_approver.approve_trade(signal=signal, context=context)
        
        if not approval.approved:
            if settings.ml_gatekeeper_enabled:
                logger.info(f"{symbol}: ❌ ML REJECTED - {approval.reason}")
                logger.debug(f"{symbol}: Model scores: {approval.model_scores}")
                return
            else:
                logger.warning(f"{symbol}: ⚠️ ML REJECTED ({approval.reason}) but Gatekeeper DISABLED - PROCEEDING")
        
        logger.info(f"{symbol}: ✅ ML APPROVED - score={approval.score:.2f} | {approval.reason}")
        logger.debug(f"{symbol}: Model scores: {approval.model_scores}")
        
        # Update signal confidence with ML-adjusted value
        signal.confidence = approval.adjusted_confidence
        # -------------------------------------------------------------------------
        
        # Store in memory
        self.active_signals[symbol].append(signal)
        self.signal_history[symbol].append(signal)
        
        # Store in database
        signal_type = "SCALPER"
        
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
                market_state=f"StochK={signal.stoch_k:.1f}_EMA={signal.ema_aligned}",
                lvn_price=signal.ema13,
                poc_price=signal.take_profit,
                cvd_value=signal.stoch_k,
                aggression_score=signal.confidence
            )
            logger.info(f"{symbol}: Scalper Signal saved to database (ID: {signal_id})")
            
            # Record for ML pattern learning
            try:
                online_trainer.record_lvn_touch(
                    symbol=symbol,
                    lvn_price=signal.ema13,
                    touch_price=signal.entry_price,
                    direction=signal.direction,
                    context={
                        'volume_ratio': 1.0,
                        'cvd_value': signal.stoch_k,
                        'order_flow_imbalance': signal.confidence,
                        'distance_to_poc': abs(signal.entry_price - signal.take_profit) / signal.entry_price,
                        'market_state': 'SCALPER',
                        'momentum': signal.stoch_k,
                        'volatility': signal.risk_percent / 100,
                        'stoch_k': signal.stoch_k,
                        'ema_aligned': signal.ema_aligned
                    },
                    reaction='pending'
                )
                logger.debug(f"{symbol}: Pattern recorded for ML learning")
            except Exception as e:
                logger.error(f"Failed to record pattern: {e}")
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
            scalper_signal="NONE",  # Updated when signal generated
            scalper_stoch=50.0,
            scalper_trend=market_analysis.state.value,
            active_signals=[s.to_dict() for s in self.active_signals[symbol]]
        )
        
        self.analysis_snapshots[symbol] = snapshot
        
        # Record state observation for ML classifier (sample every ~10 updates to avoid over-sampling)
        if not hasattr(self, '_state_sample_counter'):
            self._state_sample_counter = {}
        counter = self._state_sample_counter.get(symbol, 0) + 1
        self._state_sample_counter[symbol] = counter
        
        if counter % 10 == 0:
            try:
                # Map market state to label: 0=balanced, 1=trending_up, 2=trending_down, 3=choppy
                state_labels = {'balanced': 0, 'trending_up': 1, 'trending_down': 2, 'choppy': 3, 'breakout_up': 1, 'breakout_down': 2}
                label = state_labels.get(market_analysis.state.value, 3)
                
                online_trainer.record_state_observation(
                    symbol=symbol,
                    features={
                        'atr_ratio': 0.01,  # Would calculate from actual data
                        'range_width': (vp.vah - vp.val) / klines[-1].close if vp.vah > vp.val else 0.02,
                        'momentum': market_analysis.momentum,
                        'volume_distribution': market_analysis.balance_score,
                        'poc_crosses': 0,  # Would track from klines
                        'vah_val_touches': 0,
                        'higher_highs': market_analysis.higher_highs,
                        'higher_lows': market_analysis.higher_lows,
                        'lower_highs': market_analysis.lower_highs,
                        'lower_lows': market_analysis.lower_lows
                    },
                    verified_label=label
                )
                logger.debug(f"{symbol}: State observation recorded for ML classifier")
            except Exception as e:
                logger.error(f"Failed to record state observation: {e}")
        
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
