"""
Order Executor - Converts trading signals into executed orders.
Main entry point for trade execution.
"""
import asyncio
from typing import Optional, Union
from dataclasses import dataclass
from loguru import logger

from config import settings
from signals.trend_model import TrendSignal
from signals.mean_reversion import ReversionSignal
from trading.binance_trader import binance_trader, OrderResult
from trading.position_manager import position_manager, ManagedPosition
from trading.risk_manager import risk_manager


@dataclass
class ExecutionResult:
    """Result of signal execution."""
    success: bool
    position: Optional[ManagedPosition] = None
    message: str = ""
    entry_result: Optional[OrderResult] = None
    tp_result: Optional[OrderResult] = None
    sl_result: Optional[OrderResult] = None
    
    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'message': self.message,
            'position_id': self.position.id if self.position else None,
            'entry': self.entry_result.to_dict() if self.entry_result else None,
            'tp': self.tp_result.to_dict() if self.tp_result else None,
            'sl': self.sl_result.to_dict() if self.sl_result else None
        }


class OrderExecutor:
    """
    Execute trading signals.
    
    Flow:
    1. Validate signal meets criteria
    2. Check risk manager approval
    3. Calculate position size
    4. Place bracket order (entry + TP + SL)
    5. Track in position manager
    """
    
    def __init__(self):
        self.is_running = False
        self._execution_lock = asyncio.Lock()
    
    async def start(self):
        """Start the order executor."""
        self.is_running = True
        
        # Get initial balance for risk manager
        balance = await binance_trader.get_account_balance()
        risk_manager.set_starting_balance(balance)
        
        # Sync positions with exchange
        exchange_positions = await binance_trader.get_open_positions()
        position_manager.sync_with_exchange([
            {'symbol': p.symbol, 'quantity': p.quantity}
            for p in exchange_positions
        ])
        
        logger.info(f"Order Executor started. Balance: ${balance:.2f}")
    
    async def stop(self):
        """Stop the order executor."""
        self.is_running = False
        risk_manager.kill_switch(False)
        logger.info("Order Executor stopped")
    
    def validate_signal(
        self,
        signal: Union[TrendSignal, ReversionSignal]
    ) -> tuple[bool, str]:
        """Validate a signal before execution."""
        
        # Check signal has required fields
        if not signal.entry_price or not signal.stop_loss or not signal.take_profit:
            return False, "Signal missing entry/SL/TP prices"
        
        # Check direction is valid
        if signal.direction not in ("LONG", "SHORT"):
            return False, f"Invalid direction: {signal.direction}"
        
        # Check risk/reward
        rr = getattr(signal, 'risk_reward', 0)
        if rr < 1.0:
            return False, f"Risk/reward too low: {rr:.2f}"
        
        # Check we don't already have position for this symbol
        if position_manager.has_position(signal.symbol):
            return False, f"Already have position for {signal.symbol}"
        
        return True, "OK"
    
    async def execute_signal(
        self,
        signal: Union[TrendSignal, ReversionSignal]
    ) -> ExecutionResult:
        """
        Execute a trading signal.
        
        This is the main entry point called by signal_manager.
        """
        async with self._execution_lock:
            return await self._do_execute(signal)
    
    async def _do_execute(
        self,
        signal: Union[TrendSignal, ReversionSignal]
    ) -> ExecutionResult:
        """Internal execution logic."""
        
        symbol = signal.symbol
        direction = signal.direction
        
        logger.info(f"Executing signal: {direction} {symbol}")
        
        # 1. Validate signal
        valid, reason = self.validate_signal(signal)
        if not valid:
            logger.warning(f"Signal validation failed: {reason}")
            return ExecutionResult(success=False, message=reason)
        
        # 2. Check risk manager
        current_positions = position_manager.get_positions_count()
        can_trade, risk_reason = risk_manager.can_trade(signal.confidence, current_positions)
        
        if not can_trade:
            logger.warning(f"Risk manager blocked trade: {risk_reason}")
            return ExecutionResult(success=False, message=risk_reason)
        
        # 3. Get account balance and calculate position size
        balance = await binance_trader.get_account_balance()
        
        if balance < 10:  # Minimum balance check
            return ExecutionResult(success=False, message="Insufficient balance")
        
        position_size = risk_manager.calculate_position_size(
            account_balance=balance,
            entry_price=signal.entry_price,
            stop_loss_price=signal.stop_loss
        )
        
        logger.info(f"Position size: {position_size:.4f} {symbol.replace('USDT', '')}")
        
        # 4. Place bracket order
        try:
            order_results = await binance_trader.place_bracket_order(
                symbol=symbol,
                direction=direction,
                quantity=position_size,
                take_profit_price=signal.take_profit,
                stop_loss_price=signal.stop_loss
            )
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return ExecutionResult(success=False, message=str(e))
        
        entry_result = order_results.get('entry')
        tp_result = order_results.get('tp')
        sl_result = order_results.get('sl')
        
        # Check if entry succeeded
        if not entry_result or not entry_result.success:
            error_msg = entry_result.error if entry_result else "Entry order failed"
            return ExecutionResult(
                success=False,
                message=error_msg,
                entry_result=entry_result
            )
        
        # 5. Track position
        signal_type = "TREND" if isinstance(signal, TrendSignal) else "MEAN_REVERSION"
        
        position = position_manager.add_position(
            symbol=symbol,
            direction=direction,
            quantity=entry_result.quantity,
            entry_price=entry_result.price or signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_order_id=entry_result.order_id,
            sl_order_id=sl_result.order_id if sl_result and sl_result.success else None,
            tp_order_id=tp_result.order_id if tp_result and tp_result.success else None,
            signal_type=signal_type,
            signal_confidence=signal.confidence
        )
        
        logger.info(
            f"Trade executed successfully! Position ID: {position.id} | "
            f"Entry: {entry_result.price:.2f} | TP: {signal.take_profit:.2f} | SL: {signal.stop_loss:.2f}"
        )
        
        return ExecutionResult(
            success=True,
            position=position,
            message="Trade executed successfully",
            entry_result=entry_result,
            tp_result=tp_result,
            sl_result=sl_result
        )
    
    async def close_position_manually(
        self,
        position_id: str,
        current_price: Optional[float] = None
    ) -> tuple[bool, str]:
        """Manually close a position."""
        position = position_manager.get_position(position_id)
        
        if not position:
            return False, "Position not found"
        
        # Cancel existing TP/SL orders
        if position.tp_order_id:
            await binance_trader.cancel_order(position.symbol, position.tp_order_id)
        if position.sl_order_id:
            await binance_trader.cancel_order(position.symbol, position.sl_order_id)
        
        # Close with market order
        result = await binance_trader.close_position(
            symbol=position.symbol,
            direction=position.direction,
            quantity=position.quantity
        )
        
        if not result.success:
            return False, f"Close failed: {result.error}"
        
        exit_price = result.price or current_price or position.entry_price
        pnl = position_manager.close_position(position_id, exit_price, "MANUAL")
        
        # Record in risk manager
        if pnl is not None:
            risk_manager.record_trade_result(pnl, pnl > 0)
        
        return True, f"Position closed. P&L: ${pnl:.2f}"
    
    def get_status(self) -> dict:
        """Get executor status."""
        return {
            'is_running': self.is_running,
            'trading_enabled': settings.trading_enabled,
            'positions': position_manager.get_stats(),
            'risk': risk_manager.get_status()
        }


# Global instance
order_executor = OrderExecutor()
