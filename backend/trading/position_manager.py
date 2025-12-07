"""
Position Manager - Track and manage open trading positions.
Handles position lifecycle, P&L calculation, and persistence.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from loguru import logger

from config import settings


@dataclass
class ManagedPosition:
    """A position opened by the bot."""
    id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    quantity: float
    entry_price: float
    entry_time: str
    stop_loss: float
    take_profit: float
    
    # Order IDs from exchange
    entry_order_id: str
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    
    # Signal info
    signal_type: str = ""  # "TREND" or "MEAN_REVERSION"
    signal_confidence: float = 0.0
    
    # Status
    status: str = "OPEN"  # OPEN, CLOSED_TP, CLOSED_SL, CLOSED_MANUAL
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    realized_pnl: Optional[float] = None
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity
    
    def to_dict(self) -> dict:
        return asdict(self)


class PositionManager:
    """
    Manage bot-opened positions.
    
    Responsibilities:
    - Track open positions
    - Persist position state
    - Calculate P&L
    - Record closed trades
    """
    
    def __init__(self):
        self.positions: Dict[str, ManagedPosition] = {}  # id -> position
        self.trade_history: List[dict] = []
        
        # Persistence paths
        self._positions_file = Path(settings.ml_model_path) / "positions.json"
        self._history_file = Path(settings.ml_model_path) / "trade_history.json"
        
        self._load_state()
    
    def _load_state(self):
        """Load positions and history from disk."""
        try:
            if self._positions_file.exists():
                with open(self._positions_file) as f:
                    data = json.load(f)
                    for pos_data in data:
                        pos = ManagedPosition(**pos_data)
                        if pos.status == "OPEN":
                            self.positions[pos.id] = pos
                logger.info(f"Loaded {len(self.positions)} open positions")
            
            if self._history_file.exists():
                with open(self._history_file) as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} historical trades")
                
        except Exception as e:
            logger.error(f"Failed to load position state: {e}")
    
    def _save_state(self):
        """Save positions and history to disk."""
        try:
            self._positions_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save open positions
            positions_data = [pos.to_dict() for pos in self.positions.values()]
            with open(self._positions_file, 'w') as f:
                json.dump(positions_data, f, indent=2)
            
            # Save history (keep last 500)
            with open(self._history_file, 'w') as f:
                json.dump(self.trade_history[-500:], f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save position state: {e}")
    
    def add_position(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        entry_order_id: str,
        sl_order_id: Optional[str] = None,
        tp_order_id: Optional[str] = None,
        signal_type: str = "",
        signal_confidence: float = 0.0
    ) -> ManagedPosition:
        """Add a new position."""
        position_id = f"{symbol}_{int(datetime.now().timestamp() * 1000)}"
        
        position = ManagedPosition(
            id=position_id,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now().isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_order_id=entry_order_id,
            sl_order_id=sl_order_id,
            tp_order_id=tp_order_id,
            signal_type=signal_type,
            signal_confidence=signal_confidence
        )
        
        self.positions[position_id] = position
        self._save_state()
        
        logger.info(
            f"Position opened: {position_id} | {direction} {quantity} {symbol} @ {entry_price:.2f}"
        )
        
        return position
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "MANUAL"
    ) -> Optional[float]:
        """
        Close a position and calculate realized P&L.
        Returns the realized P&L.
        """
        if position_id not in self.positions:
            logger.warning(f"Position not found: {position_id}")
            return None
        
        position = self.positions[position_id]
        
        # Calculate P&L
        pnl = position.calculate_pnl(exit_price)
        
        # Update position
        position.status = f"CLOSED_{reason}"
        position.exit_price = exit_price
        position.exit_time = datetime.now().isoformat()
        position.realized_pnl = pnl
        
        # Move to history
        self.trade_history.append(position.to_dict())
        del self.positions[position_id]
        
        self._save_state()
        
        logger.info(
            f"Position closed: {position_id} | "
            f"Exit: {exit_price:.2f} | P&L: ${pnl:.2f} | Reason: {reason}"
        )
        
        return pnl
    
    def get_position(self, position_id: str) -> Optional[ManagedPosition]:
        """Get a specific position."""
        return self.positions.get(position_id)
    
    def get_position_by_symbol(self, symbol: str) -> Optional[ManagedPosition]:
        """Get open position for a symbol (if any)."""
        for pos in self.positions.values():
            if pos.symbol == symbol:
                return pos
        return None
    
    def get_all_positions(self) -> List[ManagedPosition]:
        """Get all open positions."""
        return list(self.positions.values())
    
    def get_positions_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)
    
    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol."""
        return any(pos.symbol == symbol for pos in self.positions.values())
    
    def get_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate total unrealized P&L given current prices."""
        total = 0.0
        for pos in self.positions.values():
            if pos.symbol in prices:
                total += pos.calculate_pnl(prices[pos.symbol])
        return total
    
    def get_realized_pnl(self, days: int = 1) -> float:
        """Get realized P&L for recent trades."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        
        total = 0.0
        for trade in self.trade_history:
            try:
                trade_time = datetime.fromisoformat(trade['exit_time'])
                if trade_time > cutoff:
                    total += trade.get('realized_pnl', 0)
            except (KeyError, ValueError):
                continue
        
        return total
    
    def get_stats(self) -> dict:
        """Get position statistics."""
        wins = sum(1 for t in self.trade_history if t.get('realized_pnl', 0) > 0)
        losses = sum(1 for t in self.trade_history if t.get('realized_pnl', 0) < 0)
        total_trades = len(self.trade_history)
        
        return {
            'open_positions': len(self.positions),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
            'realized_pnl_24h': self.get_realized_pnl(1),
            'realized_pnl_7d': self.get_realized_pnl(7)
        }
    
    def sync_with_exchange(self, exchange_positions: List[dict]):
        """
        Sync local state with exchange positions.
        Called on startup to handle positions opened externally.
        """
        exchange_symbols = {p['symbol'] for p in exchange_positions}
        local_symbols = {p.symbol for p in self.positions.values()}
        
        # Find orphaned local positions (closed on exchange)
        orphaned = local_symbols - exchange_symbols
        for symbol in orphaned:
            pos = self.get_position_by_symbol(symbol)
            if pos:
                logger.warning(f"Position {pos.id} not found on exchange, marking as closed")
                # We don't know the exit price, estimate based on TP/SL midpoint
                estimated_exit = (pos.take_profit + pos.stop_loss) / 2
                self.close_position(pos.id, estimated_exit, "SYNC")
        
        logger.info(f"Position sync complete. {len(self.positions)} open positions")


# Global instance
position_manager = PositionManager()
