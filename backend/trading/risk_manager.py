"""
Risk Manager - Safety controls for automated trading.
Enforces limits on positions, daily losses, and provides kill switch.
"""
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger

from config import settings


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: str
    starting_balance: float
    realized_pnl: float = 0.0
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    
    @property
    def loss_percent(self) -> float:
        """Calculate daily loss percentage."""
        if self.starting_balance <= 0:
            return 0.0
        return (self.realized_pnl / self.starting_balance) * 100 if self.realized_pnl < 0 else 0.0


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    max_position_size_usdt: float = 100.0
    risk_per_trade_pct: float = 1.0
    max_concurrent_positions: int = 3
    daily_loss_limit_pct: float = 5.0
    min_signal_confidence: float = 0.7
    cooldown_after_losses: int = 3  # Pause after N consecutive losses
    cooldown_minutes: int = 30


class RiskManager:
    """
    Trading risk controls and safety features.
    
    Responsibilities:
    - Enforce daily loss limits
    - Limit concurrent positions
    - Validate trade size
    - Manage cooldowns after losses
    - Provide emergency kill switch
    """
    
    def __init__(self):
        self.is_enabled = True  # Kill switch
        self.limits = RiskLimits(
            max_position_size_usdt=settings.max_position_size_usdt,
            risk_per_trade_pct=settings.risk_per_trade_pct,
            max_concurrent_positions=settings.max_concurrent_positions,
            daily_loss_limit_pct=settings.daily_loss_limit_pct,
            min_signal_confidence=settings.min_signal_confidence
        )
        
        self.daily_stats = DailyStats(
            date=str(date.today()),
            starting_balance=0.0
        )
        
        self.consecutive_losses = 0
        self.cooldown_until: Optional[datetime] = None
        self.current_positions_count = 0
        
        # Persistence
        self._stats_file = Path(settings.ml_model_path) / "risk_stats.json"
        self._load_stats()
    
    def _load_stats(self):
        """Load daily stats from file."""
        try:
            if self._stats_file.exists():
                with open(self._stats_file) as f:
                    data = json.load(f)
                    if data.get('date') == str(date.today()):
                        self.daily_stats = DailyStats(**data)
                        logger.info(f"Loaded daily stats: P&L={self.daily_stats.realized_pnl:.2f}")
        except Exception as e:
            logger.warning(f"Could not load risk stats: {e}")
    
    def _save_stats(self):
        """Save daily stats to file."""
        try:
            self._stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._stats_file, 'w') as f:
                json.dump({
                    'date': self.daily_stats.date,
                    'starting_balance': self.daily_stats.starting_balance,
                    'realized_pnl': self.daily_stats.realized_pnl,
                    'trades_count': self.daily_stats.trades_count,
                    'wins': self.daily_stats.wins,
                    'losses': self.daily_stats.losses
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save risk stats: {e}")
    
    def set_starting_balance(self, balance: float):
        """Set starting balance for the day."""
        today = str(date.today())
        
        if self.daily_stats.date != today:
            # New day, reset stats
            self.daily_stats = DailyStats(
                date=today,
                starting_balance=balance
            )
            self.consecutive_losses = 0
            self.cooldown_until = None
            logger.info(f"New trading day. Starting balance: ${balance:.2f}")
        elif self.daily_stats.starting_balance == 0:
            self.daily_stats.starting_balance = balance
        
        self._save_stats()
    
    def kill_switch(self, enable: bool = False):
        """Emergency stop for all trading."""
        self.is_enabled = enable
        status = "ENABLED" if enable else "DISABLED"
        logger.warning(f"Trading kill switch: {status}")
    
    def can_trade(self, confidence: float, current_positions: int) -> tuple[bool, str]:
        """
        Check if trading is allowed.
        Returns (allowed, reason).
        """
        # Kill switch
        if not self.is_enabled:
            return False, "Trading disabled (kill switch active)"
        
        # Daily loss limit
        if abs(self.daily_stats.loss_percent) >= self.limits.daily_loss_limit_pct:
            return False, f"Daily loss limit reached ({self.daily_stats.loss_percent:.1f}%)"
        
        # Max positions
        if current_positions >= self.limits.max_concurrent_positions:
            return False, f"Max positions reached ({current_positions}/{self.limits.max_concurrent_positions})"
        
        # Confidence threshold
        if confidence < self.limits.min_signal_confidence:
            return False, f"Signal confidence too low ({confidence:.2f} < {self.limits.min_signal_confidence})"
        
        # Cooldown check
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).seconds // 60
            return False, f"Cooldown active ({remaining} minutes remaining)"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate position size based on risk percentage.
        
        Uses fixed-risk position sizing:
        Position Size = (Account * Risk%) / (Entry - SL)
        """
        risk_amount = account_balance * (self.limits.risk_per_trade_pct / 100)
        stop_distance = abs(entry_price - stop_loss_price)
        
        if stop_distance == 0:
            logger.warning("Stop distance is 0, using minimum position")
            return self.limits.max_position_size_usdt / entry_price
        
        # Position size in base currency
        position_size = risk_amount / stop_distance
        
        # Apply max limit
        position_value = position_size * entry_price
        if position_value > self.limits.max_position_size_usdt:
            position_size = self.limits.max_position_size_usdt / entry_price
            logger.info(f"Position capped at max size: ${self.limits.max_position_size_usdt}")
        
        return position_size
    
    def record_trade_result(self, pnl: float, is_win: bool):
        """Record a trade result for risk tracking."""
        self.daily_stats.trades_count += 1
        self.daily_stats.realized_pnl += pnl
        
        if is_win:
            self.daily_stats.wins += 1
            self.consecutive_losses = 0
            self.cooldown_until = None
        else:
            self.daily_stats.losses += 1
            self.consecutive_losses += 1
            
            # Check for cooldown trigger
            if self.consecutive_losses >= self.limits.cooldown_after_losses:
                from datetime import timedelta
                self.cooldown_until = datetime.now() + timedelta(minutes=self.limits.cooldown_minutes)
                logger.warning(
                    f"Entering cooldown after {self.consecutive_losses} consecutive losses. "
                    f"Resuming at {self.cooldown_until.strftime('%H:%M')}"
                )
        
        self._save_stats()
        
        logger.info(
            f"Trade recorded: P&L=${pnl:.2f} | "
            f"Daily: {self.daily_stats.wins}W/{self.daily_stats.losses}L | "
            f"Total P&L: ${self.daily_stats.realized_pnl:.2f}"
        )
    
    def get_status(self) -> dict:
        """Get current risk manager status."""
        return {
            'is_enabled': self.is_enabled,
            'daily_pnl': self.daily_stats.realized_pnl,
            'daily_pnl_percent': self.daily_stats.loss_percent,
            'trades_today': self.daily_stats.trades_count,
            'wins': self.daily_stats.wins,
            'losses': self.daily_stats.losses,
            'consecutive_losses': self.consecutive_losses,
            'cooldown_active': self.cooldown_until is not None and datetime.now() < self.cooldown_until,
            'limits': {
                'max_position_size': self.limits.max_position_size_usdt,
                'risk_per_trade': self.limits.risk_per_trade_pct,
                'max_positions': self.limits.max_concurrent_positions,
                'daily_loss_limit': self.limits.daily_loss_limit_pct,
                'min_confidence': self.limits.min_signal_confidence
            }
        }


# Global instance
risk_manager = RiskManager()
