"""
Trading module for automated order execution.
Converts signals into Binance Futures orders with TP/SL.
"""
from trading.binance_trader import binance_trader
from trading.order_executor import order_executor
from trading.position_manager import position_manager
from trading.risk_manager import risk_manager

__all__ = [
    'binance_trader',
    'order_executor', 
    'position_manager',
    'risk_manager'
]
