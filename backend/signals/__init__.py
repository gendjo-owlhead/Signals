"""Signals package - Trading signal generation."""
from signals.trend_model import trend_model, TrendModelGenerator, TrendSignal
from signals.mean_reversion import mean_reversion_model, MeanReversionGenerator, ReversionSignal
from signals.signal_manager import signal_manager, SignalManager, SignalUpdate, AnalysisSnapshot

__all__ = [
    'trend_model', 'TrendModelGenerator', 'TrendSignal',
    'mean_reversion_model', 'MeanReversionGenerator', 'ReversionSignal',
    'signal_manager', 'SignalManager', 'SignalUpdate', 'AnalysisSnapshot'
]
