"""Signals package - Trading signal generation."""
from signals.scalper_strategy import scalper_generator, ScalperGenerator, ScalperSignal
from signals.signal_manager import signal_manager, SignalManager, SignalUpdate, AnalysisSnapshot

__all__ = [
    'scalper_generator', 'ScalperGenerator', 'ScalperSignal',
    'signal_manager', 'SignalManager', 'SignalUpdate', 'AnalysisSnapshot'
]
