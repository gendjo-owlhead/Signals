"""Signals package - Trading signal generation with multi-timeframe analysis."""
from signals.scalper_strategy import scalper_generator, ScalperGenerator, ScalperSignal
from signals.signal_manager import signal_manager, SignalManager, SignalUpdate, AnalysisSnapshot
from signals.multi_timeframe_analyzer import mtf_analyzer, MultiTimeframeAnalyzer

__all__ = [
    'scalper_generator', 'ScalperGenerator', 'ScalperSignal',
    'signal_manager', 'SignalManager', 'SignalUpdate', 'AnalysisSnapshot',
    'mtf_analyzer', 'MultiTimeframeAnalyzer'
]
