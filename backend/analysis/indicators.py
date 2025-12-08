"""
Technical Indicators Module.

Implements common technical analysis indicators used by trading strategies.
"""
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class MACDResult:
    """MACD indicator result."""
    macd_line: np.ndarray
    signal_line: np.ndarray
    histogram: np.ndarray


@dataclass
class StochRSIResult:
    """Stochastic RSI result."""
    k_line: np.ndarray
    d_line: np.ndarray


def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    ema = np.zeros_like(data)
    multiplier = 2 / (period + 1)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema


def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average."""
    sma = np.zeros_like(data)
    
    for i in range(len(data)):
        if i < period - 1:
            sma[i] = np.mean(data[:i+1])
        else:
            sma[i] = np.mean(data[i-period+1:i+1])
    
    return sma


def calculate_macd(
    closes: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> MACDResult:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        closes: Array of closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)
    
    Returns:
        MACDResult with macd_line, signal_line, and histogram
    """
    fast_ema = calculate_ema(closes, fast_period)
    slow_ema = calculate_ema(closes, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return MACDResult(
        macd_line=macd_line,
        signal_line=signal_line,
        histogram=histogram
    )


def calculate_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index."""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(len(closes))
    avg_loss = np.zeros(len(closes))
    
    # First average
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    # Subsequent averages using smoothing
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_stoch_rsi(
    closes: np.ndarray,
    k_period: int = 3,
    d_period: int = 3,
    rsi_period: int = 14,
    stoch_period: int = 14
) -> StochRSIResult:
    """
    Calculate Stochastic RSI.
    
    Args:
        closes: Array of closing prices
        k_period: %K smoothing period
        d_period: %D smoothing period
        rsi_period: RSI calculation period
        stoch_period: Stochastic calculation period
    
    Returns:
        StochRSIResult with k_line and d_line
    """
    rsi = calculate_rsi(closes, rsi_period)
    
    # Calculate Stochastic of RSI
    stoch_rsi = np.zeros_like(rsi)
    
    for i in range(stoch_period - 1, len(rsi)):
        rsi_window = rsi[i-stoch_period+1:i+1]
        rsi_min = np.min(rsi_window)
        rsi_max = np.max(rsi_window)
        
        if rsi_max - rsi_min != 0:
            stoch_rsi[i] = ((rsi[i] - rsi_min) / (rsi_max - rsi_min)) * 100
        else:
            stoch_rsi[i] = 50
    
    # Smooth with SMA for K and D lines
    k_line = calculate_sma(stoch_rsi, k_period)
    d_line = calculate_sma(k_line, d_period)
    
    return StochRSIResult(k_line=k_line, d_line=d_line)


def calculate_awesome_oscillator(
    highs: np.ndarray,
    lows: np.ndarray,
    fast_period: int = 5,
    slow_period: int = 34
) -> np.ndarray:
    """
    Calculate Awesome Oscillator (AO).
    
    AO = SMA(HL/2, 5) - SMA(HL/2, 34)
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        fast_period: Fast SMA period (default 5)
        slow_period: Slow SMA period (default 34)
    
    Returns:
        Array of AO values
    """
    hl2 = (highs + lows) / 2
    fast_sma = calculate_sma(hl2, fast_period)
    slow_sma = calculate_sma(hl2, slow_period)
    
    return fast_sma - slow_sma


def calculate_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        period: ATR period
    
    Returns:
        Array of ATR values
    """
    tr = np.zeros(len(highs))
    
    for i in range(1, len(highs)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr[i] = max(hl, hc, lc)
    
    tr[0] = highs[0] - lows[0]
    
    # Use EMA for ATR smoothing
    atr = calculate_ema(tr, period)
    
    return atr


def calculate_volume_sma(volumes: np.ndarray, period: int = 30) -> np.ndarray:
    """Calculate Volume Simple Moving Average."""
    return calculate_sma(volumes, period)


def detect_crossover(series: np.ndarray, threshold: float = 0) -> np.ndarray:
    """Detect when series crosses above threshold."""
    crosses = np.zeros(len(series), dtype=bool)
    
    for i in range(1, len(series)):
        if series[i-1] <= threshold and series[i] > threshold:
            crosses[i] = True
    
    return crosses


def detect_crossunder(series: np.ndarray, threshold: float = 0) -> np.ndarray:
    """Detect when series crosses below threshold."""
    crosses = np.zeros(len(series), dtype=bool)
    
    for i in range(1, len(series)):
        if series[i-1] >= threshold and series[i] < threshold:
            crosses[i] = True
    
    return crosses
