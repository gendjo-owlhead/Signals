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


# ════════════════════════════════════════════════════════════════════════════
# APEX STRATEGY INDICATORS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DMIResult:
    """DMI/ADX indicator result."""
    plus_di: np.ndarray
    minus_di: np.ndarray
    adx: np.ndarray


@dataclass  
class WaveTrendResult:
    """WaveTrend oscillator result."""
    wt1: np.ndarray  # Main WaveTrend line
    wt2: np.ndarray  # Signal line


def calculate_rma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Running Moving Average (Wilder's Smoothing).
    
    This is the same as the RMA function in Pine Script.
    RMA = (prev_rma * (period - 1) + current_value) / period
    """
    rma = np.zeros_like(data, dtype=float)
    rma[0] = data[0]
    
    alpha = 1.0 / period
    for i in range(1, len(data)):
        rma[i] = alpha * data[i] + (1 - alpha) * rma[i-1]
    
    return rma


def calculate_hma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Hull Moving Average.
    
    HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    
    This reduces lag while maintaining smoothness.
    """
    def wma(arr: np.ndarray, length: int) -> np.ndarray:
        """Weighted Moving Average."""
        weights = np.arange(1, length + 1, dtype=float)
        result = np.zeros_like(arr, dtype=float)
        
        for i in range(len(arr)):
            if i < length - 1:
                w = weights[:i+1]
                result[i] = np.sum(arr[:i+1] * w) / np.sum(w)
            else:
                result[i] = np.sum(arr[i-length+1:i+1] * weights) / np.sum(weights)
        
        return result
    
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma_half = wma(data, half_period)
    wma_full = wma(data, period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma


def calculate_dmi(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> DMIResult:
    """
    Calculate Directional Movement Index (+DI, -DI) and ADX.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        period: ADX/DI smoothing period (default 14)
    
    Returns:
        DMIResult with plus_di, minus_di, and adx arrays
    """
    length = len(highs)
    
    # True Range
    tr = np.zeros(length)
    tr[0] = highs[0] - lows[0]
    for i in range(1, length):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Directional Movement
    plus_dm = np.zeros(length)
    minus_dm = np.zeros(length)
    
    for i in range(1, length):
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with RMA (Wilder's method)
    atr = calculate_rma(tr, period)
    smooth_plus_dm = calculate_rma(plus_dm, period)
    smooth_minus_dm = calculate_rma(minus_dm, period)
    
    # Calculate +DI and -DI
    plus_di = np.zeros(length)
    minus_di = np.zeros(length)
    
    for i in range(length):
        if atr[i] != 0:
            plus_di[i] = 100 * smooth_plus_dm[i] / atr[i]
            minus_di[i] = 100 * smooth_minus_dm[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(length)
    for i in range(length):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = calculate_rma(dx, period)
    
    return DMIResult(plus_di=plus_di, minus_di=minus_di, adx=adx)


def calculate_mfi(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    Calculate Money Flow Index (Volume-weighted RSI).
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        volumes: Array of volumes
        period: MFI period (default 14)
    
    Returns:
        Array of MFI values (0-100)
    """
    # Typical Price
    tp = (highs + lows + closes) / 3
    
    # Raw Money Flow
    raw_mf = tp * volumes
    
    # Positive and Negative Money Flow
    pos_mf = np.zeros(len(tp))
    neg_mf = np.zeros(len(tp))
    
    for i in range(1, len(tp)):
        if tp[i] > tp[i-1]:
            pos_mf[i] = raw_mf[i]
        elif tp[i] < tp[i-1]:
            neg_mf[i] = raw_mf[i]
    
    # Sum over period
    mfi = np.zeros(len(tp))
    
    for i in range(period, len(tp)):
        sum_pos = np.sum(pos_mf[i-period+1:i+1])
        sum_neg = np.sum(neg_mf[i-period+1:i+1])
        
        if sum_neg != 0:
            mf_ratio = sum_pos / sum_neg
            mfi[i] = 100 - (100 / (1 + mf_ratio))
        else:
            mfi[i] = 100
    
    return mfi


def calculate_wavetrend(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    channel_length: int = 10,
    avg_length: int = 21
) -> WaveTrendResult:
    """
    Calculate WaveTrend oscillator.
    
    This is an institutional-grade oscillator that detects market cycles.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        channel_length: Channel period (default 10)
        avg_length: Average period (default 21)
    
    Returns:
        WaveTrendResult with wt1 and wt2 lines
    """
    # HLC3 (typical price)
    hlc3 = (highs + lows + closes) / 3
    
    # ESA = EMA of HLC3
    esa = calculate_ema(hlc3, channel_length)
    
    # D = EMA of |HLC3 - ESA|
    d = calculate_ema(np.abs(hlc3 - esa), channel_length)
    
    # CI = (HLC3 - ESA) / (0.015 * D)
    ci = np.zeros_like(hlc3)
    for i in range(len(hlc3)):
        if d[i] != 0:
            ci[i] = (hlc3[i] - esa[i]) / (0.015 * d[i])
        else:
            ci[i] = 0
    
    # WT1 = EMA of CI
    wt1 = calculate_ema(ci, avg_length)
    
    # WT2 = SMA of WT1
    wt2 = calculate_sma(wt1, 4)
    
    return WaveTrendResult(wt1=wt1, wt2=wt2)


def detect_pivot_high(
    highs: np.ndarray,
    left_bars: int = 5,
    right_bars: int = 5
) -> np.ndarray:
    """
    Detect pivot high points.
    
    A pivot high is a bar whose high is higher than 'left_bars' bars 
    before and 'right_bars' bars after.
    
    Args:
        highs: Array of high prices
        left_bars: Bars to look back
        right_bars: Bars to look forward
    
    Returns:
        Array of pivot high values (nan where no pivot)
    """
    pivots = np.full(len(highs), np.nan)
    
    for i in range(left_bars, len(highs) - right_bars):
        is_pivot = True
        current = highs[i]
        
        # Check left side
        for j in range(1, left_bars + 1):
            if highs[i - j] >= current:
                is_pivot = False
                break
        
        if is_pivot:
            # Check right side
            for j in range(1, right_bars + 1):
                if highs[i + j] >= current:
                    is_pivot = False
                    break
        
        if is_pivot:
            pivots[i] = current
    
    return pivots


def detect_pivot_low(
    lows: np.ndarray,
    left_bars: int = 5,
    right_bars: int = 5
) -> np.ndarray:
    """
    Detect pivot low points.
    
    A pivot low is a bar whose low is lower than 'left_bars' bars 
    before and 'right_bars' bars after.
    
    Args:
        lows: Array of low prices
        left_bars: Bars to look back
        right_bars: Bars to look forward
    
    Returns:
        Array of pivot low values (nan where no pivot)
    """
    pivots = np.full(len(lows), np.nan)
    
    for i in range(left_bars, len(lows) - right_bars):
        is_pivot = True
        current = lows[i]
        
        # Check left side
        for j in range(1, left_bars + 1):
            if lows[i - j] <= current:
                is_pivot = False
                break
        
        if is_pivot:
            # Check right side
            for j in range(1, right_bars + 1):
                if lows[i + j] <= current:
                    is_pivot = False
                    break
        
        if is_pivot:
            pivots[i] = current
    
    return pivots


def get_ma(ma_type: str, data: np.ndarray, period: int) -> np.ndarray:
    """
    Get moving average based on type.
    
    Args:
        ma_type: Type of MA - 'SMA', 'EMA', 'HMA', or 'RMA'
        data: Input data array
        period: MA period
    
    Returns:
        Array of MA values
    """
    ma_type = ma_type.upper()
    
    if ma_type == "SMA":
        return calculate_sma(data, period)
    elif ma_type == "EMA":
        return calculate_ema(data, period)
    elif ma_type == "HMA":
        return calculate_hma(data, period)
    elif ma_type == "RMA":
        return calculate_rma(data, period)
    else:
        # Default to EMA
        return calculate_ema(data, period)


@dataclass
class SupertrendResult:
    """Supertrend indicator result."""
    upper_band: np.ndarray
    lower_band: np.ndarray
    trend: np.ndarray  # 1 = bullish, -1 = bearish
    supertrend: np.ndarray  # The actual trend line (follows price)


def calculate_supertrend(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0
) -> SupertrendResult:
    """
    Calculate Supertrend indicator.
    
    Supertrend = ATR-based trend following indicator that flips 
    between bullish and bearish based on price breaching the bands.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        period: ATR period (default 10)
        multiplier: ATR multiplier for bands (default 3.0)
    
    Returns:
        SupertrendResult with upper_band, lower_band, trend, supertrend
    """
    length = len(closes)
    
    # Calculate ATR
    atr = calculate_atr(highs, lows, closes, period)
    
    # Calculate basic upper and lower bands
    hl2 = (highs + lows) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    
    # Initialize arrays
    upper_band = np.zeros(length)
    lower_band = np.zeros(length)
    trend = np.zeros(length)
    supertrend = np.zeros(length)
    
    # First values
    upper_band[0] = basic_upper[0]
    lower_band[0] = basic_lower[0]
    trend[0] = 1  # Start bullish
    supertrend[0] = lower_band[0]
    
    for i in range(1, length):
        # Upper band - only decreases (in downtrend)
        if basic_upper[i] < upper_band[i-1] or closes[i-1] > upper_band[i-1]:
            upper_band[i] = basic_upper[i]
        else:
            upper_band[i] = upper_band[i-1]
        
        # Lower band - only increases (in uptrend)
        if basic_lower[i] > lower_band[i-1] or closes[i-1] < lower_band[i-1]:
            lower_band[i] = basic_lower[i]
        else:
            lower_band[i] = lower_band[i-1]
        
        # Determine trend
        if trend[i-1] == 1:  # Was bullish
            if closes[i] < lower_band[i]:
                trend[i] = -1  # Flip to bearish
            else:
                trend[i] = 1  # Stay bullish
        else:  # Was bearish
            if closes[i] > upper_band[i]:
                trend[i] = 1  # Flip to bullish
            else:
                trend[i] = -1  # Stay bearish
        
        # Supertrend line follows the appropriate band
        if trend[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    return SupertrendResult(
        upper_band=upper_band,
        lower_band=lower_band,
        trend=trend,
        supertrend=supertrend
    )
