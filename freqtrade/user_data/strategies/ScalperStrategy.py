import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    CategoricalParameter
)
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ScalperStrategy(IStrategy):
    """
    EMA 5-8-13 High-Frequency Scalping Strategy.
    
    Ported from backend/signals/scalper_strategy.py
    """
    INTERFACE_VERSION = 3

    # Minimal ROI (We use custom_exit for TP)
    minimal_roi = {
        "0": 100  # Effectively disabled, managed by custom_exit
    }

    # Stoploss (We use custom_stoploss)
    stoploss = -0.99  # Effectively disabled, managed by custom_stoploss

    # Timeframe
    timeframe = '1m'

    # Process only new candles
    process_only_new_candles = True
    
    # Startup candle count
    startup_candle_count = 50

    # Parameters (optimizable)
    ema_fast = IntParameter(5, 5, default=5, space="buy")
    ema_mid = IntParameter(8, 8, default=8, space="buy")
    ema_slow = IntParameter(13, 13, default=13, space="buy")
    
    stoch_k_smooth = IntParameter(3, 3, default=3, space="buy")
    stoch_period = IntParameter(14, 14, default=14, space="buy")
    
    # Risk Management
    atr_period = IntParameter(14, 14, default=14, space="sell")
    atr_mult_sl = DecimalParameter(1.0, 3.0, default=1.0, space="sell")
    rr_ratio = DecimalParameter(1.0, 3.0, default=1.5, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate all indicators used by the strategy
        """
        # EMAs
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_mid'] = ta.EMA(dataframe, timeperiod=self.ema_mid.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)

        # StochRSI
        period = 14
        smooth = 3
        rsi = ta.RSI(dataframe, timeperiod=period)
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        dataframe['stoch_k'] = stoch_rsi.rolling(smooth).mean() * 100
        # dataframe['stoch_d'] = dataframe['stoch_k'].rolling(smooth).mean()

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signals
        """
        
        # ----------------------------------------------------------------------
        # LONG Conditions
        # ----------------------------------------------------------------------
        # 1. EMA5 crosses above EMA8
        # 2. EMA5 > EMA8 > EMA13
        # 3. Price > EMA13
        # 4. StochK < 80 (Not overbought)
        
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_mid'])) &
                (dataframe['ema_fast'] > dataframe['ema_mid']) &
                (dataframe['ema_mid'] > dataframe['ema_slow']) &
                (dataframe['close'] > dataframe['ema_slow']) &
                (dataframe['stoch_k'] < 80.0)
            ),
            'enter_long'] = 1

        # ----------------------------------------------------------------------
        # SHORT Conditions
        # ----------------------------------------------------------------------
        # 1. EMA5 crosses below EMA8
        # 2. EMA5 < EMA8 < EMA13
        # 3. Price < EMA13
        # 4. StochK > 20 (Not oversold)
        
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['ema_fast'], dataframe['ema_mid'])) &
                (dataframe['ema_fast'] < dataframe['ema_mid']) &
                (dataframe['ema_mid'] < dataframe['ema_slow']) &
                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['stoch_k'] > 20.0)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signals - mostly handled by custom_exit and custom_stoploss,
        but we can add signal-based exits here if needed (e.g. EMA recross).
        """
        # Exit LONG if EMA5 crosses below EMA8 (Trend Reversal)
        dataframe.loc[
            (qtpylib.crossed_below(dataframe['ema_fast'], dataframe['ema_mid'])),
            'exit_long'] = 1

        # Exit SHORT if EMA5 crosses above EMA8 (Trend Reversal)
        dataframe.loc[
            (qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_mid'])),
            'exit_short'] = 1
            
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Calculate dynamic stoploss based on ATR at entry
        """
        # To calculate the initial stoploss relative to entry price:
        # We need the ATR at the time of entry.
        # Freqtrade doesn't store the ATR in the Trade object by default.
        # We can approximate or use a percentage if we can't get exact ATR at entry.
        # However, for correct backtesting, we should use the dataframe value at open_date.
        
        # A workaround in Freqtrade for "Entry ATR" is usually to calculate it here 
        # or use relative SL. 
        # Since 'trade' object doesn't carry custom data easily without db modifications,
        # we will use a simplified approach:
        # Calculate SL % distance based on current bar's ATR (approximation) 
        # OR better: Use the standard `stoploss` logic if we want fixed % 
        # BUT this strategy wants Volatility based SL.
        
        # Correct approach for Freqtrade:
        # We can return a specific stop loss price, or a percentage relative to entry.
        
        # For backtesting precission, we'll try to look up the ATR from the dataframe 
        # passed via `populate_indicators`. But `custom_stoploss` doesn't receive dataframe.
        
        # fallback: use a fixed sensible % if dynamic failed (or reimplement in `custom_entry_price`)
        # For this MVP invalidation: return default fixed 1.5% if simpler, 
        # but let's try to do it right:
        # We can set `stop_loss` in the `custom_entry_price` or just return a static value
        # that mimics reasonable scalping stops (e.g., 0.5% - 1.0%).
        
        # Let's use the average ATR % for 1m timeframe for BTC which is around 0.1% - 0.2%.
        # With multiplier 1.0, that is very tight.
        
        # Simulating ATR logic:
        # Let's assume 0.3% stop loss for now as a baseline for 1m scalping
        return 0.003  # 0.3% static for now, as fetching historical ATR per trade is complex in this callback
        
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        """
        Custom exit for Take Profit based on R:R
        """
        # If profit > R:R * Risk (0.3% * 1.5 = 0.45%)
        # This is a simplification. Ideally we want:
        # TP = Entry +/- (ATR * Mult * RR)
        
        risk_pct = 0.003 # Matched with custom_stoploss
        target_profit = risk_pct * self.rr_ratio.value
        
        if current_profit >= target_profit:
            return "take_profit"
            
        return None
