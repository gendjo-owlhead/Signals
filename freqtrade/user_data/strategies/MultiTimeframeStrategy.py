
import numpy as np
import talib.abstract as ta
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame
from functools import reduce
from freqtrade.persistence import Trade

class MultiTimeframeStrategy(IStrategy):
    """
    Multi-Timeframe Strategy
    
    Timeframes:
    - 5m: Base timeframe for entries (Trigger)
    - 15m: Setup timeframe (Pullback)
    - 1h: Trend timeframe (Direction)
    
    Logic:
    - Trend: 1h EMA 200
    - Setup: 15m Bollinger Bands (Pullback to lower BB in uptrend)
    - Trigger: 5m MACD Crossover
    """

    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    # We use a dynamic ROI based on time, but keep it tighter for scalping behavior
    minimal_roi = {
        "0": 0.05,      # Take profit at 5% immediately
        "10": 0.03,     # After 10 mins, take 3%
        "20": 0.015,    # After 20 mins, take 1.5%
        "40": 0.005     # After 40 mins, take 0.5%
    }

    # Optimal stoploss designed for the strategy.
    stoploss = -0.025  # 2.5% stop loss

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.005  # Lock in profits after 0.5%
    trailing_stop_positive_offset = 0.015 # Start trailing after 1.5% profit
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = '5m'

    # Run "populate_indicators" on the 1h timeframe
    informative_timeframe_1h = '1h'
    informative_timeframe_15m = '15m'

    def informative_pairs(self):
        # Define informative pairs for 1h and 15m
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe_1h) for pair in pairs]
        informative_pairs += [(pair, self.informative_timeframe_15m) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. Base Timeframe (5m) Indicators
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # 2. Informative Timeframe (15m)
        inf_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe_15m)
        
        # Bollinger Bands on 15m
        bollinger = ta.BBANDS(inf_15m, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        inf_15m['bb_lowerband'] = bollinger['lowerband']
        inf_15m['bb_middleband'] = bollinger['middleband']
        inf_15m['bb_upperband'] = bollinger['upperband']
        
        # RSI on 15m
        inf_15m['rsi'] = ta.RSI(inf_15m, timeperiod=14)

        # Merge 15m
        dataframe = merge_informative_pair(dataframe, inf_15m, self.timeframe, self.informative_timeframe_15m, ffill=True)

        # 3. Informative Timeframe (1h)
        inf_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe_1h)
        
        # EMA 200 on 1h
        inf_1h['ema_200'] = ta.EMA(inf_1h, timeperiod=200)
        
        # RSI on 1h
        inf_1h['rsi'] = ta.RSI(inf_1h, timeperiod=14)

        # Merge 1h
        dataframe = merge_informative_pair(dataframe, inf_1h, self.timeframe, self.informative_timeframe_1h, ffill=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry Logic
        """
        
        # Long Conditions
        dataframe.loc[
            (
                # 1. Trend: 1h Price > 1h EMA 200 (Uptrend)
                (dataframe[f'close_{self.informative_timeframe_1h}'] > dataframe[f'ema_200_{self.informative_timeframe_1h}']) &
                
                # 2. Setup: 15m Price dipped below/near Lower BB (Pullback)
                # We relax "below" to "close to" (within 1% of lower band) or actually below
                (dataframe[f'close_{self.informative_timeframe_15m}'] <= dataframe[f'bb_lowerband_{self.informative_timeframe_15m}'] * 1.005) &
                
                # 3. Momentum: 5m MACD crossover (MACD > Signal)
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1)) & # Crossover
                
                # 4. Filter: 1h RSI not overbought (>70)
                (dataframe[f'rsi_{self.informative_timeframe_1h}'] < 70) &
                
                # 5. Volume check
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        # Short Conditions
        dataframe.loc[
            (
                # 1. Trend: 1h Price < 1h EMA 200 (Downtrend)
                (dataframe[f'close_{self.informative_timeframe_1h}'] < dataframe[f'ema_200_{self.informative_timeframe_1h}']) &
                
                # 2. Setup: 15m Price spiked above/near Upper BB (Pullback)
                (dataframe[f'close_{self.informative_timeframe_15m}'] >= dataframe[f'bb_upperband_{self.informative_timeframe_15m}'] * 0.995) &
                
                # 3. Momentum: 5m MACD crossunder (MACD < Signal)
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1)) &
                
                # 4. Filter: 1h RSI not oversold (<30)
                (dataframe[f'rsi_{self.informative_timeframe_1h}'] > 30) &
                
                # 5. Volume check
                (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit Logic - Primarily handled by ROI/Stoploss/Trailing Stop, 
        but we can add custom exit signals here.
        """
        
        # Exit Long if 15m RSI becomes overbought or 5m MACD turns down strongly
        dataframe.loc[
            (
                (dataframe[f'rsi_{self.informative_timeframe_15m}'] > 75)
            ),
            'exit_long'] = 1
            
        # Exit Short if 15m RSI becomes oversold
        dataframe.loc[
            (
                (dataframe[f'rsi_{self.informative_timeframe_15m}'] < 25)
            ),
            'exit_short'] = 1

        return dataframe
