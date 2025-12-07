"""
Historical data fetcher for backtesting and initial data loading.
"""
import asyncio
from typing import List, Optional
from datetime import datetime, timedelta
import aiohttp
from loguru import logger
import pandas as pd

from config import settings, get_api_url
from data.binance_ws import Kline, Trade


class HistoricalDataFetcher:
    """Fetch historical data from Binance REST API."""
    
    def __init__(self, use_mainnet: bool = False):
        """
        Initialize historical data fetcher.
        
        Args:
            use_mainnet: If True, use mainnet for data (recommended for backtesting
                        since testnet has limited historical data)
        """
        if use_mainnet:
            # Use Spot mainnet for public data (no auth needed, more accessible)
            self.base_url = "https://api.binance.com"
        else:
            self.base_url = get_api_url()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Kline]:
        """
        Fetch historical klines from Binance.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Timeframe (1m, 5m, 15m, 1h, etc.)
            limit: Number of candles (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
        
        Returns:
            List of Kline objects
        """
        session = await self._get_session()
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        url = f"{self.base_url}/api/v3/klines"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Klines fetch error: {error}")
                    return []
                
                data = await response.json()
                
                klines = []
                for k in data:
                    kline = Kline(
                        symbol=symbol,
                        interval=interval,
                        open_time=k[0],
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5]),
                        close_time=k[6],
                        quote_volume=float(k[7]),
                        trades=k[8],
                        taker_buy_volume=float(k[9]),
                        taker_buy_quote_volume=float(k[10]),
                        is_closed=True
                    )
                    klines.append(kline)
                
                logger.info(f"Fetched {len(klines)} klines for {symbol} {interval}")
                return klines
                
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            return []
    
    async def fetch_recent_trades(
        self,
        symbol: str,
        limit: int = 1000
    ) -> List[Trade]:
        """
        Fetch recent trades from Binance.
        
        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)
        
        Returns:
            List of Trade objects
        """
        session = await self._get_session()
        
        url = f"{self.base_url}/api/v3/aggTrades"
        params = {
            "symbol": symbol,
            "limit": min(limit, 1000)
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Trades fetch error: {error}")
                    return []
                
                data = await response.json()
                
                trades = []
                for t in data:
                    trade = Trade(
                        trade_id=t['a'],
                        price=float(t['p']),
                        quantity=float(t['q']),
                        timestamp=t['T'],
                        is_buyer_maker=t['m']
                    )
                    trades.append(trade)
                
                logger.info(f"Fetched {len(trades)} trades for {symbol}")
                return trades
                
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []
    
    async def fetch_historical_trades(
        self,
        symbol: str,
        start_time: int,
        end_time: int
    ) -> List[Trade]:
        """
        Fetch historical trades in a time range.
        Uses pagination to get all trades.
        
        Args:
            symbol: Trading pair
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
        
        Returns:
            List of Trade objects
        """
        session = await self._get_session()
        
        all_trades = []
        current_start = start_time
        
        while current_start < end_time:
            url = f"{self.base_url}/api/v3/aggTrades"
            params = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000
            }
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        break
                    
                    data = await response.json()
                    if not data:
                        break
                    
                    for t in data:
                        trade = Trade(
                            trade_id=t['a'],
                            price=float(t['p']),
                            quantity=float(t['q']),
                            timestamp=t['T'],
                            is_buyer_maker=t['m']
                        )
                        all_trades.append(trade)
                    
                    # Move to next batch
                    current_start = data[-1]['T'] + 1
                    
                    # Rate limit protection
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error fetching historical trades: {e}")
                break
        
        logger.info(f"Fetched {len(all_trades)} historical trades for {symbol}")
        return all_trades
    
    def klines_to_dataframe(self, klines: List[Kline]) -> pd.DataFrame:
        """Convert klines to pandas DataFrame."""
        if not klines:
            return pd.DataFrame()
        
        data = {
            'timestamp': [k.open_time for k in klines],
            'open': [k.open for k in klines],
            'high': [k.high for k in klines],
            'low': [k.low for k in klines],
            'close': [k.close for k in klines],
            'volume': [k.volume for k in klines],
            'quote_volume': [k.quote_volume for k in klines],
            'trades': [k.trades for k in klines],
            'taker_buy_volume': [k.taker_buy_volume for k in klines]
        }
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        return df


# Global fetcher instance
historical_fetcher = HistoricalDataFetcher()
