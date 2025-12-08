"""
Binance WebSocket data feeds for real-time market data.
Handles klines, trades, and order book depth streams.
"""
import asyncio
import json
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger
import websockets
from collections import deque

from config import settings, get_ws_url


@dataclass
class Trade:
    """Individual trade from the aggTrades stream."""
    trade_id: int
    price: float
    quantity: float
    timestamp: int
    is_buyer_maker: bool  # True = sell aggressor, False = buy aggressor
    
    @property
    def side(self) -> str:
        """Buy or Sell based on aggressor."""
        return "SELL" if self.is_buyer_maker else "BUY"
    
    @property
    def value(self) -> float:
        """Trade value in quote currency."""
        return self.price * self.quantity


@dataclass
class Kline:
    """Candlestick/OHLCV data."""
    symbol: str
    interval: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trades: int
    taker_buy_volume: float
    taker_buy_quote_volume: float
    is_closed: bool


@dataclass
class OrderBookLevel:
    """Single price level in order book."""
    price: float
    quantity: float


@dataclass
class OrderBook:
    """Order book snapshot."""
    symbol: str
    timestamp: int
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0
    
    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


class BinanceWebSocket:
    """
    Real-time data feeds from Binance WebSocket.
    Streams: aggTrades, klines, depth.
    """
    
    def __init__(self):
        self.ws_url = get_ws_url()
        self.connections: Dict[str, Any] = {}
        self.running = False
        
        # Data storage - circular buffers for memory efficiency
        self.trades: Dict[str, deque] = {}  # symbol -> trades
        self.klines: Dict[str, Dict[str, deque]] = {}  # symbol -> timeframe -> klines
        self.order_books: Dict[str, OrderBook] = {}  # symbol -> latest order book
        
        # Callbacks for real-time processing
        self.trade_callbacks: List[Callable] = []
        self.kline_callbacks: List[Callable] = []
        self.orderbook_callbacks: List[Callable] = []
        
        # Initialize storage for configured symbols
        for symbol in settings.trading_pairs:
            self.trades[symbol] = deque(maxlen=10000)  # Keep last 10k trades
            self.klines[symbol] = {}
            for tf in settings.timeframes:
                self.klines[symbol][tf] = deque(maxlen=500)
    
    def on_trade(self, callback: Callable[[str, Trade], None]):
        """Register callback for new trades."""
        self.trade_callbacks.append(callback)
    
    def on_kline(self, callback: Callable[[Kline], None]):
        """Register callback for kline updates."""
        self.kline_callbacks.append(callback)
    
    def on_orderbook(self, callback: Callable[[OrderBook], None]):
        """Register callback for order book updates."""
        self.orderbook_callbacks.append(callback)
    
    async def start(self):
        """Start all WebSocket streams."""
        self.running = True
        
        # Seed historical data
        logger.info("Seeding historical data...")
        from trading.binance_trader import binance_trader
        
        try:
            for symbol in settings.trading_pairs:
                for tf in settings.timeframes:
                    history = await binance_trader.get_historical_klines(symbol, tf, limit=100)
                    if history:
                        # Convert to Kline objects
                        klines = []
                        for k in history:
                            klines.append(Kline(
                                symbol=symbol,
                                interval=tf,
                                open_time=k['t'],
                                open=float(k['o']),
                                high=float(k['h']),
                                low=float(k['l']),
                                close=float(k['c']),
                                volume=float(k['v']),
                                close_time=k['T'],
                                quote_volume=float(k['q']),
                                trades=k['n'],
                                taker_buy_volume=float(k['V']),
                                taker_buy_quote_volume=float(k['Q']),
                                is_closed=True
                            ))
                        
                        # Populate buffer
                        self.klines[symbol][tf].extend(klines)
                        logger.info(f"Seeded {len(klines)} historical candles for {symbol} {tf}")
                        
        except Exception as e:
            logger.error(f"Failed to seed historical data: {e}")
            
        tasks = []
        for symbol in settings.trading_pairs:
            # Aggregate trades stream for order flow
            tasks.append(self._stream_trades(symbol))
            
            # Kline streams for each timeframe
            for tf in settings.timeframes:
                tasks.append(self._stream_klines(symbol, tf))
            
            # Order book depth stream
            tasks.append(self._stream_depth(symbol))
        
        logger.info(f"Starting WebSocket streams for {settings.trading_pairs}")
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop all WebSocket connections."""
        self.running = False
        # Create a copy of the values to avoid dictionary changed size during iteration
        connections_to_close = list(self.connections.values())
        for ws in connections_to_close:
            try:
                await ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
        self.connections.clear()
        logger.info("WebSocket streams stopped")
    
    async def _stream_trades(self, symbol: str):
        """Stream aggregate trades for order flow analysis."""
        stream_name = f"{symbol.lower()}@aggTrade"
        url = f"{self.ws_url}/{stream_name}"
        
        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    self.connections[stream_name] = ws
                    logger.info(f"Connected to {stream_name}")
                    
                    async for message in ws:
                        if not self.running:
                            break
                        
                        data = json.loads(message)
                        trade = Trade(
                            trade_id=data['a'],
                            price=float(data['p']),
                            quantity=float(data['q']),
                            timestamp=data['T'],
                            is_buyer_maker=data['m']
                        )
                        
                        # Store trade
                        self.trades[symbol].append(trade)
                        
                        # Notify callbacks
                        for callback in self.trade_callbacks:
                            try:
                                await callback(symbol, trade) if asyncio.iscoroutinefunction(callback) else callback(symbol, trade)
                            except Exception as e:
                                logger.error(f"Trade callback error: {e}")
                        
            except Exception as e:
                if self.running:
                    logger.error(f"Trade stream error for {symbol}: {e}")
                    await asyncio.sleep(5)  # Reconnect delay
    
    async def _stream_klines(self, symbol: str, interval: str):
        """Stream kline/candlestick data."""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        url = f"{self.ws_url}/{stream_name}"
        
        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    self.connections[stream_name] = ws
                    logger.info(f"Connected to {stream_name}")
                    
                    async for message in ws:
                        if not self.running:
                            break
                        
                        data = json.loads(message)
                        k = data['k']
                        
                        kline = Kline(
                            symbol=symbol,
                            interval=interval,
                            open_time=k['t'],
                            open=float(k['o']),
                            high=float(k['h']),
                            low=float(k['l']),
                            close=float(k['c']),
                            volume=float(k['v']),
                            close_time=k['T'],
                            quote_volume=float(k['q']),
                            trades=k['n'],
                            taker_buy_volume=float(k['V']),
                            taker_buy_quote_volume=float(k['Q']),
                            is_closed=k['x']
                        )
                        
                        # Update or append kline
                        kline_buffer = self.klines[symbol][interval]
                        if kline_buffer and kline_buffer[-1].open_time == kline.open_time:
                            kline_buffer[-1] = kline  # Update current
                        else:
                            kline_buffer.append(kline)  # New candle
                        
                        # Notify callbacks
                        for callback in self.kline_callbacks:
                            try:
                                await callback(kline) if asyncio.iscoroutinefunction(callback) else callback(kline)
                            except Exception as e:
                                logger.error(f"Kline callback error: {e}")
                        
            except Exception as e:
                if self.running:
                    logger.error(f"Kline stream error for {symbol} {interval}: {e}")
                    await asyncio.sleep(5)
    
    async def _stream_depth(self, symbol: str, levels: int = 20):
        """Stream order book depth."""
        stream_name = f"{symbol.lower()}@depth{levels}@100ms"
        url = f"{self.ws_url}/{stream_name}"
        
        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    self.connections[stream_name] = ws
                    logger.info(f"Connected to {stream_name}")
                    
                    async for message in ws:
                        if not self.running:
                            break
                        
                        data = json.loads(message)
                        
                        # Skip messages without order book data (status/connection messages)
                        if 'bids' not in data or 'asks' not in data:
                            continue
                        
                        order_book = OrderBook(
                            symbol=symbol,
                            timestamp=data.get('E', int(datetime.now().timestamp() * 1000)),
                            bids=[OrderBookLevel(float(p), float(q)) for p, q in data['bids']],
                            asks=[OrderBookLevel(float(p), float(q)) for p, q in data['asks']]
                        )
                        
                        self.order_books[symbol] = order_book
                        
                        # Notify callbacks
                        for callback in self.orderbook_callbacks:
                            try:
                                await callback(order_book) if asyncio.iscoroutinefunction(callback) else callback(order_book)
                            except Exception as e:
                                logger.error(f"OrderBook callback error: {e}")
                        
            except Exception as e:
                if self.running:
                    logger.error(f"Depth stream error for {symbol}: {e}")
                    await asyncio.sleep(5)
    
    def get_recent_trades(self, symbol: str, count: int = 100) -> List[Trade]:
        """Get recent trades for a symbol."""
        trades = list(self.trades.get(symbol, []))
        return trades[-count:] if len(trades) > count else trades
    
    def get_klines(self, symbol: str, interval: str, count: int = 100) -> List[Kline]:
        """Get recent klines for a symbol and interval."""
        klines = list(self.klines.get(symbol, {}).get(interval, []))
        return klines[-count:] if len(klines) > count else klines
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get latest order book for a symbol."""
        return self.order_books.get(symbol)


# Global WebSocket instance
binance_ws = BinanceWebSocket()
