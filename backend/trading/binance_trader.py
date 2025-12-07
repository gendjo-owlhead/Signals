"""
Binance Futures API wrapper for trade execution.
Handles order placement, position queries, and TP/SL management.
"""
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
import aiohttp
from loguru import logger

from config import settings, get_api_url


@dataclass
class OrderResult:
    """Result of an order placement."""
    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    status: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status,
            'error': self.error
        }


@dataclass
class Position:
    """Current open position."""
    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    margin_type: str


class BinanceTrader:
    """
    Low-level Binance Futures API wrapper.
    
    Handles:
    - Order placement (market, limit, stop-loss, take-profit)
    - Position queries
    - Account balance
    - Order cancellation
    """
    
    def __init__(self):
        self.api_key = settings.binance_api_key
        self.api_secret = settings.binance_api_secret
        self.base_url = get_api_url()
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Symbol precision cache
        self._precision_cache: Dict[str, dict] = {}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _sign(self, params: dict) -> str:
        """Generate HMAC SHA256 signature."""
        # Binance expects params in original order, NOT sorted
        query_string = '&'.join(f"{k}={v}" for k, v in params.items())
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        signed: bool = True
    ) -> dict:
        """Make authenticated request to Binance API."""
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        params = params or {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._sign(params)
        
        headers = {'X-MBX-APIKEY': self.api_key}
        
        try:
            if method == 'GET':
                async with session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            elif method == 'POST':
                # Binance requires POST params in URL query string
                async with session.post(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            elif method == 'DELETE':
                async with session.delete(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            else:
                async with session.request(method, url, params=params, headers=headers) as resp:
                    data = await resp.json()
            
            if 'code' in data and data['code'] < 0:
                logger.error(f"Binance API Error: {data}")
                return {'error': data.get('msg', 'Unknown error'), 'code': data['code']}
            
            return data
            
        except Exception as e:
            logger.error(f"Binance request failed: {e}")
            return {'error': str(e)}
    
    async def get_exchange_info(self, symbol: str) -> dict:
        """Get symbol precision info."""
        if symbol in self._precision_cache:
            return self._precision_cache[symbol]
        
        data = await self._request('GET', '/fapi/v1/exchangeInfo', signed=False)
        
        if 'symbols' in data:
            for sym in data['symbols']:
                if sym['symbol'] == symbol:
                    info = {
                        'price_precision': sym['pricePrecision'],
                        'quantity_precision': sym['quantityPrecision'],
                        'min_qty': float(next(
                            f['minQty'] for f in sym['filters'] 
                            if f['filterType'] == 'LOT_SIZE'
                        )),
                        'tick_size': float(next(
                            f['tickSize'] for f in sym['filters']
                            if f['filterType'] == 'PRICE_FILTER'
                        ))
                    }
                    self._precision_cache[symbol] = info
                    return info
        
        return {'price_precision': 2, 'quantity_precision': 3, 'min_qty': 0.001, 'tick_size': 0.01}
    
    def _round_quantity(self, quantity: float, precision: int) -> float:
        """Round quantity to exchange precision."""
        return round(quantity, precision)
    
    def _round_price(self, price: float, precision: int) -> float:
        """Round price to exchange precision."""
        return round(price, precision)
    
    async def get_account_balance(self) -> float:
        """Get USDT balance."""
        data = await self._request('GET', '/fapi/v2/balance')
        
        if 'error' in data:
            logger.error(f"Failed to get balance: {data['error']}")
            return 0.0
        
        for asset in data:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
        
        return 0.0
    
    async def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        data = await self._request('GET', '/fapi/v2/positionRisk')
        
        if 'error' in data:
            logger.error(f"Failed to get positions: {data['error']}")
            return []
        
        positions = []
        for pos in data:
            amt = float(pos['positionAmt'])
            if amt != 0:
                positions.append(Position(
                    symbol=pos['symbol'],
                    side="LONG" if amt > 0 else "SHORT",
                    quantity=abs(amt),
                    entry_price=float(pos['entryPrice']),
                    unrealized_pnl=float(pos['unRealizedProfit']),
                    leverage=int(pos['leverage']),
                    margin_type=pos['marginType']
                ))
        
        return positions
    
    async def place_market_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float
    ) -> OrderResult:
        """Place a market order."""
        info = await self.get_exchange_info(symbol)
        qty = self._round_quantity(quantity, info['quantity_precision'])
        
        if qty < info['min_qty']:
            return OrderResult(
                success=False,
                symbol=symbol,
                error=f"Quantity {qty} below minimum {info['min_qty']}"
            )
        
        # Get current price to check minimum notional
        price_data = await self._request('GET', '/fapi/v1/ticker/price', 
                                         {'symbol': symbol}, signed=False)
        if 'price' in price_data:
            current_price = float(price_data['price'])
            notional = qty * current_price
            min_notional = 100.0  # Binance Futures minimum
            
            if notional < min_notional:
                # Round up quantity to meet minimum
                min_qty_needed = min_notional / current_price
                qty = self._round_quantity(min_qty_needed + (10 ** -info['quantity_precision']), 
                                          info['quantity_precision'])
                logger.info(f"Quantity adjusted to {qty} to meet ${min_notional} minimum notional")
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': qty
        }
        
        logger.info(f"Placing market order: {side} {qty} {symbol}")
        
        data = await self._request('POST', '/fapi/v1/order', params)
        
        if 'error' in data:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                quantity=qty,
                error=data['error']
            )
        
        return OrderResult(
            success=True,
            order_id=str(data.get('orderId', '')),
            symbol=symbol,
            side=side,
            quantity=qty,
            price=float(data.get('avgPrice', 0)),
            status=data.get('status', 'UNKNOWN')
        )
    
    async def place_stop_loss_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        stop_price: float
    ) -> OrderResult:
        """Place a stop-loss order."""
        info = await self.get_exchange_info(symbol)
        qty = self._round_quantity(quantity, info['quantity_precision'])
        price = self._round_price(stop_price, info['price_precision'])
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'STOP_MARKET',
            'quantity': qty,
            'stopPrice': price,
            'closePosition': 'false',
            'workingType': 'MARK_PRICE'
        }
        
        logger.info(f"Placing SL order: {side} {qty} {symbol} @ {price}")
        
        data = await self._request('POST', '/fapi/v1/order', params)
        
        if 'error' in data:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                error=data['error']
            )
        
        return OrderResult(
            success=True,
            order_id=str(data.get('orderId', '')),
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            status=data.get('status', 'UNKNOWN')
        )
    
    async def place_take_profit_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        take_profit_price: float
    ) -> OrderResult:
        """Place a take-profit order."""
        info = await self.get_exchange_info(symbol)
        qty = self._round_quantity(quantity, info['quantity_precision'])
        price = self._round_price(take_profit_price, info['price_precision'])
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'TAKE_PROFIT_MARKET',
            'quantity': qty,
            'stopPrice': price,
            'closePosition': 'false',
            'workingType': 'MARK_PRICE'
        }
        
        logger.info(f"Placing TP order: {side} {qty} {symbol} @ {price}")
        
        data = await self._request('POST', '/fapi/v1/order', params)
        
        if 'error' in data:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side,
                error=data['error']
            )
        
        return OrderResult(
            success=True,
            order_id=str(data.get('orderId', '')),
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            status=data.get('status', 'UNKNOWN')
        )
    
    async def place_bracket_order(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float
    ) -> Dict[str, OrderResult]:
        """
        Place a market order with TP and SL.
        Returns dict with 'entry', 'tp', 'sl' results.
        """
        # Determine sides
        entry_side = "BUY" if direction == "LONG" else "SELL"
        exit_side = "SELL" if direction == "LONG" else "BUY"
        
        results = {}
        
        # 1. Place entry order
        entry_result = await self.place_market_order(symbol, entry_side, quantity)
        results['entry'] = entry_result
        
        if not entry_result.success:
            logger.error(f"Entry order failed: {entry_result.error}")
            return results
        
        logger.info(f"Entry filled: {entry_result.order_id}")
        
        # Use actual filled quantity for TP/SL
        filled_qty = entry_result.quantity
        
        # 2. Place take profit
        tp_result = await self.place_take_profit_order(
            symbol, exit_side, filled_qty, take_profit_price
        )
        results['tp'] = tp_result
        
        if not tp_result.success:
            logger.warning(f"TP order failed: {tp_result.error}")
        
        # 3. Place stop loss
        sl_result = await self.place_stop_loss_order(
            symbol, exit_side, filled_qty, stop_loss_price
        )
        results['sl'] = sl_result
        
        if not sl_result.success:
            logger.warning(f"SL order failed: {sl_result.error}")
        
        return results
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        data = await self._request('DELETE', '/fapi/v1/order', params)
        
        if 'error' in data:
            logger.error(f"Cancel failed: {data['error']}")
            return False
        
        return True
    
    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol."""
        params = {'symbol': symbol}
        
        data = await self._request('DELETE', '/fapi/v1/allOpenOrders', params)
        
        if 'error' in data:
            logger.error(f"Cancel all failed: {data['error']}")
            return False
        
        return True
    
    async def close_position(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        quantity: float
    ) -> OrderResult:
        """Close an open position with market order."""
        # To close: sell if long, buy if short
        side = "SELL" if direction == "LONG" else "BUY"
        return await self.place_market_order(symbol, side, quantity)


# Global instance
binance_trader = BinanceTrader()
