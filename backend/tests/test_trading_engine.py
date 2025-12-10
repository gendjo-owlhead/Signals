import pytest
import aiohttp
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.binance_trader import BinanceTrader, OrderResult

class TestBinanceTrader:
    
    @pytest.fixture
    async def trader(self):
        """Create trader instance with mocked session."""
        trader = BinanceTrader()
        trader.api_key = "test_key"
        trader.api_secret = "test_secret"
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.closed = False  # Important to prevent _get_session from overwriting it
        trader._session = mock_session
        
        yield trader
        
        await trader.close()

    @pytest.mark.asyncio
    async def test_get_account_balance(self, trader):
        """Test getting USDT balance."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.json.return_value = [
            {'asset': 'BTC', 'availableBalance': '0.1'},
            {'asset': 'USDT', 'availableBalance': '1000.0'}
        ]
        
        # Setup session.get context manager
        trader._session.get.return_value.__aenter__.return_value = mock_response
        
        balance = await trader.get_account_balance()
        assert balance == 1000.0
        
        # Verify URL (relative to base URL, but full URL is constructed in _request)
        # We can check specific call args if needed
        trader._session.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_open_positions(self, trader):
        """Test getting open positions."""
        mock_response = AsyncMock()
        mock_response.json.return_value = [
            {
                'symbol': 'BTCUSDT',
                'positionAmt': '0.5',
                'entryPrice': '50000',
                'unRealizedProfit': '100',
                'leverage': '10',
                'marginType': 'ISOLATED'
            },
            {
                'symbol': 'ETHUSDT',
                'positionAmt': '0.0', # Should be ignored
                'entryPrice': '0',
                'unRealizedProfit': '0',
                'leverage': '10',
                'marginType': 'ISOLATED'
            }
        ]
        trader._session.get.return_value.__aenter__.return_value = mock_response
        
        positions = await trader.get_open_positions()
        
        assert len(positions) == 1
        assert positions[0].symbol == 'BTCUSDT'
        assert positions[0].quantity == 0.5
        assert positions[0].side == 'LONG'

    @pytest.mark.asyncio
    async def test_place_market_order_success(self, trader):
        """Test successful market order."""
        # Mock exchange info first ensuring min qty is met
        with patch.object(trader, 'get_exchange_info') as mock_info:
            mock_info.return_value = {
                'price_precision': 2,
                'quantity_precision': 3,
                'min_qty': 0.001,
                'tick_size': 0.01
            }
            
            # Mock price check
            mock_price_response = AsyncMock()
            mock_price_response.json.return_value = {'price': '50000'}
            
            # Mock order response
            mock_order_response = AsyncMock()
            mock_order_response.json.return_value = {
                'orderId': 12345,
                'avgPrice': '50000',
                'status': 'FILLED'
            }
            
            # We have multiple calls. 
            # 1. Price check (GET)
            # 2. Order placement (POST)
            
            # side_effect for get/post
            # We can mock _request directly for simpler testing of logic
            with patch.object(trader, '_request') as mock_request:
                mock_request.side_effect = [
                    {'price': '50000'}, # Price check
                    {
                        'orderId': 12345, # Order result
                        'avgPrice': '50000',
                        'status': 'FILLED'
                    }
                ]
                
                result = await trader.place_market_order('BTCUSDT', 'BUY', 0.01)
                
                assert result.success is True
                assert result.order_id == '12345'
                assert result.quantity == 0.01

    @pytest.mark.asyncio
    async def test_place_bracket_order(self, trader):
        """Test full bracket order flow."""
        # We'll mock the individual place methods to verify orchestration
        with patch.object(trader, 'place_market_order') as mock_entry, \
             patch.object(trader, 'place_take_profit_order') as mock_tp, \
             patch.object(trader, 'place_stop_loss_order') as mock_sl:
            
            # Setup success returns
            mock_entry.return_value = OrderResult(True, '1', 'BTCUSDT', 'BUY', 1.0, 50000, 'FILLED')
            mock_tp.return_value = OrderResult(True, '2', 'BTCUSDT', 'SELL', 1.0, 51000, 'NEW')
            mock_sl.return_value = OrderResult(True, '3', 'BTCUSDT', 'SELL', 1.0, 49000, 'NEW')
            
            results = await trader.place_bracket_order('BTCUSDT', 'LONG', 1.0, 51000, 49000)
            
            assert results['entry'].success
            assert results['tp'].success
            assert results['sl'].success
            
            mock_entry.assert_awaited_once()
            mock_tp.assert_awaited_once()
            mock_sl.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_logic(self, trader):
        """Test that retries work for TP/SL."""
         # We need to simulate failure then success
        mock_fn = AsyncMock()
        mock_fn.side_effect = [
            OrderResult(False, error="Fail 1"),
            OrderResult(False, error="Fail 2"),
            OrderResult(True, order_id="Success")
        ]
        
        result = await trader._place_order_with_retry(mock_fn, "TEST", max_retries=3)
        
        assert result.success is True
        assert result.order_id == "Success"
        assert mock_fn.call_count == 3
