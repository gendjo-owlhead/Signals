import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.signal_manager import SignalManager, SignalUpdate, AnalysisSnapshot
from signals.scalper_strategy import ScalperSignal

@pytest.fixture
def mock_ws():
    """Mock binance_ws."""
    with patch('signals.signal_manager.binance_ws') as mock:
        yield mock

@pytest.fixture
def mock_storage():
    """Mock storage."""
    with patch('signals.signal_manager.storage') as mock:
        mock.save_signal = AsyncMock(return_value=123)
        yield mock

@pytest.fixture
def mock_trainer():
    """Mock online_trainer."""
    with patch('signals.signal_manager.online_trainer') as mock:
        yield mock

@pytest.fixture
def mock_scalper():
    """Mock scalper_generator."""
    with patch('signals.signal_manager.scalper_generator') as mock:
        yield mock

@pytest.fixture
def mock_order_executor():
    """Mock order_executor."""
    with patch('signals.signal_manager.order_executor', create=True) as mock:
        mock.execute_signal = AsyncMock()
        result = MagicMock()
        result.success = True
        result.position.id = "pos_123"
        mock.execute_signal.return_value = result
        yield mock

class TestSignalManager:
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test concurrent signal manager initialization."""
        manager = SignalManager()
        assert manager.running is False
        assert isinstance(manager.active_signals, dict)
        assert isinstance(manager.signal_history, dict)
    
    @pytest.mark.asyncio
    async def test_broadcast_update(self):
        """Test update broadcasting to callbacks."""
        manager = SignalManager()
        
        # Mock callback
        callback = AsyncMock()
        manager.on_update(callback)
        
        update = SignalUpdate(
            type="test",
            timestamp=1234567890,
            symbol="BTCUSDT",
            data={}
        )
        
        await manager._broadcast_update(update)
        callback.assert_called_once_with(update)
    
    @pytest.mark.asyncio
    async def test_handle_new_signal(self, mock_storage, mock_trainer, mock_order_executor):
        """Test processing of a new signal."""
        manager = SignalManager()
        
        # Test signal
        signal = ScalperSignal(
            timestamp=1234567890,
            symbol="BTCUSDT",
            timeframe="1m",
            direction="LONG",
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,
            atr_value=100.0,
            confidence=0.8,
            ema5=50000, ema8=49900, ema13=49800,
            ema_aligned=True, ema_crossover=True,
            stoch_k=30.0, stoch_d=30.0,
            risk_reward=2.0,
            risk_percent=1.0
        )
        
        # Patch config to enable trading locally
        with patch('signals.signal_manager.settings') as mock_settings:
            mock_settings.trading_enabled = True
            mock_settings.trading_pairs = ["BTCUSDT"]
            mock_settings.primary_timeframe = "1m"
            
            # Mock the trade approver to return an approved result
            mock_approval = MagicMock()
            mock_approval.approved = True
            mock_approval.score = 0.8
            mock_approval.adjusted_confidence = 0.85
            mock_approval.reason = "Test approval"
            mock_approval.model_scores = {'signal_accuracy': 0.9}
            
            with patch('ml.trade_approver.trade_approver') as mock_approver, \
                 patch('trading.order_executor.OrderExecutor.execute_signal', new_callable=AsyncMock) as mock_exec:
                
                mock_approver.approve_trade = AsyncMock(return_value=mock_approval)
                
                result = MagicMock()
                result.success = True
                result.position.id = "pos_123"
                mock_exec.return_value = result
                
                await manager._handle_new_signal("BTCUSDT", signal)
                
                # Verify storage
                mock_storage.save_signal.assert_awaited_once()
                
                # Verify trainer
                mock_trainer.record_lvn_touch.assert_called_once()
                
                # Verify execution
                mock_exec.assert_awaited_once_with(signal)
                
                # Verify internal state
                assert len(manager.active_signals["BTCUSDT"]) == 1
                assert manager.active_signals["BTCUSDT"][0] == signal

    @pytest.mark.asyncio
    async def test_update_snapshot(self, mock_ws):
        """Test analysis snapshot generation."""
        manager = SignalManager()
            
        with patch('signals.signal_manager.volume_profile_calculator') as mock_vp, \
             patch('signals.signal_manager.market_state_analyzer') as mock_ms, \
             patch('analysis.order_flow.order_flow_analyzer') as mock_of:
            
            # Setup VP return
            vp_res = MagicMock()
            vp_res.poc_price = 50000
            vp_res.vah = 51000
            vp_res.val = 49000
            vp_res.lvn_zones = []
            mock_vp.calculate_from_klines.return_value = vp_res
            
            # Setup Market State return
            ms_res = MagicMock()
            ms_res.state.value = "balanced"
            ms_res.confidence = 0.8
            ms_res.is_balanced = True
            mock_ms.analyze.return_value = ms_res
            
            # Mock order flow
            mock_of.get_cvd_pressure.return_value = {'cvd': 100, 'trend': 'up'}
            mock_of.analyze_aggression.side_effect = Exception("Should handle missing trades path")

            # Mock klines and trades interactions
            mock_ws.get_recent_trades.return_value = []
            
            # Mock Klines for input
            klines = [MagicMock()]
            klines[0].close = 50000.0
            klines[0].close_time = 1000
            
            await manager._update_snapshot("BTCUSDT", klines)
            
            snapshot = manager.get_snapshot("BTCUSDT")
            assert snapshot is not None
            assert snapshot.symbol == "BTCUSDT"
            assert snapshot.current_price == 50000.0
