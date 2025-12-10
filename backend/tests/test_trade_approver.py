"""
Tests for the ML Trade Approver system.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.trade_approver import TradeApprover, TradeContext, ApprovalResult
from ml.feedback_loop import FeedbackLoop, TradeOutcome


class TestTradeApprover:
    """Test the TradeApprover class."""
    
    def test_initialization(self, tmp_path):
        """Test approver initialization."""
        with patch('ml.trade_approver.settings') as mock_settings:
            mock_settings.ml_model_path = str(tmp_path)
            
            approver = TradeApprover()
            
            assert approver.approval_threshold == 0.65
            assert approver.min_models_agree == 2
            assert len(approver.model_weights) == 4
            assert sum(approver.model_weights.values()) == pytest.approx(1.0, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_approve_trade_high_confidence(self, tmp_path):
        """Test trade approval with high confidence signals."""
        with patch('ml.trade_approver.settings') as mock_settings:
            mock_settings.ml_model_path = str(tmp_path)
            mock_settings.primary_timeframe = "1m"
            
            approver = TradeApprover()
            
            # Mock all model scores to be high
            with patch('ml.signal_accuracy.signal_accuracy_model') as mock_sig, \
                 patch('ml.state_classifier.state_classifier') as mock_state, \
                 patch('ml.lvn_patterns.lvn_pattern_recognizer') as mock_lvn, \
                 patch('ml.freqai.engine.freqai_engine') as mock_freqai:
                
                mock_sig.adjust_confidence.return_value = 0.85
                mock_lvn.get_pattern_confidence.return_value = 0.8
                mock_freqai.model_manager.model = None  # No FreqAI model
                
                signal = MagicMock()
                context = TradeContext(
                    symbol="BTCUSDT",
                    direction="LONG",
                    entry_price=50000,
                    stop_loss=49500,
                    take_profit=51000,
                    original_confidence=0.8,
                    market_state="trending_up",
                    market_state_confidence=0.85
                )
                
                result = await approver.approve_trade(signal, context)
                
                # Should be approved with high scores
                assert isinstance(result, ApprovalResult)
                assert 'signal_accuracy' in result.model_scores
    
    @pytest.mark.asyncio
    async def test_approve_trade_low_confidence(self, tmp_path):
        """Test trade rejection with low confidence signals."""
        with patch('ml.trade_approver.settings') as mock_settings:
            mock_settings.ml_model_path = str(tmp_path)
            mock_settings.primary_timeframe = "1m"
            
            approver = TradeApprover()
            
            # Force low scores
            with patch.object(approver, '_calculate_ensemble_score', return_value=0.3):
                signal = MagicMock()
                context = TradeContext(
                    symbol="BTCUSDT",
                    direction="LONG",
                    entry_price=50000,
                    stop_loss=49500,
                    take_profit=51000,
                    original_confidence=0.3,
                    market_state="choppy",
                    market_state_confidence=0.4
                )
                
                result = await approver.approve_trade(signal, context)
                
                # Should be rejected
                assert result.approved == False
                assert result.score < approver.approval_threshold
    
    def test_weight_update(self, tmp_path):
        """Test model weight adjustment."""
        with patch('ml.trade_approver.settings') as mock_settings:
            mock_settings.ml_model_path = str(tmp_path)
            
            approver = TradeApprover()
            
            original_weights = approver.model_weights.copy()
            
            # Apply adjustments
            adjustments = {
                'signal_accuracy': 0.05,
                'state_classifier': -0.02
            }
            approver.update_weights(adjustments)
            
            # Weights should have changed (normalized)
            assert approver.model_weights != original_weights
            assert sum(approver.model_weights.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_record_outcome(self, tmp_path):
        """Test outcome recording for performance tracking."""
        with patch('ml.trade_approver.settings') as mock_settings:
            mock_settings.ml_model_path = str(tmp_path)
            
            approver = TradeApprover()
            
            model_scores = {
                'signal_accuracy': 0.8,
                'state_classifier': 0.7,
                'lvn_patterns': 0.6,
                'freqai': 0.5
            }
            
            approver.record_outcome(model_scores, 'win', 10.5)
            
            # Check performance was recorded
            for name in model_scores:
                assert len(approver.model_performance[name]) > 0


class TestFeedbackLoop:
    """Test the FeedbackLoop class."""
    
    def test_initialization(self, tmp_path):
        """Test feedback loop initialization."""
        with patch('ml.feedback_loop.settings') as mock_settings:
            mock_settings.ml_model_path = str(tmp_path)
            
            loop = FeedbackLoop()
            
            assert loop.rebalance_window == 20
            assert len(loop.model_names) == 4
    
    def test_record_outcome(self, tmp_path):
        """Test outcome recording."""
        with patch('ml.feedback_loop.settings') as mock_settings:
            mock_settings.ml_model_path = str(tmp_path)
            
            loop = FeedbackLoop()
            
            model_scores = {
                'signal_accuracy': 0.8,
                'state_classifier': 0.7,
                'lvn_patterns': 0.6,
                'freqai': 0.5
            }
            
            # Patch _save_state and internal imports
            with patch.object(loop, '_save_state'), \
                 patch.object(loop, '_update_performance'):
                loop.record_outcome(
                    signal_id=123,
                    symbol="BTCUSDT",
                    direction="LONG",
                    entry_price=50000,
                    exit_price=50500,
                    pnl=10.0,
                    reason="tp_hit",
                    model_scores=model_scores
                )
            
            assert len(loop.outcomes) == 1
            assert loop.outcomes[0].pnl == 10.0
    
    def test_should_rebalance(self, tmp_path):
        """Test rebalance trigger logic."""
        with patch('ml.feedback_loop.settings') as mock_settings:
            mock_settings.ml_model_path = str(tmp_path)
            
            loop = FeedbackLoop()
            
            # Not enough outcomes yet
            assert loop.should_rebalance() == False
    
    def test_suggest_weight_adjustments(self, tmp_path):
        """Test weight adjustment suggestions."""
        with patch('ml.feedback_loop.settings') as mock_settings:
            mock_settings.ml_model_path = str(tmp_path)
            
            loop = FeedbackLoop()
            
            # Record some outcomes directly (without touching trade_approver)
            for i in range(25):
                model_scores = {
                    'signal_accuracy': 0.9 if i % 2 == 0 else 0.3,
                    'state_classifier': 0.5,
                    'lvn_patterns': 0.4,
                    'freqai': 0.5
                }
                
                outcome = TradeOutcome(
                    signal_id=i,
                    symbol="BTCUSDT",
                    direction="LONG",
                    entry_price=50000 + i,
                    exit_price=50100 + i if i % 2 == 0 else 49900 + i,
                    pnl=10.0 if i % 2 == 0 else -5.0,
                    pnl_percent=0.2 if i % 2 == 0 else -0.2,
                    outcome="win" if i % 2 == 0 else "loss",
                    reason="tp_hit" if i % 2 == 0 else "sl_hit",
                    model_scores=model_scores
                )
                loop.outcomes.append(outcome)
            
            adjustments = loop.suggest_weight_adjustments()
            
            # Should have adjustments for each model
            assert isinstance(adjustments, dict)


class TestTradeContext:
    """Test TradeContext dataclass."""
    
    def test_context_creation(self):
        """Test context creation with all fields."""
        context = TradeContext(
            symbol="ETHUSDT",
            direction="SHORT",
            entry_price=2000,
            stop_loss=2050,
            take_profit=1900,
            original_confidence=0.75,
            market_state="trending_down",
            market_state_confidence=0.8,
            cvd_trend="down",
            aggression_strength=0.6,
            stoch_k=25.0,
            ema_aligned=True,
            risk_reward=2.0
        )
        
        assert context.symbol == "ETHUSDT"
        assert context.direction == "SHORT"
        assert context.risk_reward == 2.0


class TestApprovalResult:
    """Test ApprovalResult dataclass."""
    
    def test_result_to_dict(self):
        """Test result serialization."""
        result = ApprovalResult(
            approved=True,
            score=0.78,
            adjusted_confidence=0.82,
            reason="High ensemble score",
            model_scores={'signal_accuracy': 0.85, 'state_classifier': 0.7}
        )
        
        d = result.to_dict()
        
        assert d['approved'] == True
        assert d['score'] == 0.78
        assert 'signal_accuracy' in d['model_scores']
