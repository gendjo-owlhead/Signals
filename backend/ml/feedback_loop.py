"""
Feedback Loop - Continuous learning from trade outcomes.

Tracks trade results and updates model weights based on performance.
This enables the ML system to improve over time by learning which
models are most reliable under different market conditions.
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from collections import deque
from loguru import logger

from config import settings


@dataclass
class TradeOutcome:
    """Record of a completed trade."""
    signal_id: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    outcome: str  # 'win' or 'loss'
    reason: str  # 'tp_hit', 'sl_hit', 'manual_close'
    model_scores: Dict[str, float]  # Score each model gave at approval time
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    
    def to_dict(self) -> dict:
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': round(self.pnl, 4),
            'pnl_percent': round(self.pnl_percent, 4),
            'outcome': self.outcome,
            'reason': self.reason,
            'model_scores': {k: round(v, 4) for k, v in self.model_scores.items()},
            'timestamp': self.timestamp
        }


@dataclass
class ModelPerformance:
    """Performance metrics for a single model."""
    model_name: str
    correct_predictions: int = 0
    total_predictions: int = 0
    total_pnl: float = 0.0
    avg_score_on_wins: float = 0.0
    avg_score_on_losses: float = 0.0
    
    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'accuracy': round(self.accuracy, 4),
            'correct_predictions': self.correct_predictions,
            'total_predictions': self.total_predictions,
            'total_pnl': round(self.total_pnl, 4),
            'avg_score_on_wins': round(self.avg_score_on_wins, 4),
            'avg_score_on_losses': round(self.avg_score_on_losses, 4)
        }


class FeedbackLoop:
    """
    Continuous learning system that tracks trade outcomes.
    
    Responsibilities:
    - Record trade outcomes with model scores
    - Track per-model accuracy and contribution
    - Suggest weight adjustments based on recent performance
    - Provide metrics for monitoring
    """
    
    def __init__(self):
        self._state_file = Path(settings.ml_model_path) / "feedback_loop_state.json"
        
        # Outcome history
        self.outcomes: deque = deque(maxlen=500)
        
        # Model performance tracking
        self.model_names = ['signal_accuracy', 'state_classifier', 'lvn_patterns', 'freqai']
        self.performance: Dict[str, ModelPerformance] = {
            name: ModelPerformance(model_name=name)
            for name in self.model_names
        }
        
        # Rebalance configuration
        self.rebalance_window = 20  # Trades to consider for rebalancing
        self.rebalance_threshold = 0.15  # Min accuracy difference to trigger
        self.weight_adjustment_rate = 0.05  # Max weight change per rebalance
        
        self._load_state()
        logger.info("FeedbackLoop initialized")
    
    def record_outcome(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str,
        model_scores: Dict[str, float]
    ):
        """
        Record a trade outcome for learning.
        
        Called when a position is closed.
        """
        pnl_percent = (exit_price - entry_price) / entry_price * 100
        if direction == 'SHORT':
            pnl_percent = -pnl_percent
        
        outcome_type = 'win' if pnl > 0 else 'loss'
        
        outcome = TradeOutcome(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            outcome=outcome_type,
            reason=reason,
            model_scores=model_scores
        )
        
        self.outcomes.append(outcome)
        
        # Update model performance
        self._update_performance(outcome)
        
        # Notify trade approver
        try:
            from ml.trade_approver import trade_approver
            trade_approver.record_outcome(model_scores, outcome_type, pnl)
        except Exception as e:
            logger.warning(f"Failed to notify trade approver: {e}")
        
        self._save_state()
        
        logger.info(
            f"FeedbackLoop: Recorded {outcome_type} | PnL: {pnl:.4f} | "
            f"Scores: {model_scores}"
        )
    
    def _update_performance(self, outcome: TradeOutcome):
        """Update model performance metrics based on outcome."""
        is_win = outcome.outcome == 'win'
        
        for model_name, score in outcome.model_scores.items():
            if model_name not in self.performance:
                self.performance[model_name] = ModelPerformance(model_name=model_name)
            
            perf = self.performance[model_name]
            perf.total_predictions += 1
            perf.total_pnl += outcome.pnl
            
            # Model is "correct" if it predicted well:
            # - High score (>0.5) and win
            # - Low score (<=0.5) and loss (correctly skeptical)
            model_predicted_success = score > 0.5
            model_correct = (model_predicted_success and is_win) or \
                           (not model_predicted_success and not is_win)
            
            if model_correct:
                perf.correct_predictions += 1
            
            # Track average scores by outcome
            if is_win:
                # Running average for wins
                n = sum(1 for o in self.outcomes if o.outcome == 'win')
                if n > 0:
                    old_avg = perf.avg_score_on_wins
                    perf.avg_score_on_wins = old_avg + (score - old_avg) / n
            else:
                # Running average for losses
                n = sum(1 for o in self.outcomes if o.outcome == 'loss')
                if n > 0:
                    old_avg = perf.avg_score_on_losses
                    perf.avg_score_on_losses = old_avg + (score - old_avg) / n
    
    def should_rebalance(self) -> bool:
        """Check if model weights should be rebalanced."""
        recent = list(self.outcomes)[-self.rebalance_window:]
        if len(recent) < self.rebalance_window:
            return False
        
        # Check if any model's recent performance differs significantly
        recent_accuracy = self._calculate_recent_accuracy(recent)
        
        accuracies = list(recent_accuracy.values())
        if not accuracies:
            return False
        
        # Trigger if spread between best and worst is large
        return max(accuracies) - min(accuracies) > self.rebalance_threshold
    
    def _calculate_recent_accuracy(self, recent_outcomes: List[TradeOutcome]) -> Dict[str, float]:
        """Calculate accuracy for each model on recent trades."""
        accuracy = {}
        
        for model_name in self.model_names:
            correct = 0
            total = 0
            
            for outcome in recent_outcomes:
                if model_name in outcome.model_scores:
                    score = outcome.model_scores[model_name]
                    is_win = outcome.outcome == 'win'
                    
                    predicted_success = score > 0.5
                    if (predicted_success and is_win) or (not predicted_success and not is_win):
                        correct += 1
                    total += 1
            
            accuracy[model_name] = correct / total if total > 0 else 0.5
        
        return accuracy
    
    def suggest_weight_adjustments(self) -> Dict[str, float]:
        """
        Calculate suggested weight adjustments based on recent performance.
        
        Returns a dict of model_name -> adjustment_delta.
        Positive delta = increase weight, negative = decrease.
        """
        recent = list(self.outcomes)[-self.rebalance_window:]
        if len(recent) < 10:
            return {}
        
        recent_accuracy = self._calculate_recent_accuracy(recent)
        
        if not recent_accuracy:
            return {}
        
        # Calculate deviation from mean accuracy
        mean_accuracy = sum(recent_accuracy.values()) / len(recent_accuracy)
        
        adjustments = {}
        for model_name, accuracy in recent_accuracy.items():
            deviation = accuracy - mean_accuracy
            
            # Scale adjustment by deviation, capped by rate
            adjustment = deviation * self.weight_adjustment_rate
            adjustment = max(-self.weight_adjustment_rate, min(self.weight_adjustment_rate, adjustment))
            
            adjustments[model_name] = adjustment
        
        return adjustments
    
    def get_performance_summary(self) -> Dict[str, Dict]:
        """Get performance summary for all models."""
        return {
            name: perf.to_dict()
            for name, perf in self.performance.items()
        }
    
    def get_recent_outcomes(self, limit: int = 20) -> List[Dict]:
        """Get recent trade outcomes."""
        recent = list(self.outcomes)[-limit:]
        return [o.to_dict() for o in recent]
    
    def get_stats(self) -> Dict:
        """Get overall feedback loop statistics."""
        outcomes_list = list(self.outcomes)
        
        if not outcomes_list:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0
            }
        
        wins = sum(1 for o in outcomes_list if o.outcome == 'win')
        total_pnl = sum(o.pnl for o in outcomes_list)
        
        return {
            'total_trades': len(outcomes_list),
            'win_rate': wins / len(outcomes_list),
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(outcomes_list),
            'recent_window': self.rebalance_window,
            'should_rebalance': self.should_rebalance()
        }
    
    def _save_state(self):
        """Persist state to disk."""
        try:
            state = {
                'outcomes': [o.to_dict() for o in self.outcomes],
                'performance': {
                    name: {
                        'correct_predictions': perf.correct_predictions,
                        'total_predictions': perf.total_predictions,
                        'total_pnl': perf.total_pnl,
                        'avg_score_on_wins': perf.avg_score_on_wins,
                        'avg_score_on_losses': perf.avg_score_on_losses
                    }
                    for name, perf in self.performance.items()
                }
            }
            
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save FeedbackLoop state: {e}")
    
    def _load_state(self):
        """Load state from disk."""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore outcomes
                if 'outcomes' in state:
                    for o_dict in state['outcomes']:
                        outcome = TradeOutcome(
                            signal_id=o_dict['signal_id'],
                            symbol=o_dict['symbol'],
                            direction=o_dict['direction'],
                            entry_price=o_dict['entry_price'],
                            exit_price=o_dict['exit_price'],
                            pnl=o_dict['pnl'],
                            pnl_percent=o_dict['pnl_percent'],
                            outcome=o_dict['outcome'],
                            reason=o_dict['reason'],
                            model_scores=o_dict['model_scores'],
                            timestamp=o_dict['timestamp']
                        )
                        self.outcomes.append(outcome)
                
                # Restore performance
                if 'performance' in state:
                    for name, perf_dict in state['performance'].items():
                        if name in self.performance:
                            self.performance[name].correct_predictions = perf_dict['correct_predictions']
                            self.performance[name].total_predictions = perf_dict['total_predictions']
                            self.performance[name].total_pnl = perf_dict['total_pnl']
                            self.performance[name].avg_score_on_wins = perf_dict.get('avg_score_on_wins', 0.0)
                            self.performance[name].avg_score_on_losses = perf_dict.get('avg_score_on_losses', 0.0)
                
                logger.info(f"Loaded FeedbackLoop state: {len(self.outcomes)} outcomes")
        except Exception as e:
            logger.warning(f"Failed to load FeedbackLoop state: {e}")


# Global instance
feedback_loop = FeedbackLoop()
