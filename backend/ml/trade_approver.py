"""
Trade Approver - ML Gatekeeper for Trade Decisions.

This is the primary decision-maker that determines whether a signal should
be executed as a trade. It aggregates multiple ML models in an ensemble
voting system with dynamic weight adjustment based on performance.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
from pathlib import Path
import json
from loguru import logger

from config import settings


@dataclass
class ApprovalResult:
    """Result of trade approval decision."""
    approved: bool
    score: float  # Ensemble confidence score [0, 1]
    adjusted_confidence: float  # Final confidence to use if approved
    reason: str
    model_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    
    def to_dict(self) -> dict:
        return {
            'approved': self.approved,
            'score': round(self.score, 4),
            'adjusted_confidence': round(self.adjusted_confidence, 4),
            'reason': self.reason,
            'model_scores': {k: round(v, 4) for k, v in self.model_scores.items()},
            'timestamp': self.timestamp
        }


@dataclass
class TradeContext:
    """Context information for trade approval."""
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    original_confidence: float
    
    # Market context
    market_state: str = "unknown"
    market_state_confidence: float = 0.0
    cvd_trend: str = "neutral"
    aggression_strength: float = 0.0
    
    # Technical indicators
    stoch_k: float = 50.0
    ema_aligned: bool = False
    risk_reward: float = 1.0


class TradeApprover:
    """
    Primary ML gatekeeper for trade decisions.
    
    Combines multiple ML models in an ensemble voting system:
    - Signal Accuracy Model: Historical pattern matching
    - State Classifier: Market regime detection
    - LVN Pattern Recognizer: Price level analysis
    - FreqAI Engine: Adaptive prediction
    
    Features:
    - Weighted ensemble voting
    - Dynamic weight adjustment based on performance
    - Explainable decisions with reasoning
    - Confidence calibration
    """
    
    def __init__(self):
        # Model weights (sum to 1.0)
        self.model_weights = {
            'signal_accuracy': 0.35,
            'state_classifier': 0.25,
            'lvn_patterns': 0.20,
            'freqai': 0.20
        }
        
        # Approval thresholds
        self.approval_threshold = 0.65  # Minimum ensemble score
        self.min_models_agree = 2       # Minimum models scoring > 0.5
        self.high_confidence_boost = 0.05  # Boost when all models agree
        
        # Performance tracking
        self.performance_window = 50  # Track last N decisions
        self.model_performance: Dict[str, deque] = {
            name: deque(maxlen=self.performance_window)
            for name in self.model_weights.keys()
        }
        
        # Decision history for API/debugging
        self.recent_decisions: deque = deque(maxlen=100)
        
        # Persistence
        self._state_file = Path(settings.ml_model_path) / "trade_approver_state.json"
        self._load_state()
        
        logger.info(f"TradeApprover initialized | threshold={self.approval_threshold} | weights={self.model_weights}")
    
    async def approve_trade(
        self,
        signal,  # ScalperSignal
        context: TradeContext
    ) -> ApprovalResult:
        """
        Main entry point - evaluate whether trade should be executed.
        
        Returns ApprovalResult with decision and reasoning.
        """
        model_scores = {}
        
        # 1. Signal Accuracy Model Score
        try:
            from ml.signal_accuracy import signal_accuracy_model
            
            features = {
                'market_state_conf': context.market_state_confidence,
                'aggression_strength': context.aggression_strength,
                'cvd_confirming': (context.cvd_trend == 'up' and context.direction == 'LONG') or 
                                  (context.cvd_trend == 'down' and context.direction == 'SHORT'),
                'risk_reward': context.risk_reward,
                'lvn_precision': 1.0  # Default
            }
            
            adjusted = signal_accuracy_model.adjust_confidence(
                context.original_confidence, 
                features
            )
            # Normalize to [0, 1]
            model_scores['signal_accuracy'] = min(1.0, max(0.0, adjusted))
        except Exception as e:
            logger.warning(f"Signal accuracy model error: {e}")
            model_scores['signal_accuracy'] = context.original_confidence
        
        # 2. State Classifier Score
        try:
            from ml.state_classifier import state_classifier
            
            # Check if current market state favors the trade direction
            state_score = self._evaluate_state_for_direction(
                context.market_state,
                context.market_state_confidence,
                context.direction
            )
            model_scores['state_classifier'] = state_score
        except Exception as e:
            logger.warning(f"State classifier error: {e}")
            model_scores['state_classifier'] = 0.5  # Neutral
        
        # 3. LVN Pattern Score
        try:
            from ml.lvn_patterns import lvn_pattern_recognizer
            
            # Get pattern recognition confidence
            lvn_score = lvn_pattern_recognizer.get_pattern_confidence(
                symbol=context.symbol,
                direction=context.direction,
                entry_price=context.entry_price
            )
            model_scores['lvn_patterns'] = lvn_score if lvn_score is not None else 0.5
        except Exception as e:
            logger.warning(f"LVN pattern recognizer error: {e}")
            model_scores['lvn_patterns'] = 0.5  # Neutral
        
        # 4. FreqAI Score
        try:
            from ml.freqai.engine import freqai_engine
            
            if freqai_engine.model_manager.model is not None:
                # FreqAI predicts price direction/magnitude
                # Convert to directional confidence
                freqai_score = await self._get_freqai_score(context)
                model_scores['freqai'] = freqai_score
            else:
                model_scores['freqai'] = 0.5  # No model yet
        except Exception as e:
            logger.warning(f"FreqAI error: {e}")
            model_scores['freqai'] = 0.5  # Neutral
        
        # Calculate ensemble score
        ensemble_score = self._calculate_ensemble_score(model_scores)
        
        # Count agreeing models (score > 0.5)
        models_agreeing = sum(1 for score in model_scores.values() if score > 0.5)
        
        # Make decision
        approved = False
        reasons = []
        
        if ensemble_score >= self.approval_threshold:
            if models_agreeing >= self.min_models_agree:
                approved = True
                reasons.append(f"Ensemble score {ensemble_score:.2f} ≥ {self.approval_threshold}")
                reasons.append(f"{models_agreeing}/{len(model_scores)} models agree")
            else:
                reasons.append(f"Only {models_agreeing}/{self.min_models_agree} required models agree")
        else:
            reasons.append(f"Ensemble score {ensemble_score:.2f} < {self.approval_threshold}")
        
        # Boost confidence if unanimous agreement
        adjusted_confidence = ensemble_score
        if approved and models_agreeing == len(model_scores):
            adjusted_confidence = min(1.0, ensemble_score + self.high_confidence_boost)
            reasons.append("Unanimous agreement boost applied")
        
        result = ApprovalResult(
            approved=approved,
            score=ensemble_score,
            adjusted_confidence=adjusted_confidence,
            reason=" | ".join(reasons),
            model_scores=model_scores
        )
        
        # Track decision
        self.recent_decisions.append({
            'symbol': context.symbol,
            'direction': context.direction,
            'result': result.to_dict()
        })
        
        return result
    
    def _calculate_ensemble_score(self, model_scores: Dict[str, float]) -> float:
        """Calculate weighted ensemble score."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for model_name, score in model_scores.items():
            weight = self.model_weights.get(model_name, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _evaluate_state_for_direction(
        self,
        market_state: str,
        confidence: float,
        direction: str
    ) -> float:
        """Evaluate if market state aligns with trade direction."""
        
        # State-direction alignment matrix
        alignment = {
            ('trending_up', 'LONG'): 0.9,
            ('trending_down', 'SHORT'): 0.9,
            ('breakout_up', 'LONG'): 0.85,
            ('breakout_down', 'SHORT'): 0.85,
            ('balanced', 'LONG'): 0.5,
            ('balanced', 'SHORT'): 0.5,
            ('choppy', 'LONG'): 0.3,
            ('choppy', 'SHORT'): 0.3,
            # Opposing directions
            ('trending_up', 'SHORT'): 0.2,
            ('trending_down', 'LONG'): 0.2,
            ('breakout_up', 'SHORT'): 0.15,
            ('breakout_down', 'LONG'): 0.15,
        }
        
        base_score = alignment.get((market_state, direction), 0.5)
        
        # Adjust by state confidence
        # High confidence in favorable state = boost
        # High confidence in unfavorable state = reduce
        if base_score > 0.5:
            return base_score * confidence + 0.5 * (1 - confidence)
        else:
            return base_score * confidence + 0.5 * (1 - confidence)
    
    async def _get_freqai_score(self, context: TradeContext) -> float:
        """Get directional score from FreqAI prediction."""
        try:
            from ml.freqai.engine import freqai_engine
            from data.binance_ws import binance_ws
            from data.historical import historical_fetcher
            
            # Get recent klines for prediction
            klines = binance_ws.get_klines(context.symbol, settings.primary_timeframe, 100)
            if not klines:
                return 0.5
            
            # Convert to DataFrame
            df = historical_fetcher.klines_to_dataframe(klines)
            
            # Get prediction (expected price change)
            prediction = await freqai_engine.predict(df)
            
            # Convert prediction to directional score
            # Positive prediction = bullish, negative = bearish
            if context.direction == 'LONG':
                # Higher positive prediction = higher score
                return min(1.0, max(0.0, 0.5 + prediction * 10))
            else:
                # Higher negative prediction = higher score for SHORT
                return min(1.0, max(0.0, 0.5 - prediction * 10))
        except Exception as e:
            logger.debug(f"FreqAI score calculation failed: {e}")
            return 0.5
    
    def record_outcome(
        self,
        model_scores: Dict[str, float],
        outcome: str,  # 'win' or 'loss'
        pnl: float
    ):
        """
        Record trade outcome for performance tracking.
        
        Called by feedback loop when trade closes.
        """
        is_win = outcome == 'win'
        
        for model_name, score in model_scores.items():
            if model_name in self.model_performance:
                # Track if model prediction aligned with outcome
                # High score + win = good prediction
                # Low score + loss = good prediction (correctly skeptical)
                model_correct = (score > 0.5 and is_win) or (score <= 0.5 and not is_win)
                
                self.model_performance[model_name].append({
                    'score': score,
                    'outcome': outcome,
                    'correct': model_correct,
                    'pnl': pnl,
                    'timestamp': datetime.now().isoformat()
                })
        
        self._save_state()
    
    def update_weights(self, adjustments: Dict[str, float]):
        """
        Apply weight adjustments from feedback loop.
        
        Adjustments are deltas that get applied to current weights.
        """
        for model_name, delta in adjustments.items():
            if model_name in self.model_weights:
                new_weight = max(0.1, min(0.5, self.model_weights[model_name] + delta))
                self.model_weights[model_name] = new_weight
        
        # Normalize weights to sum to 1.0
        total = sum(self.model_weights.values())
        if total > 0:
            for name in self.model_weights:
                self.model_weights[name] /= total
        
        logger.info(f"TradeApprover weights updated: {self.model_weights}")
        self._save_state()
    
    def get_model_stats(self) -> Dict[str, Dict]:
        """Get performance statistics for each model."""
        stats = {}
        
        for model_name, history in self.model_performance.items():
            if not history:
                stats[model_name] = {
                    'accuracy': 0.0,
                    'sample_count': 0,
                    'weight': self.model_weights.get(model_name, 0.0)
                }
                continue
            
            correct_count = sum(1 for h in history if h['correct'])
            total_count = len(history)
            
            stats[model_name] = {
                'accuracy': correct_count / total_count if total_count > 0 else 0.0,
                'sample_count': total_count,
                'weight': self.model_weights.get(model_name, 0.0),
                'recent_pnl': sum(h['pnl'] for h in list(history)[-10:])
            }
        
        return stats
    
    def get_status(self) -> Dict:
        """Get current approver status for API."""
        return {
            'enabled': True,
            'approval_threshold': self.approval_threshold,
            'min_models_agree': self.min_models_agree,
            'model_weights': self.model_weights,
            'model_stats': self.get_model_stats(),
            'recent_decisions_count': len(self.recent_decisions),
            'performance_window': self.performance_window
        }
    
    def get_recent_decisions(self, limit: int = 20) -> List[Dict]:
        """Get recent approval decisions."""
        decisions = list(self.recent_decisions)
        return decisions[-limit:]
    
    def _save_state(self):
        """Persist state to disk."""
        try:
            state = {
                'model_weights': self.model_weights,
                'model_performance': {
                    name: list(history)
                    for name, history in self.model_performance.items()
                }
            }
            
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save TradeApprover state: {e}")
    
    def _load_state(self):
        """Load state from disk."""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    state = json.load(f)
                    
                if 'model_weights' in state:
                    self.model_weights = state['model_weights']
                    
                if 'model_performance' in state:
                    for name, history in state['model_performance'].items():
                        if name in self.model_performance:
                            self.model_performance[name] = deque(history, maxlen=self.performance_window)
                
                logger.info(f"Loaded TradeApprover state from {self._state_file}")
        except Exception as e:
            logger.warning(f"Failed to load TradeApprover state: {e}")


# Global instance
trade_approver = TradeApprover()
