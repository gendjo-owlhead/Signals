"""
Signal Accuracy ML - Feedback loop for signal improvement.

Tracks signal outcomes and adjusts confidence scoring over time.
This is the core self-improvement mechanism for the signal generator.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from loguru import logger

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using simplified signal accuracy model")

from config import settings


@dataclass
class SignalOutcome:
    """Recorded outcome of a signal."""
    signal_id: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_percent: float
    outcome: str  # "WIN", "LOSS", "BREAKEVEN"
    confidence: float
    features: Dict
    model_type: str
    timestamp: int


class SignalAccuracyModel:
    """
    ML model for signal accuracy prediction and improvement.
    
    Features used:
    - Market state confidence
    - Aggression strength
    - CVD confirmation
    - Risk/reward ratio
    - LVN distance
    - Time of day
    - Volatility
    
    The model learns which feature combinations lead to winning trades
    and adjusts the confidence scoring weights accordingly.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize signal accuracy model."""
        self.model_path = Path(model_path or settings.ml_model_path) / "signal_accuracy"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Feature weights (learned over time)
        self.feature_weights = {
            'market_state_conf': 0.3,
            'aggression_strength': 0.4,
            'cvd_confirming': 0.15,
            'risk_reward': 0.1,
            'lvn_precision': 0.05
        }
        
        # Outcome history for online learning
        self.outcomes: List[SignalOutcome] = []
        
        # Performance metrics
        self.metrics = {
            'total_signals': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'avg_confidence': 0.0,
            'confidence_calibration': 0.0
        }
        
        # Load saved state
        self._load_state()
    
    def record_outcome(self, outcome: SignalOutcome):
        """
        Record a signal outcome for learning.
        
        This is called when a signal's SL or TP is hit.
        """
        # Idempotency check: Don't record if we already have this signal_id
        if any(o.signal_id == outcome.signal_id for o in self.outcomes):
            logger.warning(f"Signal {outcome.signal_id} outcome already recorded. Skipping.")
            return

        self.outcomes.append(outcome)
        
        # Update metrics
        if self.metrics.get('total_signals') is None: self.metrics['total_signals'] = 0
        if self.metrics.get('wins') is None: self.metrics['wins'] = 0
        if self.metrics.get('losses') is None: self.metrics['losses'] = 0

        self.metrics['total_signals'] += 1
        if outcome.outcome == "WIN":
            self.metrics['wins'] += 1
        elif outcome.outcome == "LOSS":
            self.metrics['losses'] += 1
        
        if self.metrics['total_signals'] > 0:
            wins = self.metrics.get('wins', 0) or 0
            total = self.metrics.get('total_signals', 1) or 1
            self.metrics['win_rate'] = wins / total
        
        # Trigger learning if enough samples
        if len(self.outcomes) >= settings.min_samples_for_training:
            self._update_weights()
        
        # Save state
        self._save_state()
        
        logger.info(
            f"Signal outcome recorded: {outcome.outcome} | "
            f"Win rate: {self.metrics['win_rate']:.1%}"
        )
    
    def adjust_confidence(
        self,
        base_confidence: float,
        features: Dict
    ) -> float:
        """
        Adjust signal confidence based on learned patterns.
        
        Args:
            base_confidence: Original confidence from signal generator
            features: Signal features
        
        Returns:
            Adjusted confidence score
        """
        if not self.outcomes:
            return base_confidence
        
        # Calculate weighted feature score
        feature_score = 0.0
        
        if 'market_state_conf' in features:
            feature_score += features['market_state_conf'] * self.feature_weights['market_state_conf']
        
        if 'aggression_strength' in features:
            feature_score += features['aggression_strength'] * self.feature_weights['aggression_strength']
        
        if 'cvd_confirming' in features:
            cvd_val = 1.0 if features['cvd_confirming'] else 0.5
            feature_score += cvd_val * self.feature_weights['cvd_confirming']
        
        if 'risk_reward' in features:
            rr_score = min(features['risk_reward'] / 3, 1.0)
            feature_score += rr_score * self.feature_weights['risk_reward']
        
        if 'lvn_precision' in features:
            feature_score += features['lvn_precision'] * self.feature_weights['lvn_precision']
        
        # Apply calibration based on historical accuracy
        calibration = self._get_calibration_factor()
        
        # Blend base confidence with learned adjustment
        adjusted = (base_confidence * 0.6) + (feature_score * 0.4 * calibration)
        
        return min(max(adjusted, 0.0), 1.0)
    
    def _update_weights(self):
        """Update feature weights based on outcome patterns."""
        if len(self.outcomes) < settings.min_samples_for_training:
            return
        
        # Analyze recent outcomes
        wins = [o for o in self.outcomes if o.outcome == "WIN"]
        losses = [o for o in self.outcomes if o.outcome == "LOSS"]
        
        if not wins or not losses:
            return
        
        # Calculate feature importance based on difference between wins and losses
        for feature_name in self.feature_weights.keys():
            win_values = [o.features.get(feature_name) if o.features.get(feature_name) is not None else 0.5 for o in wins if feature_name in o.features]
            loss_values = [o.features.get(feature_name) if o.features.get(feature_name) is not None else 0.5 for o in losses if feature_name in o.features]
            
            if win_values and loss_values:
                win_avg = np.mean(win_values)
                loss_avg = np.mean(loss_values)
                
                # Scale weight based on discriminative power
                importance = abs(win_avg - loss_avg)
                
                # Smooth update
                current = self.feature_weights[feature_name]
                self.feature_weights[feature_name] = current * 0.8 + importance * 0.2
        
        # Normalize weights
        total = sum(self.feature_weights.values())
        if total > 0:
            for k in self.feature_weights:
                self.feature_weights[k] /= total
        
        logger.info(f"Feature weights updated: {self.feature_weights}")
    
    def _get_calibration_factor(self) -> float:
        """
        Get calibration factor based on prediction accuracy.
        
        If we're overconfident (predict high, often wrong), reduce scores.
        If we're underconfident (predict low, often right), increase scores.
        """
        if self.metrics['total_signals'] < 20:
            return 1.0
        
        # Calculate expected vs actual win rate at different confidence levels
        # This is simplified - production would use binned calibration
        
        avg_conf = self.metrics.get('avg_confidence', 0.7)
        actual_wr = self.metrics['win_rate']
        
        if avg_conf > 0:
            calibration = actual_wr / avg_conf
            return min(max(calibration, 0.5), 1.5)
        
        return 1.0
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        return {
            **self.metrics,
            'feature_weights': self.feature_weights,
            'samples_collected': len(self.outcomes)
        }
    
    def _save_state(self):
        """Save model state to disk."""
        state = {
            'feature_weights': self.feature_weights,
            'metrics': self.metrics,
            'outcomes': [
                {
                    'signal_id': o.signal_id,
                    'symbol': o.symbol,
                    'direction': o.direction,
                    'pnl_percent': o.pnl_percent,
                    'outcome': o.outcome,
                    'confidence': o.confidence,
                    'model_type': o.model_type,
                    'timestamp': o.timestamp
                }
                for o in self.outcomes[-500:]  # Keep last 500
            ]
        }
        
        state_file = self.model_path / "state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load model state from disk."""
        state_file = self.model_path / "state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.feature_weights = state.get('feature_weights', self.feature_weights)
                self.metrics = state.get('metrics', self.metrics)
                
                # Restore outcomes for idempotency check
                saved_outcomes = state.get('outcomes', [])
                for o in saved_outcomes:
                    self.outcomes.append(SignalOutcome(
                        signal_id=o['signal_id'],
                        symbol=o['symbol'],
                        direction=o['direction'],
                        entry_price=0,
                        exit_price=0,
                        pnl_percent=o.get('pnl_percent', 0),
                        outcome=o['outcome'],
                        confidence=o['confidence'],
                        features={},
                        model_type=o['model_type'],
                        timestamp=o['timestamp']
                    ))
                
                logger.info(f"Signal accuracy model loaded: {self.metrics['total_signals']} signals, {len(self.outcomes)} outcomes")
            except Exception as e:
                logger.error(f"Failed to load model state: {e}")


# Neural network version (optional, more sophisticated)
if TORCH_AVAILABLE:
    class SignalAccuracyNN(nn.Module):
        """Neural network for signal accuracy prediction."""
        
        def __init__(self, input_size: int = 10):
            super().__init__()
            
            self.network = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.network(x)


# Global instance
signal_accuracy_model = SignalAccuracyModel()
