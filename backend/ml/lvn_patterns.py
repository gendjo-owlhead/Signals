"""
LVN Pattern Recognition - ML for predicting price reaction at LVN zones.

Classifies how price is likely to react when it touches an LVN:
- BOUNCE: Price reverses from LVN (ideal for entry)
- BREAK: Price breaks through LVN
- ABSORPTION: Price stalls at LVN, volume absorbed
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from collections import deque
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import settings


class ReactionType:
    """LVN reaction classification."""
    BOUNCE = "bounce"      # Price reverses from LVN
    BREAK = "break"        # Price breaks through LVN
    ABSORPTION = "absorption"  # Volume absorbed, price stalls


@dataclass
class LVNPattern:
    """Recorded LVN touch event with outcome."""
    timestamp: int
    symbol: str
    lvn_price: float
    touch_price: float
    direction: str  # "up_touch" or "down_touch"
    
    # Context features
    volume_ratio: float  # Volume at touch vs average
    cvd_value: float
    order_flow_imbalance: float
    distance_to_poc: float
    market_state: str
    
    # Price action before touch
    momentum_before: float
    volatility: float
    
    # Outcome
    reaction: str  # BOUNCE, BREAK, or ABSORPTION
    price_after_5: float  # Price 5 candles later
    max_adverse: float  # Max adverse move
    
    def to_features(self) -> List[float]:
        """Convert to feature vector for ML."""
        return [
            1.0 if self.direction == "up_touch" else 0.0,
            self.volume_ratio,
            self.cvd_value,
            self.order_flow_imbalance,
            self.distance_to_poc,
            1.0 if self.market_state == "balanced" else 0.0,
            self.momentum_before,
            self.volatility
        ]
    
    def to_label(self) -> int:
        """Convert reaction to numeric label."""
        labels = {ReactionType.BOUNCE: 0, ReactionType.BREAK: 1, ReactionType.ABSORPTION: 2}
        return labels.get(self.reaction, 1)


class LVNPatternRecognizer:
    """
    ML model for predicting LVN reactions.
    
    Uses historical LVN touch events to learn patterns that predict:
    - When price will bounce (good for entry)
    - When price will break through (avoid entry)
    - When price will absorb (wait for confirmation)
    """
    
    def __init__(self, model_path: str = None):
        """Initialize pattern recognizer."""
        self.model_path = Path(model_path or settings.ml_model_path) / "lvn_patterns"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Pattern storage
        self.patterns: List[LVNPattern] = []
        self.pattern_buffer: deque = deque(maxlen=1000)
        
        # Simple classifier weights (upgraded to NN when enough data)
        self.feature_means = np.zeros(8)
        self.feature_stds = np.ones(8)
        self.class_priors = {0: 0.4, 1: 0.35, 2: 0.25}  # BOUNCE, BREAK, ABSORPTION
        
        # Feature importance for each reaction type
        self.bounce_features = np.array([0.1, 0.3, 0.3, 0.15, 0.05, 0.05, 0.05, 0.0])
        self.break_features = np.array([0.1, -0.1, -0.2, -0.2, 0.1, -0.2, 0.3, 0.2])
        
        # Neural model (when available)
        self.nn_model = None
        self.nn_trained = False
        
        self._load_state()
    
    def record_pattern(self, pattern: LVNPattern):
        """Record an LVN touch pattern with its outcome."""
        self.patterns.append(pattern)
        self.pattern_buffer.append(pattern)
        
        # Update statistics
        self._update_statistics()
        
        # Train model if enough data
        if len(self.patterns) >= settings.min_samples_for_training and TORCH_AVAILABLE:
            self._train_model()
        
        self._save_state()
        
        logger.info(f"LVN pattern recorded: {pattern.reaction} at {pattern.lvn_price:.2f}")
    
    def predict_reaction(
        self,
        features: Dict
    ) -> Tuple[str, float]:
        """
        Predict likely reaction when price touches LVN.
        
        Args:
            features: Current market features at LVN
        
        Returns:
            (predicted_reaction, confidence)
        """
        # Build feature vector
        feature_vector = np.array([
            features.get('direction', 0),  # 0=up_touch, 1=down_touch
            features.get('volume_ratio', 1.0),
            features.get('cvd_value', 0),
            features.get('order_flow_imbalance', 0),
            features.get('distance_to_poc', 0),
            features.get('is_balanced', 0),
            features.get('momentum', 0),
            features.get('volatility', 0)
        ])
        
        # Use neural model if trained
        if self.nn_trained and self.nn_model is not None:
            return self._predict_with_nn(feature_vector)
        
        # Otherwise use simple heuristic model
        return self._predict_simple(feature_vector)
    
    def _predict_simple(self, features: np.ndarray) -> Tuple[str, float]:
        """Simple prediction using learned feature weights."""
        
        # Normalize features
        normalized = (features - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Score each reaction type
        bounce_score = np.dot(normalized, self.bounce_features)
        break_score = np.dot(normalized, self.break_features)
        absorption_score = 0  # Default
        
        # Determine prediction
        scores = {
            ReactionType.BOUNCE: bounce_score + self.class_priors[0],
            ReactionType.BREAK: break_score + self.class_priors[1],
            ReactionType.ABSORPTION: absorption_score + self.class_priors[2]
        }
        
        predicted = max(scores, key=scores.get)
        confidence = self._softmax(list(scores.values()))[list(scores.keys()).index(predicted)]
        
        return predicted, confidence
    
    def _predict_with_nn(self, features: np.ndarray) -> Tuple[str, float]:
        """Prediction using trained neural network."""
        if not TORCH_AVAILABLE or self.nn_model is None:
            return self._predict_simple(features)
        
        # Normalize
        normalized = (features - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Predict
        self.nn_model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(normalized).unsqueeze(0)
            output = self.nn_model(x)
            probs = torch.softmax(output, dim=1)
            
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        reactions = [ReactionType.BOUNCE, ReactionType.BREAK, ReactionType.ABSORPTION]
        return reactions[pred_idx], confidence
    
    def _update_statistics(self):
        """Update feature statistics from recorded patterns."""
        if len(self.patterns) < 10:
            return
        
        features = np.array([p.to_features() for p in self.patterns])
        self.feature_means = np.mean(features, axis=0)
        self.feature_stds = np.std(features, axis=0)
        
        # Update class priors
        labels = [p.to_label() for p in self.patterns]
        for label in range(3):
            self.class_priors[label] = labels.count(label) / len(labels)
        
        # Update feature weights based on correlation with outcomes
        self._update_feature_weights()
    
    def _update_feature_weights(self):
        """Learn feature importance from data."""
        if len(self.patterns) < 30:
            return
        
        bounce_patterns = [p for p in self.patterns if p.reaction == ReactionType.BOUNCE]
        break_patterns = [p for p in self.patterns if p.reaction == ReactionType.BREAK]
        
        if len(bounce_patterns) < 5 or len(break_patterns) < 5:
            return
        
        # Calculate mean features for each class
        bounce_feats = np.mean([p.to_features() for p in bounce_patterns], axis=0)
        break_feats = np.mean([p.to_features() for p in break_patterns], axis=0)
        
        # Direction that predicts bounce vs break
        self.bounce_features = (bounce_feats - break_feats) / (np.abs(bounce_feats - break_feats).max() + 1e-8)
        self.break_features = -self.bounce_features
    
    def _train_model(self):
        """Train neural network on collected patterns."""
        if not TORCH_AVAILABLE or len(self.patterns) < 50:
            return
        
        logger.info("Training LVN pattern recognition model...")
        
        # Prepare data
        features = np.array([p.to_features() for p in self.patterns])
        labels = np.array([p.to_label() for p in self.patterns])
        
        # Normalize
        features = (features - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Create model
        self.nn_model = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 3)
        )
        
        # Train
        optimizer = optim.Adam(self.nn_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        X = torch.FloatTensor(features)
        y = torch.LongTensor(labels)
        
        self.nn_model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            output = self.nn_model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        self.nn_trained = True
        logger.info(f"LVN pattern model trained on {len(self.patterns)} samples")
    
    def _softmax(self, x: List[float]) -> List[float]:
        """Compute softmax."""
        exp_x = np.exp(np.array(x) - np.max(x))
        return (exp_x / exp_x.sum()).tolist()
    
    def get_pattern_confidence(
        self,
        symbol: str,
        direction: str,
        entry_price: float
    ) -> Optional[float]:
        """
        Get pattern confidence score for trade approver.
        
        Returns a score between 0 and 1 indicating how favorable
        the current pattern is for the given trade direction.
        
        Args:
            symbol: Trading pair
            direction: LONG or SHORT
            entry_price: Entry price for the trade
            
        Returns:
            Confidence score [0, 1] or None if insufficient data
        """
        if len(self.patterns) < 5:
            return None  # Not enough data
        
        # Build features from recent patterns for this symbol
        symbol_patterns = [p for p in self.patterns if p.symbol == symbol]
        if len(symbol_patterns) < 3:
            return 0.5  # Neutral
        
        # Use last pattern's features as baseline
        last = symbol_patterns[-1] if symbol_patterns else None
        
        features = {
            'direction': 0 if direction == 'LONG' else 1,
            'volume_ratio': last.volume_ratio if last else 1.0,
            'cvd_value': last.cvd_value if last else 0.0,
            'order_flow_imbalance': last.order_flow_imbalance if last else 0.0,
            'distance_to_poc': last.distance_to_poc if last else 0.0,
            'is_balanced': 1 if (last and last.market_state == 'balanced') else 0,
            'momentum': last.momentum_before if last else 0.0,
            'volatility': last.volatility if last else 0.0
        }
        
        reaction, confidence = self.predict_reaction(features)
        
        # For LONG, BOUNCE is good. For SHORT, BREAK might be neutral
        if direction == 'LONG':
            if reaction == ReactionType.BOUNCE:
                return min(1.0, 0.5 + confidence * 0.5)  # Boost for bounce
            elif reaction == ReactionType.BREAK:
                return max(0.0, 0.5 - confidence * 0.3)  # Reduce for break
            else:
                return 0.5  # Neutral for absorption
        else:  # SHORT
            if reaction == ReactionType.BREAK:
                return min(1.0, 0.5 + confidence * 0.4)  # Modest boost for break
            elif reaction == ReactionType.BOUNCE:
                return max(0.0, 0.5 - confidence * 0.4)  # Reduce for bounce
            else:
                return 0.5
    
    def get_metrics(self) -> Dict:
        """Get pattern recognition metrics."""
        if not self.patterns:
            return {'patterns_recorded': 0}
        
        labels = [p.reaction for p in self.patterns]
        
        return {
            'patterns_recorded': len(self.patterns),
            'bounce_rate': labels.count(ReactionType.BOUNCE) / len(labels),
            'break_rate': labels.count(ReactionType.BREAK) / len(labels),
            'absorption_rate': labels.count(ReactionType.ABSORPTION) / len(labels),
            'nn_trained': self.nn_trained
        }
    
    def _save_state(self):
        """Save model state."""
        state = {
            'feature_means': self.feature_means.tolist(),
            'feature_stds': self.feature_stds.tolist(),
            'class_priors': self.class_priors,
            'bounce_features': self.bounce_features.tolist(),
            'break_features': self.break_features.tolist(),
            'patterns_count': len(self.patterns)
        }
        
        with open(self.model_path / "state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save neural model
        if self.nn_trained and self.nn_model is not None:
            torch.save(self.nn_model.state_dict(), self.model_path / "nn_model.pt")
    
    def _load_state(self):
        """Load model state."""
        state_file = self.model_path / "state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.feature_means = np.array(state.get('feature_means', self.feature_means))
                self.feature_stds = np.array(state.get('feature_stds', self.feature_stds))
                self.class_priors = state.get('class_priors', self.class_priors)
                self.bounce_features = np.array(state.get('bounce_features', self.bounce_features))
                self.break_features = np.array(state.get('break_features', self.break_features))
                
                logger.info(f"LVN pattern model loaded: {state.get('patterns_count', 0)} patterns")
                
                # Load neural model
                nn_path = self.model_path / "nn_model.pt"
                if nn_path.exists() and TORCH_AVAILABLE:
                    self.nn_model = nn.Sequential(
                        nn.Linear(8, 32),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(16, 3)
                    )
                    self.nn_model.load_state_dict(torch.load(nn_path))
                    self.nn_trained = True
                    
            except Exception as e:
                logger.error(f"Failed to load pattern model: {e}")


# Global instance
lvn_pattern_recognizer = LVNPatternRecognizer()
