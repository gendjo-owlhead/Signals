"""
Market State Classifier - ML-based classification of market conditions.

Classifies market into:
- BALANCED: Range-bound, rotating around POC
- TRENDING_UP: Out of balance, bullish
- TRENDING_DOWN: Out of balance, bearish
- CHOPPY: Unclear, mixed signals
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
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, using rule-based state classifier")

from config import settings


class StateLabel:
    """Market state labels."""
    BALANCED = 0
    TRENDING_UP = 1
    TRENDING_DOWN = 2
    CHOPPY = 3


@dataclass
class StateObservation:
    """Recorded market state observation."""
    timestamp: int
    symbol: str
    
    # Features
    atr_ratio: float  # ATR / Price
    range_width: float  # Range / Price
    momentum: float
    volume_distribution: float  # How evenly distributed
    poc_crosses: int  # How often price crosses POC
    vah_val_touches: int  # Touches of value area bounds
    swing_structure: List[int]  # [HH, HL, LH, LL]
    
    # Label (verified state)
    label: int
    
    def to_features(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.atr_ratio,
            self.range_width,
            self.momentum,
            self.volume_distribution,
            self.poc_crosses,
            self.vah_val_touches,
            self.swing_structure[0],  # HH
            self.swing_structure[1],  # HL
            self.swing_structure[2],  # LH
            self.swing_structure[3]   # LL
        ])


class MarketStateClassifier:
    """
    ML classifier for market state.
    
    Uses gradient boosting to learn patterns that distinguish:
    - Balanced markets (use Mean Reversion model)
    - Trending markets (use Trend model)
    - Choppy markets (avoid trading)
    """
    
    def __init__(self, model_path: str = None):
        """Initialize state classifier."""
        self.model_path = Path(model_path or settings.ml_model_path) / "state_classifier"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Training data
        self.observations: List[StateObservation] = []
        
        # Model components
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.is_trained = False
        
        # Rule-based fallback thresholds
        self.thresholds = {
            'balance_momentum': 0.3,
            'balance_range': 0.02,
            'trend_momentum': 0.5,
            'choppy_atr': 0.03
        }
        
        # Class weights for imbalanced data
        self.class_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.8}
        
        self._load_state()
    
    def record_observation(self, observation: StateObservation):
        """Record a market state observation."""
        self.observations.append(observation)
        
        # Train if enough data
        if len(self.observations) >= settings.min_samples_for_training:
            self._train_model()
        
        self._save_state()
    
    def classify(self, features: Dict) -> Tuple[str, float]:
        """
        Classify current market state.
        
        Args:
            features: Market features dict
        
        Returns:
            (state_name, confidence)
        """
        # Build feature vector
        feature_vector = np.array([
            features.get('atr_ratio', 0.01),
            features.get('range_width', 0.02),
            features.get('momentum', 0),
            features.get('volume_distribution', 0.5),
            features.get('poc_crosses', 2),
            features.get('vah_val_touches', 1),
            features.get('higher_highs', 0),
            features.get('higher_lows', 0),
            features.get('lower_highs', 0),
            features.get('lower_lows', 0)
        ])
        
        # Use ML model if trained
        if self.is_trained and self.model is not None:
            return self._classify_ml(feature_vector)
        
        # Fallback to rules
        return self._classify_rules(features)
    
    def _classify_ml(self, features: np.ndarray) -> Tuple[str, float]:
        """Classification using trained ML model."""
        if not SKLEARN_AVAILABLE or self.model is None:
            return self._classify_rules({})
        
        # Normalize
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction]
        
        state_names = ['balanced', 'trending_up', 'trending_down', 'choppy']
        return state_names[prediction], confidence
    
    def _classify_rules(self, features: Dict) -> Tuple[str, float]:
        """Rule-based classification fallback."""
        momentum = abs(features.get('momentum', 0))
        atr_ratio = features.get('atr_ratio', 0.01)
        range_width = features.get('range_width', 0.02)
        hh = features.get('higher_highs', 0)
        ll = features.get('lower_lows', 0)
        
        # Trending detection
        if momentum > self.thresholds['trend_momentum']:
            if features.get('momentum', 0) > 0 and hh > ll:
                return 'trending_up', min(momentum, 0.9)
            elif features.get('momentum', 0) < 0 and ll > hh:
                return 'trending_down', min(momentum, 0.9)
        
        # Balanced detection
        if momentum < self.thresholds['balance_momentum'] and range_width < self.thresholds['balance_range']:
            return 'balanced', 0.7
        
        # Choppy detection
        if atr_ratio > self.thresholds['choppy_atr'] and momentum < 0.3:
            return 'choppy', 0.6
        
        # Default
        return 'balanced', 0.5
    
    def _train_model(self):
        """Train the gradient boosting classifier."""
        if not SKLEARN_AVAILABLE or len(self.observations) < 50:
            return
        
        logger.info("Training market state classifier...")
        
        # Prepare data
        X = np.array([o.to_features() for o in self.observations])
        y = np.array([o.label for o in self.observations])
        
        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        
        # Calculate training accuracy
        train_acc = self.model.score(X_scaled, y)
        logger.info(f"State classifier trained: {len(self.observations)} samples, accuracy: {train_acc:.2%}")
    
    def get_metrics(self) -> Dict:
        """Get classifier metrics."""
        if not self.observations:
            return {'observations': 0}
        
        labels = [o.label for o in self.observations]
        
        return {
            'observations': len(self.observations),
            'balanced_count': labels.count(StateLabel.BALANCED),
            'trending_up_count': labels.count(StateLabel.TRENDING_UP),
            'trending_down_count': labels.count(StateLabel.TRENDING_DOWN),
            'choppy_count': labels.count(StateLabel.CHOPPY),
            'is_trained': self.is_trained
        }
    
    def _save_state(self):
        """Save classifier state."""
        state = {
            'thresholds': self.thresholds,
            'class_weights': self.class_weights,
            'observations_count': len(self.observations),
            'is_trained': self.is_trained
        }
        
        with open(self.model_path / "state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save sklearn model
        if self.is_trained and SKLEARN_AVAILABLE:
            joblib.dump(self.model, self.model_path / "model.joblib")
            joblib.dump(self.scaler, self.model_path / "scaler.joblib")
    
    def _load_state(self):
        """Load classifier state."""
        state_file = self.model_path / "state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.thresholds = state.get('thresholds', self.thresholds)
                self.class_weights = state.get('class_weights', self.class_weights)
                self.is_trained = state.get('is_trained', False)
                
                # Load sklearn model
                if self.is_trained and SKLEARN_AVAILABLE:
                    model_file = self.model_path / "model.joblib"
                    scaler_file = self.model_path / "scaler.joblib"
                    
                    if model_file.exists() and scaler_file.exists():
                        self.model = joblib.load(model_file)
                        self.scaler = joblib.load(scaler_file)
                        logger.info(f"State classifier loaded: {state.get('observations_count', 0)} observations")
                        
            except Exception as e:
                logger.error(f"Failed to load state classifier: {e}")


# Global instance
state_classifier = MarketStateClassifier()
