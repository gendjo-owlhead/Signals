"""
Online Learning Trainer - Coordinates ML model training and updates.

Manages the continuous learning pipeline for all ML components.
"""
import asyncio
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger

from config import settings
from data.storage import storage
from ml.signal_accuracy import signal_accuracy_model, SignalOutcome
from ml.lvn_patterns import lvn_pattern_recognizer, LVNPattern
from ml.state_classifier import state_classifier, StateObservation


class OnlineLearningTrainer:
    """
    Coordinates online learning across all ML models.
    
    Responsibilities:
    - Monitor signal outcomes and feed to signal accuracy model
    - Record LVN touch events and train pattern recognizer
    - Collect market state observations for classifier
    - Periodically trigger model updates
    """
    
    def __init__(self):
        self.running = False
        self.last_training = {}
        self.training_interval_seconds = 300  # 5 minutes
        
        # Metrics
        self.metrics = {
            'signal_outcomes_processed': 0,
            'lvn_patterns_recorded': 0,
            'state_observations_recorded': 0,
            'last_model_update': None
        }
    
    async def start(self):
        """Start the online learning trainer."""
        self.running = True
        logger.info("Online Learning Trainer started")
        
        # Start background tasks
        asyncio.create_task(self._training_loop())
    
    async def stop(self):
        """Stop the trainer."""
        self.running = False
        logger.info("Online Learning Trainer stopped")
    
    async def _training_loop(self):
        """Periodic training loop."""
        while self.running:
            try:
                # Check if any model needs updating
                await self._check_and_train()
                
                # Wait for next training interval
                await asyncio.sleep(self.training_interval_seconds)
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(60)
    
    async def _check_and_train(self):
        """Check if models need training and trigger updates."""
        
        # Get new completed signals from database
        signals = await storage.get_signals(with_outcome=True, limit=100)
        
        new_outcomes = [
            s for s in signals
            if s['id'] not in self.last_training.get('processed_signals', [])
        ]
        
        # Process new outcomes for signal accuracy model
        for signal_data in new_outcomes:
            if signal_data.get('outcome'):
                outcome = SignalOutcome(
                    signal_id=signal_data['id'],
                    symbol=signal_data['symbol'],
                    direction=signal_data['direction'],
                    entry_price=signal_data['entry_price'],
                    exit_price=0,  # Need to get from trade
                    pnl_percent=signal_data.get('outcome_pnl', 0),
                    outcome=signal_data['outcome'],
                    confidence=signal_data['confidence'],
                    features={
                        'market_state_conf': 0.7,  # Would extract from stored features
                        'aggression_strength': signal_data.get('aggression_score', 0.5),
                        'cvd_confirming': signal_data.get('cvd_value', False),
                        'risk_reward': 2.0
                    },
                    model_type=signal_data['model_type'],
                    timestamp=signal_data['timestamp']
                )
                
                signal_accuracy_model.record_outcome(outcome)
                self.metrics['signal_outcomes_processed'] += 1
        
        # Track processed signals
        processed_ids = self.last_training.get('processed_signals', [])
        processed_ids.extend([s['id'] for s in new_outcomes])
        self.last_training['processed_signals'] = processed_ids[-500:]  # Keep last 500
        
        self.metrics['last_model_update'] = datetime.now().isoformat()
    
    def record_lvn_touch(
        self,
        symbol: str,
        lvn_price: float,
        touch_price: float,
        direction: str,
        context: Dict,
        reaction: str
    ):
        """
        Record an LVN touch event for pattern learning.
        
        Called when price touches an LVN and we later observe the reaction.
        """
        pattern = LVNPattern(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            lvn_price=lvn_price,
            touch_price=touch_price,
            direction=direction,
            volume_ratio=context.get('volume_ratio', 1.0),
            cvd_value=context.get('cvd_value', 0),
            order_flow_imbalance=context.get('order_flow_imbalance', 0),
            distance_to_poc=context.get('distance_to_poc', 0),
            market_state=context.get('market_state', 'balanced'),
            momentum_before=context.get('momentum', 0),
            volatility=context.get('volatility', 0),
            reaction=reaction,
            price_after_5=context.get('price_after_5', touch_price),
            max_adverse=context.get('max_adverse', 0)
        )
        
        lvn_pattern_recognizer.record_pattern(pattern)
        self.metrics['lvn_patterns_recorded'] += 1
    
    def record_state_observation(
        self,
        symbol: str,
        features: Dict,
        verified_label: int
    ):
        """
        Record a verified market state observation.
        
        Called when we can verify what the market state actually was.
        """
        observation = StateObservation(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol=symbol,
            atr_ratio=features.get('atr_ratio', 0.01),
            range_width=features.get('range_width', 0.02),
            momentum=features.get('momentum', 0),
            volume_distribution=features.get('volume_distribution', 0.5),
            poc_crosses=features.get('poc_crosses', 0),
            vah_val_touches=features.get('vah_val_touches', 0),
            swing_structure=[
                features.get('higher_highs', 0),
                features.get('higher_lows', 0),
                features.get('lower_highs', 0),
                features.get('lower_lows', 0)
            ],
            label=verified_label
        )
        
        state_classifier.record_observation(observation)
        self.metrics['state_observations_recorded'] += 1
    
    def get_ml_metrics(self) -> Dict:
        """Get combined ML metrics from all models."""
        return {
            'training': self.metrics,
            'signal_accuracy': signal_accuracy_model.get_metrics(),
            'lvn_patterns': lvn_pattern_recognizer.get_metrics(),
            'state_classifier': state_classifier.get_metrics()
        }
    
    def get_learning_status(self) -> Dict:
        """Get current learning status for dashboard."""
        sig_metrics = signal_accuracy_model.get_metrics()
        lvn_metrics = lvn_pattern_recognizer.get_metrics()
        state_metrics = state_classifier.get_metrics()
        
        return {
            'signal_accuracy': {
                'total_signals': sig_metrics.get('total_signals', 0),
                'win_rate': sig_metrics.get('win_rate', 0) * 100,
                'feature_weights': sig_metrics.get('feature_weights', {}),
                'status': 'active' if sig_metrics.get('total_signals', 0) > 0 else 'collecting'
            },
            'lvn_patterns': {
                'patterns_recorded': lvn_metrics.get('patterns_recorded', 0),
                'bounce_rate': lvn_metrics.get('bounce_rate', 0) * 100,
                'nn_trained': lvn_metrics.get('nn_trained', False),
                'status': 'trained' if lvn_metrics.get('nn_trained') else 'learning'
            },
            'state_classifier': {
                'observations': state_metrics.get('observations', 0),
                'is_trained': state_metrics.get('is_trained', False),
                'status': 'trained' if state_metrics.get('is_trained') else 'learning'
            },
            'overall_status': self._get_overall_status()
        }
    
    def _get_overall_status(self) -> str:
        """Get overall ML system status."""
        sig_metrics = signal_accuracy_model.get_metrics()
        
        if sig_metrics.get('total_signals', 0) < 10:
            return 'initializing'
        elif sig_metrics.get('total_signals', 0) < 50:
            return 'learning'
        else:
            return 'active'


# Global trainer instance
online_trainer = OnlineLearningTrainer()
