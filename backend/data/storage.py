"""
Data storage for trades and signals using SQLite.
"""
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import aiosqlite
from pathlib import Path
from loguru import logger
import json

from config import settings


class DataStorage:
    """SQLite-based storage for trades, signals, and ML training data."""
    
    def __init__(self, db_path: str = "data/signals.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: Optional[aiosqlite.Connection] = None
    
    async def connect(self):
        """Initialize database connection and create tables."""
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._create_tables()
        logger.info(f"Connected to database: {self.db_path}")
    
    async def close(self):
        """Close database connection."""
        if self._db:
            await self._db.close()
            logger.info("Database connection closed")
    
    async def _create_tables(self):
        """Create required tables if they don't exist."""
        
        # Signals table - stores all generated signals
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                confidence REAL NOT NULL,
                model_type TEXT NOT NULL,
                market_state TEXT,
                lvn_price REAL,
                poc_price REAL,
                cvd_value REAL,
                aggression_score REAL,
                features TEXT,
                outcome TEXT,
                outcome_pnl REAL,
                outcome_timestamp INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trade outcomes for ML training
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS trade_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl_percent REAL,
                outcome TEXT,
                duration_seconds INTEGER,
                features TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            )
        """)
        
        # LVN reaction patterns for pattern recognition
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS lvn_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                lvn_price REAL NOT NULL,
                touch_price REAL NOT NULL,
                reaction_type TEXT NOT NULL,
                price_before TEXT,
                price_after TEXT,
                volume_at_touch REAL,
                cvd_at_touch REAL,
                order_flow_features TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Market state history for state classifier
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS market_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                state TEXT NOT NULL,
                features TEXT,
                confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model metrics for tracking improvement
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                sample_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await self._db.commit()
    
    async def save_signal(
        self,
        symbol: str,
        timeframe: str,
        signal_type: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        model_type: str,
        market_state: Optional[str] = None,
        lvn_price: Optional[float] = None,
        poc_price: Optional[float] = None,
        cvd_value: Optional[float] = None,
        aggression_score: Optional[float] = None,
        features: Optional[Dict] = None
    ) -> int:
        """Save a generated signal to the database."""
        
        cursor = await self._db.execute("""
            INSERT INTO signals (
                timestamp, symbol, timeframe, signal_type, direction,
                entry_price, stop_loss, take_profit, confidence, model_type,
                market_state, lvn_price, poc_price, cvd_value, aggression_score,
                features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(datetime.now().timestamp() * 1000),
            symbol, timeframe, signal_type, direction,
            entry_price, stop_loss, take_profit, confidence, model_type,
            market_state, lvn_price, poc_price, cvd_value, aggression_score,
            json.dumps(features) if features else None
        ))
        
        await self._db.commit()
        return cursor.lastrowid
    
    async def update_signal_outcome(
        self,
        signal_id: int,
        outcome: str,
        outcome_pnl: float
    ):
        """Update signal with its outcome (WIN/LOSS/BE)."""
        
        await self._db.execute("""
            UPDATE signals 
            SET outcome = ?, outcome_pnl = ?, outcome_timestamp = ?
            WHERE id = ?
        """, (outcome, outcome_pnl, int(datetime.now().timestamp() * 1000), signal_id))
        
        await self._db.commit()
    
    async def save_lvn_pattern(
        self,
        symbol: str,
        timeframe: str,
        lvn_price: float,
        touch_price: float,
        reaction_type: str,
        price_before: List[float],
        price_after: List[float],
        volume_at_touch: float,
        cvd_at_touch: float,
        order_flow_features: Optional[Dict] = None
    ) -> int:
        """Save LVN reaction pattern for ML training."""
        
        cursor = await self._db.execute("""
            INSERT INTO lvn_patterns (
                timestamp, symbol, timeframe, lvn_price, touch_price,
                reaction_type, price_before, price_after, volume_at_touch,
                cvd_at_touch, order_flow_features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(datetime.now().timestamp() * 1000),
            symbol, timeframe, lvn_price, touch_price,
            reaction_type,
            json.dumps(price_before),
            json.dumps(price_after),
            volume_at_touch, cvd_at_touch,
            json.dumps(order_flow_features) if order_flow_features else None
        ))
        
        await self._db.commit()
        return cursor.lastrowid
    
    async def save_market_state(
        self,
        symbol: str,
        timeframe: str,
        state: str,
        features: Dict,
        confidence: float
    ) -> int:
        """Save market state classification for ML training."""
        
        cursor = await self._db.execute("""
            INSERT INTO market_states (
                timestamp, symbol, timeframe, state, features, confidence
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            int(datetime.now().timestamp() * 1000),
            symbol, timeframe, state,
            json.dumps(features),
            confidence
        ))
        
        await self._db.commit()
        return cursor.lastrowid
    
    async def get_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        with_outcome: bool = False
    ) -> List[Dict]:
        """Get recent signals."""
        
        query = "SELECT * FROM signals"
        params = []
        conditions = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        
        if with_outcome:
            conditions.append("outcome IS NOT NULL")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    async def get_signal_statistics(self, symbol: Optional[str] = None) -> Dict:
        """Get signal performance statistics."""
        
        query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(outcome_pnl) as avg_pnl,
                AVG(confidence) as avg_confidence
            FROM signals
            WHERE outcome IS NOT NULL
        """
        
        if symbol:
            query += " AND symbol = ?"
            params = [symbol]
        else:
            params = []
        
        async with self._db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            
            total = row[0] or 0
            wins = row[1] or 0
            losses = row[2] or 0
            
            return {
                'total_signals': total,
                'wins': wins,
                'losses': losses,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'avg_pnl': row[3] or 0,
                'avg_confidence': row[4] or 0
            }
    
    async def get_lvn_patterns(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get LVN patterns for ML training."""
        
        async with self._db.execute("""
            SELECT * FROM lvn_patterns 
            WHERE symbol = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (symbol, limit)) as cursor:
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    async def save_model_metric(
        self,
        model_name: str,
        metric_name: str,
        metric_value: float,
        sample_count: int
    ):
        """Save model performance metric."""
        
        await self._db.execute("""
            INSERT INTO model_metrics (
                timestamp, model_name, metric_name, metric_value, sample_count
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            int(datetime.now().timestamp() * 1000),
            model_name, metric_name, metric_value, sample_count
        ))
        
        await self._db.commit()


# Global storage instance
storage = DataStorage()
