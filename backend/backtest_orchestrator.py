"""
Backtest Orchestrator - Manages continuous backtests across multiple timeframes and symbols.

Automatically schedules and runs backtests with ML training enabled to improve model accuracy.
Target: Achieve 60-70% win rate through continuous learning.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
from pathlib import Path
import subprocess
import json

from loguru import logger


class BacktestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    symbol: str
    timeframe: str
    days: int
    schedule_interval_hours: int = 24  # How often to re-run
    
    @property
    def id(self) -> str:
        return f"{self.symbol}_{self.timeframe}"


@dataclass
class BacktestRun:
    """Represents a single backtest execution."""
    config: BacktestConfig
    status: BacktestStatus = BacktestStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_pct: float = 0.0
    
    # Results
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "days": self.config.days,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress_pct": self.progress_pct,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_pnl_pct": self.total_pnl_pct,
            "error_message": self.error_message
        }


# Backtest matrix: All symbol/timeframe combinations
BACKTEST_MATRIX: List[BacktestConfig] = [
    # 1m timeframe - 1 year (365 days), re-run every 24 hours
    BacktestConfig("BTCUSDT", "1m", 365, 24),
    BacktestConfig("ETHUSDT", "1m", 365, 24),
    
    # 5m timeframe - 3 years (1095 days), re-run every 48 hours
    BacktestConfig("BTCUSDT", "5m", 1095, 48),
    BacktestConfig("ETHUSDT", "5m", 1095, 48),
    
    # 15m timeframe - 3 years (1095 days), re-run every 48 hours
    BacktestConfig("BTCUSDT", "15m", 1095, 48),
    BacktestConfig("ETHUSDT", "15m", 1095, 48),
    
    # 1h timeframe - 5 years (1825 days), re-run every 72 hours
    BacktestConfig("BTCUSDT", "1h", 1825, 72),
    BacktestConfig("ETHUSDT", "1h", 1825, 72),
]


class BacktestOrchestrator:
    """
    Manages continuous backtest execution across all configured symbol/timeframe pairs.
    
    Features:
    - Automatic scheduling based on configured intervals
    - Sequential execution to avoid resource contention
    - ML training flag always enabled for continuous learning
    - Real-time progress tracking
    - Results aggregation and performance monitoring
    """
    
    def __init__(self, parallel_limit: int = 1):
        self.parallel_limit = parallel_limit
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
        # Track runs and schedule
        self.current_runs: Dict[str, BacktestRun] = {}
        self.completed_runs: List[BacktestRun] = []
        self.last_run_times: Dict[str, datetime] = {}
        
        # Performance tracking
        self.aggregate_stats: Dict[str, Dict[str, float]] = {
            "BTCUSDT": {},
            "ETHUSDT": {}
        }
        
        # State persistence path
        self.state_path = Path("data/orchestrator_state.json")
        self._load_state()
    
    def start(self):
        """Start the orchestrator background task."""
        if self.running:
            logger.warning("Backtest orchestrator already running")
            return
        
        self.running = True
        self._task = asyncio.create_task(self._orchestration_loop())
        logger.info("🚀 Backtest Orchestrator started")
    
    def stop(self):
        """Stop the orchestrator."""
        self.running = False
        if self._task:
            self._task.cancel()
        self._save_state()
        logger.info("⏹️ Backtest Orchestrator stopped")
    
    async def _orchestration_loop(self):
        """Main orchestration loop - checks for due backtests and runs them."""
        logger.info("Orchestration loop started")
        
        while self.running:
            try:
                # Find backtests that are due
                due_configs = self._get_due_backtests()
                
                if due_configs:
                    logger.info(f"Found {len(due_configs)} backtests due for execution")
                    
                    # Run backtests sequentially (or with parallel limit)
                    for config in due_configs:
                        if not self.running:
                            break
                        await self._run_backtest(config)
                
                # Save state periodically
                self._save_state()
                
                # Check every 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                logger.info("Orchestration loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(60)
    
    def _get_due_backtests(self) -> List[BacktestConfig]:
        """Get list of backtests that are due to run."""
        due = []
        now = datetime.now()
        
        for config in BACKTEST_MATRIX:
            config_id = config.id
            
            # Check if already running
            if config_id in self.current_runs:
                if self.current_runs[config_id].status == BacktestStatus.RUNNING:
                    continue
            
            # Check if enough time has passed since last run
            last_run = self.last_run_times.get(config_id)
            if last_run:
                interval = timedelta(hours=config.schedule_interval_hours)
                if now - last_run < interval:
                    continue
            
            due.append(config)
        
        return due
    
    async def _run_backtest(self, config: BacktestConfig):
        """Execute a single backtest with ML training enabled."""
        run = BacktestRun(config=config)
        run.status = BacktestStatus.RUNNING
        run.started_at = datetime.now()
        
        self.current_runs[config.id] = run
        
        logger.info(f"🔄 Starting backtest: {config.symbol} {config.timeframe} ({config.days} days)")
        
        try:
            # Build command - always with --train flag for ML learning
            cmd = [
                "python", "backtest.py",
                "--symbol", config.symbol,
                "--timeframe", config.timeframe,
                "--days", str(config.days),
                "--train"  # Always train ML with backtest data
            ]
            
            # Run backtest as subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Parse results from output
                output = stdout.decode()
                run = self._parse_backtest_output(run, output)
                run.status = BacktestStatus.COMPLETED
                run.completed_at = datetime.now()
                
                logger.info(
                    f"✅ Completed: {config.symbol} {config.timeframe} | "
                    f"Win Rate: {run.win_rate:.1f}% | "
                    f"Trades: {run.total_trades} | "
                    f"PnL: {run.total_pnl_pct:+.2f}%"
                )
                
                # Update aggregate stats
                self._update_aggregate_stats(run)
            else:
                run.status = BacktestStatus.FAILED
                run.error_message = stderr.decode()[:500]
                logger.error(f"❌ Backtest failed: {config.symbol} {config.timeframe}")
            
        except Exception as e:
            run.status = BacktestStatus.FAILED
            run.error_message = str(e)
            logger.error(f"Exception running backtest: {e}")
        
        finally:
            self.last_run_times[config.id] = datetime.now()
            self.completed_runs.append(run)
            
            # Keep only last 50 completed runs
            if len(self.completed_runs) > 50:
                self.completed_runs = self.completed_runs[-50:]
    
    def _parse_backtest_output(self, run: BacktestRun, output: str) -> BacktestRun:
        """Parse backtest output to extract results."""
        lines = output.split('\n')
        
        for line in lines:
            try:
                if "Win Rate:" in line:
                    # Format: "Win Rate:         55.5% (100 wins / 80 losses)"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "%" in part:
                            run.win_rate = float(part.replace("%", ""))
                            break
                
                elif "Trades Executed:" in line:
                    parts = line.split()
                    run.total_trades = int(parts[-1].replace(",", ""))
                
                elif "Profit Factor:" in line:
                    parts = line.split()
                    run.profit_factor = float(parts[-1])
                
                elif "Total Return:" in line:
                    parts = line.split()
                    run.total_pnl_pct = float(parts[-1].replace("%", "").replace("+", ""))
                    
            except (ValueError, IndexError):
                continue
        
        return run
    
    def _update_aggregate_stats(self, run: BacktestRun):
        """Update aggregate statistics for symbol/timeframe."""
        symbol = run.config.symbol
        timeframe = run.config.timeframe
        
        if symbol not in self.aggregate_stats:
            self.aggregate_stats[symbol] = {}
        
        self.aggregate_stats[symbol][timeframe] = {
            "win_rate": run.win_rate,
            "profit_factor": run.profit_factor,
            "total_pnl_pct": run.total_pnl_pct,
            "total_trades": run.total_trades,
            "last_updated": datetime.now().isoformat()
        }
    
    async def run_single_backtest(self, symbol: str, timeframe: str, days: int) -> BacktestRun:
        """Manually trigger a single backtest."""
        config = BacktestConfig(symbol, timeframe, days)
        await self._run_backtest(config)
        return self.current_runs.get(config.id)
    
    def get_status(self) -> dict:
        """Get current orchestrator status."""
        # Find currently running
        running = [
            run.to_dict() for run in self.current_runs.values()
            if run.status == BacktestStatus.RUNNING
        ]
        
        # Find pending (due but not started)
        due = self._get_due_backtests()
        pending = [{"symbol": c.symbol, "timeframe": c.timeframe, "days": c.days} for c in due]
        
        # Recent completed
        recent_completed = [run.to_dict() for run in self.completed_runs[-10:]]
        
        return {
            "running": self.running,
            "parallel_limit": self.parallel_limit,
            "current_backtests": running,
            "pending_backtests": pending,
            "recent_completed": recent_completed,
            "aggregate_stats": self.aggregate_stats,
            "total_completed": len(self.completed_runs),
            "matrix_size": len(BACKTEST_MATRIX)
        }
    
    def get_performance_summary(self) -> dict:
        """Get aggregated performance across all symbol/timeframe combinations."""
        summary = {
            "by_symbol": {},
            "by_timeframe": {},
            "overall": {
                "avg_win_rate": 0.0,
                "total_trades": 0
            }
        }
        
        all_win_rates = []
        
        for symbol, timeframes in self.aggregate_stats.items():
            summary["by_symbol"][symbol] = {"avg_win_rate": 0.0, "timeframes": {}}
            symbol_rates = []
            
            for tf, stats in timeframes.items():
                summary["by_symbol"][symbol]["timeframes"][tf] = stats
                symbol_rates.append(stats.get("win_rate", 0))
                all_win_rates.append(stats.get("win_rate", 0))
                
                # By timeframe
                if tf not in summary["by_timeframe"]:
                    summary["by_timeframe"][tf] = {"symbols": {}, "avg_win_rate": 0.0}
                summary["by_timeframe"][tf]["symbols"][symbol] = stats
            
            if symbol_rates:
                summary["by_symbol"][symbol]["avg_win_rate"] = sum(symbol_rates) / len(symbol_rates)
        
        # Calculate timeframe averages
        for tf in summary["by_timeframe"]:
            rates = [s.get("win_rate", 0) for s in summary["by_timeframe"][tf]["symbols"].values()]
            if rates:
                summary["by_timeframe"][tf]["avg_win_rate"] = sum(rates) / len(rates)
        
        # Overall average
        if all_win_rates:
            summary["overall"]["avg_win_rate"] = sum(all_win_rates) / len(all_win_rates)
        
        return summary
    
    def _save_state(self):
        """Persist state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                "last_run_times": {k: v.isoformat() for k, v in self.last_run_times.items()},
                "aggregate_stats": self.aggregate_stats,
                "completed_runs": [run.to_dict() for run in self.completed_runs[-20:]]
            }
            
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save orchestrator state: {e}")
    
    def _load_state(self):
        """Load state from disk."""
        try:
            if self.state_path.exists():
                with open(self.state_path, 'r') as f:
                    state = json.load(f)
                
                # Restore last run times
                for k, v in state.get("last_run_times", {}).items():
                    self.last_run_times[k] = datetime.fromisoformat(v)
                
                # Restore aggregate stats
                self.aggregate_stats = state.get("aggregate_stats", self.aggregate_stats)
                
                logger.info("Loaded orchestrator state from disk")
                
        except Exception as e:
            logger.warning(f"Could not load orchestrator state: {e}")


# Global instance
backtest_orchestrator = BacktestOrchestrator()
