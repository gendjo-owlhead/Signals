"""
Backtesting script for Auction Market Strategy.
Tests Trend Model and Mean Reversion Model on historical data.

Usage:
    python backtest.py --symbol BTCUSDT --days 30
    python backtest.py --symbol ETHUSDT --days 7 --save
"""
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from data.historical import HistoricalDataFetcher
from data.binance_ws import Kline, Trade
from signals.trend_model import TrendModelGenerator
from signals.mean_reversion import MeanReversionGenerator
from analysis.volume_profile import VolumeProfileCalculator
from analysis.order_flow import OrderFlowAnalyzer
from config import settings

# Use mainnet for historical data (testnet has limited history)
historical_fetcher = HistoricalDataFetcher(use_mainnet=True)


@dataclass
class BacktestTrade:
    """Represents a simulated trade in backtest."""
    timestamp: int
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    model_type: str  # TREND or MEAN_REVERSION
    
    # Outcome (filled after simulation)
    exit_price: Optional[float] = None
    exit_timestamp: Optional[int] = None
    exit_reason: Optional[str] = None  # TP_HIT, SL_HIT, TIMEOUT
    pnl_pct: Optional[float] = None
    is_win: Optional[bool] = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    total_candles: int
    
    # Trade stats
    total_signals: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    
    # Performance
    total_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    
    # By model
    trend_trades: int = 0
    trend_wins: int = 0
    reversion_trades: int = 0
    reversion_wins: int = 0
    
    # Trade list
    trades: List[BacktestTrade] = field(default_factory=list)
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.wins / self.total_trades) * 100
    
    @property
    def trend_win_rate(self) -> float:
        if self.trend_trades == 0:
            return 0.0
        return (self.trend_wins / self.trend_trades) * 100
    
    @property
    def reversion_win_rate(self) -> float:
        if self.reversion_trades == 0:
            return 0.0
        return (self.reversion_wins / self.reversion_trades) * 100


class Backtester:
    """
    Backtest the Auction Market Strategy on historical data.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "5m",
        confidence_threshold: float = 0.7,
        lookback_periods: int = 100
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.confidence_threshold = confidence_threshold
        self.lookback_periods = lookback_periods
        
        # Strategy generators
        self.trend_model = TrendModelGenerator(confidence_threshold=confidence_threshold)
        self.reversion_model = MeanReversionGenerator(confidence_threshold=confidence_threshold)
        self.volume_profile = VolumeProfileCalculator()
        self.order_flow = OrderFlowAnalyzer()
        
        # Track active position
        self.active_trade: Optional[BacktestTrade] = None
    
    async def run(self, days: int = 30) -> BacktestResult:
        """
        Run backtest for specified number of days.
        
        Args:
            days: Number of days to backtest
            
        Returns:
            BacktestResult with performance metrics
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        logger.info(f"Starting backtest for {self.symbol} {self.timeframe}")
        logger.info(f"Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        # Fetch historical klines
        klines = await self._fetch_historical_klines(start_time, end_time)
        
        if not klines:
            logger.error("No historical data fetched!")
            return BacktestResult(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=start_time,
                end_date=end_time,
                total_candles=0
            )
        
        logger.info(f"Fetched {len(klines)} candles for backtesting")
        
        # Initialize result
        result = BacktestResult(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=start_time,
            end_date=end_time,
            total_candles=len(klines)
        )
        
        # Simulate trading
        await self._simulate_trading(klines, result)
        
        # Calculate final metrics
        self._calculate_metrics(result)
        
        return result
    
    async def _fetch_historical_klines(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Kline]:
        """Fetch historical klines with pagination."""
        all_klines = []
        current_start = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        while current_start < end_ms:
            klines = await historical_fetcher.fetch_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                limit=1000,
                start_time=current_start,
                end_time=end_ms
            )
            
            if not klines:
                break
            
            all_klines.extend(klines)
            current_start = klines[-1].close_time + 1
            
            # Progress update
            if len(all_klines) % 1000 == 0:
                logger.info(f"Fetched {len(all_klines)} candles...")
            
            # Rate limit
            await asyncio.sleep(0.1)
        
        # Remove duplicates and sort
        seen = set()
        unique_klines = []
        for k in all_klines:
            if k.open_time not in seen:
                seen.add(k.open_time)
                unique_klines.append(k)
        
        unique_klines.sort(key=lambda x: x.open_time)
        return unique_klines
    
    async def _simulate_trading(self, klines: List[Kline], result: BacktestResult):
        """
        Simulate trading through historical data.
        Walk through each candle and check for signals/exits.
        """
        logger.info("Simulating trading...")
        
        for i in range(self.lookback_periods, len(klines)):
            # Get lookback window
            window = klines[i - self.lookback_periods:i + 1]
            current_kline = klines[i]
            
            # Check if we have an active trade
            if self.active_trade:
                # Check for exit
                exit_result = self._check_exit(self.active_trade, current_kline)
                
                if exit_result:
                    exit_price, exit_reason = exit_result
                    self._close_trade(self.active_trade, exit_price, 
                                     current_kline.close_time, exit_reason, result)
            
            # If no active trade, look for signals
            if not self.active_trade:
                # Generate mock trades list (simplified - no real trades needed for signal gen)
                mock_trades = self._generate_mock_trades(window)
                
                # Try Trend Model
                trend_signal = self.trend_model.generate_signal(
                    window, mock_trades, self.symbol
                )
                
                if trend_signal and trend_signal.confidence >= self.confidence_threshold:
                    result.total_signals += 1
                    self._open_trade(trend_signal, current_kline, "TREND", result)
                    continue
                
                # Try Mean Reversion Model
                reversion_signal = self.reversion_model.generate_signal(
                    window, mock_trades, self.symbol
                )
                
                if reversion_signal and reversion_signal.confidence >= self.confidence_threshold:
                    result.total_signals += 1
                    self._open_trade(reversion_signal, current_kline, "MEAN_REVERSION", result)
            
            # Progress
            if i % 1000 == 0:
                pct = (i / len(klines)) * 100
                logger.info(f"Progress: {pct:.1f}% ({result.total_trades} trades so far)")
        
        # Close any remaining trade at end
        if self.active_trade:
            final_price = klines[-1].close
            self._close_trade(self.active_trade, final_price,
                             klines[-1].close_time, "TIMEOUT", result)
    
    def _generate_mock_trades(self, klines: List[Kline]) -> List[Trade]:
        """
        Generate mock trade data from klines for order flow analysis.
        Uses volume and taker buy volume to simulate trade flow.
        """
        trades = []
        trade_id = 0
        
        for kline in klines[-20:]:  # Last 20 candles
            # Simulate trades based on candle data
            buy_volume = kline.taker_buy_volume
            sell_volume = kline.volume - buy_volume
            
            # Create representative trades
            if buy_volume > 0:
                trades.append(Trade(
                    trade_id=trade_id,
                    price=kline.close,
                    quantity=buy_volume,
                    timestamp=kline.close_time,
                    is_buyer_maker=False  # Buyer is taker
                ))
                trade_id += 1
            
            if sell_volume > 0:
                trades.append(Trade(
                    trade_id=trade_id,
                    price=kline.close,
                    quantity=sell_volume,
                    timestamp=kline.close_time,
                    is_buyer_maker=True  # Seller is taker
                ))
                trade_id += 1
        
        return trades
    
    def _open_trade(self, signal, kline: Kline, model_type: str, result: BacktestResult):
        """Open a new trade from a signal."""
        trade = BacktestTrade(
            timestamp=kline.close_time,
            symbol=self.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            model_type=model_type
        )
        
        self.active_trade = trade
        result.total_trades += 1
        
        if model_type == "TREND":
            result.trend_trades += 1
        else:
            result.reversion_trades += 1
    
    def _check_exit(
        self, 
        trade: BacktestTrade, 
        kline: Kline
    ) -> Optional[Tuple[float, str]]:
        """
        Check if trade should exit on this candle.
        Returns (exit_price, reason) or None.
        """
        is_long = trade.direction == "LONG"
        
        if is_long:
            # Check SL hit (low touched SL)
            if kline.low <= trade.stop_loss:
                return (trade.stop_loss, "SL_HIT")
            
            # Check TP hit (high touched TP)
            if kline.high >= trade.take_profit:
                return (trade.take_profit, "TP_HIT")
        else:
            # SHORT
            # Check SL hit (high touched SL)
            if kline.high >= trade.stop_loss:
                return (trade.stop_loss, "SL_HIT")
            
            # Check TP hit (low touched TP)
            if kline.low <= trade.take_profit:
                return (trade.take_profit, "TP_HIT")
        
        return None
    
    def _close_trade(
        self,
        trade: BacktestTrade,
        exit_price: float,
        exit_timestamp: int,
        exit_reason: str,
        result: BacktestResult
    ):
        """Close a trade and record the result."""
        trade.exit_price = exit_price
        trade.exit_timestamp = exit_timestamp
        trade.exit_reason = exit_reason
        
        # Calculate P&L
        if trade.direction == "LONG":
            trade.pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            trade.pnl_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100
        
        trade.is_win = trade.pnl_pct > 0
        
        # Update result
        result.trades.append(trade)
        result.total_pnl_pct += trade.pnl_pct
        
        if trade.is_win:
            result.wins += 1
            if trade.model_type == "TREND":
                result.trend_wins += 1
            else:
                result.reversion_wins += 1
        else:
            result.losses += 1
        
        # Clear active trade
        self.active_trade = None
    
    def _calculate_metrics(self, result: BacktestResult):
        """Calculate final performance metrics."""
        if not result.trades:
            return
        
        winning_trades = [t for t in result.trades if t.is_win]
        losing_trades = [t for t in result.trades if not t.is_win]
        
        # Average win/loss
        if winning_trades:
            result.avg_win_pct = sum(t.pnl_pct for t in winning_trades) / len(winning_trades)
        
        if losing_trades:
            result.avg_loss_pct = sum(t.pnl_pct for t in losing_trades) / len(losing_trades)
        
        # Profit factor
        gross_profit = sum(t.pnl_pct for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl_pct for t in losing_trades)) if losing_trades else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # Max drawdown
        equity_curve = []
        cumulative = 0
        for trade in result.trades:
            cumulative += trade.pnl_pct
            equity_curve.append(cumulative)
        
        peak = 0
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        
        result.max_drawdown_pct = max_dd


def print_results(result: BacktestResult):
    """Print backtest results in a nice format."""
    print("\n")
    print("═" * 65)
    print(f"              BACKTEST RESULTS - {result.symbol} {result.timeframe}")
    print("═" * 65)
    print(f"Period:           {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
    print(f"Total Candles:    {result.total_candles:,}")
    print(f"Total Signals:    {result.total_signals}")
    print(f"Trades Executed:  {result.total_trades}")
    print()
    
    if result.total_trades == 0:
        print("No trades executed during this period.")
        print("Try lowering the confidence threshold or using a longer period.")
        print("═" * 65)
        return
    
    print(f"Win Rate:         {result.win_rate:.1f}% ({result.wins} wins / {result.losses} losses)")
    print(f"Profit Factor:    {result.profit_factor:.2f}")
    print(f"Average Win:      +{result.avg_win_pct:.2f}%")
    print(f"Average Loss:     {result.avg_loss_pct:.2f}%")
    print(f"Total Return:     {'+' if result.total_pnl_pct >= 0 else ''}{result.total_pnl_pct:.2f}%")
    print(f"Max Drawdown:     -{result.max_drawdown_pct:.2f}%")
    print()
    print("By Model:")
    if result.trend_trades > 0:
        print(f"  Trend Model:       {result.trend_win_rate:.1f}% win rate ({result.trend_trades} trades)")
    if result.reversion_trades > 0:
        print(f"  Mean Reversion:    {result.reversion_win_rate:.1f}% win rate ({result.reversion_trades} trades)")
    print()
    
    # Show recent trades
    print("Recent Trades:")
    print("-" * 65)
    for trade in result.trades[-10:]:
        dt = datetime.fromtimestamp(trade.timestamp / 1000)
        win_loss = "✓ WIN " if trade.is_win else "✗ LOSS"
        print(f"  {dt.strftime('%m-%d %H:%M')} | {trade.direction:5} | "
              f"{trade.model_type:15} | {win_loss} | {trade.pnl_pct:+.2f}%")
    
    print("═" * 65)


def save_results(result: BacktestResult, filename: str):
    """Save results to a file."""
    import json
    
    data = {
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
        "total_candles": result.total_candles,
        "total_signals": result.total_signals,
        "total_trades": result.total_trades,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "total_pnl_pct": result.total_pnl_pct,
        "avg_win_pct": result.avg_win_pct,
        "avg_loss_pct": result.avg_loss_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "trend_trades": result.trend_trades,
        "trend_win_rate": result.trend_win_rate,
        "reversion_trades": result.reversion_trades,
        "reversion_win_rate": result.reversion_win_rate,
        "trades": [
            {
                "timestamp": t.timestamp,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "pnl_pct": t.pnl_pct,
                "is_win": t.is_win,
                "exit_reason": t.exit_reason,
                "model_type": t.model_type,
                "confidence": t.confidence
            }
            for t in result.trades
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {filename}")


async def save_to_database(result: BacktestResult):
    """
    Save backtest trades to the signals database for ML training.
    This allows the ML model to learn from historical simulated trades.
    """
    from data.storage import storage
    
    # Connect to database
    await storage.connect()
    
    saved_count = 0
    
    for trade in result.trades:
        try:
            # Determine outcome
            outcome = "WIN" if trade.is_win else "LOSS"
            
            # Save to signals table
            signal_id = await storage.save_signal(
                symbol=trade.symbol,
                timeframe=result.timeframe,
                signal_type=trade.model_type,
                direction=trade.direction,
                entry_price=trade.entry_price,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                confidence=trade.confidence,
                model_type=trade.model_type
            )
            
            # Update with outcome
            await storage.update_signal_outcome(
                signal_id=signal_id,
                outcome=outcome,
                outcome_pnl=trade.pnl_pct
            )
            
            saved_count += 1
            
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
    
    await storage.close()
    
    print(f"\n✅ Saved {saved_count} trades to database for ML training")
    print(f"   The OnlineLearningTrainer will use these to improve predictions")


async def main():
    parser = argparse.ArgumentParser(description="Backtest Auction Market Strategy")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--days", type=int, default=30, help="Days of history (default: 30)")
    parser.add_argument("--timeframe", type=str, default="5m", help="Timeframe (default: 5m)")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold (default: 0.7)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--train", action="store_true", help="Save results to database for ML training")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), 
               format="{time:HH:mm:ss} | {level:7} | {message}",
               level="INFO")
    
    print(f"\n🎯 Auction Market Strategy Backtester")
    print(f"Symbol: {args.symbol} | Timeframe: {args.timeframe} | Days: {args.days}")
    print(f"Confidence Threshold: {args.confidence}")
    if args.train:
        print(f"Mode: TRAINING - Results will be saved to database")
    print()
    
    # Run backtest
    backtester = Backtester(
        symbol=args.symbol,
        timeframe=args.timeframe,
        confidence_threshold=args.confidence
    )
    
    result = await backtester.run(days=args.days)
    
    # Close HTTP session
    await historical_fetcher.close()
    
    # Print results
    print_results(result)
    
    # Optionally save to JSON file
    if args.save:
        filename = f"backtest_{args.symbol}_{args.days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(result, filename)
    
    # Save to database for ML training
    if args.train:
        await save_to_database(result)


if __name__ == "__main__":
    asyncio.run(main())

