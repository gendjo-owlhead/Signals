"""
FastAPI main application for Auction Market Signal Generator.
Provides REST API and WebSocket endpoints for real-time signals.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
from datetime import datetime
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn

from config import settings
from data import binance_ws, storage, historical_fetcher
from signals import signal_manager, SignalUpdate
from ml import online_trainer
from trading import order_executor, position_manager, risk_manager, binance_trader


# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("Starting Auction Market Signal Generator...")
    
    # Initialize database
    await storage.connect()
    
    # Start Binance WebSocket streams
    asyncio.create_task(binance_ws.start())
    
    # Wait for initial data
    await asyncio.sleep(3)
    
    # Start signal manager
    asyncio.create_task(signal_manager.start())
    
    # Register broadcast callback
    signal_manager.on_update(broadcast_update)
    
    # Start ML trainer
    if settings.online_learning_enabled:
        await online_trainer.start()
    
    # Start trading engine (if enabled)
    if settings.trading_enabled:
        await order_executor.start()
        risk_manager.kill_switch(True)  # Enable trading on startup
        logger.info("Trading Engine ENABLED - Will execute trades!")
    else:
        logger.info("Trading Engine disabled - Signal-only mode")
    
    logger.info("All systems initialized!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    
    # Stop trading first
    if settings.trading_enabled:
        await order_executor.stop()
        await binance_trader.close()
    
    await signal_manager.stop()
    await binance_ws.stop()
    await storage.close()
    await historical_fetcher.close()
    
    if settings.online_learning_enabled:
        await online_trainer.stop()


# Broadcast callback
async def broadcast_update(update: SignalUpdate):
    """Broadcast signal updates to all WebSocket clients."""
    await manager.broadcast({
        'type': update.type,
        'timestamp': update.timestamp,
        'symbol': update.symbol,
        'data': update.data
    })


# Create FastAPI app
app = FastAPI(
    title="Auction Market Signal Generator",
    description="Real-time trading signals based on TradeZella Auction Market Playbook",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== REST API Routes ==============

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "pairs": settings.trading_pairs,
        "timeframe": settings.primary_timeframe
    }


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return {
        "trading_pairs": settings.trading_pairs,
        "timeframes": settings.timeframes,
        "primary_timeframe": settings.primary_timeframe,
        "testnet": settings.binance_testnet,
        "ml_enabled": settings.online_learning_enabled
    }


@app.get("/api/analysis/{symbol}")
async def get_analysis(symbol: str):
    """Get current analysis snapshot for a symbol."""
    snapshot = signal_manager.get_snapshot(symbol)
    
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"No analysis available for {symbol}")
    
    return snapshot.to_dict()


@app.get("/api/signals")
async def get_signals(symbol: Optional[str] = None):
    """Get active signals."""
    signals = signal_manager.get_active_signals(symbol)
    return {"signals": signals, "count": len(signals)}


@app.get("/api/signals/history/{symbol}")
async def get_signal_history(symbol: str, limit: int = 50):
    """Get signal history for a symbol."""
    history = signal_manager.get_signal_history(symbol, limit)
    return {"history": history, "count": len(history)}


@app.get("/api/statistics")
async def get_statistics(symbol: Optional[str] = None):
    """Get signal performance statistics."""
    stats = await storage.get_signal_statistics(symbol)
    return stats


@app.get("/api/volume-profile/{symbol}")
async def get_volume_profile(symbol: str, periods: int = 100):
    """Get Volume Profile data for a symbol."""
    from analysis import volume_profile_calculator
    
    klines = binance_ws.get_klines(symbol, settings.primary_timeframe, periods)
    
    if not klines:
        raise HTTPException(status_code=404, detail=f"No kline data for {symbol}")
    
    vp = volume_profile_calculator.calculate_from_klines(klines)
    
    return {
        "symbol": symbol,
        "poc": vp.poc_price,
        "vah": vp.vah,
        "val": vp.val,
        "profile_high": vp.profile_high,
        "profile_low": vp.profile_low,
        "total_volume": vp.total_volume,
        "lvn_zones": [{"price": p, "volume": v} for p, v in vp.lvn_zones],
        "hvn_zones": [{"price": p, "volume": v} for p, v in vp.hvn_zones],
        "levels": [
            {"price": l.price, "volume": l.volume, "delta": l.delta}
            for l in vp.levels
        ]
    }


@app.get("/api/order-flow/{symbol}")
async def get_order_flow(symbol: str):
    """Get order flow analysis for a symbol."""
    from analysis import order_flow_analyzer
    
    trades = binance_ws.get_recent_trades(symbol, 500)
    
    if not trades:
        raise HTTPException(status_code=404, detail=f"No trade data for {symbol}")
    
    # CVD pressure
    cvd_pressure = order_flow_analyzer.get_cvd_pressure(symbol)
    
    # Aggression analysis
    buy_agg = order_flow_analyzer.analyze_aggression(trades, symbol, "BUY")
    sell_agg = order_flow_analyzer.analyze_aggression(trades, symbol, "SELL")
    
    return {
        "symbol": symbol,
        "cvd": cvd_pressure,
        "buy_aggression": {
            "strength": buy_agg.strength,
            "imbalances": buy_agg.imbalance_count,
            "large_prints": buy_agg.large_prints_count,
            "description": buy_agg.description
        },
        "sell_aggression": {
            "strength": sell_agg.strength,
            "imbalances": sell_agg.imbalance_count,
            "large_prints": sell_agg.large_prints_count,
            "description": sell_agg.description
        }
    }


@app.get("/api/market-state/{symbol}")
async def get_market_state(symbol: str):
    """Get market state analysis for a symbol."""
    from analysis import market_state_analyzer
    
    klines = binance_ws.get_klines(symbol, settings.primary_timeframe, 100)
    
    if not klines:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    
    analysis = market_state_analyzer.analyze(klines)
    
    return {
        "symbol": symbol,
        "state": analysis.state.value,
        "confidence": analysis.confidence,
        "is_balanced": analysis.is_balanced,
        "balance_score": analysis.balance_score,
        "momentum": analysis.momentum,
        "momentum_strength": analysis.momentum_strength,
        "poc": analysis.poc,
        "vah": analysis.vah,
        "val": analysis.val,
        "direction": analysis.direction
    }


@app.get("/api/ml/status")
async def get_ml_status():
    """Get ML learning status."""
    return online_trainer.get_learning_status()


@app.get("/api/ml/metrics")
async def get_ml_metrics():
    """Get detailed ML metrics."""
    return online_trainer.get_ml_metrics()


@app.get("/api/ml/approver-status")
async def get_approver_status():
    """Get trade approver status including model weights and performance."""
    from ml.trade_approver import trade_approver
    return trade_approver.get_status()


@app.get("/api/ml/recent-decisions")
async def get_recent_decisions(limit: int = 20):
    """Get recent approval/rejection decisions with reasoning."""
    from ml.trade_approver import trade_approver
    return {
        "decisions": trade_approver.get_recent_decisions(limit),
        "count": len(trade_approver.recent_decisions)
    }


@app.get("/api/ml/feedback-stats")
async def get_feedback_stats():
    """Get feedback loop statistics and model performance."""
    from ml.feedback_loop import feedback_loop
    return {
        "stats": feedback_loop.get_stats(),
        "performance": feedback_loop.get_performance_summary(),
        "recent_outcomes": feedback_loop.get_recent_outcomes(10)
    }


@app.delete("/api/signals/{symbol}")
async def clear_signals(symbol: str):
    """Clear active signals for a symbol."""
    await signal_manager.clear_signal(symbol)
    return {"status": "cleared", "symbol": symbol}


# ============== Trading API Routes ==============

@app.get("/api/trading/status")
async def get_trading_status():
    """Get trading engine status."""
    return order_executor.get_status()


@app.get("/api/trading/positions")
async def get_positions():
    """Get all open positions."""
    positions = position_manager.get_all_positions()
    return {
        "positions": [p.to_dict() for p in positions],
        "count": len(positions),
        "stats": position_manager.get_stats()
    }


@app.post("/api/trading/start")
async def start_trading():
    """Enable trading execution."""
    if not settings.trading_enabled:
        return {"status": "error", "message": "Trading not enabled in config. Set TRADING_ENABLED=true"}
    
    risk_manager.kill_switch(True)
    if not order_executor.is_running:
        await order_executor.start()
    
    return {"status": "started", "message": "Trading engine started"}


@app.post("/api/trading/stop")
async def stop_trading():
    """Emergency stop - disable all trading."""
    risk_manager.kill_switch(False)
    logger.warning("EMERGENCY STOP - Trading disabled via API")
    return {"status": "stopped", "message": "Trading disabled"}


@app.post("/api/trading/close/{position_id}")
async def close_position(position_id: str):
    """Manually close a position."""
    success, message = await order_executor.close_position_manually(position_id)
    return {"success": success, "message": message}


@app.get("/api/trading/risk")
async def get_risk_status():
    """Get risk manager status."""
    return risk_manager.get_status()


@app.get("/api/trading/account")
async def get_account_info():
    """Get live account info from Binance (balance, exchange positions)."""
    try:
        # Get USDT balance
        balance = await binance_trader.get_account_balance()
        
        # Get positions from Binance exchange
        exchange_positions = await binance_trader.get_open_positions()
        
        # Get bot-tracked positions
        bot_positions = position_manager.get_all_positions()
        
        # Get stats
        stats = position_manager.get_stats()
        
        return {
            "balance": balance,
            "exchange_positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "leverage": p.leverage
                }
                for p in exchange_positions
            ],
            "bot_positions": [p.to_dict() for p in bot_positions],
            "stats": stats,
            "risk": risk_manager.get_status(),
            "testnet": settings.binance_testnet,
            "trading_enabled": settings.trading_enabled
        }
    except Exception as e:
        logger.error(f"Failed to fetch account info: {e}")
        return {
            "error": str(e),
            "balance": 0,
            "exchange_positions": [],
            "bot_positions": [],
            "stats": {},
            "testnet": settings.binance_testnet,
            "trading_enabled": settings.trading_enabled
        }


# ============== Backtest API Routes ==============

@app.get("/api/backtest/results")
async def get_backtest_results():
    """Get latest backtest results from freqtrade directory."""
    import os
    import zipfile
    from pathlib import Path
    
    backtest_dir = Path(__file__).parent.parent / "freqtrade" / "user_data" / "backtest_results"
    
    if not backtest_dir.exists():
        return {"error": "Backtest results directory not found", "summary": None}
    
    # Find the most recent backtest result
    zip_files = sorted(backtest_dir.glob("backtest-result-*.zip"), reverse=True)
    
    if not zip_files:
        return {"error": "No backtest results found", "summary": None}
    
    latest_zip = zip_files[0]
    
    try:
        with zipfile.ZipFile(latest_zip, 'r') as z:
            # Find the JSON result file inside
            json_files = [f for f in z.namelist() if f.endswith('.json') and 'meta' not in f]
            
            if not json_files:
                return {"error": "No result data in backtest file", "summary": None}
            
            with z.open(json_files[0]) as f:
                result_data = json.load(f)
        
        # Extract strategy data
        strategy_name = list(result_data.get('strategy', {}).keys())[0] if result_data.get('strategy') else 'unknown'
        strategy_data = result_data.get('strategy', {}).get(strategy_name, {})
        
        # Build summary
        total_trades = strategy_data.get('total_trades', 0)
        wins = strategy_data.get('wins', 0)
        losses = strategy_data.get('losses', 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        profit_total = strategy_data.get('profit_total', 0)
        profit_factor = strategy_data.get('profit_factor', 0)
        max_drawdown = strategy_data.get('max_drawdown_abs', 0)
        
        summary = {
            "strategy": strategy_name,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "net_profit": profit_total * 100,  # Convert to percentage
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown
        }
        
        # Generate suggestions based on metrics
        suggestions = []
        
        if win_rate < 50:
            suggestions.append("Consider tightening entry conditions - win rate below 50%")
        if profit_factor and profit_factor < 1.5:
            suggestions.append("Increase R:R ratio or improve exit timing - profit factor below 1.5")
        if total_trades < 10:
            suggestions.append("Extend backtest period for more reliable statistics")
        if win_rate >= 60 and profit_factor >= 2:
            suggestions.append("✅ Strong performance! Consider live testing with small position size")
        if abs(max_drawdown) > 20:
            suggestions.append("⚠️ High drawdown detected - review risk management parameters")
        
        return {
            "summary": summary,
            "latest_backtest": {
                "file": latest_zip.name,
                "timestamp": latest_zip.stem.replace("backtest-result-", "")
            },
            "suggestions": suggestions
        }
        
    except Exception as e:
        logger.error(f"Failed to parse backtest results: {e}")
        return {"error": str(e), "summary": None}


# ============== Test/Debug Routes ==============

@app.post("/api/test/signal")
async def create_test_signal(symbol: str = "BTCUSDT", direction: str = "LONG"):
    """
    [TESTING ONLY] Create a manual test signal to verify order execution.
    DELETE THIS IN PRODUCTION!
    """
    from signals.trend_model import TrendSignal
    from analysis.order_flow import AggressionSignal
    from datetime import datetime
    
    # Get current price
    klines = binance_ws.get_klines(symbol, settings.primary_timeframe, 5)
    if not klines:
        return {"error": f"No price data for {symbol}"}
    
    current_price = klines[-1].close
    
    # Calculate SL/TP based on direction
    if direction.upper() == "LONG":
        stop_loss = current_price * 0.995  # 0.5% SL
        take_profit = current_price * 1.01  # 1% TP
        impulse_start = current_price * 0.99
        impulse_end = current_price
    else:
        stop_loss = current_price * 1.005  # 0.5% SL
        take_profit = current_price * 0.99  # 1% TP
        impulse_start = current_price * 1.01
        impulse_end = current_price
    
    # Create mock aggression signal
    mock_aggression = AggressionSignal(
        timestamp=int(datetime.now().timestamp() * 1000),
        symbol=symbol,
        direction="BUY" if direction.upper() == "LONG" else "SELL",
        strength=0.8,
        cvd_confirming=True,
        imbalance_count=3,
        large_prints_count=2,
        description="Test signal aggression"
    )
    
    # Create test signal with correct fields
    test_signal = TrendSignal(
        timestamp=int(datetime.now().timestamp() * 1000),
        symbol=symbol,
        timeframe=settings.primary_timeframe,
        direction=direction.upper(),
        entry_price=current_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        lvn_price=current_price,
        poc_target=take_profit,
        impulse_start=impulse_start,
        impulse_end=impulse_end,
        confidence=0.75,
        aggression=mock_aggression,
        market_state="trending_up" if direction.upper() == "LONG" else "trending_down",
        risk_reward=2.0,
        risk_percent=0.5
    )
    
    # Process through signal manager
    await signal_manager._handle_new_signal(symbol, test_signal)
    
    logger.warning(f"TEST SIGNAL CREATED: {symbol} {direction} @ {current_price}")
    
    return {
        "status": "created",
        "signal": {
            "symbol": symbol,
            "direction": direction.upper(),
            "entry": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": 0.75
        },
        "message": "Test signal created. Check trading panel for execution."
    }


@app.get("/api/test/why-no-signals/{symbol}")
async def diagnose_signals(symbol: str):
    """
    Debug endpoint to see why signals aren't being generated.
    """
    from analysis import market_state_analyzer, volume_profile_calculator, order_flow_analyzer
    
    klines = binance_ws.get_klines(symbol, settings.primary_timeframe, 100)
    trades = binance_ws.get_recent_trades(symbol, 500)
    
    if not klines:
        return {"error": "No kline data"}
    
    # Market state
    market_analysis = market_state_analyzer.analyze(klines)
    
    # Volume profile
    vp = volume_profile_calculator.calculate_from_klines(klines)
    
    # Check conditions
    current_price = klines[-1].close
    
    # Find nearest LVN
    nearest_lvn = None
    lvn_distance = None
    if vp.lvn_zones:
        nearest_lvn = min(vp.lvn_zones, key=lambda x: abs(x[0] - current_price))
        lvn_distance = abs(current_price - nearest_lvn[0]) / current_price * 100
    
    # Aggression
    buy_agg = order_flow_analyzer.analyze_aggression(trades, symbol, "BUY") if trades else None
    sell_agg = order_flow_analyzer.analyze_aggression(trades, symbol, "SELL") if trades else None
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "checks": {
            "1_market_out_of_balance": {
                "required": True,
                "actual": not market_analysis.is_balanced,
                "state": market_analysis.state.value,
                "confidence": float(market_analysis.confidence)
            },
            "2_lvn_zones_exist": {
                "required": True,
                "actual": len(vp.lvn_zones) > 0,
                "count": len(vp.lvn_zones)
            },
            "3_price_at_lvn": {
                "required": "< 0.5%",
                "actual": f"{lvn_distance:.2f}%" if lvn_distance else "N/A",
                "passes": lvn_distance < 0.5 if lvn_distance else False,
                "nearest_lvn": nearest_lvn[0] if nearest_lvn else None
            },
            "4_aggression": {
                "required": "> 0.4",
                "buy_strength": float(buy_agg.strength) if buy_agg else 0,
                "sell_strength": float(sell_agg.strength) if sell_agg else 0
            }
        },
        "thresholds": {
            "trend_confidence": settings.trend_confidence_threshold,
            "reversion_confidence": settings.reversion_confidence_threshold
        }
    }


# ============== WebSocket Routes ==============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        # Send initial state
        for symbol in settings.trading_pairs:
            snapshot = signal_manager.get_snapshot(symbol)
            if snapshot:
                await websocket.send_json({
                    "type": "initial_state",
                    "symbol": symbol,
                    "data": snapshot.to_dict()
                })
        
        # Keep connection alive and handle messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                message = json.loads(data)
                
                # Handle client messages
                if message.get("type") == "subscribe":
                    symbol = message.get("symbol")
                    logger.info(f"Client subscribed to {symbol}")
                
                elif message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now().isoformat()})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
