import json
import sqlite3
import pandas as pd
from datetime import datetime
import asyncio
import sys
import os
import zipfile

# Add backend to path to import backend modules if needed, 
# but for now we will interact directly with the DB to avoid complex dependency injection issues
# or we can use the storage module if we can import it.
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

DB_PATH = '../backend/data/signals.db'
BACKTEST_RESULTS_DIR = 'user_data/backtest_results'

import argparse

def load_specific_file(filepath):
    """Load a specific backtest result file (JSON or ZIP)."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
    print(f"Loading backtest results from: {filepath}")
    
    try:
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as z:
                # Find the JSON file inside
                json_files = [f for f in z.namelist() if f.endswith('.json')]
                if not json_files:
                    print("No JSON file found inside zip.")
                    return None
                
                with z.open(json_files[0]) as f:
                    return json.load(f)
        else:
             with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def load_latest_backtest(directory):
    # Look for ZIP and JSON files
    files = [
        f for f in os.listdir(directory) 
        if (f.endswith('.zip') or f.endswith('.json')) and not f.startswith('.')
    ]
    if not files:
        print("No backtest results found (checked for .zip and .json).")
        return None
    
    # Sort by modification time
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    latest_file = files[0]
    
    return load_specific_file(os.path.join(directory, latest_file))

def ingest_trades_to_db(backtest_data):
    """
    Ingest Freqtrade trades as Signals with Outcomes into the database.
    This mimics the bot having taken these trades, allowing the ML model to learn.
    """
    if not backtest_data:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Ensure tables exist (basic check)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
    if not cursor.fetchone():
        print("Error: 'signals' table not found in database. Run the backend first to initialize DB.")
        return

    strategy_name = list(backtest_data['strategy'].keys())[0]
    trades = backtest_data['strategy'][strategy_name]['trades']
    
    print(f"Found {len(trades)} trades to ingest.")
    
    ingested_count = 0
    
    for trade in trades:
        # Freqtrade trade structure:
        # { ... }
        
        # Freqtrade pair: BTC/USDT:USDT -> BTCUSDT
        symbol = trade['pair'].split(":")[0].replace("/", "")
        timestamp = int(pd.Timestamp(trade['open_date']).timestamp() * 1000) # MS timestamp
        
        # Determine direction
        if 'trade_direction' in trade:
            direction = "LONG" if trade['trade_direction'] == "long" else "SHORT"
        elif 'is_short' in trade:
            direction = "SHORT" if trade['is_short'] else "LONG"
        else:
            direction = "LONG" # Default assumption
            
        direction = direction.upper()
        
        entry_price = trade['open_rate']
        pnl_percent = trade['profit_ratio'] * 100
        
        outcome = "WIN" if pnl_percent > 0 else "LOSS"
        if abs(pnl_percent) < 0.05: # Breakeven threshold
            outcome = "BE"
            
        # Deduplication Check
        # Check if we already have a signal for this symbol at this timestamp (approx)
        cursor.execute("""
            SELECT id FROM signals 
            WHERE symbol = ? AND timestamp = ? AND model_type = 'SCALPER'
        """, (symbol, timestamp))
        
        if cursor.fetchone():
            continue # Skip existing
            
        # Insert Signal
        # Scheme: symbol, signal_type, direction, timeframe, timestamp, entry_price, 
        # stop_loss, take_profit, confidence, outcome, outcome_pnl, created_at, model_type
        
        cursor.execute("""
            INSERT INTO signals (
                symbol, signal_type, direction, timeframe, timestamp, 
                entry_price, stop_loss, take_profit, confidence, 
                outcome, outcome_pnl, created_at, model_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            "SCALPER_BACKTEST",
            direction,
            "1m",
            timestamp,
            entry_price,
            entry_price * 0.99, # Dummy SL
            entry_price * 1.01, # Dummy TP
            0.8, # Dummy Confidence
            outcome,
            pnl_percent,
            datetime.now().isoformat(),
            "SCALPER"
        ))
        
        ingested_count += 1
        
    conn.commit()
    conn.close()
    
    print(f"Successfully ingested {ingested_count} historical trades into the ML database.")
    print("The Online Trainer will pick these up on its next cycle (or restart).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export backtest results to ML database')
    parser.add_argument('--file', type=str, help='Specific backtest result file to ingest')
    args = parser.parse_args()
    
    if args.file:
        backtest_data = load_specific_file(args.file)
    else:
        backtest_data = load_latest_backtest(BACKTEST_RESULTS_DIR)
        
    if backtest_data:
        ingest_trades_to_db(backtest_data)
