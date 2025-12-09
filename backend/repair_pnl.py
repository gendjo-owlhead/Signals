import json
import sqlite3
from pathlib import Path

def patch_records():
    print("🩹 Starting PnL Repair...")
    
    # Paths
    base_dir = Path("data/models")
    history_file = base_dir / "trade_history.json"
    risk_file = base_dir / "risk_stats.json"
    db_file = Path("data/signals.db")
    
    # 1. Patch Trade History
    if history_file.exists():
        history = json.loads(history_file.read_text())
        # Find the last ETHUSDT trade
        target_trade = None
        for t in reversed(history):
            if t['symbol'] == 'ETHUSDT' and t['status'] == 'CLOSED_MANUAL': # Or whatever status it got
                target_trade = t
                break
        
        # If not found by status, just take the last ETHUSDT
        if not target_trade:
            for t in reversed(history):
                if t['symbol'] == 'ETHUSDT':
                    target_trade = t
                    break
        
        if target_trade:
            print(f"Found trade: {target_trade['id']}")
            print(f"Current PnL: {target_trade.get('realized_pnl')}")
            
            # Update PnL
            # Entry 3352, Exit ~3337.05, Qty 0.033. SHORT.
            # Profit = (Entry - Exit) * Qty = (3352 - 3337.05) * 0.033 = 0.49335
            actual_pnl = 0.49
            target_trade['realized_pnl'] = actual_pnl
            target_trade['exit_price'] = 3337.05
            target_trade['status'] = 'CLOSED_MANUAL_WIN'
            
            history_file.write_text(json.dumps(history, indent=2))
            print(f"✅ Trade history updated. New PnL: {actual_pnl}")
        else:
            print("❌ Could not find target trade in history.")
    
    # 2. Patch Risk Stats
    if risk_file.exists():
        stats = json.loads(risk_file.read_text())
        print(f"Current Daily PnL: {stats.get('realized_pnl')}")
        
        # Add the profit (assuming it was 0 before)
        stats['realized_pnl'] += 0.49
        stats['wins'] += 1
        # trades_count probably already incremented by executor
        
        risk_file.write_text(json.dumps(stats, indent=2))
        print(f"✅ Risk stats updated. New Daily PnL: {stats['realized_pnl']}")
        
    # 3. Patch Database (optional, for ML)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Find the signal associated with this trade (if any)
    # We can assume it's the last ETHUSDT signal
    cursor.execute("SELECT id, outcome FROM signals WHERE symbol='ETHUSDT' ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    if row:
        sig_id, outcome = row
        print(f"Found DB signal {sig_id}. Outcome: {outcome}")
        
        new_outcome = "WIN"
        cursor.execute("UPDATE signals SET outcome=?, pnl=? WHERE id=?", (new_outcome, 0.49, sig_id))
        conn.commit()
        print(f"✅ Database updated. Signal {sig_id} -> WIN")
    
    conn.close()
    print("✨ All patches applied.")

if __name__ == "__main__":
    patch_records()
