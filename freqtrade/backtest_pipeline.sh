#!/bin/bash

# Configuration
PAIRS=("BTC/USDT" "ETH/USDT" "SOL/USDT" "BNB/USDT" "XRP/USDT")
TIMEFRAMES=("5m" "15m" "1h" "4h")
START_DATE="20200101"
DAYS=1825 # Approx 5 years

echo "🚀 Starting Comprehensive Backtest Pipeline"
echo "-----------------------------------------"
echo "Pairs: ${PAIRS[*]}"
echo "Timeframes: ${TIMEFRAMES[*]}"
echo "Duration: 5 Years ($DAYS days)"
echo "-----------------------------------------"

# Activate Virtual Environment
source .venv/bin/activate || source venv/bin/activate || echo "⚠️ Could not activate venv"

# 1. Download Data
echo "📦 Downloading Historical Data..."
for PAIR in "${PAIRS[@]}"; do
    for TF in "${TIMEFRAMES[@]}"; do
        echo "   -> Downloading $PAIR for $TF..."
        freqtrade download-data \
            --pairs $PAIR \
            --exchange binance \
            --timeframes $TF \
            --days $DAYS \
            --data-format-ohlcv result \
            --timerange ${START_DATE}-
    done
done

# 2. Run Backtests
echo "🔬 Running Backtests..."
for PAIR in "${PAIRS[@]}"; do
    # We run strategies on the primary 5m timeframe, but they use 15m/1h/4h internally
    # So we mainly iterate pairs here for the backtest run
    
    # Clean pair name for filename (remove /)
    PAIR_CLEAN=${PAIR/\//}
    
    echo "   -> Backtesting $PAIR on 5m (MultiTimeframeStrategy)..."
    freqtrade backtesting \
        --config user_data/config.json \
        --strategy MultiTimeframeStrategy \
        --timeframe 5m \
        --pairs $PAIR \
        --timerange ${START_DATE}- \
        --export trades \
        --export-filename user_data/backtest_results/backtest_${PAIR_CLEAN}_5m.json
        
    # Ingest results immediately after each backtest
    echo "   🧠 Ingesting $PAIR results into ML..."
    python3 export_to_ml.py --file user_data/backtest_results/backtest_${PAIR_CLEAN}_5m.json
done

echo "✅ Pipeline Complete!"
