#!/bin/bash
source .venv/bin/activate

# Run backtest
# --export trades : Save trades to JSON for ML ingestion
freqtrade backtesting --config user_data/config.json --strategy MultiTimeframeStrategy --timeframe 5m --export trades

# Ingest results to ML Bot
echo "🧠 Ingesting backtest results into ML Bot..."
python3 export_to_ml.py

