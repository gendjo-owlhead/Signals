#!/bin/bash
source .venv/bin/activate

# Download 1m data for BTC/USDT for the last 30 days
freqtrade download-data --pairs BTC/USDT:USDT --exchange binance --days 1100 -t 5m 15m 1h
