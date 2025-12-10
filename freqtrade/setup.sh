#!/bin/bash
set -e

echo "🚀 Setting up Freqtrade..."

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel

# Install Freqtrade
echo "Installing freqtrade..."
pip install freqtrade

echo "✅ Freqtrade installed successfully!"
echo "To activate: source freqtrade/.venv/bin/activate"
