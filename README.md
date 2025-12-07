# Auction Market Signal Generator

A Python-based trading signal generator implementing **Fabio Valentini's TradeZella Auction Market Playbook** for Binance crypto futures. Features a modern React frontend dashboard and ML-powered self-improvement.

![Status](https://img.shields.io/badge/status-development-yellow)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![React](https://img.shields.io/badge/react-18.2-61dafb)

## 🎯 Strategy Overview

This system implements the exact TradeZella Auction Market Playbook strategy:

### Core Concepts
- **Volume Profile**: Identifies LVNs (Low Volume Nodes) and POC (Point of Control)
- **Order Flow**: Analyzes buy/sell aggression via footprint-style analysis and CVD
- **Market State**: Determines if market is balanced or out of balance

### Two Trading Models

1. **Trend Model** (Out-of-Balance → New Balance)
   - Market is trending with momentum
   - Entry at LVN retracement with order flow confirmation
   - Target: Prior balance POC
   - Stop Loss: Beyond the LVN zone

2. **Mean Reversion Model** (Failed Breakout → Back to Balance)
   - Price breaks out but fails to hold
   - Entry on reclaim leg LVN pullback
   - Target: Balance POC (center of value)
   - Stop Loss: Beyond the failed breakout

## 🏗️ Architecture

```
Signals/
├── backend/                    # Python backend
│   ├── data/                   # Data layer
│   │   ├── binance_ws.py      # WebSocket real-time data
│   │   ├── historical.py      # Historical data fetcher
│   │   └── storage.py         # SQLite database
│   ├── analysis/              # Analysis engine
│   │   ├── volume_profile.py  # Volume Profile, LVN, POC
│   │   ├── order_flow.py      # CVD, footprint, aggression
│   │   └── market_state.py    # Balance/imbalance detection
│   ├── signals/               # Signal generation
│   │   ├── trend_model.py     # Trend Model implementation
│   │   ├── mean_reversion.py  # Mean Reversion Model
│   │   └── signal_manager.py  # Signal coordination
│   ├── ml/                    # Machine learning
│   │   ├── signal_accuracy.py # Win rate feedback loop
│   │   ├── lvn_patterns.py    # LVN reaction patterns
│   │   ├── state_classifier.py# Market state ML
│   │   └── trainer.py         # Online learning
│   ├── config.py              # Configuration
│   ├── main.py                # FastAPI server
│   └── requirements.txt       # Dependencies
│
└── frontend/                   # React dashboard
    ├── src/
    │   ├── components/        # UI components
    │   ├── hooks/             # Data hooks
    │   └── App.jsx            # Main app
    └── package.json
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ (on macOS, use `python3` and `pip3`)
- Node.js 18+
- Binance account (Testnet recommended for development)

### Backend Setup

```bash
cd backend

# Create virtual environment (required on macOS due to PEP 668)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (inside venv, both 'pip' and 'python' work)
pip install -r requirements.txt

# Configure environment
cp ../.env.example .env
# Edit .env with your Binance API keys

# Run the server
python main.py
```

> **Note for macOS users**: Always activate the virtual environment first with `source venv/bin/activate` before running the backend.

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The dashboard will be available at `http://localhost:3000`

### Running Both Services

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate && python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Troubleshooting

**Port 8000 already in use:**
```bash
# Kill existing process on port 8000
lsof -ti:8000 | xargs kill -9

# Then restart the backend
source venv/bin/activate && python main.py
```

**"command not found: python" on macOS:**
Use `python3` instead, or activate the virtual environment first where `python` is aliased.

## ⚙️ Configuration

Create a `.env` file in the `backend/` directory:

```env
# Binance API
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=true

# Trading
TRADING_PAIRS=["BTCUSDT", "ETHUSDT"]
PRIMARY_TIMEFRAME=5m

# ML
ONLINE_LEARNING_ENABLED=true
```

## 📊 Features

### Real-Time Analysis
- Live Volume Profile with LVN/POC/VAH/VAL levels
- Order flow analysis with CVD and aggression detection
- Market state classification

### Trading Signals
- Automated signal generation based on TradeZella strategy
- Entry, Stop Loss, and Take Profit levels
- Confidence scoring with order flow confirmation

### ML Self-Improvement
- **Signal Accuracy**: Learns from trade outcomes
- **LVN Patterns**: Predicts price reaction at LVN zones
- **State Classifier**: Improves market state detection

### Dashboard
- Premium dark theme with glassmorphism
- Real-time WebSocket updates
- Volume Profile visualization
- ML learning progress display

## 🔌 API Endpoints

### REST API

| Endpoint | Description |
|----------|-------------|
| `GET /api/analysis/{symbol}` | Current analysis snapshot |
| `GET /api/signals` | Active trading signals |
| `GET /api/volume-profile/{symbol}` | Volume Profile data |
| `GET /api/order-flow/{symbol}` | Order flow analysis |
| `GET /api/market-state/{symbol}` | Market state |
| `GET /api/ml/status` | ML learning status |

### WebSocket

Connect to `ws://localhost:8000/ws` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Types: 'analysis_update', 'new_signal', 'heartbeat'
};
```

## 📈 Strategy Implementation

Based on [TradeZella Auction Market Playbook](https://www.tradezella.com/playbooks/auction-market-playbook):

1. **Market State Check** → Balanced or Out of Balance?
2. **Location** → Is price at an LVN zone?
3. **Aggression** → Order flow confirmation (CVD, imbalances, large prints)
4. **Signal Generation** → Entry with defined SL/TP

## ⚠️ Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Always use Testnet for development and testing. Past performance does not guarantee future results.

## 📝 License

MIT License - See LICENSE file for details.
