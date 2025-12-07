import { useState, useEffect } from 'react'
import { 
  useWebSocket, 
  useAnalysis, 
  useVolumeProfile, 
  useOrderFlow, 
  useSignals, 
  useMLStatus,
  useStatistics,
  useTrading
} from './hooks/useWebSocket'
import SignalPanel from './components/SignalPanel'
import VolumeProfileChart from './components/VolumeProfile'
import OrderFlowPanel from './components/OrderFlowPanel'
import MarketStatePanel from './components/MarketState'
import MLMetrics from './components/MLMetrics'
import TradingPanel from './components/TradingPanel'
import { TechnicalAnalysisWidget, MiniChartWidget, TopStoriesWidget } from './components/TradingViewWidgets'

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT')
  const [symbols] = useState(['BTCUSDT', 'ETHUSDT'])
  
  // WebSocket connection
  const { isConnected, lastMessage } = useWebSocket()
  
  // Data hooks
  const { analysis } = useAnalysis(selectedSymbol)
  const { profile } = useVolumeProfile(selectedSymbol)
  const { orderFlow } = useOrderFlow(selectedSymbol)
  const { signals } = useSignals(selectedSymbol)
  const { status: mlStatus } = useMLStatus()
  const { stats } = useStatistics()
  
  // Trading data hook - REAL data from Binance
  const { 
    account, 
    loading: tradingLoading, 
    startTrading, 
    stopTrading, 
    closePosition 
  } = useTrading()
  
  // Handle WebSocket updates
  const [realtimeData, setRealtimeData] = useState(null)
  
  useEffect(() => {
    if (lastMessage && lastMessage.symbol === selectedSymbol) {
      if (lastMessage.type === 'analysis_update') {
        setRealtimeData(lastMessage.data)
      }
    }
  }, [lastMessage, selectedSymbol])
  
  // Use realtime data if available, otherwise use REST data
  const currentAnalysis = realtimeData || analysis
  
  // Win rate from real trading stats
  const winRate = account?.stats?.win_rate || 0
  
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header__logo">
          <div className="header__logo-icon">📊</div>
          <span className="header__title">Auction Market Signals</span>
        </div>
        
        <div className="symbol-selector">
          {symbols.map(symbol => (
            <button
              key={symbol}
              className={`symbol-btn ${selectedSymbol === symbol ? 'active' : ''}`}
              onClick={() => setSelectedSymbol(symbol)}
            >
              {symbol}
            </button>
          ))}
        </div>
        
        <div className="header__status">
          <div className="status-badge">
            <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
            {isConnected ? 'Live' : 'Reconnecting...'}
          </div>
          
          {account && (
            <>
              <div className="status-badge balance-badge">
                💰 ${account.balance?.toFixed(2) || '0.00'}
              </div>
              <div className="status-badge">
                Win Rate: {winRate.toFixed(1)}%
              </div>
              {account.testnet && (
                <div className="status-badge testnet-badge">
                  TESTNET
                </div>
              )}
            </>
          )}
        </div>
      </header>
      
      {/* Main Content */}
      <main className="main">
        {/* Left Column - Charts/Analysis */}
        <div className="left-column">
          {/* Mini Chart - TradingView */}
          <MiniChartWidget symbol={selectedSymbol} />
          
          {/* Market State */}
          <MarketStatePanel analysis={currentAnalysis} symbol={selectedSymbol} />
          
          {/* Volume Profile */}
          <div className="card fade-in" style={{ gridColumn: '1 / -1' }}>
            <div className="card__header">
              <span className="card__title">Volume Profile</span>
              {profile && (
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  POC: {profile.poc?.toFixed(2)} | VAH: {profile.vah?.toFixed(2)} | VAL: {profile.val?.toFixed(2)}
                </span>
              )}
            </div>
            <VolumeProfileChart profile={profile} analysis={currentAnalysis} />
          </div>
          
          {/* Order Flow */}
          <OrderFlowPanel orderFlow={orderFlow} symbol={selectedSymbol} />
          
          {/* Crypto News */}
          <TopStoriesWidget />
        </div>
        
        {/* Right Column - Signals & Trading */}
        <div className="right-column">
          {/* Trading Panel - REAL DATA */}
          <TradingPanel 
            account={account}
            loading={tradingLoading}
            onStartTrading={startTrading}
            onStopTrading={stopTrading}
            onClosePosition={closePosition}
          />
          
          {/* Technical Analysis - TradingView */}
          <TechnicalAnalysisWidget symbol={selectedSymbol} />
          
          {/* Active Signal */}
          <SignalPanel signals={signals} symbol={selectedSymbol} />
          
          {/* ML Status */}
          <MLMetrics status={mlStatus} stats={stats} />
        </div>
      </main>
    </div>
  )
}

export default App

