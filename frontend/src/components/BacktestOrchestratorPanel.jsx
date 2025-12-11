import { useState } from 'react'
import { useBacktestOrchestrator } from '../hooks/useWebSocket'

/**
 * Backtest Orchestrator Panel - Shows continuous backtest status and ML training progress
 */
function BacktestOrchestratorPanel() {
  const { status, performance, loading, startOrchestrator, stopOrchestrator } = useBacktestOrchestrator()
  const [expanded, setExpanded] = useState(false)
  
  if (loading && !status) {
    return (
      <div className="card fade-in">
        <div className="card__header">
          <span className="card__title">🔄 ML Training Pipeline</span>
        </div>
        <div className="card__body" style={{ textAlign: 'center', padding: '2rem' }}>
          Loading orchestrator status...
        </div>
      </div>
    )
  }
  
  const isRunning = status?.running || false
  const currentBacktests = status?.current_backtests || []
  const pendingBacktests = status?.pending_backtests || []
  const recentCompleted = status?.recent_completed || []
  const overallWinRate = performance?.overall?.avg_win_rate || 0
  
  // Get performance by symbol
  const symbolPerformance = performance?.by_symbol || {}
  
  return (
    <div className="card fade-in">
      <div className="card__header">
        <span className="card__title">🔄 ML Training Pipeline</span>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className={`status-badge ${isRunning ? 'status-badge--success' : 'status-badge--warning'}`}>
            {isRunning ? '● Running' : '○ Stopped'}
          </span>
          <button 
            className={`btn btn--sm ${isRunning ? 'btn--danger' : 'btn--primary'}`}
            onClick={isRunning ? stopOrchestrator : startOrchestrator}
          >
            {isRunning ? 'Stop' : 'Start'}
          </button>
        </div>
      </div>
      
      <div className="card__body">
        {/* Overall Win Rate Target */}
        <div className="orchestrator-target">
          <div className="target-label">Target Win Rate: 60-70%</div>
          <div className="target-progress">
            <div 
              className="target-bar" 
              style={{ 
                width: `${Math.min(overallWinRate, 100)}%`,
                background: overallWinRate >= 60 
                  ? 'var(--color-success)' 
                  : overallWinRate >= 50 
                    ? 'var(--color-warning)' 
                    : 'var(--color-danger)'
              }}
            />
            <div className="target-marker" style={{ left: '60%' }} />
            <div className="target-marker" style={{ left: '70%' }} />
          </div>
          <div className="target-current">
            Current: <strong>{overallWinRate.toFixed(1)}%</strong>
          </div>
        </div>
        
        {/* Currently Running */}
        {currentBacktests.length > 0 && (
          <div className="orchestrator-section">
            <h4>🏃 Running</h4>
            {currentBacktests.map((bt, i) => (
              <div key={i} className="backtest-item backtest-item--running">
                <span className="backtest-symbol">{bt.symbol}</span>
                <span className="backtest-timeframe">{bt.timeframe}</span>
                <span className="backtest-days">{bt.days}d</span>
                <div className="backtest-progress">
                  <div 
                    className="backtest-progress-bar" 
                    style={{ width: `${bt.progress_pct || 0}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
        
        {/* Win Rates by Symbol/Timeframe */}
        <div className="orchestrator-section">
          <h4 onClick={() => setExpanded(!expanded)} style={{ cursor: 'pointer' }}>
            📊 Performance by Symbol {expanded ? '▼' : '▶'}
          </h4>
          
          {expanded && Object.entries(symbolPerformance).map(([symbol, data]) => (
            <div key={symbol} className="symbol-performance">
              <div className="symbol-header">
                <span className="symbol-name">{symbol}</span>
                <span className="symbol-avg">Avg: {(data.avg_win_rate || 0).toFixed(1)}%</span>
              </div>
              <div className="timeframe-grid">
                {['1m', '5m', '15m', '1h'].map(tf => {
                  const tfData = data.timeframes?.[tf]
                  const winRate = tfData?.win_rate || 0
                  return (
                    <div key={tf} className="timeframe-cell">
                      <span className="tf-label">{tf}</span>
                      <span 
                        className="tf-value"
                        style={{ 
                          color: winRate >= 60 ? 'var(--color-success)' : 
                                 winRate >= 50 ? 'var(--color-warning)' : 'var(--color-danger)'
                        }}
                      >
                        {winRate > 0 ? `${winRate.toFixed(1)}%` : '—'}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
        
        {/* Pending Queue */}
        {pendingBacktests.length > 0 && (
          <div className="orchestrator-section">
            <h4>⏳ Queue ({pendingBacktests.length})</h4>
            <div className="pending-list">
              {pendingBacktests.slice(0, 4).map((bt, i) => (
                <span key={i} className="pending-item">
                  {bt.symbol} {bt.timeframe}
                </span>
              ))}
              {pendingBacktests.length > 4 && (
                <span className="pending-more">+{pendingBacktests.length - 4} more</span>
              )}
            </div>
          </div>
        )}
        
        {/* Recent Completed */}
        {recentCompleted.length > 0 && (
          <div className="orchestrator-section">
            <h4>✅ Recent ({recentCompleted.length})</h4>
            <div className="completed-list">
              {recentCompleted.slice(-3).reverse().map((bt, i) => (
                <div key={i} className="completed-item">
                  <span>{bt.symbol} {bt.timeframe}</span>
                  <span 
                    className="completed-winrate"
                    style={{ 
                      color: bt.win_rate >= 60 ? 'var(--color-success)' : 
                             bt.win_rate >= 50 ? 'var(--color-warning)' : 'var(--color-danger)'
                    }}
                  >
                    {bt.win_rate?.toFixed(1)}%
                  </span>
                  <span className="completed-trades">{bt.total_trades} trades</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Matrix Info */}
        <div className="orchestrator-footer">
          <span>Matrix: {status?.matrix_size || 8} configs</span>
          <span>Completed: {status?.total_completed || 0}</span>
        </div>
      </div>
    </div>
  )
}

export default BacktestOrchestratorPanel
