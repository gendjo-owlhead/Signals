import { useState } from 'react';

/**
 * Trading Panel - Shows real account data from Binance
 * Displays: Balance, Open Positions, Win Rate, P&L
 */
function TradingPanel({ account, loading, onStartTrading, onStopTrading, onClosePosition }) {
  const [actionLoading, setActionLoading] = useState(false);
  
  if (loading && !account) {
    return (
      <div className="card fade-in">
        <div className="card__header">
          <span className="card__title">💼 Trading Account</span>
        </div>
        <div className="loading-state">Loading account data...</div>
      </div>
    );
  }
  
  const balance = account?.balance || 0;
  const exchangePositions = account?.exchange_positions || [];
  const botPositions = account?.bot_positions || [];
  const stats = account?.stats || {};
  const risk = account?.risk || {};
  const isTestnet = account?.testnet;
  const tradingEnabled = account?.trading_enabled;
  const isKillSwitchOn = risk?.is_enabled;
  
  const winRate = stats.win_rate || 0;
  const totalTrades = stats.total_trades || 0;
  const wins = stats.wins || 0;
  const losses = stats.losses || 0;
  const dailyPnl = risk?.daily_pnl || 0;
  
  const handleStartStop = async () => {
    setActionLoading(true);
    if (isKillSwitchOn) {
      await onStopTrading();
    } else {
      await onStartTrading();
    }
    setActionLoading(false);
  };
  
  const handleClosePosition = async (positionId) => {
    setActionLoading(true);
    await onClosePosition(positionId);
    setActionLoading(false);
  };
  
  // Combine exchange and bot positions (prioritize bot positions for TP/SL info)
  const allPositions = [...exchangePositions];
  
  return (
    <div className="card fade-in trading-panel">
      <div className="card__header">
        <span className="card__title">💼 Trading Account</span>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {isTestnet && (
            <span className="badge badge--warning">TESTNET</span>
          )}
          {tradingEnabled ? (
            <span className="badge badge--success">AUTO-TRADE ON</span>
          ) : (
            <span className="badge badge--muted">AUTO-TRADE OFF</span>
          )}
        </div>
      </div>
      
      {/* Balance & Stats Row */}
      <div className="trading-stats-grid">
        <div className="stat-box">
          <div className="stat-box__label">Balance (USDT)</div>
          <div className="stat-box__value" style={{ color: 'var(--accent)' }}>
            ${balance.toFixed(2)}
          </div>
        </div>
        
        <div className="stat-box">
          <div className="stat-box__label">Win Rate</div>
          <div className="stat-box__value" style={{ color: winRate >= 50 ? 'var(--success)' : 'var(--danger)' }}>
            {winRate.toFixed(1)}%
          </div>
        </div>
        
        <div className="stat-box">
          <div className="stat-box__label">Trades (W/L)</div>
          <div className="stat-box__value">
            <span style={{ color: 'var(--success)' }}>{wins}</span>
            <span style={{ color: 'var(--text-muted)' }}> / </span>
            <span style={{ color: 'var(--danger)' }}>{losses}</span>
          </div>
        </div>
        
        <div className="stat-box">
          <div className="stat-box__label">Daily P&L</div>
          <div className="stat-box__value" style={{ color: dailyPnl >= 0 ? 'var(--success)' : 'var(--danger)' }}>
            {dailyPnl >= 0 ? '+' : ''}${dailyPnl.toFixed(2)}
          </div>
        </div>
      </div>
      
      {/* Trading Control */}
      {tradingEnabled && (
        <div className="trading-controls">
          <button 
            className={`trading-btn ${isKillSwitchOn ? 'trading-btn--stop' : 'trading-btn--start'}`}
            onClick={handleStartStop}
            disabled={actionLoading}
          >
            {actionLoading ? '...' : (isKillSwitchOn ? '⏹ STOP TRADING' : '▶ START TRADING')}
          </button>
        </div>
      )}
      
      {/* Open Positions */}
      <div className="positions-section">
        <h4 className="section-title">
          Open Positions ({allPositions.length})
        </h4>
        
        {allPositions.length === 0 ? (
          <div className="empty-state">No open positions</div>
        ) : (
          <div className="positions-list">
            {allPositions.map((pos, idx) => {
              // Find matching bot position for TP/SL info
              const botPos = botPositions.find(bp => bp.symbol === pos.symbol);
              
              return (
                <div key={`${pos.symbol}-${idx}`} className="position-card">
                  <div className="position-header">
                    <span className="position-symbol">{pos.symbol}</span>
                    <span className={`position-side ${pos.side.toLowerCase()}`}>
                      {pos.side}
                    </span>
                  </div>
                  
                  <div className="position-details">
                    <div className="position-row">
                      <span>Entry</span>
                      <span>${pos.entry_price?.toFixed(2)}</span>
                    </div>
                    <div className="position-row">
                      <span>Size</span>
                      <span>{pos.quantity}</span>
                    </div>
                    {botPos && (
                      <>
                        <div className="position-row">
                          <span style={{ color: 'var(--success)' }}>TP</span>
                          <span style={{ color: 'var(--success)' }}>${botPos.take_profit?.toFixed(2)}</span>
                        </div>
                        <div className="position-row">
                          <span style={{ color: 'var(--danger)' }}>SL</span>
                          <span style={{ color: 'var(--danger)' }}>${botPos.stop_loss?.toFixed(2)}</span>
                        </div>
                      </>
                    )}
                    <div className="position-row">
                      <span>Unrealized P&L</span>
                      <span style={{ color: pos.unrealized_pnl >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                        {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl?.toFixed(2)}
                      </span>
                    </div>
                  </div>
                  
                  {botPos && (
                    <button 
                      className="close-position-btn"
                      onClick={() => handleClosePosition(botPos.id)}
                      disabled={actionLoading}
                    >
                      Close Position
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default TradingPanel;
