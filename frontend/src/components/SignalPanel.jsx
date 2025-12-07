/**
 * Signal Panel Component
 * Displays active trading signals with entry, SL, TP levels
 */
export default function SignalPanel({ signals, symbol }) {
  if (!signals || signals.length === 0) {
    return (
      <div className="card signal-card fade-in">
        <div className="card__header">
          <span className="card__title">Active Signals</span>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">🎯</div>
          <p className="empty-state__text">
            No active signals for {symbol}<br />
            <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>
              Waiting for setup conditions...
            </span>
          </p>
        </div>
      </div>
    )
  }
  
  const signal = signals[0] // Show most recent signal
  const isLong = signal.direction === 'LONG'
  
  return (
    <div className={`card signal-card ${isLong ? 'long' : 'short'} fade-in`}>
      <div className="card__header">
        <span className="card__title">Active Signal</span>
        <div style={{ display: 'flex', gap: '8px' }}>
          <span className={`signal-badge ${signal.model?.toLowerCase() === 'trend' ? 'trend' : 'reversion'}`}>
            {signal.model === 'TREND' ? '📈 Trend' : '🔄 Reversion'}
          </span>
          <span className={`signal-badge ${isLong ? 'long' : 'short'}`}>
            {isLong ? '↑ LONG' : '↓ SHORT'}
          </span>
        </div>
      </div>
      
      {/* Price Levels */}
      <div className="price-levels">
        <div className="price-level entry">
          <div className="price-level__label">Entry</div>
          <div className="price-level__value">{signal.entry_price?.toFixed(2)}</div>
        </div>
        <div className="price-level sl">
          <div className="price-level__label">Stop Loss</div>
          <div className="price-level__value">{signal.stop_loss?.toFixed(2)}</div>
        </div>
        <div className="price-level tp">
          <div className="price-level__label">Take Profit</div>
          <div className="price-level__value">{signal.take_profit?.toFixed(2)}</div>
        </div>
      </div>
      
      {/* Key Levels */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 1fr', 
        gap: '12px', 
        marginTop: '16px',
        padding: '12px',
        background: 'var(--bg-tertiary)',
        borderRadius: 'var(--radius-md)'
      }}>
        <div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px' }}>LVN Entry Zone</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{signal.lvn_price?.toFixed(2)}</div>
        </div>
        <div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px' }}>POC Target</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{signal.poc_target?.toFixed(2)}</div>
        </div>
      </div>
      
      {/* Confidence Meter */}
      <div className="confidence-meter">
        <div className="confidence-meter__header">
          <span style={{ color: 'var(--text-muted)' }}>Confidence</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            {(signal.confidence * 100).toFixed(0)}%
          </span>
        </div>
        <div className="confidence-meter__bar">
          <div 
            className="confidence-meter__fill" 
            style={{ width: `${signal.confidence * 100}%` }}
          />
        </div>
      </div>
      
      {/* Risk/Reward */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        marginTop: '16px',
        padding: '12px',
        background: 'var(--bg-tertiary)',
        borderRadius: 'var(--radius-md)'
      }}>
        <div>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Risk/Reward</span>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.1rem', fontWeight: 600, color: 'var(--color-long)' }}>
            1:{signal.risk_reward?.toFixed(1)}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Aggression</span>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.1rem', fontWeight: 600 }}>
            {(signal.aggression_strength * 100).toFixed(0)}%
          </div>
        </div>
      </div>
      
      {/* Aggression Description */}
      {signal.aggression_description && (
        <div style={{ 
          marginTop: '12px', 
          padding: '8px 12px', 
          background: 'rgba(99, 102, 241, 0.1)',
          borderRadius: 'var(--radius-sm)',
          fontSize: '0.8rem',
          color: 'var(--accent-primary)'
        }}>
          ✓ {signal.aggression_description}
        </div>
      )}
    </div>
  )
}
