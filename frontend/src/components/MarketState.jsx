/**
 * Market State Panel
 * Shows current market condition (Balanced, Trending, Choppy)
 */
export default function MarketStatePanel({ analysis, symbol }) {
  if (!analysis) {
    return (
      <div className="card fade-in">
        <div className="card__header">
          <span className="card__title">Market State</span>
        </div>
        <div className="loading">
          <div className="loading__spinner"></div>
          Analyzing market...
        </div>
      </div>
    )
  }
  
  const { 
    market_state, 
    market_state_confidence, 
    is_balanced,
    current_price,
    poc,
    vah,
    val
  } = analysis
  
  const stateConfig = getStateConfig(market_state)
  
  return (
    <div className="card fade-in">
      <div className="card__header">
        <span className="card__title">Market State - {symbol}</span>
        <span style={{ 
          fontFamily: 'var(--font-mono)', 
          fontSize: '1.1rem',
          fontWeight: 700,
          color: 'var(--text-primary)'
        }}>
          {current_price?.toFixed(2)}
        </span>
      </div>
      
      <div className="market-state">
        <div className={`market-state__indicator ${market_state}`}>
          {stateConfig.icon}
        </div>
        <div className="market-state__info">
          <h4 style={{ color: stateConfig.color }}>{stateConfig.label}</h4>
          <p>{stateConfig.description}</p>
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Confidence</div>
          <div style={{ 
            fontFamily: 'var(--font-mono)', 
            fontSize: '1.2rem', 
            fontWeight: 700,
            color: stateConfig.color
          }}>
            {((market_state_confidence || 0) * 100).toFixed(0)}%
          </div>
        </div>
      </div>
      
      {/* Key Levels Quick View */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(3, 1fr)', 
        gap: '12px', 
        marginTop: '16px' 
      }}>
        <QuickLevel label="POC" value={poc} current={current_price} />
        <QuickLevel label="VAH" value={vah} current={current_price} type="high" />
        <QuickLevel label="VAL" value={val} current={current_price} type="low" />
      </div>
      
      {/* Trading Model Suggestion */}
      <div style={{ 
        marginTop: '16px', 
        padding: '12px',
        background: is_balanced ? 'rgba(139, 92, 246, 0.1)' : 'rgba(99, 102, 241, 0.1)',
        borderRadius: 'var(--radius-md)',
        borderLeft: `3px solid ${is_balanced ? 'var(--accent-secondary)' : 'var(--accent-primary)'}`
      }}>
        <div style={{ fontSize: '0.75rem', fontWeight: 600, marginBottom: '4px' }}>
          {is_balanced ? '🔄 Mean Reversion Model' : '📈 Trend Model'}
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          {is_balanced 
            ? 'Market is balanced. Look for failed breakouts and entries at LVN zones targeting POC.'
            : 'Market is out of balance. Look for LVN retracements with order flow confirmation.'
          }
        </div>
      </div>
    </div>
  )
}

function QuickLevel({ label, value, current, type }) {
  const distance = current && value ? ((current - value) / value * 100) : 0
  const isNear = Math.abs(distance) < 0.5
  
  return (
    <div style={{ 
      padding: '12px',
      background: 'var(--bg-tertiary)',
      borderRadius: 'var(--radius-md)',
      textAlign: 'center',
      border: isNear ? '1px solid var(--accent-primary)' : '1px solid transparent'
    }}>
      <div style={{ 
        fontSize: '0.65rem', 
        color: 'var(--text-muted)',
        textTransform: 'uppercase',
        letterSpacing: '0.5px'
      }}>
        {label}
      </div>
      <div style={{ 
        fontFamily: 'var(--font-mono)', 
        fontSize: '1rem',
        fontWeight: 600,
        marginTop: '4px'
      }}>
        {value?.toFixed(2) || '—'}
      </div>
      {current && value && (
        <div style={{ 
          fontSize: '0.7rem',
          marginTop: '2px',
          color: distance > 0 ? 'var(--color-long)' : distance < 0 ? 'var(--color-short)' : 'var(--text-muted)'
        }}>
          {distance > 0 ? '+' : ''}{distance.toFixed(2)}%
        </div>
      )}
    </div>
  )
}

function getStateConfig(state) {
  switch (state) {
    case 'balanced':
      return {
        label: 'BALANCED',
        icon: '⚖️',
        color: 'var(--accent-primary)',
        description: 'Price rotating around POC'
      }
    case 'trending_up':
      return {
        label: 'TRENDING UP',
        icon: '📈',
        color: 'var(--color-long)',
        description: 'Out of balance, bullish momentum'
      }
    case 'trending_down':
      return {
        label: 'TRENDING DOWN',
        icon: '📉',
        color: 'var(--color-short)',
        description: 'Out of balance, bearish momentum'
      }
    case 'breakout_up':
      return {
        label: 'BREAKOUT UP',
        icon: '🚀',
        color: 'var(--color-long)',
        description: 'Breaking above value area'
      }
    case 'breakout_down':
      return {
        label: 'BREAKOUT DOWN',
        icon: '⬇️',
        color: 'var(--color-short)',
        description: 'Breaking below value area'
      }
    case 'choppy':
    default:
      return {
        label: 'CHOPPY',
        icon: '〰️',
        color: 'var(--color-neutral)',
        description: 'Unclear conditions, wait for clarity'
      }
  }
}
