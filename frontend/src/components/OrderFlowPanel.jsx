/**
 * Order Flow Panel
 * Displays CVD, buy/sell aggression, and imbalance indicators
 */
export default function OrderFlowPanel({ orderFlow, symbol }) {
  if (!orderFlow) {
    return (
      <div className="card fade-in">
        <div className="card__header">
          <span className="card__title">Order Flow Analysis</span>
        </div>
        <div className="loading">
          <div className="loading__spinner"></div>
          Loading order flow...
        </div>
      </div>
    )
  }
  
  const { cvd, buy_aggression, sell_aggression } = orderFlow
  
  // Calculate CVD position (0-100, 50 is neutral)
  const cvdPosition = 50 + (cvd?.strength || 0) * 50 * (cvd?.direction || 0)
  
  return (
    <div className="card fade-in">
      <div className="card__header">
        <span className="card__title">Order Flow Analysis</span>
        <span style={{ 
          fontFamily: 'var(--font-mono)', 
          fontSize: '0.75rem',
          padding: '4px 8px',
          borderRadius: '4px',
          background: cvd?.trend === 'bullish' 
            ? 'var(--color-long-bg)' 
            : cvd?.trend === 'bearish' 
              ? 'var(--color-short-bg)' 
              : 'var(--bg-tertiary)',
          color: cvd?.trend === 'bullish' 
            ? 'var(--color-long)' 
            : cvd?.trend === 'bearish' 
              ? 'var(--color-short)' 
              : 'var(--text-muted)'
        }}>
          CVD: {cvd?.trend?.toUpperCase() || 'NEUTRAL'}
        </span>
      </div>
      
      {/* Buy/Sell Aggression */}
      <div className="order-flow">
        <div className="order-flow__side buy">
          <div className="order-flow__label">Buy Aggression</div>
          <div className="order-flow__value">{((buy_aggression?.strength || 0) * 100).toFixed(0)}%</div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            {buy_aggression?.imbalances || 0} imbalances • {buy_aggression?.large_prints || 0} large prints
          </div>
        </div>
        
        <div className="order-flow__side sell">
          <div className="order-flow__label">Sell Aggression</div>
          <div className="order-flow__value">{((sell_aggression?.strength || 0) * 100).toFixed(0)}%</div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            {sell_aggression?.imbalances || 0} imbalances • {sell_aggression?.large_prints || 0} large prints
          </div>
        </div>
      </div>
      
      {/* CVD Gauge */}
      <div className="cvd-indicator">
        <div className="cvd-indicator__header">
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Cumulative Volume Delta
          </span>
          <span style={{ 
            fontFamily: 'var(--font-mono)',
            fontSize: '0.8rem',
            fontWeight: 600,
            color: cvd?.direction > 0 ? 'var(--color-long)' : cvd?.direction < 0 ? 'var(--color-short)' : 'var(--text-secondary)'
          }}>
            {cvd?.cvd_change > 0 ? '+' : ''}{cvd?.cvd_change?.toFixed(2) || '0.00'}
          </span>
        </div>
        <div className="cvd-indicator__bar">
          <div 
            className="cvd-indicator__marker" 
            style={{ left: `${Math.min(Math.max(cvdPosition, 5), 95)}%` }}
          />
        </div>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          fontSize: '0.65rem',
          color: 'var(--text-muted)',
          marginTop: '4px'
        }}>
          <span>Sellers</span>
          <span>Neutral</span>
          <span>Buyers</span>
        </div>
      </div>
      
      {/* Flow Description */}
      <div style={{ 
        marginTop: '16px',
        padding: '12px',
        background: 'var(--bg-tertiary)',
        borderRadius: 'var(--radius-md)',
        fontSize: '0.8rem',
        color: 'var(--text-secondary)'
      }}>
        <strong>Flow Summary: </strong>
        {getFlowDescription(cvd, buy_aggression, sell_aggression)}
      </div>
    </div>
  )
}

function getFlowDescription(cvd, buyAgg, sellAgg) {
  const buyStrength = buyAgg?.strength || 0
  const sellStrength = sellAgg?.strength || 0
  const cvdTrend = cvd?.trend || 'neutral'
  
  if (cvdTrend === 'bullish' && buyStrength > 0.5) {
    return 'Strong buying pressure with CVD confirmation. Favorable for LONG entries.'
  }
  
  if (cvdTrend === 'bearish' && sellStrength > 0.5) {
    return 'Strong selling pressure with CVD confirmation. Favorable for SHORT entries.'
  }
  
  if (buyStrength > 0.3 && sellStrength > 0.3) {
    return 'Active two-way order flow. Wait for clear direction before entry.'
  }
  
  if (cvdTrend === 'bullish') {
    return 'Moderate buying interest developing. Watch for aggression confirmation.'
  }
  
  if (cvdTrend === 'bearish') {
    return 'Moderate selling interest developing. Watch for aggression confirmation.'
  }
  
  return 'Neutral order flow. No clear aggression in either direction.'
}
