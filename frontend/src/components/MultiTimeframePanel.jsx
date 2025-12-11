import { useMTFSignals } from '../hooks/useWebSocket'

/**
 * Multi-Timeframe Panel - Shows signals across all timeframes for a symbol
 */
function MultiTimeframePanel({ symbol }) {
  const { mtfData, loading } = useMTFSignals(symbol)
  
  if (loading && !mtfData) {
    return (
      <div className="card fade-in">
        <div className="card__header">
          <span className="card__title">📊 Multi-Timeframe Analysis</span>
        </div>
        <div className="card__body" style={{ textAlign: 'center', padding: '1rem' }}>
          Loading MTF data...
        </div>
      </div>
    )
  }
  
  const confluence = mtfData?.confluence || {}
  const signalsByTf = mtfData?.signals_by_timeframe || {}
  
  const overallDirection = confluence.overall_direction || 'NEUTRAL'
  const confluenceScore = confluence.confluence_score || 0
  const alignedCount = confluence.aligned_timeframes || 0
  const totalCount = confluence.total_timeframes || 0
  
  // Direction colors
  const getDirectionColor = (direction) => {
    if (direction === 'LONG') return 'var(--color-success)'
    if (direction === 'SHORT') return 'var(--color-danger)'
    return 'var(--text-muted)'
  }
  
  // Direction icon
  const getDirectionIcon = (direction) => {
    if (direction === 'LONG') return '↑'
    if (direction === 'SHORT') return '↓'
    return '—'
  }
  
  return (
    <div className="card fade-in">
      <div className="card__header">
        <span className="card__title">📊 Multi-Timeframe Analysis</span>
        <span className="card__badge" style={{ color: getDirectionColor(overallDirection) }}>
          {getDirectionIcon(overallDirection)} {overallDirection}
        </span>
      </div>
      
      <div className="card__body">
        {/* Confluence Score */}
        <div className="mtf-confluence">
          <div className="confluence-header">
            <span>Confluence Score</span>
            <span style={{ fontWeight: 'bold' }}>
              {(confluenceScore * 100).toFixed(0)}%
            </span>
          </div>
          <div className="confluence-bar">
            <div 
              className="confluence-fill"
              style={{ 
                width: `${confluenceScore * 100}%`,
                background: confluenceScore >= 0.6 
                  ? 'var(--color-success)' 
                  : confluenceScore >= 0.4 
                    ? 'var(--color-warning)' 
                    : 'var(--color-danger)'
              }}
            />
            <div className="confluence-threshold" style={{ left: '50%' }} />
          </div>
          <div className="confluence-info">
            {alignedCount}/{totalCount} timeframes aligned
          </div>
        </div>
        
        {/* Timeframe Grid */}
        <div className="mtf-grid">
          {['1h', '15m', '5m', '1m'].map(tf => {
            const signal = signalsByTf[tf]
            const direction = signal?.direction || 'NEUTRAL'
            const strength = signal?.strength || 0
            const stochK = signal?.stoch_k || 50
            const confidence = signal?.confidence || 0
            
            return (
              <div key={tf} className="mtf-cell">
                <div className="mtf-cell-header">
                  <span className="mtf-tf">{tf}</span>
                  <span 
                    className="mtf-direction"
                    style={{ color: getDirectionColor(direction) }}
                  >
                    {getDirectionIcon(direction)}
                  </span>
                </div>
                
                <div className="mtf-cell-body">
                  <div className="mtf-metric">
                    <span className="mtf-label">Strength</span>
                    <div className="mtf-bar-container">
                      <div 
                        className="mtf-bar"
                        style={{ 
                          width: `${strength * 100}%`,
                          background: getDirectionColor(direction)
                        }}
                      />
                    </div>
                  </div>
                  
                  <div className="mtf-metric">
                    <span className="mtf-label">Stoch K</span>
                    <span 
                      className="mtf-value"
                      style={{ 
                        color: stochK > 80 ? 'var(--color-danger)' : 
                               stochK < 20 ? 'var(--color-success)' : 'inherit'
                      }}
                    >
                      {stochK.toFixed(0)}
                    </span>
                  </div>
                  
                  <div className="mtf-metric">
                    <span className="mtf-label">Conf</span>
                    <span className="mtf-value">{(confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
        
        {/* Reasoning */}
        {confluence.reasoning && (
          <div className="mtf-reasoning">
            <code>{confluence.reasoning}</code>
          </div>
        )}
      </div>
    </div>
  )
}

export default MultiTimeframePanel
