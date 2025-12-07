/**
 * ML Metrics Panel
 * Displays machine learning model status and learning progress
 */
export default function MLMetrics({ status, stats }) {
  if (!status) {
    return (
      <div className="card fade-in">
        <div className="card__header">
          <span className="card__title">AI Learning Progress</span>
        </div>
        <div className="loading">
          <div className="loading__spinner"></div>
          Loading ML status...
        </div>
      </div>
    )
  }
  
  const { signal_accuracy, lvn_patterns, state_classifier, overall_status } = status
  
  return (
    <div className="card fade-in">
      <div className="card__header">
        <span className="card__title">AI Learning Progress</span>
        <span className={`ml-metric__status ${overall_status === 'active' ? 'active' : 'learning'}`}>
          {overall_status === 'active' ? '✓ Active' : overall_status === 'learning' ? '◉ Learning' : '○ Initializing'}
        </span>
      </div>
      
      <div className="ml-metrics">
        {/* Signal Accuracy */}
        <div className="ml-metric">
          <div className="ml-metric__icon">🎯</div>
          <div className="ml-metric__info">
            <div className="ml-metric__label">Signal Accuracy</div>
            <div className="ml-metric__value">
              {signal_accuracy?.win_rate?.toFixed(1) || 0}%
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              Signals
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              {signal_accuracy?.total_signals || 0}
            </div>
          </div>
        </div>
        
        {/* LVN Patterns */}
        <div className="ml-metric">
          <div className="ml-metric__icon">📊</div>
          <div className="ml-metric__info">
            <div className="ml-metric__label">LVN Pattern Recognition</div>
            <div className="ml-metric__value">
              {lvn_patterns?.bounce_rate?.toFixed(1) || 0}% Bounce
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              Patterns
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              {lvn_patterns?.patterns_recorded || 0}
            </div>
          </div>
        </div>
        
        {/* State Classifier */}
        <div className="ml-metric">
          <div className="ml-metric__icon">🧠</div>
          <div className="ml-metric__info">
            <div className="ml-metric__label">Market State Classifier</div>
            <div className="ml-metric__value">
              {state_classifier?.is_trained ? 'Trained' : 'Training...'}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              Observations
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              {state_classifier?.observations || 0}
            </div>
          </div>
        </div>
      </div>
      
      {/* Feature Weights (if available) */}
      {signal_accuracy?.feature_weights && Object.keys(signal_accuracy.feature_weights).length > 0 && (
        <div style={{ marginTop: '16px' }}>
          <div style={{ 
            fontSize: '0.75rem', 
            color: 'var(--text-muted)',
            marginBottom: '8px'
          }}>
            Learned Feature Importance
          </div>
          <div style={{ 
            display: 'flex', 
            gap: '4px',
            height: '24px',
            borderRadius: 'var(--radius-sm)',
            overflow: 'hidden'
          }}>
            {Object.entries(signal_accuracy.feature_weights).map(([key, value], idx) => (
              <div 
                key={key}
                style={{ 
                  flex: value,
                  background: getFeatureColor(idx),
                  position: 'relative'
                }}
                title={`${key}: ${(value * 100).toFixed(1)}%`}
              />
            ))}
          </div>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            marginTop: '4px',
            fontSize: '0.65rem',
            color: 'var(--text-muted)'
          }}>
            {Object.entries(signal_accuracy.feature_weights).slice(0, 4).map(([key], idx) => (
              <span key={key} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ 
                  width: '8px', 
                  height: '8px', 
                  borderRadius: '2px',
                  background: getFeatureColor(idx)
                }}></span>
                {key.replace('_', ' ').slice(0, 12)}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {/* Performance Stats */}
      {stats && (
        <div style={{ 
          marginTop: '16px',
          padding: '12px',
          background: 'var(--bg-tertiary)',
          borderRadius: 'var(--radius-md)'
        }}>
          <div style={{ 
            fontSize: '0.75rem', 
            color: 'var(--text-muted)',
            marginBottom: '8px'
          }}>
            Lifetime Performance
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
            <StatBox label="Win Rate" value={`${stats.win_rate?.toFixed(1) || 0}%`} positive={stats.win_rate > 50} />
            <StatBox label="Wins" value={stats.wins || 0} positive={true} />
            <StatBox label="Losses" value={stats.losses || 0} positive={false} />
          </div>
        </div>
      )}
    </div>
  )
}

function StatBox({ label, value, positive }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{label}</div>
      <div style={{ 
        fontFamily: 'var(--font-mono)', 
        fontSize: '1rem',
        fontWeight: 600,
        color: positive ? 'var(--color-long)' : 'var(--color-short)'
      }}>
        {value}
      </div>
    </div>
  )
}

function getFeatureColor(index) {
  const colors = [
    '#6366f1', // Indigo
    '#8b5cf6', // Purple
    '#a855f7', // Fuchsia
    '#ec4899', // Pink
    '#f43f5e', // Rose
  ]
  return colors[index % colors.length]
}
