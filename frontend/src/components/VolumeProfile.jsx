/**
 * Volume Profile Visualization
 * Shows horizontal volume histogram with LVN and POC levels
 */
export default function VolumeProfileChart({ profile, analysis }) {
  if (!profile || !profile.levels || profile.levels.length === 0) {
    return (
      <div className="loading">
        <div className="loading__spinner"></div>
        Loading volume profile...
      </div>
    )
  }
  
  const { levels, poc, vah, val, lvn_zones = [] } = profile
  const currentPrice = analysis?.current_price || 0
  
  // Get max volume for scaling
  const maxVolume = Math.max(...levels.map(l => l.volume))
  
  // Take subset of levels for display (too many makes it cluttered)
  const displayLevels = levels
    .filter(l => l.volume > 0)
    .slice(-30) // Last 30 levels
  
  // Check if a price is an LVN
  const isLVN = (price) => {
    return lvn_zones.some(lvn => Math.abs(lvn.price - price) / price < 0.001)
  }
  
  // Check if price is POC
  const isPOC = (price) => {
    return Math.abs(price - poc) / poc < 0.001
  }
  
  return (
    <div className="volume-profile">
      <div className="volume-profile__chart">
        {displayLevels.map((level, idx) => {
          const widthPercent = (level.volume / maxVolume) * 100
          const deltaColor = level.delta > 0 ? 'buy' : level.delta < 0 ? 'sell' : 'neutral'
          const isCurrentPrice = currentPrice && Math.abs(level.price - currentPrice) / currentPrice < 0.002
          
          return (
            <div 
              key={idx} 
              className={`volume-level ${isPOC(level.price) ? 'poc' : ''} ${isLVN(level.price) ? 'lvn' : ''}`}
              style={{ 
                opacity: isCurrentPrice ? 1 : 0.9,
                background: isCurrentPrice ? 'rgba(99, 102, 241, 0.1)' : 'transparent'
              }}
            >
              <span className="volume-level__price">
                {level.price.toFixed(1)}
              </span>
              <div className="volume-level__bar">
                <div 
                  className={`volume-level__fill ${deltaColor}`}
                  style={{ width: `${widthPercent}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>
      
      {/* Legend */}
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        gap: '8px', 
        paddingLeft: '16px',
        borderLeft: '1px solid var(--border-color)'
      }}>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
          KEY LEVELS
        </div>
        
        <LevelIndicator 
          label="POC" 
          value={poc?.toFixed(2)} 
          color="var(--accent-gradient)" 
          description="Point of Control"
        />
        <LevelIndicator 
          label="VAH" 
          value={vah?.toFixed(2)} 
          color="var(--color-long)" 
          description="Value Area High"
        />
        <LevelIndicator 
          label="VAL" 
          value={val?.toFixed(2)} 
          color="var(--color-short)" 
          description="Value Area Low"
        />
        
        {lvn_zones.length > 0 && (
          <>
            <div style={{ 
              fontSize: '0.7rem', 
              color: 'var(--text-muted)', 
              marginTop: '12px' 
            }}>
              LVN ZONES
            </div>
            {lvn_zones.slice(0, 3).map((lvn, idx) => (
              <LevelIndicator 
                key={idx}
                label={`LVN ${idx + 1}`}
                value={lvn.price?.toFixed(2)}
                color="var(--color-neutral)"
                description="Entry Zone"
              />
            ))}
          </>
        )}
        
        <div style={{ marginTop: 'auto', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ width: '12px', height: '4px', background: 'var(--color-long)', borderRadius: '2px' }}></span>
            Buy Volume
          </div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '4px' }}>
            <span style={{ width: '12px', height: '4px', background: 'var(--color-short)', borderRadius: '2px' }}></span>
            Sell Volume
          </div>
        </div>
      </div>
    </div>
  )
}

function LevelIndicator({ label, value, color, description }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <span style={{ 
        width: '8px', 
        height: '8px', 
        borderRadius: '2px',
        background: color
      }}></span>
      <div>
        <div style={{ 
          fontFamily: 'var(--font-mono)', 
          fontSize: '0.85rem',
          fontWeight: 600 
        }}>
          {value || '—'}
        </div>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
          {label}
        </div>
      </div>
    </div>
  )
}
