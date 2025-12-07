import { useEffect, useRef, memo } from 'react'

/**
 * TradingView Technical Analysis Widget
 * Shows buy/sell/neutral ratings with oscillators & moving averages
 */
export const TechnicalAnalysisWidget = memo(function TechnicalAnalysisWidget({ symbol = 'BTCUSDT' }) {
  const containerRef = useRef(null)
  
  useEffect(() => {
    if (!containerRef.current) return
    
    // Clear previous widget
    containerRef.current.innerHTML = ''
    
    // TradingView uses exchange:symbol format
    const tvSymbol = symbol.includes(':') ? symbol : `BINANCE:${symbol}`
    
    const script = document.createElement('script')
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js'
    script.type = 'text/javascript'
    script.async = true
    script.innerHTML = JSON.stringify({
      interval: '5m',
      width: '100%',
      isTransparent: true,
      height: '100%',
      symbol: tvSymbol,
      showIntervalTabs: true,
      displayMode: 'single',
      locale: 'en',
      colorTheme: 'dark'
    })
    
    containerRef.current.appendChild(script)
    
    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = ''
      }
    }
  }, [symbol])
  
  return (
    <div className="card fade-in">
      <div className="card__header">
        <span className="card__title">📊 Technical Analysis</span>
        <span style={{ 
          fontSize: '0.7rem', 
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)'
        }}>
          TradingView
        </span>
      </div>
      <div 
        ref={containerRef}
        className="tradingview-widget-container"
        style={{ height: '380px' }}
      />
    </div>
  )
})

/**
 * TradingView Mini Chart Widget
 * Compact price chart for the selected symbol
 */
export const MiniChartWidget = memo(function MiniChartWidget({ symbol = 'BTCUSDT' }) {
  const containerRef = useRef(null)
  
  useEffect(() => {
    if (!containerRef.current) return
    
    containerRef.current.innerHTML = ''
    
    const tvSymbol = symbol.includes(':') ? symbol : `BINANCE:${symbol}`
    
    const script = document.createElement('script')
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js'
    script.type = 'text/javascript'
    script.async = true
    script.innerHTML = JSON.stringify({
      symbol: tvSymbol,
      width: '100%',
      height: '100%',
      locale: 'en',
      dateRange: '1D',
      colorTheme: 'dark',
      isTransparent: true,
      autosize: true,
      largeChartUrl: '',
      chartOnly: false,
      noTimeScale: false
    })
    
    containerRef.current.appendChild(script)
    
    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = ''
      }
    }
  }, [symbol])
  
  return (
    <div className="card fade-in">
      <div className="card__header">
        <span className="card__title">📈 Price Chart</span>
        <span style={{ 
          fontSize: '0.7rem', 
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)'
        }}>
          {symbol}
        </span>
      </div>
      <div 
        ref={containerRef}
        className="tradingview-widget-container"
        style={{ height: '200px' }}
      />
    </div>
  )
})

/**
 * TradingView Top Stories Widget
 * Crypto news feed with daily market snapshots
 */
export const TopStoriesWidget = memo(function TopStoriesWidget() {
  const containerRef = useRef(null)
  
  useEffect(() => {
    if (!containerRef.current) return
    
    containerRef.current.innerHTML = ''
    
    const script = document.createElement('script')
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-timeline.js'
    script.type = 'text/javascript'
    script.async = true
    script.innerHTML = JSON.stringify({
      feedMode: 'market',
      market: 'crypto',
      isTransparent: true,
      displayMode: 'regular',
      width: '100%',
      height: '100%',
      colorTheme: 'dark',
      locale: 'en'
    })
    
    containerRef.current.appendChild(script)
    
    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = ''
      }
    }
  }, [])
  
  return (
    <div className="card fade-in">
      <div className="card__header">
        <span className="card__title">📰 Crypto News</span>
        <span style={{ 
          fontSize: '0.7rem', 
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)'
        }}>
          Top Stories
        </span>
      </div>
      <div 
        ref={containerRef}
        className="tradingview-widget-container"
        style={{ height: '400px' }}
      />
    </div>
  )
})

/**
 * TradingView Crypto Heatmap Widget
 * Visual overview of crypto market performance
 */
export const CryptoHeatmapWidget = memo(function CryptoHeatmapWidget() {
  const containerRef = useRef(null)
  
  useEffect(() => {
    if (!containerRef.current) return
    
    containerRef.current.innerHTML = ''
    
    const script = document.createElement('script')
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-crypto-coins-heatmap.js'
    script.type = 'text/javascript'
    script.async = true
    script.innerHTML = JSON.stringify({
      dataSource: 'Crypto',
      blockSize: 'market_cap_calc',
      blockColor: 'change',
      locale: 'en',
      symbolUrl: '',
      colorTheme: 'dark',
      hasTopBar: false,
      isDataSetEnabled: false,
      isZoomEnabled: true,
      hasSymbolTooltip: true,
      isMonoSize: false,
      width: '100%',
      height: '100%'
    })
    
    containerRef.current.appendChild(script)
    
    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = ''
      }
    }
  }, [])
  
  return (
    <div className="card fade-in">
      <div className="card__header">
        <span className="card__title">🔥 Crypto Heatmap</span>
        <span style={{ 
          fontSize: '0.7rem', 
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)'
        }}>
          Market Overview
        </span>
      </div>
      <div 
        ref={containerRef}
        className="tradingview-widget-container"
        style={{ height: '300px' }}
      />
    </div>
  )
})

export default {
  TechnicalAnalysisWidget,
  MiniChartWidget,
  TopStoriesWidget,
  CryptoHeatmapWidget
}
