const MLMetrics = ({ status, stats }) => {
    // If no data yet, show loading state
    if (!status && !stats) return (
        <div className="card loading">
            <span>Loading ML Intelligence...</span>
        </div>
    );

    // Use status or fallbacks
    const metrics = status || {};
    const featureWeights = metrics.signal_accuracy?.feature_weights || {};
    const isActive = metrics.overall_status === 'active';

    return (
        <div className="card fade-in">
            <div className="card__header">
                <div className="ml-metric__icon">🧠</div>
                <h2 className="card__title">ML Learning Status</h2>
                <span className={`ml-metric__status ${isActive ? 'active' : 'learning'}`}>
                    {metrics.overall_status || 'INITIALIZING'}
                </span>
            </div>
            
            <div className="ml-metrics">
                {/* Signal Accuracy */}
                <div className="ml-metric">
                    <div className="ml-metric__icon" style={{ background: 'rgba(16, 185, 129, 0.15)' }}>📊</div>
                    <div className="ml-metric__info">
                        <div className="ml-metric__label">Signal Accuracy</div>
                        <div className="ml-metric__value" style={{ color: 'var(--color-long)' }}>
                            {(metrics.signal_accuracy?.win_rate || 0).toFixed(1)}%
                        </div>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        {metrics.signal_accuracy?.total_signals || 0} signals
                    </div>
                </div>
                
                {/* Pattern Recognition */}
                <div className="ml-metric">
                    <div className="ml-metric__icon" style={{ background: 'rgba(99, 102, 241, 0.15)' }}>🔍</div>
                    <div className="ml-metric__info">
                        <div className="ml-metric__label">Pattern Recognition</div>
                        <div className="ml-metric__value" style={{ color: 'var(--accent-primary)' }}>
                            {metrics.lvn_patterns?.patterns_recorded || 0}
                        </div>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        LVN Patterns
                    </div>
                </div>
                
                {/* Market State */}
                <div className="ml-metric">
                    <div className="ml-metric__icon" style={{ background: 'rgba(139, 92, 246, 0.15)' }}>📈</div>
                    <div className="ml-metric__info">
                        <div className="ml-metric__label">Market State</div>
                        <div className="ml-metric__value" style={{ color: 'var(--accent-secondary)', textTransform: 'capitalize' }}>
                            {metrics.state_classifier?.status || 'Active'}
                        </div>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        {metrics.state_classifier?.observations || 0} obs
                    </div>
                </div>

                {/* Online Trainer */}
                <div className="ml-metric">
                    <div className="ml-metric__icon" style={{ background: 'rgba(6, 182, 212, 0.15)' }}>⚡</div>
                    <div className="ml-metric__info">
                        <div className="ml-metric__label">Online Trainer</div>
                        <div className="ml-metric__value" style={{ color: '#06b6d4' }}>
                            READY
                        </div>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        v{(metrics.model_version || 1.0).toFixed(1)}
                    </div>
                </div>

                {/* FreqAI Status */}
                {metrics.freqai_status && (
                    <div className="ml-metric">
                        <div className="ml-metric__icon" style={{ background: 'rgba(236, 72, 153, 0.15)' }}>🤖</div>
                        <div className="ml-metric__info">
                            <div className="ml-metric__label">FreqAI</div>
                            <div className="ml-metric__value" style={{ color: metrics.freqai_status.enabled ? 'var(--color-long)' : 'var(--text-muted)' }}>
                                {metrics.freqai_status.enabled ? 'ENABLED' : 'DISABLED'}
                            </div>
                        </div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                            {metrics.freqai_status.model || 'N/A'}
                        </div>
                    </div>
                )}
            </div>

            {/* Feature Importance Section */}
            <div style={{ 
                marginTop: 'var(--spacing-lg)', 
                borderTop: '1px solid var(--border-color)', 
                paddingTop: 'var(--spacing-md)' 
            }}>
                <h4 style={{ 
                    fontSize: '0.75rem', 
                    fontWeight: '600', 
                    color: 'var(--text-muted)', 
                    textTransform: 'uppercase', 
                    letterSpacing: '0.5px',
                    marginBottom: 'var(--spacing-md)'
                }}>
                    Feature Importance (Top Factors)
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
                    {Object.entries(featureWeights)
                        .sort(([,a], [,b]) => b - a)
                        .slice(0, 3)
                        .map(([feature, weight]) => (
                            <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                                <span style={{ 
                                    fontSize: '0.75rem', 
                                    color: 'var(--text-secondary)', 
                                    flex: '1',
                                    textTransform: 'capitalize'
                                }}>
                                    {feature.replace(/_/g, ' ')}
                                </span>
                                <div className="confidence-meter" style={{ flex: '2', margin: 0 }}>
                                    <div className="confidence-meter__bar">
                                        <div 
                                            className="confidence-meter__fill" 
                                            style={{ width: `${weight * 100}%` }}
                                        />
                                    </div>
                                </div>
                                <span style={{ 
                                    fontSize: '0.75rem', 
                                    fontFamily: 'var(--font-mono)', 
                                    color: 'var(--color-long)',
                                    width: '40px',
                                    textAlign: 'right'
                                }}>
                                    {(weight * 100).toFixed(0)}%
                                </span>
                            </div>
                        ))
                    }
                    {Object.keys(featureWeights).length === 0 && (
                        <div className="empty-state" style={{ padding: 'var(--spacing-md)' }}>
                            <span className="empty-state__text">Gathering feature data...</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default MLMetrics;
