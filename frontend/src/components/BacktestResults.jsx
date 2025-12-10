import { useState, useEffect } from 'react';

const BacktestResults = () => {
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchResults = async () => {
            try {
                const response = await fetch('http://localhost:8000/api/backtest/results');
                if (!response.ok) throw new Error('Failed to fetch backtest results');
                const data = await response.json();
                setResults(data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        
        fetchResults();
        const interval = setInterval(fetchResults, 60000); // Refresh every minute
        return () => clearInterval(interval);
    }, []);

    if (loading) return (
        <div className="card loading">
            <span>Loading Backtest Results...</span>
        </div>
    );

    if (error || !results) return (
        <div className="card">
            <div className="card__header">
                <div className="ml-metric__icon">📈</div>
                <h2 className="card__title">Backtest Results</h2>
            </div>
            <div className="empty-state" style={{ padding: 'var(--spacing-lg)' }}>
                <span className="empty-state__text">
                    {error || 'No backtest results available'}
                </span>
            </div>
        </div>
    );

    const { summary, latest_backtest } = results;
    
    // Calculate color based on performance
    const getColor = (value, thresholds) => {
        if (value >= thresholds.good) return 'var(--color-long)';
        if (value >= thresholds.okay) return 'var(--color-warning)';
        return 'var(--color-short)';
    };

    return (
        <div className="card fade-in">
            <div className="card__header">
                <div className="ml-metric__icon">📈</div>
                <h2 className="card__title">Backtest Results</h2>
                {latest_backtest && (
                    <span style={{ 
                        fontSize: '0.7rem', 
                        color: 'var(--text-muted)',
                        background: 'var(--bg-tertiary)',
                        padding: '2px 8px',
                        borderRadius: '4px'
                    }}>
                        {new Date(latest_backtest.timestamp).toLocaleDateString()}
                    </span>
                )}
            </div>

            {summary && (
                <div className="ml-metrics" style={{ marginBottom: 'var(--spacing-md)' }}>
                    {/* Win Rate */}
                    <div className="ml-metric">
                        <div className="ml-metric__icon" style={{ background: 'rgba(16, 185, 129, 0.15)' }}>🎯</div>
                        <div className="ml-metric__info">
                            <div className="ml-metric__label">Win Rate</div>
                            <div className="ml-metric__value" style={{ 
                                color: getColor(summary.win_rate, { good: 55, okay: 45 })
                            }}>
                                {summary.win_rate?.toFixed(1) || 0}%
                            </div>
                        </div>
                    </div>

                    {/* Profit Factor */}
                    <div className="ml-metric">
                        <div className="ml-metric__icon" style={{ background: 'rgba(99, 102, 241, 0.15)' }}>💰</div>
                        <div className="ml-metric__info">
                            <div className="ml-metric__label">Profit Factor</div>
                            <div className="ml-metric__value" style={{ 
                                color: getColor(summary.profit_factor, { good: 1.5, okay: 1.0 })
                            }}>
                                {summary.profit_factor?.toFixed(2) || 'N/A'}
                            </div>
                        </div>
                    </div>

                    {/* Total Trades */}
                    <div className="ml-metric">
                        <div className="ml-metric__icon" style={{ background: 'rgba(139, 92, 246, 0.15)' }}>📊</div>
                        <div className="ml-metric__info">
                            <div className="ml-metric__label">Total Trades</div>
                            <div className="ml-metric__value" style={{ color: 'var(--accent-primary)' }}>
                                {summary.total_trades || 0}
                            </div>
                        </div>
                    </div>

                    {/* Net Profit */}
                    <div className="ml-metric">
                        <div className="ml-metric__icon" style={{ background: 'rgba(6, 182, 212, 0.15)' }}>💵</div>
                        <div className="ml-metric__info">
                            <div className="ml-metric__label">Net Profit</div>
                            <div className="ml-metric__value" style={{ 
                                color: (summary.net_profit || 0) >= 0 ? 'var(--color-long)' : 'var(--color-short)'
                            }}>
                                {summary.net_profit >= 0 ? '+' : ''}{summary.net_profit?.toFixed(2) || 0}%
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Suggestions Section */}
            {results.suggestions && results.suggestions.length > 0 && (
                <div style={{ 
                    marginTop: 'var(--spacing-md)', 
                    borderTop: '1px solid var(--border-color)', 
                    paddingTop: 'var(--spacing-md)' 
                }}>
                    <h4 style={{ 
                        fontSize: '0.75rem', 
                        fontWeight: '600', 
                        color: 'var(--text-muted)', 
                        textTransform: 'uppercase', 
                        letterSpacing: '0.5px',
                        marginBottom: 'var(--spacing-sm)'
                    }}>
                        💡 Suggestions
                    </h4>
                    <ul style={{ 
                        listStyle: 'none', 
                        padding: 0, 
                        margin: 0,
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 'var(--spacing-xs)'
                    }}>
                        {results.suggestions.map((suggestion, idx) => (
                            <li key={idx} style={{ 
                                fontSize: '0.8rem', 
                                color: 'var(--text-secondary)',
                                padding: 'var(--spacing-xs) var(--spacing-sm)',
                                background: 'var(--bg-tertiary)',
                                borderRadius: '4px',
                                borderLeft: '3px solid var(--accent-primary)'
                            }}>
                                {suggestion}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default BacktestResults;
