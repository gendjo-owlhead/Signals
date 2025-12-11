import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * WebSocket hook for real-time data updates
 */
export function useWebSocket(url) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  
  const connect = useCallback(() => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = url || `${protocol}//${window.location.host}/ws`;
      
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        setIsConnected(true);
        setError(null);
        console.log('WebSocket connected');
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };
      
      wsRef.current.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('Connection error');
      };
      
      wsRef.current.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket disconnected, reconnecting...');
        
        // Reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 3000);
      };
    } catch (e) {
      setError(e.message);
    }
  }, [url]);
  
  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);
  
  const sendMessage = useCallback((message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);
  
  return { isConnected, lastMessage, error, sendMessage };
}

/**
 * API hook for REST endpoints
 */
export function useAPI() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const fetchData = useCallback(async (endpoint) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api${endpoint}`);
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
      const data = await response.json();
      setLoading(false);
      return data;
    } catch (e) {
      setError(e.message);
      setLoading(false);
      return null;
    }
  }, []);
  
  return { fetchData, loading, error };
}

/**
 * Hook for fetching analysis data for a symbol
 */
export function useAnalysis(symbol) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    if (!symbol) return;
    
    const fetchAnalysis = async () => {
      setLoading(true);
      try {
        const response = await fetch(`/api/analysis/${symbol}`);
        if (response.ok) {
          const data = await response.json();
          setAnalysis(data);
        }
      } catch (e) {
        setError(e.message);
      }
      setLoading(false);
    };
    
    fetchAnalysis();
    const interval = setInterval(fetchAnalysis, 5000);
    
    return () => clearInterval(interval);
  }, [symbol]);
  
  return { analysis, loading, error };
}

/**
 * Hook for ML status
 */
export function useMLStatus() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('/api/ml/status');
        if (response.ok) {
          const data = await response.json();
          setStatus(data);
        }
      } catch (e) {
        console.error('Failed to fetch ML status:', e);
      }
      setLoading(false);
    };
    
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    
    return () => clearInterval(interval);
  }, []);
  
  return { status, loading };
}

/**
 * Hook for Volume Profile data
 */
export function useVolumeProfile(symbol) {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    if (!symbol) return;
    
    const fetchProfile = async () => {
      try {
        const response = await fetch(`/api/volume-profile/${symbol}`);
        if (response.ok) {
          const data = await response.json();
          setProfile(data);
        }
      } catch (e) {
        console.error('Failed to fetch volume profile:', e);
      }
      setLoading(false);
    };
    
    fetchProfile();
    const interval = setInterval(fetchProfile, 15000);
    
    return () => clearInterval(interval);
  }, [symbol]);
  
  return { profile, loading };
}

/**
 * Hook for Order Flow data
 */
export function useOrderFlow(symbol) {
  const [orderFlow, setOrderFlow] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    if (!symbol) return;
    
    const fetchOrderFlow = async () => {
      try {
        const response = await fetch(`/api/order-flow/${symbol}`);
        if (response.ok) {
          const data = await response.json();
          setOrderFlow(data);
        }
      } catch (e) {
        console.error('Failed to fetch order flow:', e);
      }
      setLoading(false);
    };
    
    fetchOrderFlow();
    const interval = setInterval(fetchOrderFlow, 3000);
    
    return () => clearInterval(interval);
  }, [symbol]);
  
  return { orderFlow, loading };
}

/**
 * Hook for active signals
 */
export function useSignals(symbol) {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchSignals = async () => {
      try {
        const url = symbol ? `/api/signals?symbol=${symbol}` : '/api/signals';
        const response = await fetch(url);
        if (response.ok) {
          const data = await response.json();
          setSignals(data.signals || []);
        }
      } catch (e) {
        console.error('Failed to fetch signals:', e);
      }
      setLoading(false);
    };
    
    fetchSignals();
    const interval = setInterval(fetchSignals, 2000);
    
    return () => clearInterval(interval);
  }, [symbol]);
  
  return { signals, loading };
}

/**
 * Hook for statistics
 */
export function useStatistics() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/statistics');
        if (response.ok) {
          const data = await response.json();
          setStats(data);
        }
      } catch (e) {
        console.error('Failed to fetch statistics:', e);
      }
      setLoading(false);
    };
    
    fetchStats();
    const interval = setInterval(fetchStats, 30000);
    
    return () => clearInterval(interval);
  }, []);
  
  return { stats, loading };
}

/**
 * Hook for trading account data (balance, positions, stats)
 */
export function useTrading() {
  const [account, setAccount] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const fetchAccount = useCallback(async () => {
    try {
      const response = await fetch('/api/trading/account');
      if (response.ok) {
        const data = await response.json();
        setAccount(data);
        setError(null);
      }
    } catch (e) {
      setError(e.message);
      console.error('Failed to fetch trading account:', e);
    }
    setLoading(false);
  }, []);
  
  useEffect(() => {
    fetchAccount();
    const interval = setInterval(fetchAccount, 5000); // Update every 5s
    
    return () => clearInterval(interval);
  }, [fetchAccount]);
  
  const startTrading = useCallback(async () => {
    try {
      const response = await fetch('/api/trading/start', { method: 'POST' });
      const data = await response.json();
      fetchAccount();
      return data;
    } catch (e) {
      console.error('Failed to start trading:', e);
      return { error: e.message };
    }
  }, [fetchAccount]);
  
  const stopTrading = useCallback(async () => {
    try {
      const response = await fetch('/api/trading/stop', { method: 'POST' });
      const data = await response.json();
      fetchAccount();
      return data;
    } catch (e) {
      console.error('Failed to stop trading:', e);
      return { error: e.message };
    }
  }, [fetchAccount]);
  
  const closePosition = useCallback(async (positionId) => {
    try {
      const response = await fetch(`/api/trading/close/${positionId}`, { method: 'POST' });
      const data = await response.json();
      fetchAccount();
      return data;
    } catch (e) {
      console.error('Failed to close position:', e);
      return { error: e.message };
    }
  }, [fetchAccount]);
  
  return { 
    account, 
    loading, 
    error, 
    startTrading, 
    stopTrading, 
    closePosition,
    refresh: fetchAccount 
  };
}

/**
 * Hook for backtest orchestrator status
 */
export function useBacktestOrchestrator() {
  const [status, setStatus] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [loading, setLoading] = useState(true);
  
  const fetchStatus = useCallback(async () => {
    try {
      const [statusRes, perfRes] = await Promise.all([
        fetch('/api/backtest/orchestrator/status'),
        fetch('/api/backtest/orchestrator/performance')
      ]);
      
      if (statusRes.ok) {
        setStatus(await statusRes.json());
      }
      if (perfRes.ok) {
        setPerformance(await perfRes.json());
      }
    } catch (e) {
      console.error('Failed to fetch orchestrator status:', e);
    }
    setLoading(false);
  }, []);
  
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Update every 30s
    
    return () => clearInterval(interval);
  }, [fetchStatus]);
  
  const startOrchestrator = useCallback(async () => {
    try {
      const response = await fetch('/api/backtest/orchestrator/start', { method: 'POST' });
      const data = await response.json();
      fetchStatus();
      return data;
    } catch (e) {
      console.error('Failed to start orchestrator:', e);
      return { error: e.message };
    }
  }, [fetchStatus]);
  
  const stopOrchestrator = useCallback(async () => {
    try {
      const response = await fetch('/api/backtest/orchestrator/stop', { method: 'POST' });
      const data = await response.json();
      fetchStatus();
      return data;
    } catch (e) {
      console.error('Failed to stop orchestrator:', e);
      return { error: e.message };
    }
  }, [fetchStatus]);
  
  return { 
    status, 
    performance, 
    loading, 
    startOrchestrator, 
    stopOrchestrator,
    refresh: fetchStatus 
  };
}

/**
 * Hook for multi-timeframe signals
 */
export function useMTFSignals(symbol) {
  const [mtfData, setMtfData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    if (!symbol) return;
    
    const fetchMTF = async () => {
      try {
        const response = await fetch(`/api/signals/multi-timeframe/${symbol}`);
        if (response.ok) {
          const data = await response.json();
          setMtfData(data);
        }
      } catch (e) {
        console.error('Failed to fetch MTF signals:', e);
      }
      setLoading(false);
    };
    
    fetchMTF();
    const interval = setInterval(fetchMTF, 5000);
    
    return () => clearInterval(interval);
  }, [symbol]);
  
  return { mtfData, loading };
}

/**
 * Hook for MTF analyzer overall status
 */
export function useMTFStatus() {
  const [mtfStatus, setMtfStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('/api/signals/mtf-status');
        if (response.ok) {
          const data = await response.json();
          setMtfStatus(data);
        }
      } catch (e) {
        console.error('Failed to fetch MTF status:', e);
      }
      setLoading(false);
    };
    
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    
    return () => clearInterval(interval);
  }, []);
  
  return { mtfStatus, loading };
}
