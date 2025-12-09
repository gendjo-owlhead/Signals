"""
Configuration management for Auction Market Signal Generator.
All settings loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Binance API Configuration
    binance_api_key: str = Field(default="", env="BINANCE_API_KEY")
    binance_api_secret: str = Field(default="", env="BINANCE_API_SECRET")
    binance_testnet: bool = Field(default=True, env="BINANCE_TESTNET")
    
    # Trading Configuration
    trading_pairs: List[str] = Field(
        default=["BTCUSDT", "ETHUSDT"],
        env="TRADING_PAIRS"
    )
    timeframes: List[str] = Field(
        default=["1m", "5m"],
        env="TIMEFRAMES"
    )
    primary_timeframe: str = Field(default="5m", env="PRIMARY_TIMEFRAME")
    
    # Validators for comma-separated list fields
    @field_validator('trading_pairs', 'timeframes', mode='before')
    @classmethod
    def parse_comma_separated(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        return v
    
    
    # Volume Profile Settings
    vp_lookback_periods: int = Field(default=100, env="VP_LOOKBACK_PERIODS")
    lvn_threshold_percentile: float = Field(default=20.0, env="LVN_THRESHOLD")
    value_area_percentage: float = Field(default=70.0, env="VALUE_AREA_PCT")
    
    # Order Flow Settings
    cvd_lookback: int = Field(default=50, env="CVD_LOOKBACK")
    aggression_threshold: float = Field(default=1.5, env="AGGRESSION_THRESHOLD")
    large_order_multiplier: float = Field(default=3.0, env="LARGE_ORDER_MULT")
    
    # Signal Settings
    trend_confidence_threshold: float = Field(default=0.7, env="TREND_CONF_THRESH")
    reversion_confidence_threshold: float = Field(default=0.7, env="REV_CONF_THRESH")
    
    # TTFT Strategy Settings
    ttft_trend_length: int = Field(default=20, env="TTFT_TREND_LENGTH")
    ttft_atr_length: int = Field(default=200, env="TTFT_ATR_LENGTH")
    ttft_atr_multiplier: float = Field(default=0.8, env="TTFT_ATR_MULT")
    ttft_sl_multiplier: float = Field(default=1.0, env="TTFT_SL_MULT")
    ttft_rr_ratio: float = Field(default=2.0, env="TTFT_RR_RATIO")
    ttft_confidence_threshold: float = Field(default=0.7, env="TTFT_CONF_THRESH")
    
    # ML Settings
    ml_model_path: str = Field(default="data/models", env="ML_MODEL_PATH")
    online_learning_enabled: bool = Field(default=True, env="ONLINE_LEARNING")
    min_samples_for_training: int = Field(default=50, env="MIN_TRAIN_SAMPLES")
    
    # Trading Execution Settings
    trading_enabled: bool = Field(default=False, env="TRADING_ENABLED")
    max_position_size_usdt: float = Field(default=110.0, env="MAX_POSITION_SIZE_USDT")
    risk_per_trade_pct: float = Field(default=1.0, env="RISK_PER_TRADE_PCT")
    max_concurrent_positions: int = Field(default=3, env="MAX_CONCURRENT_POSITIONS")
    daily_loss_limit_pct: float = Field(default=5.0, env="DAILY_LOSS_LIMIT_PCT")
    min_signal_confidence: float = Field(default=0.50, env="MIN_SIGNAL_CONFIDENCE")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra env vars that don't match fields


# Global settings instance
settings = Settings()


# Binance Futures WebSocket URLs
BINANCE_FUTURES_WS_URL = "wss://fstream.binance.com/ws"
BINANCE_FUTURES_TESTNET_WS_URL = "wss://stream.binancefuture.com/ws"

# Binance Spot WebSocket URLs (fallback)
BINANCE_SPOT_WS_URL = "wss://stream.binance.com:9443/ws"
BINANCE_SPOT_TESTNET_WS_URL = "wss://testnet.binance.vision/ws"

def get_ws_url() -> str:
    """Get appropriate WebSocket URL based on testnet setting."""
    if settings.binance_testnet:
        return BINANCE_FUTURES_TESTNET_WS_URL
    return BINANCE_FUTURES_WS_URL


# Binance Futures REST API URLs
BINANCE_FUTURES_API_URL = "https://fapi.binance.com"
BINANCE_FUTURES_TESTNET_API_URL = "https://testnet.binancefuture.com"

# Binance Spot REST API URLs (fallback)
BINANCE_SPOT_API_URL = "https://api.binance.com"
BINANCE_SPOT_TESTNET_API_URL = "https://testnet.binance.vision"

def get_api_url() -> str:
    """Get appropriate REST API URL based on testnet setting."""
    if settings.binance_testnet:
        return BINANCE_FUTURES_TESTNET_API_URL
    return BINANCE_FUTURES_API_URL

