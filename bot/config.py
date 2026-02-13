import os
import json
from dotenv import load_dotenv

load_dotenv() 


def _parse_json_env(var_name: str, default):
    """Best-effort parse of JSON config from env var."""
    raw = os.getenv(var_name, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"Warning: invalid JSON in {var_name}, ignoring override")
        return default


def _parse_csv_env(var_name: str, default: str = ""):
    """Parse comma-separated env values into a trimmed list."""
    raw = os.getenv(var_name, default)
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]

# API credentials
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Basic configuration
# Safe defaults: run on testnet unless explicitly disabled and never allow live
# trading unless explicitly enabled.
TESTNET = os.getenv('TESTNET', 'True').lower() in ('true', '1', 't')
LIVE_TRADING_ENABLED = os.getenv('ENABLE_LIVE_TRADING', 'False').lower() in ('true', '1', 't')
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')
DEFAULT_API_AUTH_ENABLED = "False" if TESTNET else "True"
DEFAULT_ROLLOUT_ENFORCE_PRODUCTION_GATE = "False" if TESTNET else "True"

# Trading strategy configuration
TRADING_CONFIG = {
    # Enabled strategies - set to False to disable
    "strategies": {
        "simple": {
            "enabled": os.getenv('ENABLE_SIMPLE_STRATEGY', 'True').lower() in ('true', '1', 't'),
            "timeframe": os.getenv('SIMPLE_STRATEGY_TIMEFRAME', '1m'),
            "weight": float(os.getenv('SIMPLE_STRATEGY_WEIGHT', '1.0')),
            "parameters": {}  # Simple strategy doesn't have additional parameters
        },
        "technical": {
            "enabled": os.getenv('ENABLE_TECHNICAL_STRATEGY', 'True').lower() in ('true', '1', 't'),
            "timeframe": os.getenv('TECHNICAL_STRATEGY_TIMEFRAME', '1h'),
            "weight": float(os.getenv('TECHNICAL_STRATEGY_WEIGHT', '2.0')),  # Technical analysis has higher weight by default
            "parameters": {
                # RSI parameters
                "rsi_period": int(os.getenv('RSI_PERIOD', '14')),
                "rsi_oversold": float(os.getenv('RSI_OVERSOLD', '30')),
                "rsi_overbought": float(os.getenv('RSI_OVERBOUGHT', '70')),
                
                # Bollinger Bands parameters
                "bb_period": int(os.getenv('BB_PERIOD', '20')),
                "bb_std_dev": float(os.getenv('BB_STD_DEV', '2.0')),
                
                # MACD parameters
                "macd_fast_period": int(os.getenv('MACD_FAST_PERIOD', '12')),
                "macd_slow_period": int(os.getenv('MACD_SLOW_PERIOD', '26')),
                "macd_signal_period": int(os.getenv('MACD_SIGNAL_PERIOD', '9'))
            }
        },
        # Can add more strategies here when implemented
        "custom": {
            "enabled": os.getenv('ENABLE_CUSTOM_STRATEGY', 'False').lower() in ('true', '1', 't'),
            "timeframe": os.getenv('CUSTOM_STRATEGY_TIMEFRAME', '4h'),
            "weight": float(os.getenv('CUSTOM_STRATEGY_WEIGHT', '1.0')),
            "parameters": {},  # Custom strategy parameters would be defined here
            "module_path": os.getenv('CUSTOM_STRATEGY_MODULE', '')  # Path to custom strategy module
        }
    },
    
    # Decision-making configuration
    "decision_making": {
        # LLM configuration
        "llm": {
            "enabled": os.getenv('ENABLE_LLM_DECISIONS', 'True').lower() in ('true', '1', 't'),
            "required_confidence": float(os.getenv('LLM_REQUIRED_CONFIDENCE', '0.6')),
            "models": {
                "primary": {
                    "provider": os.getenv('LLM_PRIMARY_PROVIDER', 'deepseek'),
                    "model": os.getenv('LLM_PRIMARY_MODEL', 'deepseek-reasoner'),
                    "api_key": os.getenv('LLM_API_KEY', ''),
                    "api_endpoint": os.getenv('LLM_API_ENDPOINT', 'https://api.deepseek.com/v1/chat/completions'),
                    "temperature": float(os.getenv('LLM_TEMPERATURE', '0.3'))
                },
                "secondary": {
                    "provider": os.getenv('LLM_SECONDARY_PROVIDER', 'openai'),
                    "model": os.getenv('LLM_SECONDARY_MODEL', 'gpt-5-mini'),
                    "api_key": os.getenv('OPENAI_API_KEY', ''),
                    "api_endpoint": os.getenv('OPENAI_API_ENDPOINT', 'https://api.openai.com/v1/chat/completions'),
                    "temperature": float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
                }
            }
        },
        
        # Consensus configuration (for combining multiple strategy signals)
        "consensus": {
            "method": os.getenv('CONSENSUS_METHOD', 'weighted_majority'),  # Options: 'simple_majority', 'weighted_majority', 'unanimous'
            "min_strategies": int(os.getenv('MIN_STRATEGIES_FOR_DECISION', '2')),
            "llm_agreement_required": os.getenv('LLM_AGREEMENT_REQUIRED', 'True').lower() in ('true', '1', 't')
        }
    },
    
    # Trading parameters
    "trading": {
        "trade_mode": os.getenv('TRADE_MODE', 'SPOT').upper(),
        "enable_futures_shorts": os.getenv('ENABLE_FUTURES_SHORTS', 'False').lower() in ('true', '1', 't'),
        "default_order_amount_usd": float(os.getenv('DEFAULT_ORDER_AMOUNT_USD', '10.0')),
        "max_order_amount_usd": float(os.getenv('MAX_ORDER_AMOUNT_USD', '100.0')),
        # Hard pre-trade risk limits
        "max_order_notional_usd": float(
            os.getenv('MAX_ORDER_NOTIONAL_USD', os.getenv('MAX_ORDER_AMOUNT_USD', '100.0'))
        ),
        "max_position_exposure_usd": float(os.getenv('MAX_POSITION_EXPOSURE_USD', '250.0')),
        "max_short_notional_usd": float(os.getenv('MAX_SHORT_NOTIONAL_USD', '100.0')),
        "default_futures_leverage": float(os.getenv('DEFAULT_FUTURES_LEVERAGE', '2.0')),
        "max_short_leverage": float(os.getenv('MAX_SHORT_LEVERAGE', '3.0')),
        "min_short_liquidation_buffer_pct": float(os.getenv('MIN_SHORT_LIQUIDATION_BUFFER_PCT', '20.0')),
        "risk_percentage": float(os.getenv('RISK_PERCENTAGE', '1.0')),
        "profit_target_percentage": float(os.getenv('PROFIT_TARGET_PERCENTAGE', '3.0')),
        "stop_loss_percentage": float(os.getenv('STOP_LOSS_PERCENTAGE', '2.0')),
        "enable_stop_loss": os.getenv('ENABLE_STOP_LOSS', 'True').lower() in ('true', '1', 't'),
        "enable_take_profit": os.getenv('ENABLE_TAKE_PROFIT', 'True').lower() in ('true', '1', 't')
    },

    # Market regime detection
    "regime": {
        "enabled": os.getenv('ENABLE_REGIME_DETECTION', 'True').lower() in ('true', '1', 't'),
        "lookback_candles": int(os.getenv('REGIME_LOOKBACK_CANDLES', '50')),
        "trend_threshold_pct": float(os.getenv('REGIME_TREND_THRESHOLD_PCT', '0.02')),
        "sideways_threshold_pct": float(os.getenv('REGIME_SIDEWAYS_THRESHOLD_PCT', '0.01')),
        "high_volatility_threshold_pct": float(os.getenv('REGIME_HIGH_VOLATILITY_THRESHOLD_PCT', '0.015')),
    },

    # Regime-based strategy policy and switch controls
    "policy": {
        "enabled": os.getenv('ENABLE_REGIME_POLICY', 'True').lower() in ('true', '1', 't'),
        "default_regime": os.getenv('POLICY_DEFAULT_REGIME', 'SIDEWAYS').upper(),
        "default_size_multiplier": float(os.getenv('POLICY_DEFAULT_SIZE_MULTIPLIER', '1.0')),
        "min_switch_confidence": float(os.getenv('POLICY_MIN_SWITCH_CONFIDENCE', '0.55')),
        "switch_hysteresis_confirmations": int(os.getenv('POLICY_SWITCH_HYSTERESIS_CONFIRMATIONS', '2')),
        "switch_cooldown_seconds": int(os.getenv('POLICY_SWITCH_COOLDOWN_SECONDS', '900')),
        "max_strategy_turnover_ratio": float(os.getenv('POLICY_MAX_STRATEGY_TURNOVER_RATIO', '1.0')),
        "switch_shadow_mode": os.getenv('POLICY_SWITCH_SHADOW_MODE', 'False').lower() in ('true', '1', 't'),
        "regimes": _parse_json_env('POLICY_REGIME_CONFIG_JSON', {}),
    },

    # Point-in-time data quality pipeline (P2-01)
    "data_pipeline": {
        "enabled": os.getenv('ENABLE_DATA_PIPELINE_PIT', 'True').lower() in ('true', '1', 't'),
        "require_monotonic_timestamps": os.getenv(
            'PIT_REQUIRE_MONOTONIC_TIMESTAMPS', 'True'
        ).lower() in ('true', '1', 't'),
        "feature_source": os.getenv('PIT_FEATURE_SOURCE', 'exchange_ohlcv'),
    },

    # Hard live risk engine controls (P2-05)
    "risk_engine": {
        "enabled": os.getenv('ENABLE_RISK_ENGINE', 'True').lower() in ('true', '1', 't'),
        "max_drawdown_pct": float(os.getenv('RISK_ENGINE_MAX_DRAWDOWN_PCT', '20.0')),
        "max_gross_exposure_usd": float(os.getenv('RISK_ENGINE_MAX_GROSS_EXPOSURE_USD', '300.0')),
        "daily_loss_limit_usd": float(os.getenv('RISK_ENGINE_DAILY_LOSS_LIMIT_USD', '100.0')),
        "kill_switch_enabled": os.getenv('RISK_ENGINE_KILL_SWITCH_ENABLED', 'True').lower() in ('true', '1', 't'),
        "allow_risk_reducing_orders": os.getenv(
            'RISK_ENGINE_ALLOW_RISK_REDUCING_ORDERS', 'True'
        ).lower() in ('true', '1', 't'),
    },

    # Exchange reconciliation controls (P2-06)
    "reconciliation": {
        "enabled": os.getenv('ENABLE_RECONCILIATION', 'True').lower() in ('true', '1', 't'),
        "run_on_startup": os.getenv('RECONCILIATION_RUN_ON_STARTUP', 'True').lower() in ('true', '1', 't'),
        "interval_loops": int(os.getenv('RECONCILIATION_INTERVAL_LOOPS', '5')),
        "position_tolerance": float(os.getenv('RECONCILIATION_POSITION_TOLERANCE', '0.000001')),
    },

    # Observability stack controls (P2-08)
    "observability": {
        "enabled": os.getenv('ENABLE_OBSERVABILITY', 'True').lower() in ('true', '1', 't'),
        "max_events": int(os.getenv('OBS_MAX_EVENTS', '4000')),
        "latency_alert_ms": float(os.getenv('OBS_LATENCY_ALERT_MS', '2500.0')),
        "error_rate_alert_threshold": float(os.getenv('OBS_ERROR_RATE_ALERT_THRESHOLD', '0.25')),
        "error_rate_min_events": int(os.getenv('OBS_ERROR_RATE_MIN_EVENTS', '20')),
        "persistence_enabled": os.getenv('OBS_PERSISTENCE_ENABLED', 'True').lower() in ('true', '1', 't'),
        "persistence_db_url": os.getenv('OBS_PERSISTENCE_DB_URL', 'sqlite:///data/observability.db'),
    },

    # Edge-decay monitoring (P2-07)
    "monitoring": {
        "enabled": os.getenv('ENABLE_EDGE_MONITORING', 'True').lower() in ('true', '1', 't'),
        "window_size": int(os.getenv('EDGE_MONITOR_WINDOW_SIZE', '50')),
        "min_samples": int(os.getenv('EDGE_MONITOR_MIN_SAMPLES', '20')),
        "derisk_hit_rate_threshold": float(os.getenv('EDGE_MONITOR_DERISK_HIT_RATE', '0.45')),
        "disable_hit_rate_threshold": float(os.getenv('EDGE_MONITOR_DISABLE_HIT_RATE', '0.35')),
        "derisk_mean_return_threshold": float(os.getenv('EDGE_MONITOR_DERISK_MEAN_RETURN', '-0.0002')),
        "derisk_size_multiplier": float(os.getenv('EDGE_MONITOR_DERISK_MULTIPLIER', '0.5')),
        "disable_sticky": os.getenv('EDGE_MONITOR_DISABLE_STICKY', 'True').lower() in ('true', '1', 't'),
    },

    # Telegram notification controls
    "notifications": {
        "enabled": os.getenv('ENABLE_TELEGRAM_NOTIFICATIONS', 'True').lower() in ('true', '1', 't'),
        "bot_token": os.getenv('TELEGRAM_BOT_TOKEN', ''),
        "chat_id": os.getenv('TELEGRAM_CHAT_ID', ''),
        "rate_limit_per_minute": int(os.getenv('TELEGRAM_RATE_LIMIT_PER_MINUTE', '20')),
        "min_alert_severity": os.getenv('TELEGRAM_MIN_ALERT_SEVERITY', 'high').lower(),
    },

    # API security controls
    "api_security": {
        "auth_enabled": os.getenv('API_AUTH_ENABLED', DEFAULT_API_AUTH_ENABLED).lower() in ('true', '1', 't'),
        "allow_public_health": os.getenv('API_ALLOW_PUBLIC_HEALTH', 'True').lower() in ('true', '1', 't'),
        "read_api_keys": _parse_csv_env('API_READ_KEYS', ''),
        "admin_api_keys": _parse_csv_env('API_ADMIN_KEYS', ''),
        "allowed_origins": _parse_csv_env(
            'API_ALLOWED_ORIGINS',
            'http://localhost:3000,http://127.0.0.1:3000,http://localhost:8001,http://127.0.0.1:8001',
        ),
        "allow_credentials": os.getenv('API_CORS_ALLOW_CREDENTIALS', 'False').lower() in ('true', '1', 't'),
        "allow_methods": _parse_csv_env('API_CORS_ALLOW_METHODS', 'GET,POST,PUT,DELETE,OPTIONS'),
        "allow_headers": _parse_csv_env(
            'API_CORS_ALLOW_HEADERS',
            'Authorization,Content-Type,X-API-Key,X-Request-ID',
        ),
        "rate_limit_enabled": os.getenv('API_RATE_LIMIT_ENABLED', 'True').lower() in ('true', '1', 't'),
        "rate_limit_requests": int(os.getenv('API_RATE_LIMIT_REQUESTS', '120')),
        "rate_limit_window_seconds": int(os.getenv('API_RATE_LIMIT_WINDOW_SECONDS', '60')),
    },

    # Rollout gate configuration
    "rollout": {
        "enabled": os.getenv('ENABLE_ROLLOUT_GATES', 'True').lower() in ('true', '1', 't'),
        "state_store_path": os.getenv('ROLLOUT_STATE_STORE_PATH', 'data/rollout_gate_state.json'),
        "enforce_production_gate": os.getenv(
            'ROLLOUT_ENFORCE_PRODUCTION_GATE',
            DEFAULT_ROLLOUT_ENFORCE_PRODUCTION_GATE,
        ).lower() in ('true', '1', 't'),
        "required_rollout_id": os.getenv('ROLLOUT_REQUIRED_ID', ''),
        "min_shadow_samples": int(os.getenv('ROLLOUT_MIN_SHADOW_SAMPLES', '50')),
        "max_shadow_error_rate": float(os.getenv('ROLLOUT_MAX_SHADOW_ERROR_RATE', '0.20')),
        "min_canary_samples": int(os.getenv('ROLLOUT_MIN_CANARY_SAMPLES', '30')),
        "max_canary_error_rate": float(os.getenv('ROLLOUT_MAX_CANARY_ERROR_RATE', '0.15')),
        "max_canary_drawdown_pct": float(os.getenv('ROLLOUT_MAX_CANARY_DRAWDOWN_PCT', '8.0')),
        "min_canary_total_return_pct": float(os.getenv('ROLLOUT_MIN_CANARY_RETURN_PCT', '-1.0')),
        "max_canary_latency_p95_ms": float(os.getenv('ROLLOUT_MAX_CANARY_LATENCY_P95_MS', '3000.0')),
        "require_quality_gate": os.getenv('ROLLOUT_REQUIRE_QUALITY_GATE', 'True').lower() in ('true', '1', 't'),
    },

    # Strategy quality gate thresholds
    "quality_gate": {
        "enabled": os.getenv('ENABLE_STRATEGY_QUALITY_GATE', 'True').lower() in ('true', '1', 't'),
        "min_walk_forward_folds": int(os.getenv('QUALITY_MIN_WALK_FORWARD_FOLDS', '3')),
        "min_sharpe_ratio": float(os.getenv('QUALITY_MIN_SHARPE_RATIO', '0.20')),
        "min_calmar_ratio": float(os.getenv('QUALITY_MIN_CALMAR_RATIO', '0.10')),
        "max_drawdown_pct": float(os.getenv('QUALITY_MAX_DRAWDOWN_PCT', '25.0')),
        "min_regime_samples": int(os.getenv('QUALITY_MIN_REGIME_SAMPLES', '20')),
        "min_regime_win_rate": float(os.getenv('QUALITY_MIN_REGIME_WIN_RATE', '0.45')),
        "min_regimes_passing": int(os.getenv('QUALITY_MIN_REGIMES_PASSING', '2')),
    },

    # Promotion benchmark thresholds (G0-02)
    "promotion_benchmarks": {
        "min_total_trades": int(os.getenv('BENCHMARK_MIN_TOTAL_TRADES', '25')),
        "min_net_return_pct": float(os.getenv('BENCHMARK_MIN_NET_RETURN_PCT', '1.0')),
        "min_sharpe_ratio": float(os.getenv('BENCHMARK_MIN_SHARPE_RATIO', os.getenv('QUALITY_MIN_SHARPE_RATIO', '0.20'))),
        "min_calmar_ratio": float(os.getenv('BENCHMARK_MIN_CALMAR_RATIO', os.getenv('QUALITY_MIN_CALMAR_RATIO', '0.10'))),
        "max_drawdown_pct": float(os.getenv('BENCHMARK_MAX_DRAWDOWN_PCT', os.getenv('QUALITY_MAX_DRAWDOWN_PCT', '25.0'))),
        "require_quality_gate": os.getenv('BENCHMARK_REQUIRE_QUALITY_GATE', 'True').lower() in ('true', '1', 't'),
    },
    
    # Timeframe configuration for market data
    "timeframes": {
        "primary": os.getenv('PRIMARY_TIMEFRAME', '1m'),
        "secondary": os.getenv('SECONDARY_TIMEFRAME', '1h'),
        "candle_limit": int(os.getenv('CANDLE_LIMIT', '100'))
    },
    
    # Operational settings
    "operation": {
        "loop_interval_seconds": int(os.getenv('LOOP_INTERVAL_SECONDS', '60')),
        "max_consecutive_errors": int(os.getenv('MAX_CONSECUTIVE_ERRORS', '5')),
        "max_backoff_seconds": int(os.getenv('MAX_BACKOFF_SECONDS', '3600')),
        "exchange_timeout_seconds": float(os.getenv('EXCHANGE_TIMEOUT_SECONDS', '10')),
        "exchange_max_retries": int(os.getenv('EXCHANGE_MAX_RETRIES', '2')),
        "exchange_retry_backoff_seconds": float(os.getenv('EXCHANGE_RETRY_BACKOFF_SECONDS', '0.5')),
        "exchange_circuit_breaker_threshold": int(os.getenv('EXCHANGE_CIRCUIT_BREAKER_THRESHOLD', '5')),
        "exchange_circuit_breaker_cooldown_seconds": float(os.getenv('EXCHANGE_CIRCUIT_BREAKER_COOLDOWN_SECONDS', '30')),
        "llm_timeout_seconds": float(os.getenv('LLM_TIMEOUT_SECONDS', '20')),
        "llm_max_retries": int(os.getenv('LLM_MAX_RETRIES', '2')),
        "llm_retry_backoff_seconds": float(os.getenv('LLM_RETRY_BACKOFF_SECONDS', '0.75')),
        "llm_circuit_breaker_threshold": int(os.getenv('LLM_CIRCUIT_BREAKER_THRESHOLD', '4')),
        "llm_circuit_breaker_cooldown_seconds": float(os.getenv('LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS', '45')),
        "enable_live_trading": LIVE_TRADING_ENABLED
    }
}

# Allow configuration from a JSON file if specified
config_file_path = os.getenv('CONFIG_FILE', '')
if config_file_path and os.path.exists(config_file_path):
    try:
        with open(config_file_path, 'r') as config_file:
            file_config = json.load(config_file)
            
            # Deep merge the file config with the environment config
            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        deep_update(d[k], v)
                    else:
                        d[k] = v
            
            deep_update(TRADING_CONFIG, file_config)
    except Exception as e:
        print(f"Error loading config file: {e}")

# Helper functions to access config values
def get_strategy_config(strategy_name):
    """Get the configuration for a specific strategy."""
    return TRADING_CONFIG["strategies"].get(strategy_name, {})

def is_strategy_enabled(strategy_name):
    """Check if a specific strategy is enabled."""
    strategy = get_strategy_config(strategy_name)
    return strategy.get("enabled", False)

def get_strategy_parameter(strategy_name, parameter_name, default=None):
    """Get a specific parameter for a strategy."""
    strategy = get_strategy_config(strategy_name)
    return strategy.get("parameters", {}).get(parameter_name, default)

def get_trading_parameter(parameter_name, default=None):
    """Get a specific trading parameter."""
    return TRADING_CONFIG["trading"].get(parameter_name, default)


def get_regime_config():
    """Get regime detection configuration."""
    return TRADING_CONFIG.get("regime", {})


def get_policy_config():
    """Get regime-based strategy policy configuration."""
    return TRADING_CONFIG.get("policy", {})


def get_data_pipeline_config():
    """Get point-in-time data pipeline configuration."""
    return TRADING_CONFIG.get("data_pipeline", {})


def get_risk_engine_config():
    """Get live risk-engine configuration."""
    return TRADING_CONFIG.get("risk_engine", {})


def get_reconciliation_config():
    """Get exchange reconciliation configuration."""
    return TRADING_CONFIG.get("reconciliation", {})


def get_observability_config():
    """Get observability stack configuration."""
    return TRADING_CONFIG.get("observability", {})


def get_monitoring_config():
    """Get edge-decay monitoring configuration."""
    return TRADING_CONFIG.get("monitoring", {})


def get_notification_config():
    """Get Telegram notification configuration."""
    return TRADING_CONFIG.get("notifications", {})


def get_api_security_config():
    """Get API security configuration."""
    return TRADING_CONFIG.get("api_security", {})


def get_rollout_config():
    """Get rollout gate configuration."""
    return TRADING_CONFIG.get("rollout", {})


def get_quality_gate_config():
    """Get strategy quality-gate configuration."""
    return TRADING_CONFIG.get("quality_gate", {})


def get_promotion_benchmark_config():
    """Get promotion benchmark threshold configuration."""
    return TRADING_CONFIG.get("promotion_benchmarks", {})


def get_trade_mode() -> str:
    """Get normalized trade mode (`SPOT` or `FUTURES`)."""
    mode = str(TRADING_CONFIG["trading"].get("trade_mode", "SPOT")).upper()
    if mode not in {"SPOT", "FUTURES"}:
        return "SPOT"
    return mode


def is_futures_short_enabled() -> bool:
    """Check if explicit futures shorting is enabled."""
    return bool(TRADING_CONFIG["trading"].get("enable_futures_shorts", False))

def is_llm_enabled():
    """Check if LLM-based decision making is enabled."""
    return TRADING_CONFIG["decision_making"]["llm"].get("enabled", False)

def get_required_llm_confidence():
    """Get the required confidence threshold for LLM decisions."""
    return TRADING_CONFIG["decision_making"]["llm"].get("required_confidence", 0.6)

def is_llm_agreement_required():
    """Check if LLM agreement is required for trade execution."""
    return TRADING_CONFIG["decision_making"]["consensus"].get("llm_agreement_required", True)

def get_consensus_method():
    """Get the consensus method for combining strategy signals."""
    return TRADING_CONFIG["decision_making"]["consensus"].get("method", "weighted_majority")

def get_loop_interval():
    """Get the interval between trading loop iterations in seconds."""
    return TRADING_CONFIG["operation"].get("loop_interval_seconds", 60)

def get_operation_parameter(parameter_name, default=None):
    """Get a specific operational parameter."""
    return TRADING_CONFIG["operation"].get(parameter_name, default)

def is_live_trading_enabled():
    """Check if explicit live-trading execution is enabled."""
    return TRADING_CONFIG["operation"].get("enable_live_trading", False)
