import os
import json
from dotenv import load_dotenv

load_dotenv() 

# API credentials
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Basic configuration
TESTNET = os.getenv('TESTNET', 'False').lower() in ('true', '1', 't')
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')

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
                    "model": os.getenv('LLM_SECONDARY_MODEL', 'gpt-4o'),
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
        "default_order_amount_usd": float(os.getenv('DEFAULT_ORDER_AMOUNT_USD', '10.0')),
        "max_order_amount_usd": float(os.getenv('MAX_ORDER_AMOUNT_USD', '100.0')),
        "risk_percentage": float(os.getenv('RISK_PERCENTAGE', '1.0')),
        "profit_target_percentage": float(os.getenv('PROFIT_TARGET_PERCENTAGE', '3.0')),
        "stop_loss_percentage": float(os.getenv('STOP_LOSS_PERCENTAGE', '2.0')),
        "enable_stop_loss": os.getenv('ENABLE_STOP_LOSS', 'True').lower() in ('true', '1', 't'),
        "enable_take_profit": os.getenv('ENABLE_TAKE_PROFIT', 'True').lower() in ('true', '1', 't')
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
        "max_backoff_seconds": int(os.getenv('MAX_BACKOFF_SECONDS', '3600'))
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