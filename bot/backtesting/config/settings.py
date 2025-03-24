"""
Centralized configuration settings for the backtesting engine.
This module contains all configurable parameters used across the backtesting engine.
"""
from typing import Dict, Any
from pathlib import Path
import os

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Default backtesting parameters
DEFAULT_BACKTEST_SETTINGS = {
    "initial_capital": 10000.0,
    "commission_rate": 0.001,  # 0.1%
    "min_candles_required": 30,  # Minimum candles needed for indicators
    "position_size_pct": 0.95,  # Use 95% of capital per trade
    "random_seed": 42,  # For reproducible random data in tests
}

# Chart settings
CHART_SETTINGS = {
    "default_figsize": (14, 10),
    "dpi": 100,
    "line_width": 1.5,
    "marker_size": 100,
    "color_scheme": {
        "equity_line": "blue",
        "buy_marker": "green",
        "sell_marker": "red",
        "drawdown_fill": "red",
        "rsi_line": "purple",
        "overbought_line": "red",
        "oversold_line": "green",
        "macd_line": "blue",
        "signal_line": "red",
        "positive_histogram": "green",
        "negative_histogram": "red",
    }
}

# Database settings
DATABASE_SETTINGS = {
    "batch_size": 1000,  # Number of records to process in a batch
    "cache_enabled": True,
    "cache_ttl": 3600,  # Cache time-to-live in seconds
}

# Reporting settings
REPORT_SETTINGS = {
    "max_trades_in_report": 100,
    "include_trade_details": True,
    "generate_html": True,
    "generate_csv": True,
    "generate_plots": True,
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_handler_enabled": True,
    "console_handler_enabled": True,
}

# Performance optimization settings
PERFORMANCE_SETTINGS = {
    "use_parallel_processing": True,
    "num_processes": os.cpu_count() or 4,  # Default to number of CPU cores
    "use_vectorized_operations": True,
    "chunk_size": 10000,  # For processing large datasets in chunks
}

def get_config() -> Dict[str, Any]:
    """
    Returns the complete configuration dictionary.
    This is useful for passing to functions that need access to all settings.
    
    Returns:
        Dict[str, Any]: Complete configuration dictionary
    """
    return {
        "backtest": DEFAULT_BACKTEST_SETTINGS,
        "chart": CHART_SETTINGS,
        "database": DATABASE_SETTINGS,
        "report": REPORT_SETTINGS,
        "logging": LOGGING_CONFIG,
        "performance": PERFORMANCE_SETTINGS,
        "paths": {
            "base_dir": str(BASE_DIR),
            "output_dir": str(OUTPUT_DIR),
            "logs_dir": str(LOGS_DIR),
        }
    }

def update_config(section: str, key: str, value: Any) -> None:
    """
    Update a specific configuration setting.
    
    Args:
        section: The configuration section (e.g., 'backtest', 'chart')
        key: The setting key to update
        value: The new value
    """
    config_map = {
        "backtest": DEFAULT_BACKTEST_SETTINGS,
        "chart": CHART_SETTINGS,
        "database": DATABASE_SETTINGS,
        "report": REPORT_SETTINGS,
        "logging": LOGGING_CONFIG,
        "performance": PERFORMANCE_SETTINGS,
    }
    
    if section in config_map and key in config_map[section]:
        config_map[section][key] = value
    else:
        raise ValueError(f"Invalid configuration section '{section}' or key '{key}'") 