# CogniTrade: AI-Enhanced Trading Platform with Binance & LLM Integration

A scalable Python project for automated trading that combines traditional trading strategies with modern AI/LLM orchestration for decision making. CogniTrade connects to Binance (starting with paper trading via the Testnet) and is designed to easily integrate multiple LLMs to refine trade signals and manage order execution.

## Overview

CogniTrade aims to take you from basic trading bot functionality to a robust system that leverages AI:
- **Paper Trading:** Start with simulated trades using Binance Testnet.
- **Scalable Architecture:** Modular design to support multiple strategies and future integration with various LLMs.
- **AI Orchestration:** A dedicated module to incorporate LLMs (e.g., DeepSeek R1, GPT-4o, o3-mini, Claude 3.7 Sonnet) for enhanced decision making.
- **Extensible:** Easily add new exchanges, trading strategies, and real-time data streams.
- **Custom Strategies:** Plug-and-play system for adding your own trading strategies without modifying core code.
- **Comprehensive Configuration:** Highly configurable through environment variables or JSON configuration files.
- **Advanced Backtesting:** Completely refactored backtesting engine with modular architecture, better error handling, and improved performance.
- **Comprehensive Testing:** High-coverage test suite for all components with mocks for external dependencies.
- **Vectorized Backtesting:** Optimized performance with vectorized operations for high-speed strategy testing.
- **Arithmetic Validation:** Robust validation of profit/loss calculations with controlled datasets.

## Features

- **Binance API Integration:** Uses the official Binance API (via python-binance) for fetching market data, placing orders, and managing accounts.
- **Paper Trading Mode:** Safely test strategies on Binance Testnet before going live.
- **Strategy Module:** Contains logic for trading signals (e.g., based on technical indicators) with an abstract layer for future enhancements.
- **Custom Strategy Support:** Create your own strategies that can be dynamically loaded and configured without changing core code.
- **Enhanced Configuration System:**
  - Multiple configuration sources (environment variables and JSON files)
  - Structured configuration with sensible defaults
  - Direct access to configuration values through helper functions
  - Type-safe parameter retrieval
- **Advanced LLM Integration:** 
  - DeepSeek R1 integration for reasoning-based trading decisions
  - GPT-4o structured output processing for improved confidence estimation and reasoning
  - Support for multiple LLM providers with configurable parameters
  - Fallback to rule-based decisions when LLM services are unavailable
  - Detailed confidence scores for decision making
  - Complete transparency with both primary and secondary model responses stored in database
  - Enhanced reasoning capabilities with processed structured outputs
- **Flexible Decision Making:**
  - Multiple consensus methods (simple majority, weighted majority, unanimous)
  - Configurable strategy weights and confidence thresholds
  - LLM agreement requirements for trade execution
- **Order Management:** Robust order execution system with proper error handling and logging.
- **Database System:** Comprehensive data storage solution with optimized SQLite schema
  - Robust database integration layer for saving signals, trades, market data, and system alerts
  - Transparent logging of LLM decisions with full model responses
  - Clean separation between database access and trading logic
  - Optimized schema with appropriate indexes for performance
  - Self-healing database tables that ensure correct schema on initialization
- **Robust Project Structure:** Clean separation of concerns with modules for configuration, API calls, strategy logic, order management, and service orchestration.
- **Testing Suite:** Comprehensive unit and integration tests with high coverage metrics for all system components.
- **Enhanced Backtesting System:** Completely refactored backtesting capabilities with:
  - Modular architecture with separation of concerns (core, data, models, reporting, etc.)
  - Type-safe data structures using Python dataclasses 
  - Improved error handling with specific exception types
  - Market data caching for better performance
  - Multi-timeframe analysis for more robust trading decisions
  - Clean HTML reports using Jinja2 templates
  - Advanced visualization with matplotlib
  - Strategy optimization through parallel parameter grid search
  - Comprehensive performance metrics calculation
  - Data validation and robust safeguards
  - Synthetic test data generation with configurable parameters
  - Vectorized operations for high-speed strategy testing
  - Precise arithmetic validation for profit/loss calculations
- **Backend API:** RESTful API for interacting with trading bot functions and accessing historical data.

## CogniTrade Project Structure

```
trading_bot/
├── bot/
│   ├── __init__.py
│   ├── config.py              # Enhanced configuration loader with multiple sources
│   ├── binance_api.py         # Binance API wrapper (using python-binance)
│   ├── strategy.py            # Trading strategy interface and built-in strategies
│   ├── custom_strategies/     # Directory for user-defined custom strategies
│   │   ├── __init__.py        # Package initialization
│   │   └── my_strategy.py     # Example custom strategy implementation
│   ├── llm_manager.py         # Enhanced LLM orchestration with multiple model support
│   ├── order_manager.py       # Module for order execution and logging
│   ├── database.py            # Database management for storing trading data and analytics
│   ├── db_integration.py      # Integration layer between trading system and database
│   ├── backtesting.py         # Legacy backtesting engine (deprecated, for backward compatibility)
│   ├── backtesting/           # New modular backtesting system
│   │   ├── __init__.py        # Public API for backtesting 
│   │   ├── config/            # Configuration settings
│   │   ├── core/              # Core backtesting engine
│   │   ├── data/              # Market data management
│   │   ├── exceptions/        # Custom exceptions for better error handling
│   │   ├── models/            # Data models (using dataclasses)
│   │   ├── reporting/         # Report generation with Jinja2
│   │   ├── utils/             # Utility functions for calculations 
│   │   ├── visualization/     # Charting and plotting functionality
│   │   └── strategies/        # Strategy implementations
│   └── main.py                # Main entry point for running the bot
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI backend for bot control and data access
│   └── requirements.txt       # API-specific dependencies
├── tests/                     # Comprehensive tests (pytest)
│   ├── test_binance_api.py    # Tests for Binance API wrapper with >80% coverage
│   ├── test_main.py           # Tests for the main trading loop with proper mocks
│   ├── test_order_manager.py  # Tests for order execution and management
│   ├── test_strategy.py       # Tests for trading strategies
│   ├── test_llm_manager.py    # Tests for LLM integration
│   ├── test_backtesting/      # Tests for the new backtesting modules
│   ├── test_vectorized_backtesting.py # Tests for vectorized backtesting performance
│   ├── test_arithmetic_validation.py  # Validation of profit/loss calculations
│   ├── test_database.py       # Tests for database operations
│   ├── test_db_integration.py # Tests for database integration layer
│   ├── test_main_db_integration.py # Integration tests for main and database
│   ├── conftests.py           # Pytest configuration and fixtures
│   └── __init__.py            # Test package initialization
├── examples/                  # Example scripts
│   ├── simple_backtest.py     # Simple SMA crossover strategy example
│   ├── rsi_backtest.py        # RSI strategy implementation example
│   ├── compare_strategies.py  # Tool for comparing strategy performance
│   └── backtest_example.py    # Demo of the refactored backtesting engine
├── data/                      # Directory for storing market data
├── logs/                      # Trading and backtesting logs
├── order_logs/                # Logs of executed orders
├── output/                    # Directory for backtesting outputs
│   ├── charts/                # Generated charts from backtests
│   └── reports/               # HTML and other report formats
├── config.sample.json         # Sample JSON configuration file
├── requirements.txt           # List of dependencies
├── .env.example               # Example environment variables file
├── .env                       # Environment variables (API keys, etc.)
└── README.md                  # Project documentation (this file)
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone git@github.com:KarimMesallam/CogniTrade.git
   cd CogniTrade
   ```

2. **Set Up a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   Create a `.env` file in the root folder based on the provided `.env.example`:
   ```ini
   # Basic configuration
   API_KEY=your_binance_testnet_api_key
   API_SECRET=your_binance_testnet_api_secret
   TESTNET=True
   SYMBOL=BTCUSDT
   
   # Strategy Configuration
   ENABLE_SIMPLE_STRATEGY=True
   SIMPLE_STRATEGY_TIMEFRAME=1m
   SIMPLE_STRATEGY_WEIGHT=1.0
   
   ENABLE_TECHNICAL_STRATEGY=True
   TECHNICAL_STRATEGY_TIMEFRAME=1h
   TECHNICAL_STRATEGY_WEIGHT=2.0
   
   # Additional configuration options...
   ```

## Usage

### Running CogniTrade

Start CogniTrade (for paper trading) by running:

```bash
# Method 1: Using the run_bot.py script
./run_bot.py

# Method 2: Using the module directly
python -m bot.main
```

CogniTrade will:
1. Initialize the Binance connection using your API keys
2. Check account balances and verify that the configured symbol can be traded
3. Enter a continuous trading loop that:
   - Retrieves current market data
   - Generates signals from enabled strategies (simple, technical, and custom)
   - Uses LLM (or rule-based fallback) for decision support
   - Applies the configured consensus method to determine final trading action
   - Executes trades based on the consensus and configured parameters
   - Implements exponential backoff for error handling

All activity is logged to both the console and a file named `trading_bot.log`.

### Configuration Options

CogniTrade offers two ways to configure the system:

1. **Environment Variables**: Set configuration in the `.env` file
2. **JSON Configuration**: Use a JSON file for more structured configuration 

#### Using a JSON Configuration File

To use a JSON configuration file:

1. Copy the `config.sample.json` file to create your own config:
   ```bash
   cp config.sample.json config.json
   ```

2. Set the `CONFIG_FILE` environment variable in your `.env` file:
   ```
   CONFIG_FILE=config.json
   ```

3. Edit the JSON file to customize your configuration:
   ```json
   {
     "strategies": {
       "simple": {
         "enabled": true,
         "timeframe": "1m",
         "weight": 1.0
       },
       "technical": {
         "enabled": true,
         "timeframe": "1h",
         "weight": 2.0,
         "parameters": {
           "rsi_period": 14,
           "rsi_oversold": 30,
           "rsi_overbought": 70
         }
       },
       "custom": {
         "enabled": false,
         "timeframe": "4h",
         "weight": 1.0,
         "module_path": "bot.custom_strategies.my_strategy"
       }
     },
     "decision_making": {
       "llm": {
         "enabled": true,
         "required_confidence": 0.6
       },
       "consensus_method": "weighted_majority",
       "min_strategies": 2
     }
   }
   ```

### Custom Trading Strategies

CogniTrade allows you to create your own custom trading strategies that can be loaded dynamically without modifying the core codebase.

#### Creating a Custom Strategy

1. Create a new Python file in the `bot/custom_strategies/` directory:

```python
"""
My custom trading strategy.
"""
import pandas as pd
import talib
from typing import Dict, Any

def initialize() -> Dict[str, Any]:
    """
    Initialize strategy parameters.
    """
    return {
        "name": "My Custom Strategy",
        "description": "A strategy combining EMA trend and RSI momentum",
        "parameters": {
            "ema_period": 20,
            "rsi_period": 14,
            "rsi_threshold_high": 70,
            "rsi_threshold_low": 30
        }
    }

def generate_signal(symbol, interval, client, parameters=None):
    """
    Generate a trading signal.
    
    This is the main entry point called by the bot.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        interval: Timeframe (e.g., "1h", "4h")
        client: Binance API client instance
        parameters: Optional parameters from config
        
    Returns:
        Signal string: "BUY", "SELL", or "HOLD"
    """
    # Get parameters (use provided or defaults)
    params = parameters if parameters else initialize()["parameters"]
    
    # Fetch market data
    candles = client.get_klines(symbol=symbol, interval=interval, limit=100)
    df = pd.DataFrame(candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    
    # Calculate indicators
    df['ema'] = talib.EMA(df['close'], timeperiod=params['ema_period'])
    df['rsi'] = talib.RSI(df['close'], timeperiod=params['rsi_period'])
    
    # Generate signal based on conditions
    if len(df) < 2:
        return "HOLD"  # Not enough data
    
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # BUY if price is above EMA and RSI is rising from oversold
    if (current['close'] > current['ema'] and 
        previous['rsi'] < params['rsi_threshold_low'] and 
        current['rsi'] > previous['rsi']):
        return "BUY"
    
    # SELL if price is below EMA and RSI is falling from overbought
    elif (current['close'] < current['ema'] and 
          previous['rsi'] > params['rsi_threshold_high'] and 
          current['rsi'] < previous['rsi']):
        return "SELL"
    
    return "HOLD"
```

2. Update your configuration:

In your `.env` file:
```
ENABLE_CUSTOM_STRATEGY=True
CUSTOM_STRATEGY_TIMEFRAME=4h
CUSTOM_STRATEGY_WEIGHT=1.0
CUSTOM_STRATEGY_MODULE=bot.custom_strategies.my_strategy
```

Or in your JSON config file:
```json
"custom": {
  "enabled": true,
  "timeframe": "4h",
  "weight": 1.0,
  "module_path": "bot.custom_strategies.my_strategy"
}
```

3. Optional: Add strategy-specific parameters:

```json
"custom": {
  "enabled": true,
  "timeframe": "4h",
  "weight": 1.0,
  "module_path": "bot.custom_strategies.my_strategy",
  "parameters": {
    "ema_period": 25,
    "rsi_period": 21,
    "rsi_threshold_high": 75,
    "rsi_threshold_low": 25
  }
}
```

### Using the API

The project includes a FastAPI backend that you can use to interact with CogniTrade:

```bash
# Start the API server
cd api
uvicorn main:app --reload
```

The API will be available at http://localhost:8000 with automatic API documentation at http://localhost:8000/docs.

### Backtesting Strategies

#### Using the New Backtesting Module

The refactored backtesting module provides a clean API for running backtests:

```python
from bot.backtesting import run_backtest, generate_report, generate_test_data

# Generate test data if needed
test_data = generate_test_data(
    symbol="BTCTEST",
    timeframe="1h",
    start_date="2023-01-01",
    end_date="2023-06-30",
    trend="random",  # 'up', 'down', 'sideways', or 'random'
    volatility=0.02,
    seed=42  # For reproducibility
)

# Define a strategy function
def my_strategy(data_dict, symbol):
    """Simple moving average crossover strategy."""
    # Get data for primary timeframe
    primary_tf = list(data_dict.keys())[0]
    df = data_dict[primary_tf]
    
    # Calculate indicators
    df['sma_short'] = df['close'].rolling(window=10).mean()
    df['sma_long'] = df['close'].rolling(window=30).mean()
    
    # Get latest values
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Generate signals
    if previous['sma_short'] < previous['sma_long'] and latest['sma_short'] > latest['sma_long']:
        return "BUY"  # Bullish crossover
    elif previous['sma_short'] > previous['sma_long'] and latest['sma_short'] < latest['sma_long']:
        return "SELL"  # Bearish crossover
    else:
        return "HOLD"  # No crossover

# Run backtest
result = run_backtest(
    symbol="BTCTEST",
    timeframes=["1h", "4h"],  # Primary and secondary timeframes
    start_date="2023-01-01",
    end_date="2023-06-30",
    strategy_func=my_strategy,
    initial_capital=10000.0
)

# Generate reports and charts
report_paths = generate_report(result)
print(f"HTML report: {report_paths['HTML Report']}")
print(f"Equity chart: {report_paths['Equity Chart']}")
```

#### Vectorized Backtesting

For high-performance backtesting with large datasets, use the vectorized approach:

```python
from bot.backtesting import run_backtest

def vectorized_strategy(data_dict, symbol, vectorized=True):
    """Vectorized strategy that processes all data at once."""
    primary_tf = list(data_dict.keys())[0]
    df = data_dict[primary_tf].copy()
    
    # Calculate indicators (vectorized operations)
    df['sma_short'] = df['close'].rolling(window=10).mean()
    df['sma_long'] = df['close'].rolling(window=30).mean()
    
    # Generate signals for all rows at once
    df['signal'] = 'HOLD'  # Default signal
    
    # Buy condition: Short MA crosses above Long MA
    buy_condition = (df['sma_short'].shift(1) < df['sma_long'].shift(1)) & (df['sma_short'] > df['sma_long'])
    df.loc[buy_condition, 'signal'] = 'BUY'
    
    # Sell condition: Short MA crosses below Long MA
    sell_condition = (df['sma_short'].shift(1) > df['sma_long'].shift(1)) & (df['sma_short'] < df['sma_long'])
    df.loc[sell_condition, 'signal'] = 'SELL'
    
    # If vectorized mode, return the DataFrame with signals
    if vectorized:
        return df[['timestamp', 'signal']]
    
    # For non-vectorized mode, return just the latest signal
    return df.iloc[-1]['signal']

# Run the backtest with the vectorized strategy
result = run_backtest(
    symbol="BTCTEST",
    timeframes=["1h"],
    start_date="2023-01-01",
    end_date="2023-06-30",
    strategy_func=vectorized_strategy,
    initial_capital=10000.0
)
```

#### Parameter Optimization

Find the optimal parameters for your trading strategy:

```python
from bot.backtesting import optimize_strategy

# Create a strategy factory function
def strategy_factory(params):
    def strategy(data_dict, symbol):
        # Extract parameters
        short_window = params['short_window']
        long_window = params['long_window']
        
        # Rest of the strategy using these parameters
        primary_tf = list(data_dict.keys())[0]
        df = data_dict[primary_tf]
        # ... strategy implementation ...
    
    return strategy

# Define parameter grid
param_grid = {
    'short_window': [5, 10, 15, 20],
    'long_window': [30, 50, 100]
}

# Run optimization
result = optimize_strategy(
    symbol='BTCUSDT',
    timeframes=['1h'],
    start_date='2023-01-01',
    end_date='2023-06-30',
    strategy_factory=strategy_factory,
    param_grid=param_grid,
    metric='sharpe_ratio'  # Optimize for Sharpe ratio
)

# Get best parameters
print(f"Best parameters: {result.best_parameters}")
print(f"Sharpe ratio: {result.best_backtest.metrics.sharpe_ratio:.2f}")
print(f"Return: {result.best_backtest.metrics.total_return_pct:.2f}%")
```

#### Multi-Symbol Testing

Run the same strategy across multiple symbols:

```python
from bot.backtesting import run_multi_symbol

results = run_multi_symbol(
    symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    timeframes=['1h'],
    start_date='2023-01-01',
    end_date='2023-06-30',
    strategy_func=my_strategy,
    use_parallel=True  # Run in parallel for speed
)

# Check results for each symbol
for symbol, result in results.items():
    print(f"{symbol}: {result.metrics.total_return_pct:.2f}% return, Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

See the example scripts in the `examples/` directory for comprehensive demonstrations of the backtesting system.

### Running Tests

The project uses pytest for unit testing with a focus on high code coverage. To run the tests:

```bash
# Run all tests
python -m pytest

# Run tests with verbose output
python -m pytest -v

# Run tests in a specific file
python -m pytest tests/test_strategy.py

# Run tests for the backtesting module
python -m pytest tests/test_backtesting/

# Generate test coverage report
python -m pytest --cov=bot tests/

# Generate coverage for a specific module
python -m pytest tests/test_backtesting/ --cov=bot.backtesting
```

Test categories:
- **API tests:** Comprehensive tests for Binance API interactions, including error handling, time synchronization, and edge cases
- **Order Manager tests:** Tests for order execution, tracking, and management functionality
- **Main Module tests:** Tests for the main trading loop with proper mocking of dependencies
- **Strategy tests:** Tests for trading strategy logic
- **LLM tests:** Tests for LLM integration and decision making
- **Backtesting tests:** Tests for the backtesting engine functionality
- **Vectorized Backtesting tests:** Tests for high-performance vectorized operations
- **Arithmetic Validation tests:** Ensures accurate profit/loss calculations using controlled datasets
- **Database tests:** Tests for database operations and data persistence
- **Database Integration tests:** Tests for the database integration layer that connects trading functions with data storage
- **Integration tests:** Tests for the integration between main trading functions and the database system

## Enhanced Configuration System

CogniTrade now features a comprehensive configuration system that supports:

1. **Multiple Configuration Sources:**
   - Environment variables (highest priority)
   - JSON configuration files (for more structured configuration)
   - Default values (fallback when nothing is specified)

2. **Configuration Access:**
   ```python
   # Get a specific configuration value
   from bot.config import get_config
   
   api_key = get_config("API_KEY")
   is_testnet = get_config("TESTNET", default=True, data_type=bool)
   
   # Get strategy configuration
   from bot.config import get_strategy_config, get_strategy_parameter
   
   technical_config = get_strategy_config("technical")
   rsi_period = get_strategy_parameter("technical", "rsi_period", default=14)
   
   # Check if a strategy is enabled
   from bot.config import is_strategy_enabled
   
   if is_strategy_enabled("custom"):
       # Execute custom strategy logic
   ```

3. **Structured Trading Configuration:**
   ```python
   from bot.config import TRADING_CONFIG
   
   # Access structured configuration
   llm_enabled = TRADING_CONFIG["decision_making"]["llm"]["enabled"]
   consensus_method = TRADING_CONFIG["decision_making"]["consensus_method"]
   ```

4. **Direct Integration:** All components of the system now use the configuration system for parameters, making them fully configurable without code modifications.

## LLM Integration

CogniTrade's enhanced LLM integration now supports multiple language models for sophisticated trading analysis:

### Multi-Model LLM Manager

The LLM Manager integrates multiple language models with a configurable setup:

1. **Primary Model** (e.g., DeepSeek R1) - Used for in-depth market analysis and initial trading recommendations.

2. **Secondary Model** (e.g., GPT-4o) - Processes the primary model's output to produce structured responses with:
   - A decisive trading action (BUY, SELL, or HOLD)
   - A confidence score (0.5-1.0) indicating certainty level
   - A concise reasoning summary

3. **Complete Transparency** - Both model responses are preserved:
   - Primary model's complete analytical response
   - Secondary model's structured JSON output
   - All responses stored in the database for auditability
   - Easy access to reasoning behind each trading decision

4. **Database Integration** - All LLM decisions and responses are saved:
   - Trading signals include both raw and processed LLM responses
   - Historical record of all model interactions
   - Full ability to audit and analyze model performance over time

### Advanced Configuration

Configure the LLM integration through your `.env` file or JSON configuration:

```ini
# LLM Configuration
ENABLE_LLM_DECISIONS=True
LLM_REQUIRED_CONFIDENCE=0.6

# Primary LLM (DeepSeek R1)
LLM_PRIMARY_PROVIDER=deepseek
LLM_PRIMARY_MODEL=deepseek-reasoner
LLM_API_KEY=your_deepseek_api_key
LLM_API_ENDPOINT=https://api.deepseek.com/v1/chat/completions
LLM_TEMPERATURE=0.3

# Secondary LLM (OpenAI GPT-4o)
LLM_SECONDARY_PROVIDER=openai
LLM_SECONDARY_MODEL=gpt-4o
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_ENDPOINT=https://api.openai.com/v1/chat/completions
OPENAI_TEMPERATURE=0.1

# Decision Making Configuration
CONSENSUS_METHOD=weighted_majority  # Options: simple_majority, weighted_majority, unanimous
MIN_STRATEGIES_FOR_DECISION=2
LLM_AGREEMENT_REQUIRED=True
```

### Enhanced Decision Process

The improved decision-making process now includes:

1. **Strategy Signal Generation** - Get signals from all enabled strategies (simple, technical, custom)
2. **Signal Aggregation** - Apply configurable weights to different strategy signals
3. **LLM Analysis** - Process market data and signals with the primary and secondary LLMs
4. **Confidence Estimation** - Get a confidence score for the LLM decision
5. **Consensus Determination** - Apply the selected consensus method to reach a final decision
6. **Order Execution** - Execute trades based on the consensus decision and configuration parameters

## Future Improvements

- **Advanced Trading Strategies:** Implement additional strategies beyond the basic examples.
- **Live Trading:** After thorough paper trading, switch to live mode (with caution) by toggling configuration parameters.
- **Multi-Exchange Support:** Add wrappers for additional exchanges (e.g., using CCXT) for more diversified trading.
- **Enhanced Logging & Monitoring:** Integrate more robust logging and alerting for operational insights.
- **Reinforcement Learning:** Implement RL-based strategies that can learn and adapt to changing market conditions.
- **Web Interface:** Create a dashboard for monitoring trades, backtesting results, and adjusting bot parameters (after core backend functionality is stable).
- **Parallel Processing:** Further optimize performance with multi-threaded and multi-process operations.
- **Custom Indicators:** Expand the library of technical indicators and custom signal generators.
- **LLM Evaluation:** Add tools to evaluate LLM performance and compare different models.
- **Analytics Dashboard:** Create visualizations for LLM decision making and reasoning patterns.
- **Auto-Tuning:** Implement machine learning to fine-tune strategy parameters based on historical performance.
- **Advanced Risk Management:** Enhance the order management system with more sophisticated risk controls.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests. Please follow the standard GitHub flow and include tests for any new functionality.

## License

This project is licensed under the MIT License.
