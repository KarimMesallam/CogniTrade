# AI Trading Bot with Binance & LLM Integration

A scalable Python project for automated trading that combines traditional trading strategies with modern AI/LLM orchestration for decision making. The bot connects to Binance (starting with paper trading via the Testnet) and is designed to easily integrate multiple LLMs to refine trade signals and manage order execution.

## Overview

This project aims to take you from basic trading bot functionality to a robust system that leverages AI:
- **Paper Trading:** Start with simulated trades using Binance Testnet.
- **Scalable Architecture:** Modular design to support multiple strategies and future integration with various LLMs.
- **AI Orchestration:** A dedicated module to incorporate LLMs (e.g., DeepSeek R1, GPT-4o, o3-mini, Claude 3.7 Sonnet) for enhanced decision making.
- **Extensible:** Easily add new exchanges, trading strategies, and real-time data streams.
- **Advanced Backtesting:** Completely refactored backtesting engine with modular architecture, better error handling, and improved performance.
- **Comprehensive Testing:** High-coverage test suite for all components with mocks for external dependencies.
- **Vectorized Backtesting:** Optimized performance with vectorized operations for high-speed strategy testing.
- **Arithmetic Validation:** Robust validation of profit/loss calculations with controlled datasets.

## Features

- **Binance API Integration:** Uses the official Binance API (via python-binance) for fetching market data, placing orders, and managing accounts.
- **Paper Trading Mode:** Safely test strategies on Binance Testnet before going live.
- **Strategy Module:** Contains logic for trading signals (e.g., based on technical indicators) with an abstract layer for future enhancements.
- **LLM Manager:** A placeholder module to later integrate multiple language models for trade signal orchestration.
- **Order Management:** Robust order execution system with proper error handling and logging.
- **Database System:** Comprehensive data storage solution with database integration layer for saving signals, trades, market data, and system alerts.
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

## Project Structure (Backend-Only Version)

```
trading_bot/
├── bot/
│   ├── __init__.py
│   ├── config.py              # Configuration loader (API keys, endpoints, etc.)
│   ├── binance_api.py         # Binance API wrapper (using python-binance)
│   ├── strategy.py            # Abstract trading strategy and sample strategy implementation
│   ├── llm_manager.py         # LLM orchestration placeholder (for decision support)
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
├── requirements.txt           # List of dependencies
├── .env                       # Environment variables (API keys, etc.)
└── README.md                  # Project documentation (this file)
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ai-trading-bot.git
   cd ai-trading-bot
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

   Create a `.env` file in the root folder with your API credentials:
   ```ini
   API_KEY=your_binance_testnet_api_key
   API_SECRET=your_binance_testnet_api_secret
   TESTNET=True
   SYMBOL=BTCUSDT
   ```

## Usage

### Running the Bot

Start the bot (for paper trading) by running:

```bash
# Method 1: Using the run_bot.py script
./run_bot.py

# Method 2: Using the module directly
python -m bot.main
```

The bot will:
1. Initialize the Binance connection using your API keys
2. Check account balances and verify that the configured symbol can be traded
3. Enter a continuous trading loop that:
   - Retrieves current market data
   - Generates signals from both simple and technical analysis strategies
   - Uses LLM (or rule-based fallback) for decision support
   - Executes trades when signals and LLM decisions align
   - Implements exponential backoff for error handling

All activity is logged to both the console and a file named `trading_bot.log`.

### Using the API

The project includes a FastAPI backend that you can use to interact with the trading bot:

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

## Examples

### Backtesting Examples

The project includes several example scripts to demonstrate backtesting capabilities:

#### 1. Basic Backtesting Example (`examples/backtest_example.py`)

A simple example demonstrating the refactored backtesting module:

```bash
# Run the example
python examples/backtest_example.py
```

This script:
- Generates synthetic test data with configurable parameters
- Implements a simple moving average crossover strategy
- Runs a backtest with the refactored engine
- Generates HTML reports and charts
- Displays key performance metrics

#### 2. Simple Moving Average Crossover (`examples/simple_backtest.py`)

A basic example of backtesting a Simple Moving Average (SMA) crossover strategy:

```bash
# Run the simple SMA crossover backtest
python examples/simple_backtest.py
```

#### 3. RSI Strategy Backtest (`examples/rsi_backtest.py`)

A more advanced example using the Relative Strength Index (RSI) indicator:

```bash
# Run the RSI strategy backtest
python examples/rsi_backtest.py
```

This script:
- Implements an RSI-based mean reversion strategy
- Uses overbought/oversold conditions for trading signals
- Visualizes RSI values alongside price and equity curves
- Shows how to use a different timeframe (4h) for testing

#### 4. Strategy Comparison (`examples/compare_strategies.py`)

Compare and analyze different trading strategies:

```bash
# Compare multiple strategy backtest results
python examples/compare_strategies.py
```

### Running Your Own Backtest

You can create your own backtesting script by following these steps:

1. Import the necessary functions from the refactored module:
   ```python
   from bot.backtesting import run_backtest, generate_report, generate_test_data
   ```

2. Define your strategy function that takes data and returns 'BUY', 'SELL', or 'HOLD'

3. Run a backtest with appropriate parameters:
   ```python
   result = run_backtest(symbol, timeframes, start_date, end_date, strategy_func)
   ```

4. Generate reports and visualizations:
   ```python
   report_paths = generate_report(result)
   ```

## Future Improvements

- **Advanced Trading Strategies:** Implement additional strategies beyond the basic examples.
- **LLM Integration:** Expand `bot/llm_manager.py` to interface with models like DeepSeek R1, GPT-4, etc., for real-time decision support.
- **Live Trading:** After thorough paper trading, switch to live mode (with caution) by toggling configuration parameters.
- **Multi-Exchange Support:** Add wrappers for additional exchanges (e.g., using CCXT) for more diversified trading.
- **Enhanced Logging & Monitoring:** Integrate more robust logging and alerting for operational insights.
- **Reinforcement Learning:** Implement RL-based strategies that can learn and adapt to changing market conditions.
- **Web Interface:** Create a dashboard for monitoring trades, backtesting results, and adjusting bot parameters (after core backend functionality is stable).
- **Parallel Processing:** Further optimize performance with multi-threaded and multi-process operations.
- **Custom Indicators:** Expand the library of technical indicators and custom signal generators.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests. Please follow the standard GitHub flow and include tests for any new functionality.

## License

This project is licensed under the MIT License.