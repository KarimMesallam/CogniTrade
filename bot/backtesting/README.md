# Backtesting Module

This module provides a framework for backtesting trading strategies on historical market data with improved modularity, error handling, and performance optimizations.

## Directory Structure

```
bot/backtesting/
├── __init__.py             # Public API for the module
├── config/                 # Configuration settings
│   ├── __init__.py
│   └── settings.py         # Centralized configuration
├── core/                   # Core backtesting engine
│   ├── __init__.py
│   └── engine.py           # Primary backtesting logic
├── data/                   # Market data management
│   ├── __init__.py
│   └── market_data.py      # Data loading and caching
├── exceptions/             # Custom exceptions
│   ├── __init__.py
│   └── base.py             # Exception hierarchy
├── models/                 # Data models
│   ├── __init__.py
│   ├── trade.py            # Trade-related classes
│   └── results.py          # Performance metrics and results
├── reporting/              # Report generation
│   ├── __init__.py
│   └── html_report.py      # HTML report templates
├── utils/                  # Utility functions
│   └── __init__.py
├── visualization/          # Charting functions
│   ├── __init__.py
│   └── charts.py           # Chart generation
└── strategies/             # Strategy implementations
    └── __init__.py
```

## Quick Start

```python
from bot.backtesting import run_backtest, generate_report

# Define a strategy function
def my_strategy(data_dict, symbol):
    """
    Strategy function that returns 'BUY', 'SELL', or 'HOLD' signals.
    
    Args:
        data_dict: Dictionary of timeframe -> DataFrame with market data
        symbol: Trading symbol
        
    Returns:
        str: Trading signal - 'BUY', 'SELL', or 'HOLD'
    """
    # Use primary timeframe data
    primary_tf = list(data_dict.keys())[0]
    df = data_dict[primary_tf]
    
    # Simple moving average crossover strategy
    if len(df) < 30:  # Need enough data for indicators
        return 'HOLD'
    
    # Calculate short and long moving averages
    short_ma = df['close'].rolling(window=10).mean()
    long_ma = df['close'].rolling(window=30).mean()
    
    # Generate signals based on moving average crossover
    if short_ma.iloc[-2] < long_ma.iloc[-2] and short_ma.iloc[-1] > long_ma.iloc[-1]:
        return 'BUY'  # Bullish crossover
    elif short_ma.iloc[-2] > long_ma.iloc[-2] and short_ma.iloc[-1] < long_ma.iloc[-1]:
        return 'SELL'  # Bearish crossover
    else:
        return 'HOLD'  # No crossover

# Run backtest
result = run_backtest(
    symbol='BTCUSDT',
    timeframes=['1h', '4h'],  # Primary and higher timeframes
    start_date='2023-01-01',
    end_date='2023-06-30',
    strategy_func=my_strategy,
    initial_capital=10000.0
)

# Generate reports
report_paths = generate_report(result)
print(f"HTML report generated at: {report_paths['HTML Report']}")
```

## Features

1. **Modular Design**: Each component has a clear responsibility, making the code easier to maintain and extend.
2. **Error Handling**: Specific exceptions for different error types, with proper error propagation.
3. **Data Models**: Clean, typed data representations using dataclasses.
4. **Performance Optimization**: Caching for market data, parallel processing for parameter optimization.
5. **Visualization**: Comprehensive charts for equity curves, drawdowns, and trades.
6. **Reporting**: HTML reports with detailed performance metrics.
7. **Multi-Timeframe Analysis**: Analyze data across multiple timeframes.
8. **Parameter Optimization**: Find optimal strategy parameters through grid search.

## Key Functions

### Basic Backtesting

```python
from bot.backtesting import run_backtest

result = run_backtest(
    symbol='BTCUSDT',
    timeframes=['1h'],
    start_date='2023-01-01',
    end_date='2023-06-30',
    strategy_func=my_strategy
)
```

### Parameter Optimization

```python
from bot.backtesting import optimize_strategy

# Strategy factory that creates a strategy function with given parameters
def strategy_factory(params):
    def strategy(data_dict, symbol):
        # Extract parameters
        short_window = params['short_window']
        long_window = params['long_window']
        
        # Use parameters in strategy logic
        primary_tf = list(data_dict.keys())[0]
        df = data_dict[primary_tf]
        
        if len(df) < long_window:
            return 'HOLD'
        
        short_ma = df['close'].rolling(window=short_window).mean()
        long_ma = df['close'].rolling(window=long_window).mean()
        
        if short_ma.iloc[-2] < long_ma.iloc[-2] and short_ma.iloc[-1] > long_ma.iloc[-1]:
            return 'BUY'
        elif short_ma.iloc[-2] > long_ma.iloc[-2] and short_ma.iloc[-1] < long_ma.iloc[-1]:
            return 'SELL'
        else:
            return 'HOLD'
    
    return strategy

# Define parameter grid
param_grid = {
    'short_window': [5, 10, 15, 20],
    'long_window': [30, 50, 100]
}

# Run optimization
optimization_result = optimize_strategy(
    symbol='BTCUSDT',
    timeframes=['1h'],
    start_date='2023-01-01',
    end_date='2023-06-30',
    strategy_factory=strategy_factory,
    param_grid=param_grid,
    metric='sharpe_ratio'  # Optimize for Sharpe ratio
)

print(f"Best parameters: {optimization_result.best_parameters}")
print(f"Best Sharpe ratio: {optimization_result.best_backtest.metrics.sharpe_ratio:.2f}")
```

### Multi-Symbol Testing

```python
from bot.backtesting import run_multi_symbol

results = run_multi_symbol(
    symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'],
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

### Generating Test Data

```python
from bot.backtesting import generate_test_data

# Generate synthetic data for testing
test_data = generate_test_data(
    symbol='TEST',
    timeframe='1h',
    start_date='2023-01-01',
    end_date='2023-06-30',
    trend='up',  # Generate bullish trend
    volatility=0.02,  # 2% volatility
    seed=42  # For reproducibility
)

print(f"Generated {len(test_data)} candles of test data")
```

## Configuration

The module uses a centralized configuration system in `config/settings.py`. You can update settings using:

```python
from bot.backtesting.config.settings import update_config

# Update a specific setting
update_config('backtest', 'commission_rate', 0.0005)  # Set 0.05% commission
```

## Advanced Customization

For advanced use cases, you can create your own instances of the core classes:

```python
from bot.backtesting.core.engine import BacktestEngine
from bot.backtesting.data.market_data import MarketData
from bot.backtesting.visualization.charts import ChartGenerator

# Custom data loading
data_manager = MarketData()
data = data_manager.get_market_data('BTCUSDT', '1h', '2023-01-01', '2023-06-30')
data_with_indicators = data_manager.add_indicators(data)

# Custom backtesting engine
engine = BacktestEngine(
    symbol='BTCUSDT',
    timeframes=['1h', '4h'],
    start_date='2023-01-01',
    end_date='2023-06-30',
    initial_capital=10000.0,
    commission_rate=0.001
)

# Custom visualization
chart_gen = ChartGenerator(output_dir='./custom_charts')
chart_gen.plot_equity_curve(result, show_plot=True)
``` 