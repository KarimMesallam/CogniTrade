"""
Backtesting module for trading strategies.
Provides a framework for strategy backtesting, analysis, and reporting.
"""
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
import pandas as pd  # Import pandas at the top level

# Import from sub-modules
from bot.backtesting.core.engine import BacktestEngine, MultiSymbolBacktester
from bot.backtesting.data.market_data import MarketData
from bot.backtesting.models.results import BacktestResult, OptimizationResult
from bot.backtesting.visualization.charts import ChartGenerator
from bot.backtesting.config.settings import get_config, update_config

# Configure logging
logger = logging.getLogger("trading_bot")

# Create output directories
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

# Define public API functions
def run_backtest(
    symbol: str, 
    timeframes: List[str], 
    start_date: str, 
    end_date: str, 
    strategy_func: Callable,
    initial_capital: float = None,
    commission_rate: float = None,
    db_path: str = None
) -> BacktestResult:
    """
    Run a backtest with the given strategy and parameters.
    
    Args:
        symbol: Trading symbol to backtest
        timeframes: List of timeframes to include (e.g., ['1m', '5m', '1h'])
        start_date: Start date for backtest (ISO format: 'YYYY-MM-DD')
        end_date: End date for backtest (ISO format: 'YYYY-MM-DD')
        strategy_func: A function that takes market data and returns a signal
        initial_capital: Initial capital to start with (uses default if None)
        commission_rate: Commission rate as a decimal (uses default if None)
        db_path: Optional custom path for the database
        
    Returns:
        BacktestResult: Object containing backtest results
    """
    engine = BacktestEngine(
        symbol=symbol,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        db_path=db_path
    )
    
    return engine.run_backtest(strategy_func)

def run_multi_symbol(
    symbols: List[str],
    timeframes: List[str],
    start_date: str,
    end_date: str,
    strategy_func: Callable,
    initial_capital: float = None,
    commission_rate: float = None,
    db_path: str = None,
    use_parallel: bool = None
) -> Dict[str, BacktestResult]:
    """
    Run the same strategy across multiple symbols.
    
    Args:
        symbols: List of trading symbols to test
        timeframes: List of timeframes to include
        start_date: Start date for backtest (ISO format)
        end_date: End date for backtest (ISO format)
        strategy_func: A function that takes market data and returns a signal
        initial_capital: Initial capital for each backtest
        commission_rate: Commission rate as a decimal
        db_path: Optional custom path for the database
        use_parallel: Whether to use parallel processing
        
    Returns:
        Dict[str, BacktestResult]: Results for each symbol
    """
    multi_tester = MultiSymbolBacktester(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        db_path=db_path
    )
    
    return multi_tester.run_for_all_symbols(strategy_func, use_parallel)

def optimize_strategy(
    symbol: str,
    timeframes: List[str],
    start_date: str,
    end_date: str,
    strategy_factory: Callable,
    param_grid: Dict[str, List[Any]],
    metric: str = 'sharpe_ratio',
    ascending: bool = False,
    initial_capital: float = None,
    commission_rate: float = None
) -> OptimizationResult:
    """
    Optimize strategy parameters through grid search.
    
    Args:
        symbol: Trading symbol to test
        timeframes: List of timeframes to include
        start_date: Start date for backtest (ISO format)
        end_date: End date for backtest (ISO format)
        strategy_factory: A function that takes parameters and returns a strategy function
        param_grid: Dictionary of parameter names to lists of values to try
        metric: Metric to optimize (e.g., 'sharpe_ratio', 'total_return_pct')
        ascending: Whether lower values are better (True) or higher values (False)
        initial_capital: Initial capital for each backtest
        commission_rate: Commission rate as a decimal
        
    Returns:
        OptimizationResult: Results of the optimization
    """
    engine = BacktestEngine(
        symbol=symbol,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission_rate=commission_rate
    )
    
    return engine.optimize_parameters(
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        metric=metric,
        ascending=ascending
    )

def generate_report(
    result: BacktestResult,
    output_dir: str = None,
    include_trades: bool = True,
    include_charts: bool = True
) -> Dict[str, Path]:
    """
    Generate reports for a backtest result.
    
    Args:
        result: BacktestResult object
        output_dir: Directory to save reports (uses default if None)
        include_trades: Whether to include trade details in reports
        include_charts: Whether to generate charts
        
    Returns:
        Dict[str, Path]: Paths to generated reports
    """
    # Import here to avoid circular imports
    from bot.backtesting.reporting.html_report import HTMLReportGenerator
    
    # Create report generator
    report_generator = HTMLReportGenerator(output_dir=output_dir)
    
    # Generate reports
    reports = {}
    
    # Generate HTML report
    html_path = report_generator.generate_report(
        result=result,
        include_trades=include_trades
    )
    reports["HTML Report"] = html_path
    
    # Generate charts if requested
    if include_charts:
        chart_gen = ChartGenerator(output_dir=output_dir)
        
        # Get market data for the primary timeframe if needed for charts
        data_manager = MarketData()
        market_data = None
        
        try:
            market_data = data_manager.get_market_data(
                symbol=result.symbol,
                timeframe=result.timeframes[0],
                start_time=result.start_date,
                end_time=result.end_date
            )
        except Exception as e:
            logger.warning(f"Could not load market data for charts: {e}")
        
        # Generate charts
        reports["Equity Chart"] = chart_gen.plot_equity_curve(result)
        reports["Drawdown Chart"] = chart_gen.plot_drawdown(result)
        
        if market_data is not None:
            reports["Trades Chart"] = chart_gen.plot_trades(result, market_data)
            reports["Complete Analysis"] = chart_gen.plot_complete_analysis(result, market_data)
    
    return reports

def generate_test_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    trend: str = 'random',
    volatility: float = 0.02,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate synthetic market data for backtesting.
    
    Args:
        symbol: Symbol to generate data for
        timeframe: Timeframe to generate data for
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        trend: Trend type ('up', 'down', 'sideways', 'random')
        volatility: Volatility factor (0-1)
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated synthetic market data
    """
    data_manager = MarketData()
    return data_manager.generate_test_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        trend=trend,
        volatility=volatility,
        seed=seed
    )
