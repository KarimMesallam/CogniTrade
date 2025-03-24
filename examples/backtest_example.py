#!/usr/bin/env python3
"""
Example script demonstrating the refactored backtesting module.
"""
import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Ensure the parent directory is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest_example.log")
    ]
)

logger = logging.getLogger("backtest_example")

def main():
    """Run a simple backtest example."""
    logger.info("Starting backtest example")
    
    try:
        # Import from the refactored backtesting module
        from bot.backtesting import (
            run_backtest,
            generate_report,
            generate_test_data
        )
        
        # Generate synthetic test data
        symbol = "TEST_BTC"
        timeframe = "1h"
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Generating test data for {symbol} from {start_date} to {end_date}")
        
        test_data = generate_test_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            trend="random",
            volatility=0.03,
            seed=42  # For reproducibility
        )
        
        logger.info(f"Generated {len(test_data)} candles of test data")
        
        # Define a simple strategy function
        def simple_strategy(data_dict, symbol):
            """
            Simple moving average crossover strategy.
            """
            # Get data for the primary timeframe
            primary_tf = list(data_dict.keys())[0]
            df = data_dict[primary_tf]
            
            # Return HOLD if not enough data
            if len(df) < 30:
                return "HOLD"
            
            # Calculate simple moving averages
            df['sma_short'] = df['close'].rolling(window=10).mean()
            df['sma_long'] = df['close'].rolling(window=30).mean()
            
            # Get the latest values
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Generate signals based on moving average crossover
            if previous['sma_short'] < previous['sma_long'] and latest['sma_short'] > latest['sma_long']:
                return "BUY"  # Bullish crossover
            elif previous['sma_short'] > previous['sma_long'] and latest['sma_short'] < latest['sma_long']:
                return "SELL"  # Bearish crossover
            else:
                return "HOLD"  # No crossover
        
        # Run the backtest
        logger.info("Running backtest")
        
        result = run_backtest(
            symbol=symbol,
            timeframes=[timeframe],
            start_date=start_date,
            end_date=end_date,
            strategy_func=simple_strategy,
            initial_capital=10000.0
        )
        
        # Generate reports
        logger.info("Generating reports")
        report_paths = generate_report(result)
        
        logger.info(f"Total trades: {result.total_trades}")
        logger.info(f"Win rate: {result.metrics.win_rate:.2f}%")
        logger.info(f"Return: {result.metrics.total_return_pct:.2f}%")
        logger.info(f"Sharpe ratio: {result.metrics.sharpe_ratio:.2f}")
        
        if "HTML Report" in report_paths:
            logger.info(f"HTML report generated at: {report_paths['HTML Report']}")
        
        # Print paths to all generated files
        for name, path in report_paths.items():
            if path:
                logger.info(f"{name}: {path}")
        
        logger.info("Backtest example completed successfully")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("The refactored backtesting module may not be available.")
    except Exception as e:
        logger.error(f"Error running backtest example: {e}", exc_info=True)

if __name__ == "__main__":
    main() 