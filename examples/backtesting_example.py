#!/usr/bin/env python
"""
Example script for using the backtesting module for strategy testing and optimization.
This demonstrates:
1. Single strategy backtesting
2. Multi-timeframe analysis
3. Strategy parameter optimization
4. Monitoring and alerting
5. Comprehensive trade logging
6. Performance visualization
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path to import bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.backtesting import BacktestEngine, BacktestRunner
from bot.binance_api import client
from bot.database import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/backtesting_example.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

def demo_single_strategy():
    """Demonstrate running a single strategy backtest"""
    
    # Define a simple moving average crossover strategy
    def sma_crossover_strategy(data_dict, symbol):
        """
        Simple Moving Average Crossover Strategy
        
        Buy when short SMA crosses above long SMA
        Sell when short SMA crosses below long SMA
        """
        # Use 1h timeframe data for the strategy
        timeframe = '1h'
        if timeframe not in data_dict:
            logger.warning(f"Required timeframe {timeframe} not available")
            return 'HOLD'
            
        df = data_dict[timeframe]
        
        # Require at least 50 candles for proper SMA calculation
        if len(df) < 50:
            logger.warning("Not enough data for SMA calculation")
            return 'HOLD'
        
        # Calculate SMAs
        short_period = 10
        long_period = 30
        
        df['sma_short'] = df['close'].rolling(window=short_period).mean()
        df['sma_long'] = df['close'].rolling(window=long_period).mean()
        
        # Get current and previous values
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check for crossover
        if previous['sma_short'] <= previous['sma_long'] and current['sma_short'] > current['sma_long']:
            return 'BUY'
        elif previous['sma_short'] >= previous['sma_long'] and current['sma_short'] < current['sma_long']:
            return 'SELL'
        else:
            return 'HOLD'
    
    # Create a backtest engine
    symbol = 'BTCUSDT'
    timeframes = ['1h', '4h']
    start_date = '2023-01-01'
    end_date = '2023-03-31'
    
    engine = BacktestEngine(
        symbol=symbol,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000,
        commission=0.001  # 0.1% commission
    )
    
    # Download data if needed
    if not engine.market_data or all(tf_data.empty for tf_data in engine.market_data.values()):
        logger.info("Downloading market data for backtesting...")
        for timeframe in timeframes:
            engine.download_data(client, timeframe)
    
    # Run the backtest
    logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
    results = engine.run_backtest(sma_crossover_strategy)
    
    # Check for any alerts
    alerts = engine.monitor_and_alert(results)
    if alerts:
        logger.warning(f"Backtest generated {len(alerts)} alerts")
        for alert in alerts:
            logger.warning(f"Alert: {alert['severity']} - {alert['message']}")
    
    # Generate trade log
    trade_log = engine.generate_trade_log(
        results, 
        filename=f"logs/{symbol}_sma_crossover_{start_date}_{end_date}.csv"
    )
    
    # Plot the results
    engine.plot_results(
        results,
        filename=f"logs/{symbol}_sma_crossover_{start_date}_{end_date}.png"
    )
    
    # Save results to database
    engine.save_results(results, "SMA_Crossover")
    
    # Print summary
    print("\n=== Backtest Results ===")
    print(f"Symbol: {symbol}")
    print(f"Strategy: SMA Crossover")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    return results

def demo_multi_timeframe_analysis():
    """Demonstrate multi-timeframe analysis"""
    # Create a backtest engine
    symbol = 'BTCUSDT'
    timeframes = ['1h', '4h', '1d']  # Multiple timeframes
    start_date = '2023-01-01'
    end_date = '2023-03-31'
    
    engine = BacktestEngine(
        symbol=symbol,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date
    )
    
    # Download data if needed
    for timeframe in timeframes:
        if timeframe not in engine.market_data or engine.market_data[timeframe].empty:
            engine.download_data(client, timeframe)
    
    # Prepare data with indicators
    engine.prepare_data()
    
    # Run multi-timeframe analysis
    analysis = engine.multi_timeframe_analysis(engine.market_data)
    
    # Print analysis results
    print("\n=== Multi-Timeframe Analysis ===")
    for tf in timeframes:
        print(f"\nTimeframe: {tf}")
        print(f"RSI: {analysis[tf]['rsi']:.2f}")
        print(f"Trend: {analysis[tf]['trend']}")
        print(f"Volatility: {analysis[tf]['volatility']:.4f}")
        print(f"Bollinger Band Position: {analysis[tf]['bb_position']:.2f}")
    
    print("\n=== Consolidated View ===")
    print(f"Bullish timeframes: {analysis['consolidated']['bullish_timeframes']}")
    print(f"Bearish timeframes: {analysis['consolidated']['bearish_timeframes']}")
    print(f"High volatility timeframes: {analysis['consolidated']['high_volatility_timeframes']}")
    
    return analysis

def demo_strategy_optimization():
    """Demonstrate strategy parameter optimization"""
    
    # Factory function to create strategy with different parameters
    def sma_strategy_factory(params):
        """Create an SMA strategy with the given parameters"""
        short_period = params['short_period']
        long_period = params['long_period']
        
        def strategy(data_dict, symbol):
            # Use 1h timeframe data
            timeframe = '1h'
            if timeframe not in data_dict:
                return 'HOLD'
                
            df = data_dict[timeframe]
            
            # Require enough candles
            if len(df) < long_period + 10:
                return 'HOLD'
            
            # Calculate SMAs
            df['sma_short'] = df['close'].rolling(window=short_period).mean()
            df['sma_long'] = df['close'].rolling(window=long_period).mean()
            
            # Get current and previous values
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Check for crossover
            if previous['sma_short'] <= previous['sma_long'] and current['sma_short'] > current['sma_long']:
                return 'BUY'
            elif previous['sma_short'] >= previous['sma_long'] and current['sma_short'] < current['sma_long']:
                return 'SELL'
            else:
                return 'HOLD'
                
        return strategy
    
    # Create a backtest engine
    symbol = 'BTCUSDT'
    timeframes = ['1h']
    start_date = '2023-01-01'
    end_date = '2023-02-28'  # Shorter period for optimization
    
    engine = BacktestEngine(
        symbol=symbol,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date
    )
    
    # Download data if needed
    if '1h' not in engine.market_data or engine.market_data['1h'].empty:
        engine.download_data(client, '1h')
    
    # Define parameter grid
    param_grid = {
        'short_period': [5, 10, 15, 20],
        'long_period': [20, 30, 40, 50]
    }
    
    # Run optimization
    logger.info("Starting strategy parameter optimization...")
    optimization_results = engine.optimize_parameters(
        strategy_factory=sma_strategy_factory,
        param_grid=param_grid
    )
    
    # Print optimization results
    print("\n=== Strategy Optimization Results ===")
    print(f"Best Parameters: {optimization_results['params']}")
    print(f"Best Sharpe Ratio: {optimization_results['sharpe_ratio']:.2f}")
    print(f"Total Return: {optimization_results['result']['total_return_pct']:.2f}%")
    print(f"Win Rate: {optimization_results['result']['win_rate']:.2f}%")
    
    return optimization_results

def demo_multiple_strategies():
    """Demonstrate running and comparing multiple strategies"""
    
    # Define strategies
    def sma_strategy(data_dict, symbol):
        """Moving Average Crossover Strategy"""
        timeframe = '1h'
        if timeframe not in data_dict:
            return 'HOLD'
            
        df = data_dict[timeframe]
        
        if len(df) < 50:
            return 'HOLD'
        
        df['sma_short'] = df['close'].rolling(window=10).mean()
        df['sma_long'] = df['close'].rolling(window=30).mean()
        
        if df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1]:
            return 'BUY'
        else:
            return 'SELL'
    
    def rsi_strategy(data_dict, symbol):
        """RSI Strategy - Buy oversold, sell overbought"""
        timeframe = '1h'
        if timeframe not in data_dict:
            return 'HOLD'
            
        df = data_dict[timeframe]
        
        if len(df) < 30:
            return 'HOLD'
        
        # Calculate RSI if not already done
        if 'rsi' not in df.columns:
            from bot.strategy import calculate_rsi
            df['rsi'] = calculate_rsi(df)
        
        rsi = df['rsi'].iloc[-1]
        
        if rsi < 30:
            return 'BUY'
        elif rsi > 70:
            return 'SELL'
        else:
            # Hold current position
            return 'HOLD'
    
    def bb_strategy(data_dict, symbol):
        """Bollinger Bands Strategy"""
        timeframe = '1h'
        if timeframe not in data_dict:
            return 'HOLD'
            
        df = data_dict[timeframe]
        
        if len(df) < 30:
            return 'HOLD'
        
        # Calculate Bollinger Bands if not already done
        if 'upper_band' not in df.columns:
            from bot.strategy import calculate_bollinger_bands
            bb = calculate_bollinger_bands(df)
            df['upper_band'] = bb['upper_band']
            df['middle_band'] = bb['sma']
            df['lower_band'] = bb['lower_band']
        
        current_price = df['close'].iloc[-1]
        
        if current_price < df['lower_band'].iloc[-1]:
            return 'BUY'
        elif current_price > df['upper_band'].iloc[-1]:
            return 'SELL'
        else:
            return 'HOLD'
    
    # Create backtest runner
    runner = BacktestRunner()
    
    # Run multiple backtests
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['1h', '4h']
    strategies = {
        'SMA_Crossover': sma_strategy,
        'RSI': rsi_strategy,
        'Bollinger_Bands': bb_strategy
    }
    
    start_date = '2023-01-01'
    end_date = '2023-03-31'
    
    logger.info("Running multiple backtests for different strategies and symbols...")
    results = runner.run_multiple_backtests(
        symbols=symbols,
        timeframes=timeframes,
        strategies=strategies,
        start_date=start_date,
        end_date=end_date
    )
    
    # Compare strategies
    comparison = runner.compare_strategies()
    
    # Generate summary report
    report = runner.generate_summary_report(output_file='logs/backtest_summary_report.txt')
    
    # Print report
    print("\n=== Strategy Comparison ===")
    print(report)
    
    return results, comparison, report

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Run examples
    print("\n=== Backtesting Module Examples ===\n")
    
    # Choose which examples to run
    run_single = True
    run_multi_tf = True
    run_optimization = False  # Can be time-consuming
    run_multiple = False  # Can be time-consuming
    
    if run_single:
        print("\n=> Running Single Strategy Backtest Example...")
        demo_single_strategy()
    
    if run_multi_tf:
        print("\n=> Running Multi-Timeframe Analysis Example...")
        demo_multi_timeframe_analysis()
    
    if run_optimization:
        print("\n=> Running Strategy Optimization Example...")
        demo_strategy_optimization()
    
    if run_multiple:
        print("\n=> Running Multiple Strategies Comparison Example...")
        demo_multiple_strategies()
    
    print("\n=== Examples Completed ===\n") 