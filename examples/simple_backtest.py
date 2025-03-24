#!/usr/bin/env python
"""
Simple Moving Average Crossover Strategy Example

This script demonstrates how to use the BacktestEngine to test a simple
SMA crossover strategy using historical data from Binance. It properly
integrates with the bot's backtesting module.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.backtesting import BacktestEngine
from bot.binance_api import client
from bot.config import API_KEY, API_SECRET, TESTNET, SYMBOL

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/simple_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

def sma_crossover_strategy(data_dict, symbol):
    """
    Simple Moving Average Crossover Strategy
    
    Buy when short SMA crosses above long SMA
    Sell when short SMA crosses below long SMA
    
    Args:
        data_dict: Dictionary of timeframe -> DataFrame with market data
        symbol: Trading symbol
        
    Returns:
        str: Trading signal ('BUY', 'SELL', or 'HOLD')
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

def run_simple_backtest():
    """Run a simple SMA crossover strategy backtest using BacktestEngine"""
    
    # Print API configuration info (without exposing secrets)
    print("\n=== API Configuration ===")
    print(f"API Key configured: {'Yes' if API_KEY else 'No'}")
    print(f"API Secret configured: {'Yes' if API_SECRET else 'No'}")
    print(f"Using Testnet: {TESTNET}")
    print(f"Default Symbol: {SYMBOL}")
    
    # Parameters for backtest
    symbol = SYMBOL  # Use the symbol from .env
    timeframes = ['1h']  # Use 1h timeframe for this strategy
    
    # Use a 60-day period to ensure we get enough data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print(f"\n=== Running Backtest ===")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframes[0]}")
    print(f"Period: {start_date} to {end_date}")
    
    try:
        # Try to ping Binance to verify connection
        client.ping()
        print("Successfully connected to Binance API")
        
        # Create a backtest engine
        engine = BacktestEngine(
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000,
            commission=0.001  # 0.1% commission
        )
        
        # Download data if needed
        print("\nChecking for historical data...")
        for timeframe in timeframes:
            if timeframe not in engine.market_data or engine.market_data[timeframe].empty:
                print(f"Downloading {timeframe} data for {symbol}...")
                try:
                    # Convert dates to millisecond timestamps
                    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
                    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
                    
                    # Get klines data from Binance
                    klines = client.get_historical_klines(
                        symbol=symbol,
                        interval=timeframe,
                        start_str=str(start_ts),
                        end_str=str(end_ts)
                    )
                    
                    if not klines:
                        print(f"No data returned for {symbol} at {timeframe} timeframe")
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert timestamps to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Convert numeric columns
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                    
                    # Select only the columns required by the database schema
                    # and add symbol and timeframe
                    df_filtered = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                    df_filtered['symbol'] = symbol
                    df_filtered['timeframe'] = timeframe
                    
                    # Store in database (proper way)
                    try:
                        import sqlite3
                        with sqlite3.connect(engine.db.db_path) as conn:
                            # Insert data, replacing any existing entries
                            df_filtered['timestamp'] = df_filtered['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            df_filtered.to_sql('market_data', conn, if_exists='append', index=False)
                            print(f"Successfully stored {len(df_filtered)} candles in database")
                    except Exception as e:
                        print(f"Database error: {e}")
                        # Even if DB storage fails, we can still use the in-memory data
                    
                    # Update engine's market data regardless of DB storage success
                    engine.market_data[timeframe] = df
                    print(f"Using data for {timeframe} ({len(df)} candles)")
                    
                except Exception as e:
                    print(f"Error downloading data: {e}")
                    continue
            else:
                print(f"Using existing {timeframe} data for {symbol} ({len(engine.market_data[timeframe])} candles)")
        
        # Run the backtest
        print("\nRunning backtest with SMA Crossover strategy...")
        results = engine.run_backtest(sma_crossover_strategy)
        
        if results:
            # Generate trade log
            log_filename = f"logs/{symbol}_sma_crossover_{start_date}_{end_date}.csv"
            print(f"\nGenerating trade log: {log_filename}")
            engine.generate_trade_log(results, filename=log_filename)
            
            # Plot the results
            plot_filename = f"logs/{symbol}_sma_crossover_{start_date}_{end_date}.png"
            print(f"Generating performance chart: {plot_filename}")
            engine.plot_results(results, filename=plot_filename)
            
            # Save results to database
            engine.save_results(results, "SMA_Crossover")
            
            # Check for any alerts
            alerts = engine.monitor_and_alert(results)
            if alerts:
                print(f"\nBacktest generated {len(alerts)} alerts:")
                for alert in alerts:
                    print(f"  - {alert['severity'].upper()}: {alert['message']}")
            
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
        else:
            print("Backtest failed: No results generated")
            return None
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        logger.error(f"Backtest error: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    print("\n=== Simple SMA Crossover Backtest Example ===")
    results = run_simple_backtest()
    print("\nBacktest complete! Check logs directory for results.") 