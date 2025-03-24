#!/usr/bin/env python
"""
RSI Strategy Backtesting Example

This script demonstrates how to use the BacktestEngine to test an RSI-based
trading strategy with proper integration with the bot's backtesting module.
The RSI strategy:
- Buy when RSI crosses from below oversold threshold to above it (bullish)
- Sell when RSI crosses from above overbought threshold to below it (bearish)
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd

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
        logging.FileHandler("logs/rsi_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

def rsi_strategy(data_dict, symbol, rsi_period=14, overbought=70, oversold=30):
    """
    RSI-based Trading Strategy
    
    Buy when RSI crosses from below oversold to above oversold threshold
    Sell when RSI crosses from above overbought to below overbought threshold
    
    Args:
        data_dict: Dictionary of timeframe -> DataFrame with market data
        symbol: Trading symbol
        rsi_period: Period for RSI calculation
        overbought: Overbought threshold
        oversold: Oversold threshold
        
    Returns:
        str: Trading signal ('BUY', 'SELL', or 'HOLD')
    """
    # Use 4h timeframe data for this strategy
    timeframe = '4h'
    if timeframe not in data_dict:
        logger.warning(f"Required timeframe {timeframe} not available")
        return 'HOLD'
        
    df = data_dict[timeframe]
    
    # Require enough data for RSI calculation
    if len(df) < rsi_period + 10:
        logger.warning(f"Not enough data for RSI calculation. Need at least {rsi_period + 10} candles.")
        return 'HOLD'
    
    # RSI should already be calculated by the engine's prepare_data method
    if 'rsi' not in df.columns:
        logger.warning("RSI column not found in data. Check that prepare_data was called.")
        return 'HOLD'
    
    # Check for signal conditions using previous and current values
    current_rsi = df['rsi'].iloc[-1]
    previous_rsi = df['rsi'].iloc[-2]
    
    # Buy signal: RSI crosses from below oversold to above oversold
    if previous_rsi < oversold and current_rsi > oversold:
        return 'BUY'
    
    # Sell signal: RSI crosses from above overbought to below overbought
    elif previous_rsi > overbought and current_rsi < overbought:
        return 'SELL'
    
    # No signal
    else:
        return 'HOLD'

def run_rsi_backtest():
    """Run an RSI strategy backtest using BacktestEngine"""
    
    # Parameters for backtest
    symbol = SYMBOL  # Use the symbol from .env
    timeframes = ['4h']  # 4-hour timeframe is often good for RSI
    rsi_period = 14  # Standard RSI period
    overbought = 70  # Overbought threshold
    oversold = 30    # Oversold threshold
    
    # Use a 90-day period for backtesting
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"\n=== Running RSI Strategy Backtest ===")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframes[0]}")
    print(f"Period: {start_date} to {end_date}")
    print(f"RSI Settings: Period={rsi_period}, Overbought={overbought}, Oversold={oversold}")
    
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
        
        # Create a strategy function wrapper to include parameters
        def strategy_wrapper(data_dict, symbol):
            return rsi_strategy(data_dict, symbol, rsi_period, overbought, oversold)
        
        # Run the backtest
        print("\nRunning backtest with RSI strategy...")
        results = engine.run_backtest(strategy_wrapper)
        
        if results:
            # Generate trade log
            log_filename = f"logs/{symbol}_rsi_backtest_{start_date}_{end_date}.csv"
            print(f"\nGenerating trade log: {log_filename}")
            engine.generate_trade_log(results, filename=log_filename)
            
            # Plot the results
            plot_filename = f"logs/{symbol}_rsi_backtest_{start_date}_{end_date}.png"
            print(f"Generating performance chart: {plot_filename}")
            engine.plot_results(results, filename=plot_filename)
            
            # Save results to database
            engine.save_results(results, "RSI_Strategy")
            
            # Check for any alerts
            alerts = engine.monitor_and_alert(results)
            if alerts:
                print(f"\nBacktest generated {len(alerts)} alerts:")
                for alert in alerts:
                    print(f"  - {alert['severity'].upper()}: {alert['message']}")
            
            # Print summary
            print("\n=== RSI Backtest Results ===")
            print(f"Symbol: {symbol}")
            print(f"Strategy: RSI ({timeframes[0]})")
            print(f"Period: {start_date} to {end_date}")
            print(f"Initial Capital: ${results['initial_capital']:.2f}")
            print(f"Final Equity: ${results['final_equity']:.2f}")
            print(f"Total Return: {results['total_return_pct']:.2f}%")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Win Rate: {results['win_rate']:.2f}%")
            print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            
            # Calculate average RSI at buy/sell if trades exist
            if results['trades']:
                buy_trades_rsi = []
                sell_trades_rsi = []
                
                for timeframe in timeframes:
                    if timeframe in engine.market_data:
                        df = engine.market_data[timeframe]
                        
                        for trade in results['trades']:
                            trade_time = trade['timestamp']
                            # Find the candle closest to the trade
                            nearest_idx = df['timestamp'].searchsorted(trade_time)
                            if nearest_idx < len(df):
                                if trade['side'] == 'BUY':
                                    buy_trades_rsi.append(df['rsi'].iloc[nearest_idx])
                                elif trade['side'] == 'SELL':
                                    sell_trades_rsi.append(df['rsi'].iloc[nearest_idx])
                
                if buy_trades_rsi:
                    avg_buy_rsi = sum(buy_trades_rsi) / len(buy_trades_rsi)
                    print(f"Average RSI at Buy: {avg_buy_rsi:.2f}")
                
                if sell_trades_rsi:
                    avg_sell_rsi = sum(sell_trades_rsi) / len(sell_trades_rsi)
                    print(f"Average RSI at Sell: {avg_sell_rsi:.2f}")
            
            return results
        else:
            print("Backtest failed: No results generated")
            return None
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        logger.error(f"Backtest error: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    print("\n=== RSI Strategy Backtest Example ===")
    results = run_rsi_backtest()
    print("\nBacktest complete! Check logs directory for results.") 