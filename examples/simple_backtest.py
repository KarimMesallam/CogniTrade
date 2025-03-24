#!/usr/bin/env python
"""
Simplified example script for using the backtesting module with real Binance data.
This shows the basic process of:
1. Connecting to Binance
2. Downloading historical data
3. Running a basic strategy backtest
4. Visualizing the results
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path to import bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Print API configuration info (without exposing secrets)
print("\n=== API Configuration ===")
print(f"API Key configured: {'Yes' if API_KEY else 'No'}")
print(f"API Secret configured: {'Yes' if API_SECRET else 'No'}")
print(f"Using Testnet: {TESTNET}")
print(f"Default Symbol: {SYMBOL}")

def get_historical_data(symbol, interval, start_date, end_date):
    """
    Get historical OHLCV data directly from Binance
    """
    try:
        # Convert dates to millisecond timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        # Get klines data from Binance
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=str(start_ts),
            end_str=str(end_ts)
        )
        
        if not klines:
            print(f"No data returned for {symbol} at {interval} timeframe")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
            df[col] = df[col].astype(float)
        
        print(f"Downloaded {len(df)} candles for {symbol} at {interval} timeframe")
        return df
    
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def run_simple_backtest():
    """Run a simple SMA crossover strategy backtest"""
    
    # Parameters for backtest
    symbol = SYMBOL  # Use the symbol from .env
    timeframe = '1h'  # Simplify to just one timeframe for now
    
    # Use a shorter period to ensure we get data (last 60 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print(f"\n=== Running Backtest ===")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Period: {start_date} to {end_date}")
    
    # Download data
    print("\nDownloading historical data from Binance...")
    
    try:
        # Try to ping Binance to verify connection
        client.ping()
        print("Successfully connected to Binance API")
        
        # Download historical data
        df = get_historical_data(symbol, timeframe, start_date, end_date)
        
        if df is None or len(df) < 50:
            print(f"Not enough data for backtesting. Need at least 50 candles.")
            return
        
    except Exception as e:
        print(f"Error connecting to Binance API: {e}")
        return
    
    # Calculate indicators for the strategy
    short_period = 10
    long_period = 30
    
    df['sma_short'] = df['close'].rolling(window=short_period).mean()
    df['sma_long'] = df['close'].rolling(window=long_period).mean()
    
    # Generate buy/sell signals
    df['signal'] = 0
    # Buy signal: short SMA crosses above long SMA
    df.loc[(df['sma_short'] > df['sma_long']) & 
           (df['sma_short'].shift(1) <= df['sma_long'].shift(1)), 'signal'] = 1
    # Sell signal: short SMA crosses below long SMA
    df.loc[(df['sma_short'] < df['sma_long']) & 
           (df['sma_short'].shift(1) >= df['sma_long'].shift(1)), 'signal'] = -1
    
    # Initialize backtest variables
    initial_capital = 10000.0
    commission = 0.001  # 0.1%
    position = 0
    capital = initial_capital
    holdings = 0
    equity = []
    trades = []
    
    # Run backtest
    print("\nRunning backtest...")
    
    for idx, row in df.iterrows():
        equity_value = capital
        
        if position > 0:
            equity_value += holdings * row['close']
        
        equity.append({
            'timestamp': row['timestamp'],
            'equity': equity_value
        })
        
        # Process signals
        if row['signal'] == 1 and position == 0:  # Buy signal
            # Calculate position size (invest 95% of capital)
            trade_amount = capital * 0.95
            holdings = trade_amount / row['close'] * (1 - commission)
            capital -= trade_amount
            position = 1
            
            trades.append({
                'timestamp': row['timestamp'],
                'type': 'BUY',
                'price': row['close'],
                'quantity': holdings,
                'value': trade_amount,
                'commission': trade_amount * commission
            })
            
        elif row['signal'] == -1 and position == 1:  # Sell signal
            # Sell all holdings
            trade_value = holdings * row['close'] * (1 - commission)
            capital += trade_value
            
            trades.append({
                'timestamp': row['timestamp'],
                'type': 'SELL',
                'price': row['close'],
                'quantity': holdings,
                'value': trade_value,
                'commission': trade_value * commission,
                'profit': trade_value - trades[-1]['value']
            })
            
            holdings = 0
            position = 0
    
    # Calculate final equity
    final_equity = capital
    if position > 0:
        final_equity += holdings * df.iloc[-1]['close']
    
    # Calculate performance metrics
    equity_df = pd.DataFrame(equity)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Total return
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Win rate
    if not trades_df.empty and 'profit' in trades_df.columns:
        winning_trades = trades_df[trades_df['profit'] > 0]
        win_rate = len(winning_trades) / (len(trades_df) / 2) * 100 if len(trades_df) > 0 else 0
    else:
        win_rate = 0
    
    # Max drawdown
    if not equity_df.empty:
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].min()
    else:
        max_drawdown = 0
    
    # Generate trade log
    if not trades_df.empty:
        log_filename = f"logs/{symbol}_sma_crossover_{start_date}_{end_date}.csv"
        print(f"\nGenerating trade log: {log_filename}")
        trades_df.to_csv(log_filename, index=False)
    
    # Create visualization
    if not equity_df.empty:
        plot_filename = f"logs/{symbol}_sma_crossover_{start_date}_{end_date}.png"
        print(f"Generating performance chart: {plot_filename}")
        
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity_df['timestamp'], equity_df['equity'])
        plt.title(f'Equity Curve - SMA Crossover ({symbol})')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        
        # Plot price and SMA
        plt.subplot(2, 1, 2)
        plt.plot(df['timestamp'], df['close'], label='Price')
        plt.plot(df['timestamp'], df['sma_short'], label=f'SMA {short_period}')
        plt.plot(df['timestamp'], df['sma_long'], label=f'SMA {long_period}')
        
        # Plot buy/sell points
        if not trades_df.empty:
            buy_points = trades_df[trades_df['type'] == 'BUY']
            sell_points = trades_df[trades_df['type'] == 'SELL']
            
            if not buy_points.empty:
                plt.scatter(buy_points['timestamp'], buy_points['price'], 
                           marker='^', color='g', s=100, label='Buy')
            
            if not sell_points.empty:
                plt.scatter(sell_points['timestamp'], sell_points['price'], 
                           marker='v', color='r', s=100, label='Sell')
        
        plt.title(f'Price and SMA - {symbol}')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plot_filename)
    
    # Print summary
    print("\n=== Backtest Results ===")
    print(f"Symbol: {symbol}")
    print(f"Strategy: SMA Crossover ({timeframe})")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Equity: ${final_equity:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {len(trades_df) if not trades_df.empty else 0}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")

if __name__ == "__main__":
    print("\n=== Simple Backtest Example ===")
    run_simple_backtest()
    print("\nBacktest complete! Check logs directory for results.") 