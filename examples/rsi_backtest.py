#!/usr/bin/env python
"""
RSI Strategy Backtesting Example
This script demonstrates how to backtest an RSI-based trading strategy 
using historical data from Binance.

The RSI strategy:
- Buy when RSI crosses below the oversold threshold and then back above it (bullish divergence)
- Sell when RSI crosses above the overbought threshold and then back below it (bearish divergence)
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
        logging.FileHandler("logs/rsi_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI)
    
    Args:
        data: DataFrame containing price data
        period: RSI period (default: 14)
        
    Returns:
        Series containing RSI values
    """
    # Calculate price changes
    delta = data['close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

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

def run_rsi_backtest():
    """Run a backtest using the RSI strategy"""
    
    # Parameters for backtest
    symbol = SYMBOL  # Use the symbol from .env
    timeframe = '4h'  # 4-hour timeframe is often good for RSI
    rsi_period = 14  # Standard RSI period
    overbought = 70  # Overbought threshold
    oversold = 30    # Oversold threshold
    
    # Use a 90-day period for backtesting
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"\n=== Running RSI Strategy Backtest ===")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Period: {start_date} to {end_date}")
    print(f"RSI Settings: Period={rsi_period}, Overbought={overbought}, Oversold={oversold}")
    
    # Download data
    print("\nDownloading historical data from Binance...")
    
    try:
        # Try to ping Binance to verify connection
        client.ping()
        print("Successfully connected to Binance API")
        
        # Download historical data
        df = get_historical_data(symbol, timeframe, start_date, end_date)
        
        if df is None or len(df) < rsi_period + 10:
            print(f"Not enough data for backtesting. Need at least {rsi_period + 10} candles.")
            return
        
    except Exception as e:
        print(f"Error connecting to Binance API: {e}")
        return
    
    # Calculate RSI
    print("Calculating RSI values...")
    df['rsi'] = calculate_rsi(df, period=rsi_period)
    
    # Generate RSI signals
    df['signal'] = 0
    df['prev_below_oversold'] = df['rsi'].shift(1) < oversold
    df['prev_above_overbought'] = df['rsi'].shift(1) > overbought
    
    # Buy signal: RSI crosses from below oversold to above oversold
    df.loc[(df['rsi'] > oversold) & (df['prev_below_oversold']), 'signal'] = 1
    
    # Sell signal: RSI crosses from above overbought to below overbought
    df.loc[(df['rsi'] < overbought) & (df['prev_above_overbought']), 'signal'] = -1
    
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
        if pd.isna(row['rsi']):
            continue  # Skip rows without RSI values
            
        equity_value = capital
        
        if position > 0:
            equity_value += holdings * row['close']
        
        equity.append({
            'timestamp': row['timestamp'],
            'equity': equity_value,
            'rsi': row['rsi']
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
                'commission': trade_amount * commission,
                'rsi': row['rsi']
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
                'profit': trade_value - trades[-1]['value'],
                'rsi': row['rsi']
            })
            
            holdings = 0
            position = 0
    
    # Handle final position if still open
    final_equity = capital
    if position > 0:
        final_value = holdings * df.iloc[-1]['close']
        final_equity += final_value
        
        # Add a hypothetical closing trade for reporting
        if trades:
            trades.append({
                'timestamp': df.iloc[-1]['timestamp'],
                'type': 'HOLD',  # Not closed yet
                'price': df.iloc[-1]['close'],
                'quantity': holdings,
                'value': final_value,
                'profit': final_value - trades[-1]['value']
            })
    
    # Calculate performance metrics
    equity_df = pd.DataFrame(equity)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Total return
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Win rate
    if not trades_df.empty and 'profit' in trades_df.columns:
        win_trades = trades_df[trades_df['profit'] > 0]
        win_rate = len(win_trades) / len(trades_df[trades_df['type'] == 'SELL']) * 100 if len(trades_df[trades_df['type'] == 'SELL']) > 0 else 0
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
        log_filename = f"logs/{symbol}_rsi_backtest_{start_date}_{end_date}.csv"
        print(f"\nGenerating trade log: {log_filename}")
        trades_df.to_csv(log_filename, index=False)
    
    # Create visualization
    if not equity_df.empty:
        plot_filename = f"logs/{symbol}_rsi_backtest_{start_date}_{end_date}.png"
        print(f"Generating performance chart: {plot_filename}")
        
        plt.figure(figsize=(15, 10))
        
        # Plot equity curve
        plt.subplot(3, 1, 1)
        plt.plot(equity_df['timestamp'], equity_df['equity'])
        plt.title(f'Equity Curve - RSI Strategy ({symbol})')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        
        # Plot price
        plt.subplot(3, 1, 2)
        plt.plot(df['timestamp'], df['close'], label='Price')
        
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
        
        plt.title(f'Price Chart - {symbol}')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot RSI
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['rsi'], color='purple', label='RSI')
        plt.axhline(y=overbought, color='r', linestyle='--', label=f'Overbought ({overbought})')
        plt.axhline(y=oversold, color='g', linestyle='--', label=f'Oversold ({oversold})')
        plt.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        plt.fill_between(df['timestamp'], oversold, df['rsi'], 
                         where=(df['rsi'] < oversold), color='g', alpha=0.3)
        plt.fill_between(df['timestamp'], overbought, df['rsi'], 
                         where=(df['rsi'] > overbought), color='r', alpha=0.3)
        
        plt.title('RSI Indicator')
        plt.ylabel('RSI Value')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plot_filename)
    
    # Print summary
    print("\n=== RSI Backtest Results ===")
    print(f"Symbol: {symbol}")
    print(f"Strategy: RSI ({timeframe})")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Equity: ${final_equity:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {len(trades_df[trades_df['type'] != 'HOLD']) if not trades_df.empty else 0}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    # Print additional RSI-specific metrics if available
    if not trades_df.empty:
        avg_rsi_buy = trades_df[trades_df['type'] == 'BUY']['rsi'].mean() if 'rsi' in trades_df.columns else 'N/A'
        avg_rsi_sell = trades_df[trades_df['type'] == 'SELL']['rsi'].mean() if 'rsi' in trades_df.columns else 'N/A'
        
        print(f"\nRSI Strategy Metrics:")
        print(f"Average RSI at Buy: {avg_rsi_buy:.2f}")
        print(f"Average RSI at Sell: {avg_rsi_sell:.2f}")

if __name__ == "__main__":
    print("\n=== RSI Strategy Backtest Example ===")
    run_rsi_backtest()
    print("\nBacktest complete! Check logs directory for results.") 