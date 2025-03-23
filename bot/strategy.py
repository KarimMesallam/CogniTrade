import numpy as np
import pandas as pd
import logging
from bot.binance_api import get_recent_closes, client

logger = logging.getLogger("trading_bot")

def simple_signal(symbol, interval='1m'):
    """
    A simple strategy based on price movement.
    Returns 'BUY', 'SELL', or 'HOLD'.
    """
    closes = get_recent_closes(symbol, interval)
    if len(closes) < 2:
        logger.warning("Not enough data for simple signal")
        return "HOLD"
    # Simple strategy: if the latest close is higher than the previous, signal a buy
    return 'BUY' if closes[-1] > closes[-2] else 'SELL'

def get_candles_dataframe(symbol, interval='1h', limit=100):
    """
    Fetch candles and convert to pandas DataFrame for technical analysis.
    """
    try:
        candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        return df
    except Exception as e:
        logger.error(f"Error creating candles dataframe: {e}")
        return None

def calculate_rsi(df, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    Returns values between 0-100.
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.
    Returns upper band, middle band (SMA), and lower band.
    """
    df['sma'] = df['close'].rolling(window=period).mean()
    df['std_dev'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['sma'] + (df['std_dev'] * std_dev)
    df['lower_band'] = df['sma'] - (df['std_dev'] * std_dev)
    
    return df[['upper_band', 'sma', 'lower_band']]

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    Returns MACD line, signal line, and histogram.
    """
    df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    df['macd_line'] = df['ema_fast'] - df['ema_slow']
    df['signal_line'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['signal_line']
    
    return df[['macd_line', 'signal_line', 'macd_histogram']]

def technical_analysis_signal(symbol, interval='1h'):
    """
    Generate trading signals based on multiple technical indicators.
    
    Returns 'BUY', 'SELL', or 'HOLD' based on:
    - RSI (oversold/overbought)
    - Bollinger Bands (price breakouts)
    - MACD (trend and momentum)
    """
    try:
        df = get_candles_dataframe(symbol, interval, limit=100)
        if df is None or len(df) < 30:
            logger.warning(f"Not enough data for technical analysis on {symbol}")
            return "HOLD"
        
        # Calculate indicators
        df['rsi'] = calculate_rsi(df)
        bollinger = calculate_bollinger_bands(df)
        macd = calculate_macd(df)
        
        # Combine indicators
        df = pd.concat([df, bollinger, macd], axis=1)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Initialize signal strength counters
        buy_signals = 0
        sell_signals = 0
        
        # RSI signals
        if latest['rsi'] < 30:  # Oversold
            buy_signals += 1
            logger.info(f"RSI indicates oversold condition: {latest['rsi']:.2f}")
        elif latest['rsi'] > 70:  # Overbought
            sell_signals += 1
            logger.info(f"RSI indicates overbought condition: {latest['rsi']:.2f}")
        
        # Bollinger Band signals
        if latest['close'] > latest['upper_band']:  # Price above upper band
            sell_signals += 1
            logger.info("Price above upper Bollinger Band")
        elif latest['close'] < latest['lower_band']:  # Price below lower band
            buy_signals += 1
            logger.info("Price below lower Bollinger Band")
        
        # MACD signals
        prev = df.iloc[-2]
        # MACD line crosses above signal line
        if prev['macd_line'] < prev['signal_line'] and latest['macd_line'] > latest['signal_line']:
            buy_signals += 1
            logger.info("MACD line crossed above signal line")
        # MACD line crosses below signal line
        elif prev['macd_line'] > prev['signal_line'] and latest['macd_line'] < latest['signal_line']:
            sell_signals += 1
            logger.info("MACD line crossed below signal line")
        
        # Determine overall signal
        if buy_signals > sell_signals:
            return "BUY"
        elif sell_signals > buy_signals:
            return "SELL"
        else:
            return "HOLD"
            
    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")
        return "HOLD"
