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
        
        # Combine dataframes safely
        try:
            # Ensure indices are aligned
            df = df.copy()
            
            # Merge the dataframes on their indices
            for col in bollinger.columns:
                if col not in df.columns:
                    df[col] = bollinger[col]
            
            for col in macd.columns:
                if col not in df.columns:
                    df[col] = macd[col]
            
            # Ensure we have at least two rows
            if len(df) < 2:
                logger.warning("Not enough data points after indicator calculation")
                return "HOLD"
            
            # Get latest values and previous values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Check for required fields before proceeding
            required_fields = ['rsi', 'close', 'upper_band', 'lower_band', 'macd_line', 'signal_line']
            for field in required_fields:
                if field not in df.columns or pd.isna(latest[field]) or pd.isna(prev[field]):
                    logger.warning(f"Missing or invalid field: {field}")
                    return "HOLD"
            
            # Initialize signal strength counters
            buy_signals = 0
            sell_signals = 0
            
            # Safe extraction of scalar values
            try:
                # RSI signals - use item() for Series scalar extraction
                rsi_value = latest['rsi']
                if isinstance(rsi_value, pd.Series):
                    rsi_value = rsi_value.iloc[0]
                    
                if rsi_value < 30:  # Oversold
                    buy_signals += 1
                    logger.info(f"RSI indicates oversold condition: {rsi_value:.2f}")
                elif rsi_value > 70:  # Overbought
                    sell_signals += 1
                    logger.info(f"RSI indicates overbought condition: {rsi_value:.2f}")
                
                # Bollinger Band signals
                close_price = latest['close']
                upper_band = latest['upper_band']
                lower_band = latest['lower_band']
                
                if isinstance(close_price, pd.Series):
                    close_price = close_price.iloc[0]
                if isinstance(upper_band, pd.Series):
                    upper_band = upper_band.iloc[0]
                if isinstance(lower_band, pd.Series):
                    lower_band = lower_band.iloc[0]
                
                if close_price > upper_band:  # Price above upper band
                    sell_signals += 1
                    logger.info("Price above upper Bollinger Band")
                elif close_price < lower_band:  # Price below lower band
                    buy_signals += 1
                    logger.info("Price below lower Bollinger Band")
                
                # MACD signals
                prev_macd = prev['macd_line']
                prev_signal = prev['signal_line']
                latest_macd = latest['macd_line']
                latest_signal = latest['signal_line']
                
                if isinstance(prev_macd, pd.Series):
                    prev_macd = prev_macd.iloc[0]
                if isinstance(prev_signal, pd.Series):
                    prev_signal = prev_signal.iloc[0]
                if isinstance(latest_macd, pd.Series):
                    latest_macd = latest_macd.iloc[0]
                if isinstance(latest_signal, pd.Series):
                    latest_signal = latest_signal.iloc[0]
                
                # MACD line crosses above signal line
                if prev_macd < prev_signal and latest_macd > latest_signal:
                    buy_signals += 1
                    logger.info("MACD line crossed above signal line")
                # MACD line crosses below signal line
                elif prev_macd > prev_signal and latest_macd < latest_signal:
                    sell_signals += 1
                    logger.info("MACD line crossed below signal line")
            except (ValueError, TypeError) as e:
                logger.error(f"Error extracting scalar values: {e}")
                return "HOLD"
            
            # Determine overall signal
            if buy_signals > sell_signals:
                return "BUY"
            elif sell_signals > buy_signals:
                return "SELL"
            else:
                return "HOLD"
                
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error processing indicator data: {e}")
            return "HOLD"
            
    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")
        return "HOLD"
