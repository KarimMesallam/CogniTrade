"""
Technical indicator calculation functions for the backtesting module.
These functions are designed to work with pandas DataFrames of market data.
These implementations are closely aligned with TA-Lib for consistency.
"""
import pandas as pd
import numpy as np
from typing import Optional, Union, Dict


def calculate_sma(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        df: DataFrame containing OHLCV data
        period: Moving average period
        column: Column to use for calculation
        
    Returns:
        pd.Series: Series with calculated SMA values
    """
    return df[column].rolling(window=period).mean()


def calculate_ema(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
    """
    Calculate Exponential Moving Average using TA-Lib compatible method.
    
    Args:
        df: DataFrame containing OHLCV data
        period: Moving average period
        column: Column to use for calculation
        
    Returns:
        pd.Series: Series with calculated EMA values
    """
    # TA-Lib uses alpha = 2/(period+1) with SMA initialization
    alpha = 2.0 / (period + 1)
    
    # Calculate SMA for the initial value (TA-Lib approach)
    sma = df[column].rolling(window=period).mean().iloc[period-1]
    
    # Create a copy of the data to avoid modifying the original
    ema_values = df[column].copy()
    
    # Initialize with NaN for values before the period
    ema_values.iloc[:period-1] = np.nan
    
    # Set the first value to SMA of the first period entries
    if period <= len(df):
        ema_values.iloc[period-1] = sma
    
    # Calculate EMA recursively
    for i in range(period, len(df)):
        ema_values.iloc[i] = (df[column].iloc[i] * alpha) + (ema_values.iloc[i-1] * (1 - alpha))
    
    return ema_values


def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate Relative Strength Index using TA-Lib compatible method.
    
    Args:
        df: DataFrame containing OHLCV data
        period: RSI period
        column: Column to use for calculation
        
    Returns:
        pd.Series: Series with calculated RSI values (0-100)
    """
    # Calculate price changes
    delta = df[column].diff()
    
    # Separate gains and losses
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # First values for average gain and loss
    avg_gain = np.nan
    avg_loss = np.nan
    
    # Initialize RSI series with explicit dtype to avoid warning
    rsi = pd.Series(index=df.index, dtype=float)
    
    # Fill the first period values with NaN
    rsi.iloc[:period] = np.nan
    
    # Initial average gain and loss calculation (simple average)
    if len(gain) >= period:
        avg_gain = gain.iloc[1:period+1].mean()  # Skip first row as it's NaN
        avg_loss = loss.iloc[1:period+1].mean()  # Skip first row as it's NaN
        
        # Special case for flat prices - RSI should be 50
        if avg_gain == 0 and avg_loss == 0:
            rsi.iloc[period] = 50.0
        # Normal case
        elif avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi.iloc[period] = 100 - (100 / (1 + rs))
        else:
            rsi.iloc[period] = 100  # If no losses, RSI is 100
    
    # Calculate RSI for remaining rows using Wilder's smoothing method
    for i in range(period + 1, len(df)):
        # TA-Lib uses Wilder's smoothing: avg_gain = ((prev_avg_gain * (period-1)) + current_gain) / period
        avg_gain = ((avg_gain * (period-1)) + gain.iloc[i]) / period
        avg_loss = ((avg_loss * (period-1)) + loss.iloc[i]) / period
        
        # Special case for flat prices - RSI should be 50
        if avg_gain == 0 and avg_loss == 0:
            rsi.iloc[i] = 50.0
        # Normal case
        elif avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi.iloc[i] = 100 - (100 / (1 + rs))
        else:
            rsi.iloc[i] = 100  # If no losses, RSI is 100
    
    return rsi


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, deviation: float = 2.0, 
                            column: str = 'close') -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands using TA-Lib compatible method.
    
    Args:
        df: DataFrame containing OHLCV data
        period: Moving average period
        deviation: Standard deviation multiplier
        column: Column to use for calculation
        
    Returns:
        Dict[str, pd.Series]: Dictionary with 'upper_band', 'middle_band', 'lower_band'
    """
    # Calculate SMA (middle band)
    sma = df[column].rolling(window=period).mean()
    
    # Calculate standard deviation using population method (ddof=0) as TA-Lib does
    std = df[column].rolling(window=period).std(ddof=0)
    
    # Calculate upper and lower bands
    upper_band = sma + (std * deviation)
    lower_band = sma - (std * deviation)
    
    return {
        'sma': sma,
        'upper_band': upper_band,
        'lower_band': lower_band
    }


def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, column: str = 'close') -> Dict[str, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence using TA-Lib compatible method.
    
    Args:
        df: DataFrame containing OHLCV data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        column: Column to use for calculation
        
    Returns:
        Dict[str, pd.Series]: Dictionary with 'macd_line', 'signal_line', 'macd_histogram'
    """
    # Use the fixed EMA calculation to match TA-Lib
    fast_ema = calculate_ema(df, period=fast_period, column=column)
    slow_ema = calculate_ema(df, period=slow_period, column=column)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # For signal line, we need to handle NaN values properly
    # First, determine where both EMAs have valid data
    valid_macd = macd_line.dropna()
    
    # Create a signal line of the same length as macd_line
    signal_line = pd.Series(index=macd_line.index, dtype=float)
    signal_line[:] = np.nan
    
    # Only calculate signal where we have valid MACD values
    if len(valid_macd) >= signal_period:
        # First signal value is SMA of MACD for initial signal_period points
        start_idx = valid_macd.index[0]
        valid_idx = valid_macd.index
        
        # Use a range of values for the initial SMA calculation
        initial_range = valid_macd.iloc[:signal_period]
        signal_line.loc[valid_idx[signal_period-1]] = initial_range.mean()
        
        # Apply EMA smoothing for subsequent values
        alpha = 2.0 / (signal_period + 1)
        
        for i in range(signal_period, len(valid_idx)):
            current_idx = valid_idx[i]
            prev_idx = valid_idx[i-1]
            signal_line.loc[current_idx] = (valid_macd.loc[current_idx] * alpha) + \
                                           (signal_line.loc[prev_idx] * (1 - alpha))
    
    # Calculate histogram
    macd_histogram = macd_line - signal_line
    
    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'macd_histogram': macd_histogram
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        df: DataFrame containing OHLCV data
        period: ATR period
        
    Returns:
        pd.Series: Series with calculated ATR values
    """
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Use Wilder's smoothing method
    atr = pd.Series(index=df.index)
    atr.iloc[:period] = np.nan
    
    # First ATR value is simple average of true ranges
    if len(true_range) >= period:
        atr.iloc[period-1] = true_range.iloc[:period].mean()
    
    # Calculate ATR for remaining rows using Wilder's smoothing
    for i in range(period, len(df)):
        atr.iloc[i] = ((atr.iloc[i-1] * (period-1)) + true_range.iloc[i]) / period
    
    return atr


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        df: DataFrame containing OHLCV data
        k_period: %K period
        d_period: %D period
        
    Returns:
        Dict[str, pd.Series]: Dictionary with 'k' and 'd' values
    """
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    
    k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    
    return {'k': k, 'd': d}


def calculate_ichimoku(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26,
                      senkou_b_period: int = 52, displacement: int = 26) -> Dict[str, pd.Series]:
    """
    Calculate Ichimoku Cloud.
    
    Args:
        df: DataFrame containing OHLCV data
        tenkan_period: Tenkan-sen (Conversion Line) period
        kijun_period: Kijun-sen (Base Line) period
        senkou_b_period: Senkou Span B period
        displacement: Displacement period for Senkou Span
        
    Returns:
        Dict[str, pd.Series]: Dictionary with Ichimoku components
    """
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for tenkan_period
    tenkan_sen = (df['high'].rolling(window=tenkan_period).max() + 
                 df['low'].rolling(window=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for kijun_period
    kijun_sen = (df['high'].rolling(window=kijun_period).max() + 
                df['low'].rolling(window=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, displaced forward
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for senkou_b_period, displaced forward
    senkou_span_b = ((df['high'].rolling(window=senkou_b_period).max() + 
                     df['low'].rolling(window=senkou_b_period).min()) / 2).shift(displacement)
    
    # Calculate Chikou Span (Lagging Span): Current closing price, displaced backwards
    chikou_span = df['close'].shift(-displacement)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    } 