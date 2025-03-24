"""
Technical indicator calculation functions for the backtesting module.
These functions are designed to work with pandas DataFrames of market data.
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
    Calculate Exponential Moving Average.
    
    Args:
        df: DataFrame containing OHLCV data
        period: Moving average period
        column: Column to use for calculation
        
    Returns:
        pd.Series: Series with calculated EMA values
    """
    return df[column].ewm(span=period, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        df: DataFrame containing OHLCV data
        period: RSI period
        column: Column to use for calculation
        
    Returns:
        pd.Series: Series with calculated RSI values (0-100)
    """
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # For values after the initial period
    for i in range(period, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, deviation: float = 2.0, 
                            column: str = 'close') -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame containing OHLCV data
        period: Moving average period
        deviation: Standard deviation multiplier
        column: Column to use for calculation
        
    Returns:
        Dict[str, pd.Series]: Dictionary with 'upper_band', 'middle_band', 'lower_band'
    """
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
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
    Calculate Moving Average Convergence Divergence.
    
    Args:
        df: DataFrame containing OHLCV data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        column: Column to use for calculation
        
    Returns:
        Dict[str, pd.Series]: Dictionary with 'macd_line', 'signal_line', 'macd_histogram'
    """
    fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
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
    atr = true_range.rolling(window=period).mean()
    
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