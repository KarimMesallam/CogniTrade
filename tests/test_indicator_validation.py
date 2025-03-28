"""
Test module for validating custom technical indicator implementations against TA-Lib.
This module compares our custom implementations in bot/backtesting/utils/indicators.py
with the industry-standard TA-Lib library to ensure accuracy.
"""

import pytest
import pandas as pd
import numpy as np
import talib
import logging
from datetime import datetime, timedelta

from bot.backtesting.utils.indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd
)

# Configure logging for the test
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_data():
    """
    Create a sample dataset with known price movement patterns.
    """
    # Create date range for 100 days with daily data
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=100, freq='1d')
    
    # Create price series with different patterns
    np.random.seed(42)  # For reproducibility
    
    # Create a trend with some randomness
    trend = np.linspace(100, 150, 100) + np.random.normal(0, 5, 100)
    
    # Create a series with more volatility in the middle section
    volatility = np.zeros(100)
    volatility[30:70] = 2.0  # Higher volatility in the middle
    vol_adjusted = trend + np.random.normal(0, volatility, 100)
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': vol_adjusted - np.random.uniform(0, 2, 100),
        'high': vol_adjusted + np.random.uniform(1, 3, 100),
        'low': vol_adjusted - np.random.uniform(1, 3, 100),
        'close': vol_adjusted,
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    return df

def test_sma_validation(sample_data):
    """
    Validate SMA implementation against TA-Lib's implementation.
    """
    periods_to_test = [5, 10, 20, 50]
    
    for period in periods_to_test:
        # Calculate SMA using our custom implementation
        custom_sma = calculate_sma(sample_data, period=period)
        
        # Calculate SMA using TA-Lib
        talib_sma = pd.Series(talib.SMA(sample_data['close'].values, timeperiod=period))
        
        # Compare results (skip NaN values at the beginning)
        start_idx = period
        custom_valid = custom_sma.iloc[start_idx:].reset_index(drop=True)
        talib_valid = talib_sma.iloc[start_idx:].reset_index(drop=True)
        
        # Check absolute difference is very small
        diff = (custom_valid - talib_valid).abs()
        max_diff = diff.max()
        
        # Log the maximum difference
        logger.info(f"SMA period={period}, max_diff={max_diff}")
        
        # Assert that differences are negligible (float precision issues might cause tiny differences)
        assert max_diff < 1e-10, f"SMA period={period} implementation differs from TA-Lib by {max_diff}"

def test_ema_validation(sample_data):
    """
    Validate EMA implementation against TA-Lib's implementation.
    """
    periods_to_test = [5, 12, 26, 50]
    
    for period in periods_to_test:
        # Calculate EMA using our custom implementation
        custom_ema = calculate_ema(sample_data, period=period)
        
        # Calculate EMA using TA-Lib
        talib_ema = pd.Series(talib.EMA(sample_data['close'].values, timeperiod=period))
        
        # Compare results (skip NaN values at the beginning)
        # EMA can have slight differences in the initial values due to initialization methods
        start_idx = period + 5  # Skip a few more periods to allow for convergence
        custom_valid = custom_ema.iloc[start_idx:].reset_index(drop=True)
        talib_valid = talib_ema.iloc[start_idx:].reset_index(drop=True)
        
        # Check absolute difference is very small
        diff = (custom_valid - talib_valid).abs()
        max_diff = diff.max()
        
        # Log the maximum difference
        logger.info(f"EMA period={period}, max_diff={max_diff}")
        
        # Assert that differences are negligible
        # For EMA, we allow a slightly larger tolerance due to potential differences in initialization
        assert max_diff < 1e-6, f"EMA period={period} implementation differs from TA-Lib by {max_diff}"

def test_rsi_validation(sample_data):
    """
    Validate RSI implementation against TA-Lib's implementation.
    """
    periods_to_test = [9, 14, 25]
    
    for period in periods_to_test:
        # Calculate RSI using our custom implementation
        custom_rsi = calculate_rsi(sample_data, period=period)
        
        # Calculate RSI using TA-Lib
        talib_rsi = pd.Series(talib.RSI(sample_data['close'].values, timeperiod=period))
        
        # Compare results (skip NaN values)
        start_idx = period + 10  # Skip more periods to allow for convergence
        custom_valid = custom_rsi.dropna().iloc[start_idx-period:].reset_index(drop=True)
        talib_valid = talib_rsi.dropna().iloc[start_idx-period:].reset_index(drop=True)
        
        # Make sure we have data to compare
        min_length = min(len(custom_valid), len(talib_valid))
        if min_length > 0:
            custom_valid = custom_valid.iloc[:min_length]
            talib_valid = talib_valid.iloc[:min_length]
            
            # Check absolute difference
            diff = (custom_valid - talib_valid).abs()
            max_diff = diff.max()
            
            # Log the maximum difference
            logger.info(f"RSI period={period}, max_diff={max_diff}")
            
            # RSI implementations can vary more, so we use a higher tolerance
            assert max_diff < 1.0, f"RSI period={period} implementation differs from TA-Lib by {max_diff}"
        else:
            pytest.skip(f"Not enough data to compare RSI with period={period}")

def test_bollinger_bands_validation(sample_data):
    """
    Validate Bollinger Bands implementation against TA-Lib's implementation.
    """
    periods_to_test = [20]
    std_devs_to_test = [2.0, 2.5]
    
    for period in periods_to_test:
        for std_dev in std_devs_to_test:
            # Calculate Bollinger Bands using our custom implementation
            custom_bb = calculate_bollinger_bands(sample_data, period=period, deviation=std_dev)
            custom_upper = custom_bb['upper_band']
            custom_middle = custom_bb['sma']
            custom_lower = custom_bb['lower_band']
            
            # Calculate Bollinger Bands using TA-Lib
            talib_upper, talib_middle, talib_lower = talib.BBANDS(
                sample_data['close'].values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0  # SMA type
            )
            
            # Convert to pandas Series
            talib_upper = pd.Series(talib_upper)
            talib_middle = pd.Series(talib_middle)
            talib_lower = pd.Series(talib_lower)
            
            # Compare middle band (SMA) first
            start_idx = period
            custom_middle_valid = custom_middle.iloc[start_idx:].reset_index(drop=True)
            talib_middle_valid = talib_middle.iloc[start_idx:].reset_index(drop=True)
            
            diff_middle = (custom_middle_valid - talib_middle_valid).abs().max()
            logger.info(f"BB period={period}, stddev={std_dev}, middle_band_max_diff={diff_middle}")
            assert diff_middle < 1e-10, f"BB middle band implementation differs from TA-Lib by {diff_middle}"
            
            # Compare upper band
            custom_upper_valid = custom_upper.iloc[start_idx:].reset_index(drop=True)
            talib_upper_valid = talib_upper.iloc[start_idx:].reset_index(drop=True)
            
            diff_upper = (custom_upper_valid - talib_upper_valid).abs().max()
            logger.info(f"BB period={period}, stddev={std_dev}, upper_band_max_diff={diff_upper}")
            assert diff_upper < 1e-10, f"BB upper band implementation differs from TA-Lib by {diff_upper}"
            
            # Compare lower band
            custom_lower_valid = custom_lower.iloc[start_idx:].reset_index(drop=True)
            talib_lower_valid = talib_lower.iloc[start_idx:].reset_index(drop=True)
            
            diff_lower = (custom_lower_valid - talib_lower_valid).abs().max()
            logger.info(f"BB period={period}, stddev={std_dev}, lower_band_max_diff={diff_lower}")
            assert diff_lower < 1e-10, f"BB lower band implementation differs from TA-Lib by {diff_lower}"

def test_macd_validation(sample_data):
    """
    Validate MACD implementation against TA-Lib's implementation.
    """
    configs_to_test = [
        (12, 26, 9),  # Standard MACD parameters
        (8, 17, 9),   # Alternative parameters
    ]
    
    for fast_period, slow_period, signal_period in configs_to_test:
        # Calculate MACD using our custom implementation
        custom_macd = calculate_macd(
            sample_data, 
            fast_period=fast_period, 
            slow_period=slow_period, 
            signal_period=signal_period
        )
        
        custom_macd_line = custom_macd['macd_line']
        custom_signal_line = custom_macd['signal_line']
        custom_histogram = custom_macd['macd_histogram']
        
        # Calculate MACD using TA-Lib
        talib_macd_line, talib_signal_line, talib_histogram = talib.MACD(
            sample_data['close'].values,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        
        # Convert to pandas Series
        talib_macd_line = pd.Series(talib_macd_line)
        talib_signal_line = pd.Series(talib_signal_line)
        talib_histogram = pd.Series(talib_histogram)
        
        # MACD requires more data points to converge
        start_idx = max(fast_period, slow_period) + signal_period + 10
        
        # Compare MACD line
        custom_macd_valid = custom_macd_line.iloc[start_idx:].reset_index(drop=True)
        talib_macd_valid = talib_macd_line.iloc[start_idx:].reset_index(drop=True)
        
        diff_macd = (custom_macd_valid - talib_macd_valid).abs().max()
        logger.info(f"MACD fast={fast_period}, slow={slow_period}, signal={signal_period}, macd_line_max_diff={diff_macd}")
        # MACD can have larger differences due to initialization method differences
        assert diff_macd < 0.1, f"MACD line implementation differs from TA-Lib by {diff_macd}"
        
        # Compare Signal line
        custom_signal_valid = custom_signal_line.iloc[start_idx:].reset_index(drop=True)
        talib_signal_valid = talib_signal_line.iloc[start_idx:].reset_index(drop=True)
        
        diff_signal = (custom_signal_valid - talib_signal_valid).abs().max()
        logger.info(f"MACD fast={fast_period}, slow={slow_period}, signal={signal_period}, signal_line_max_diff={diff_signal}")
        assert diff_signal < 0.1, f"MACD signal line implementation differs from TA-Lib by {diff_signal}"
        
        # Compare Histogram
        custom_hist_valid = custom_histogram.iloc[start_idx:].reset_index(drop=True)
        talib_hist_valid = talib_histogram.iloc[start_idx:].reset_index(drop=True)
        
        diff_hist = (custom_hist_valid - talib_hist_valid).abs().max()
        logger.info(f"MACD fast={fast_period}, slow={slow_period}, signal={signal_period}, histogram_max_diff={diff_hist}")
        assert diff_hist < 0.1, f"MACD histogram implementation differs from TA-Lib by {diff_hist}"

def test_all_indicators_with_edge_cases():
    """
    Test all indicators with edge cases like flat prices and extreme volatility.
    """
    # Create date range for test data
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=100, freq='1d')
    
    # Test case 1: Flat prices (zero volatility)
    flat_df = pd.DataFrame({
        'timestamp': dates,
        'open': np.full(100, 100.0),
        'high': np.full(100, 100.0),
        'low': np.full(100, 100.0),
        'close': np.full(100, 100.0),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    # Test all indicators with flat prices
    for period in [14, 20]:
        # Calculate indicators with our implementation
        custom_sma = calculate_sma(flat_df, period=period)
        custom_ema = calculate_ema(flat_df, period=period)
        custom_rsi = calculate_rsi(flat_df, period=period)
        custom_bb = calculate_bollinger_bands(flat_df, period=period)
        
        # Check SMA and EMA are constant with flat prices
        assert custom_sma.iloc[period:].std() < 1e-10, "SMA should be constant with flat prices"
        assert custom_ema.iloc[period:].std() < 1e-10, "EMA should be constant with flat prices"
        
        # Check RSI with flat prices (should be close to 50 after initial values)
        if period < len(custom_rsi) - 10:
            rsi_later_values = custom_rsi.iloc[period+10:].dropna()
            if len(rsi_later_values) > 0:
                # With flat prices, RSI should be close to 50 (no momentum)
                assert abs(50 - rsi_later_values.mean()) < 1.0, "RSI should be near 50 with flat prices"
        
        # Check Bollinger Bands with flat prices
        # Upper and lower bands should be equidistant from the middle
        start_idx = period + 5
        if start_idx < len(custom_bb['upper_band']):
            upper_dist = (custom_bb['upper_band'].iloc[start_idx] - custom_bb['sma'].iloc[start_idx])
            lower_dist = (custom_bb['sma'].iloc[start_idx] - custom_bb['lower_band'].iloc[start_idx])
            assert abs(upper_dist - lower_dist) < 1e-10, "Bollinger Bands should be equidistant with flat prices"
    
    # Test case 2: High volatility
    np.random.seed(42)
    volatile_prices = 100 + np.random.normal(0, 10, 100)
    volatile_df = pd.DataFrame({
        'timestamp': dates,
        'open': volatile_prices - np.random.uniform(0, 5, 100),
        'high': volatile_prices + np.random.uniform(2, 8, 100),
        'low': volatile_prices - np.random.uniform(2, 8, 100),
        'close': volatile_prices,
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    # Test Bollinger Bands with high volatility
    period = 20
    custom_bb = calculate_bollinger_bands(volatile_df, period=period)
    
    # With high volatility, bands should be wider
    start_idx = period + 5
    if start_idx < len(custom_bb['upper_band']):
        band_width = custom_bb['upper_band'].iloc[start_idx:] - custom_bb['lower_band'].iloc[start_idx:]
        assert band_width.mean() > 30, "Bollinger Bands should be wide with high volatility" 