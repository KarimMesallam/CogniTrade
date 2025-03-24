"""
Example custom trading strategy.

This module demonstrates how to create a custom strategy that can be
dynamically loaded by the trading bot based on configuration.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, Optional, List, Tuple


def initialize() -> Dict[str, Any]:
    """
    Initialize strategy parameters.
    
    Returns:
        Dict containing any parameters needed for the strategy
    """
    return {
        "name": "My Custom Strategy",
        "description": "An example custom strategy combining momentum and volatility",
        "required_indicators": ["ema", "atr", "rsi"],
        "lookback_periods": 50,  # How many candles we need for calculation
        "parameters": {
            "ema_period": 20,
            "atr_period": 14,
            "rsi_period": 14,
            "rsi_threshold_high": 70,
            "rsi_threshold_low": 30,
            "volatility_threshold": 0.5,  # Percentage
        }
    }


def prepare_data(df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Calculate indicators and prepare data for the strategy.
    
    Args:
        df: DataFrame containing OHLCV data
        params: Optional parameters to override defaults from initialize()
    
    Returns:
        DataFrame with additional columns for indicators
    """
    # Use params if provided, otherwise use defaults
    if params is None:
        params = initialize()["parameters"]
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate EMA
    df['ema'] = talib.EMA(df['close'], timeperiod=params['ema_period'])
    
    # Calculate ATR (Average True Range) for volatility
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=params['atr_period'])
    df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR as percentage of price
    
    # Calculate RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=params['rsi_period'])
    
    # Calculate price position relative to EMA
    df['price_to_ema'] = (df['close'] - df['ema']) / df['ema'] * 100  # Percentage distance
    
    return df


def generate_signal_from_dataframe(df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate trading signal based on the prepared data.
    
    Args:
        df: DataFrame with indicators added by prepare_data
        params: Optional parameters to override defaults
    
    Returns:
        Signal string: "BUY", "SELL", or "HOLD"
    """
    if params is None:
        params = initialize()["parameters"]
    
    if len(df) < 2:
        return "HOLD"  # Not enough data
    
    # Get the latest and previous candle data
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Check for volatility (is market volatile enough to trade?)
    is_volatile = current['atr_pct'] > params['volatility_threshold']
    
    # BUY conditions:
    # 1. Price is above EMA (uptrend)
    # 2. RSI was below threshold and is now rising (momentum)
    # 3. Sufficient volatility
    buy_condition = (
        current['close'] > current['ema'] and
        previous['rsi'] < params['rsi_threshold_low'] and
        current['rsi'] > previous['rsi'] and
        is_volatile
    )
    
    # SELL conditions:
    # 1. Price is below EMA (downtrend)
    # 2. RSI was above threshold and is now falling (momentum exhaustion)
    # 3. Sufficient volatility
    sell_condition = (
        current['close'] < current['ema'] and
        previous['rsi'] > params['rsi_threshold_high'] and
        current['rsi'] < previous['rsi'] and
        is_volatile
    )
    
    if buy_condition:
        return "BUY"
    elif sell_condition:
        return "SELL"
    else:
        return "HOLD"


def get_candles_dataframe(client, symbol, interval='1h', limit=100):
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
        import logging
        logging.getLogger("trading_bot").error(f"Error creating candles dataframe: {e}")
        return None


def generate_signal(symbol, interval, client, parameters=None):
    """
    The main entry point for custom strategy execution.
    This is the function that will be called by the bot's strategy module.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        interval: Timeframe interval (e.g., "1h", "4h")
        client: Binance API client instance
        parameters: Optional parameters to override defaults
    
    Returns:
        Signal string: "BUY", "SELL", or "HOLD"
    """
    # Get parameters (use provided or defaults)
    params = parameters if parameters else initialize()["parameters"]
    
    # Get candle data
    df = get_candles_dataframe(client, symbol, interval, limit=params.get("lookback_periods", 100))
    if df is None or len(df) < 30:
        import logging
        logging.getLogger("trading_bot").warning(f"Not enough data for custom strategy on {symbol}")
        return "HOLD"
    
    # Calculate indicators
    df = prepare_data(df, params)
    
    # Generate signal
    signal = generate_signal_from_dataframe(df, params)
    
    import logging
    logging.getLogger("trading_bot").info(f"Custom strategy signal for {symbol}: {signal}")
    
    return signal


# Advanced function for future use
def get_advanced_signal(market_data, timeframe, account_info=None, position_info=None):
    """
    Advanced signal generation with additional metadata.
    
    Args:
        market_data: Dictionary with market data
        timeframe: Timeframe to use
        account_info: Optional account information
        position_info: Optional current position information
    
    Returns:
        Tuple of (signal, metadata) where metadata contains additional
        information like confidence level, stop loss, take profit, etc.
    """
    # This is a placeholder for future development
    # The bot doesn't use this function yet, but it's here for reference
    return "HOLD", {
        "confidence": 0.0,
        "stop_loss": None,
        "take_profit": None,
        "recommended_size": "NONE",
        "reason": "Advanced signal generation not implemented yet"
    } 