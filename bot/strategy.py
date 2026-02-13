import numpy as np
import pandas as pd
import logging
import importlib
import sys
from bot.binance_api import get_recent_closes, client
from bot.config import get_strategy_config, get_strategy_parameter, is_strategy_enabled

logger = logging.getLogger("trading_bot")

def simple_signal(symbol, interval=None):
    """
    A simple strategy based on price movement.
    Returns 'BUY', 'SELL', or 'HOLD'.
    
    Args:
        symbol: Trading pair symbol
        interval: Optional override for the timeframe
    """
    # Get config for simple strategy
    strategy_config = get_strategy_config("simple")
    if not strategy_config.get("enabled", True):
        logger.info("Simple strategy is disabled")
        return "HOLD"
    
    # Use provided interval or get from config
    if interval is None:
        interval = strategy_config.get("timeframe", "1m")
    
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

def technical_analysis_signal(symbol, interval=None):
    """
    Generate trading signals based on multiple technical indicators.
    
    Returns 'BUY', 'SELL', or 'HOLD' based on:
    - RSI (oversold/overbought)
    - Bollinger Bands (price breakouts)
    - MACD (trend and momentum)
    
    Args:
        symbol: Trading pair symbol
        interval: Optional override for the timeframe
    """
    # Get config for technical strategy
    strategy_config = get_strategy_config("technical")
    if not strategy_config.get("enabled", True):
        logger.info("Technical analysis strategy is disabled")
        return "HOLD"
    
    # Use provided interval or get from config
    if interval is None:
        interval = strategy_config.get("timeframe", "1h")
    
    # Get technical parameters from config
    rsi_period = get_strategy_parameter("technical", "rsi_period", 14)
    rsi_oversold = get_strategy_parameter("technical", "rsi_oversold", 30)
    rsi_overbought = get_strategy_parameter("technical", "rsi_overbought", 70)
    
    bb_period = get_strategy_parameter("technical", "bb_period", 20)
    bb_std_dev = get_strategy_parameter("technical", "bb_std_dev", 2.0)
    
    macd_fast_period = get_strategy_parameter("technical", "macd_fast_period", 12)
    macd_slow_period = get_strategy_parameter("technical", "macd_slow_period", 26)
    macd_signal_period = get_strategy_parameter("technical", "macd_signal_period", 9)
    
    try:
        df = get_candles_dataframe(symbol, interval, limit=100)
        if df is None or len(df) < 30:
            logger.warning(f"Not enough data for technical analysis on {symbol}")
            return "HOLD"
        
        # Calculate indicators with configurable parameters
        df['rsi'] = calculate_rsi(df, period=rsi_period)
        bollinger = calculate_bollinger_bands(df, period=bb_period, std_dev=bb_std_dev)
        macd = calculate_macd(df, 
                             fast_period=macd_fast_period, 
                             slow_period=macd_slow_period, 
                             signal_period=macd_signal_period)
        
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
                    
                if rsi_value < rsi_oversold:  # Oversold (configurable)
                    buy_signals += 1
                    logger.info(f"RSI indicates oversold condition: {rsi_value:.2f}")
                elif rsi_value > rsi_overbought:  # Overbought (configurable)
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

def _compute_adx(df, period=14):
    """
    Compute ADX (Average Directional Index) and +DI/-DI from OHLC data.

    Uses Wilder's smoothing method. Returns a dict of Series aligned to df.index:
        {"adx": Series, "plus_di": Series, "minus_di": Series}
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    n = len(df)
    atr = np.full(n, np.nan)
    smooth_plus = np.full(n, np.nan)
    smooth_minus = np.full(n, np.nan)
    adx = np.full(n, np.nan)

    if n < period + 1:
        nan_series = pd.Series(adx, index=df.index)
        return {"adx": nan_series, "plus_di": nan_series.copy(), "minus_di": nan_series.copy()}

    # Seed with simple averages over the first `period` rows
    atr[period] = tr.iloc[1 : period + 1].mean()
    smooth_plus[period] = plus_dm[1 : period + 1].mean()
    smooth_minus[period] = minus_dm[1 : period + 1].mean()

    # Wilder smoothing
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr.iloc[i]) / period
        smooth_plus[i] = (smooth_plus[i - 1] * (period - 1) + plus_dm[i]) / period
        smooth_minus[i] = (smooth_minus[i - 1] * (period - 1) + minus_dm[i]) / period

    # +DI / -DI
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * smooth_plus / atr
        minus_di = 100.0 * smooth_minus / atr
        dx = 100.0 * np.abs(plus_di - minus_di) / np.where(
            (plus_di + minus_di) == 0, np.nan, plus_di + minus_di
        )

    # ADX = Wilder-smoothed DX
    first_valid = period + period  # need `period` DX values to seed ADX
    if n > first_valid:
        adx[first_valid] = np.nanmean(dx[period + 1 : first_valid + 1])
        for i in range(first_valid + 1, n):
            if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return {
        "adx": pd.Series(adx, index=df.index),
        "plus_di": pd.Series(plus_di, index=df.index),
        "minus_di": pd.Series(minus_di, index=df.index),
    }


def trend_following_signal(symbol, interval=None):
    """
    ADX regime-adaptive trend-following strategy with DI direction filter.

    TRENDING  (ADX >= trend_thresh): MACD direction -> BUY/SELL
    CHOPPY    (ADX <= chop_thresh):  RSI mean-reversion -> BUY/SELL
    TRANSITION:                      HOLD

    DI filter: +DI > -DI blocks SELL, -DI > +DI blocks BUY.
    """
    strategy_config = get_strategy_config("trend_following")
    if not strategy_config.get("enabled", False):
        logger.debug("Trend following strategy is disabled")
        return "HOLD"

    if interval is None:
        interval = strategy_config.get("timeframe", "4h")

    adx_period = get_strategy_parameter("trend_following", "adx_period", 10)
    trend_thresh = get_strategy_parameter("trend_following", "adx_trend_thresh", 25)
    chop_thresh = get_strategy_parameter("trend_following", "adx_chop_thresh", 15)
    rsi_oversold = get_strategy_parameter("trend_following", "rsi_oversold", 30)
    rsi_overbought = get_strategy_parameter("trend_following", "rsi_overbought", 70)
    use_di_filter = get_strategy_parameter("trend_following", "use_di_filter", True)

    try:
        df = get_candles_dataframe(symbol, interval, limit=100)
        if df is None or len(df) < adx_period * 2 + 10:
            logger.warning(f"Not enough data for trend following on {symbol}")
            return "HOLD"

        # Compute indicators
        df["rsi"] = calculate_rsi(df, period=14)
        calculate_macd(df)
        adx_data = _compute_adx(df, period=adx_period)

        current = df.iloc[-1]
        macd_val = current.get("macd_line")
        signal_val = current.get("signal_line")
        rsi_val = current.get("rsi")
        adx_val = adx_data["adx"].iloc[-1]
        plus_di = adx_data["plus_di"].iloc[-1]
        minus_di = adx_data["minus_di"].iloc[-1]

        if pd.isna(adx_val) or pd.isna(macd_val) or pd.isna(signal_val) or pd.isna(rsi_val):
            logger.info(f"Trend following: insufficient indicator data (ADX={adx_val})")
            return "HOLD"

        # DI direction filter
        if use_di_filter and not pd.isna(plus_di) and not pd.isna(minus_di):
            bullish_di = plus_di > minus_di
        else:
            bullish_di = None

        def _filter(raw_signal):
            if bullish_di is None:
                return raw_signal
            if raw_signal == "BUY" and not bullish_di:
                return "HOLD"
            if raw_signal == "SELL" and bullish_di:
                return "HOLD"
            return raw_signal

        # Determine regime and generate signal
        regime = "TRANSITION"
        signal = "HOLD"

        if adx_val >= trend_thresh:
            regime = "TRENDING"
            if macd_val > signal_val:
                signal = _filter("BUY")
            elif macd_val < signal_val:
                signal = _filter("SELL")
        elif adx_val <= chop_thresh:
            regime = "CHOPPY"
            if rsi_val < rsi_oversold:
                signal = _filter("BUY")
            elif rsi_val > rsi_overbought:
                signal = _filter("SELL")

        logger.info(
            f"Trend following: regime={regime} ADX={adx_val:.1f} "
            f"+DI={plus_di:.1f} -DI={minus_di:.1f} "
            f"MACD={macd_val:.2f} RSI={rsi_val:.1f} -> {signal}"
        )
        return signal

    except Exception as e:
        logger.error(f"Error in trend following strategy: {e}")
        return "HOLD"


def load_custom_strategy(module_path):
    """
    Load a custom strategy module dynamically.
    
    Args:
        module_path: Path to the custom strategy module
        
    Returns:
        Module object or None if loading fails
    """
    try:
        if not module_path:
            logger.warning("No custom strategy module path provided")
            return None
            
        # Try to import the module
        if module_path not in sys.modules:
            module = importlib.import_module(module_path)
        else:
            # If already imported, reload to get latest changes
            module = sys.modules[module_path]
            module = importlib.reload(module)
            
        return module
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading custom strategy module '{module_path}': {e}")
        return None

def custom_strategy_signal(symbol, interval=None):
    """
    Execute a custom trading strategy loaded from an external module.
    
    Args:
        symbol: Trading pair symbol
        interval: Optional override for the timeframe
        
    Returns:
        'BUY', 'SELL', or 'HOLD' based on the custom strategy
    """
    # Get config for custom strategy
    strategy_config = get_strategy_config("custom")
    if not strategy_config.get("enabled", False):
        logger.debug("Custom strategy is disabled")
        return "HOLD"
    
    # Use provided interval or get from config
    if interval is None:
        interval = strategy_config.get("timeframe", "4h")
    
    # Get the module path from config
    module_path = strategy_config.get("module_path", "")
    
    try:
        # Load the custom strategy module
        module = load_custom_strategy(module_path)
        if module is None:
            logger.warning("Could not load custom strategy module")
            return "HOLD"
        
        # Check if the module has the required function - try 'generate_signal' first, then 'get_signal'
        if hasattr(module, "generate_signal"):
            logger.debug(f"Using 'generate_signal' function from custom strategy module '{module_path}'")
            return module.generate_signal(symbol, interval, client, strategy_config.get("parameters", {}))
        elif hasattr(module, "get_signal"):
            logger.debug(f"Using 'get_signal' function from custom strategy module '{module_path}'")
            # Try to get market data for the requested interval
            try:
                candles = client.get_klines(symbol=symbol, interval=interval, limit=100)
                # Format market data as expected by get_signal
                market_data = {interval: candles}
                return module.get_signal(market_data, interval)
            except Exception as e:
                logger.error(f"Error fetching market data for custom strategy: {e}")
                return "HOLD"
        else:
            logger.error(f"Custom strategy module '{module_path}' does not have a 'generate_signal' or 'get_signal' function")
            return "HOLD"
    except Exception as e:
        logger.error(f"Error executing custom strategy: {e}")
        return "HOLD"

def get_all_strategy_signals(symbol):
    """
    Get signals from all enabled strategies.
    
    Args:
        symbol: Trading pair symbol
        
    Returns:
        Dictionary of strategy name -> signal value
    """
    signals = {}
    
    # Get all strategy configs
    strategies = {
        "simple": simple_signal,
        "technical": technical_analysis_signal,
        "custom": custom_strategy_signal,
        "trend_following": trend_following_signal,
    }
    
    # Execute each enabled strategy
    for strategy_name, strategy_func in strategies.items():
        if is_strategy_enabled(strategy_name):
            strategy_config = get_strategy_config(strategy_name)
            interval = strategy_config.get("timeframe")
            signals[strategy_name] = strategy_func(symbol, interval)
    
    return signals
