"""
Module for market data management with improved error handling and caching.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import os
import time
import hashlib

from bot.database import Database
from bot.backtesting.exceptions.base import DataError, MissingDataError, InvalidDataError
from bot.backtesting.config.settings import DATABASE_SETTINGS
from bot.backtesting.utils.indicators import (
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_ema,
    calculate_sma
)

logger = logging.getLogger("trading_bot.data")


class MarketDataCache:
    """Class for caching market data to avoid repeated database queries."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, enabled: bool = True, ttl: int = 3600):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files (default: ./cache)
            enabled: Whether caching is enabled
            ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        self.enabled = enabled
        self.ttl = ttl
        
        if cache_dir is None:
            cache_dir = Path("./cache")
        elif isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
            
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # In-memory cache for faster lookups
        self._memory_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        
    def get_cache_key(self, symbol: str, timeframe: str, start_time: str, end_time: str) -> str:
        """
        Generate a unique cache key for the data request.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1m', '1h')
            start_time: Start time
            end_time: End time
            
        Returns:
            str: Hash key for the cache
        """
        # Create a string representation of the request
        key_str = f"{symbol}_{timeframe}_{start_time}_{end_time}"
        # Generate a hash
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cache_file_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, symbol: str, timeframe: str, start_time: str, end_time: str) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache if available and not expired.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1m', '1h')
            start_time: Start time
            end_time: End time
            
        Returns:
            Optional[pd.DataFrame]: Cached data or None if not in cache or expired
        """
        if not self.enabled:
            return None
            
        cache_key = self.get_cache_key(symbol, timeframe, start_time, end_time)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            timestamp, data = self._memory_cache[cache_key]
            if time.time() - timestamp <= self.ttl:
                logger.debug(f"Cache hit (memory): {symbol} {timeframe}")
                return data.copy()
            else:
                # Expired from memory cache
                del self._memory_cache[cache_key]
        
        # Check file cache
        cache_file = self.get_cache_file_path(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                timestamp = cache_data.get('timestamp', 0)
                data = cache_data.get('data')
                
                # Check if cache is still valid
                if time.time() - timestamp <= self.ttl and isinstance(data, pd.DataFrame):
                    logger.debug(f"Cache hit (file): {symbol} {timeframe}")
                    # Update memory cache
                    self._memory_cache[cache_key] = (timestamp, data)
                    return data.copy()
                else:
                    # Cache expired, delete the file
                    os.remove(cache_file)
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                logger.warning(f"Error reading cache file: {e}")
                try:
                    os.remove(cache_file)
                except Exception:
                    pass
        
        return None
    
    def set(self, symbol: str, timeframe: str, start_time: str, end_time: str, data: pd.DataFrame) -> None:
        """
        Store data in the cache.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1m', '1h')
            start_time: Start time
            end_time: End time
            data: DataFrame to cache
        """
        if not self.enabled or data is None or data.empty:
            return
            
        cache_key = self.get_cache_key(symbol, timeframe, start_time, end_time)
        timestamp = time.time()
        
        # Update memory cache
        self._memory_cache[cache_key] = (timestamp, data.copy())
        
        # Update file cache
        cache_file = self.get_cache_file_path(cache_key)
        try:
            cache_data = {
                'timestamp': timestamp,
                'data': data.copy(),
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'start_time': start_time,
                    'end_time': end_time,
                    'rows': len(data)
                }
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.debug(f"Data cached: {symbol} {timeframe} ({len(data)} rows)")
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
    
    def clear(self, older_than: Optional[int] = None) -> int:
        """
        Clear cache files.
        
        Args:
            older_than: Clear files older than this many seconds (None for all)
            
        Returns:
            int: Number of files cleared
        """
        cleared_count = 0
        
        # Clear memory cache
        if older_than is None:
            cleared_count += len(self._memory_cache)
            self._memory_cache.clear()
        else:
            current_time = time.time()
            keys_to_remove = [
                key for key, (timestamp, _) in self._memory_cache.items()
                if current_time - timestamp > older_than
            ]
            for key in keys_to_remove:
                del self._memory_cache[key]
            cleared_count += len(keys_to_remove)
        
        # Clear file cache
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            for cache_file in cache_files:
                if older_than is None:
                    os.remove(cache_file)
                    cleared_count += 1
                else:
                    file_time = os.path.getmtime(cache_file)
                    if time.time() - file_time > older_than:
                        os.remove(cache_file)
                        cleared_count += 1
        except Exception as e:
            logger.warning(f"Error clearing cache files: {e}")
        
        return cleared_count


class MarketData:
    """Class for loading and managing market data with improved error handling."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the MarketData manager.
        
        Args:
            db_path: Optional custom path for the database
        """
        self.db = Database(db_path) if db_path else Database()
        
        # Initialize cache
        cache_enabled = DATABASE_SETTINGS.get('cache_enabled', True)
        cache_ttl = DATABASE_SETTINGS.get('cache_ttl', 3600)
        self.cache = MarketDataCache(enabled=cache_enabled, ttl=cache_ttl)
        
        # Track loaded data
        self.loaded_data: Dict[str, pd.DataFrame] = {}
    
    def get_market_data(self, symbol: str, timeframe: str, start_time: str, 
                       end_time: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Load historical market data with error handling and caching.
        
        Args:
            symbol: Trading symbol to load
            timeframe: Timeframe (e.g., '1m', '1h')
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            use_cache: Whether to use cache
            
        Returns:
            pd.DataFrame: Market data
            
        Raises:
            MissingDataError: If no data is available
            InvalidDataError: If data is invalid
        """
        cache_key = f"{symbol}_{timeframe}_{start_time}_{end_time}"
        
        # Check if already loaded in this session
        if cache_key in self.loaded_data:
            return self.loaded_data[cache_key].copy()
        
        # Check cache if enabled
        if use_cache:
            cached_data = self.cache.get(symbol, timeframe, start_time, end_time)
            if cached_data is not None:
                self.loaded_data[cache_key] = cached_data
                return cached_data.copy()
        
        try:
            # Load data from database
            data = self.db.get_market_data(symbol, timeframe, start_time, end_time)
            
            if data.empty:
                logger.warning(f"No market data available for {symbol} at {timeframe} timeframe")
                raise MissingDataError(
                    f"No market data available for {symbol} at {timeframe} timeframe",
                    {"symbol": symbol, "timeframe": timeframe, "start": start_time, "end": end_time}
                )
            
            # Ensure data is sorted by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Validate data
            self._validate_data(data, symbol, timeframe)
            
            # Cache data
            if use_cache:
                self.cache.set(symbol, timeframe, start_time, end_time, data)
            
            # Store in loaded data
            self.loaded_data[cache_key] = data
            
            logger.info(f"Loaded {len(data)} candles for {symbol} at {timeframe} timeframe")
            return data.copy()
            
        except MissingDataError:
            raise  # Re-raise missing data error
        except Exception as e:
            # Log the error with traceback
            logger.error(f"Error loading market data: {e}", exc_info=True)
            raise DataError(f"Error loading market data: {str(e)}", {
                "symbol": symbol, "timeframe": timeframe, "error": str(e)
            })
    
    def download_data(self, client, symbol: str, timeframe: str, 
                     start_date: str, end_date: str, limit: int = 1000,
                     batch_size: Optional[int] = None) -> bool:
        """
        Download historical data from exchange with batching for large requests.
        
        Args:
            client: Exchange client object (e.g., Binance client)
            symbol: Symbol to download
            timeframe: Timeframe to download (e.g., '1h')
            start_date: Start date in ISO format
            end_date: End date in ISO format
            limit: Maximum number of candles per request
            batch_size: Number of days per batch (for large date ranges)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading market data for {symbol} at {timeframe} timeframe")
            
            # Convert dates to datetime objects
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Default batch size from settings
            if batch_size is None:
                batch_size = DATABASE_SETTINGS.get('batch_size', 1000)
            
            # If date range is large, break into batches
            if (end_dt - start_dt).days > 30 and batch_size > 0:
                logger.info(f"Large date range detected, downloading in batches (batch size: {batch_size} days)")
                
                current_start = start_dt
                all_dataframes = []
                
                while current_start < end_dt:
                    # Calculate batch end date
                    current_end = min(current_start + timedelta(days=batch_size), end_dt)
                    
                    # Download batch
                    batch_start_ts = int(current_start.timestamp() * 1000)
                    batch_end_ts = int(current_end.timestamp() * 1000)
                    
                    logger.info(f"Downloading batch: {current_start.date()} to {current_end.date()}")
                    
                    klines = client.get_historical_klines(
                        symbol=symbol,
                        interval=timeframe,
                        start_str=str(batch_start_ts),
                        end_str=str(batch_end_ts),
                        limit=limit
                    )
                    
                    if klines:
                        batch_df = self._process_klines_data(klines)
                        all_dataframes.append(batch_df)
                        logger.info(f"Downloaded {len(batch_df)} candles for batch")
                    
                    # Move to next batch
                    current_start = current_end
                    
                    # Small delay to avoid API rate limits
                    time.sleep(1)
                
                # Combine all batches
                if all_dataframes:
                    combined_df = pd.concat(all_dataframes, ignore_index=True)
                    combined_df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
                    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                    
                    success = self.db.store_market_data(combined_df, symbol, timeframe)
                    
                    if success:
                        # Update local cache
                        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
                        self.loaded_data[cache_key] = combined_df
                        self.cache.set(symbol, timeframe, start_date, end_date, combined_df)
                        
                        logger.info(f"Successfully downloaded and stored {len(combined_df)} candles")
                        return True
                    else:
                        logger.error("Failed to store downloaded market data")
                        return False
                else:
                    logger.warning("No data returned for any batch")
                    return False
            else:
                # Download entire range at once
                start_ts = int(start_dt.timestamp() * 1000)
                end_ts = int(end_dt.timestamp() * 1000)
                
                klines = client.get_historical_klines(
                    symbol=symbol,
                    interval=timeframe,
                    start_str=str(start_ts),
                    end_str=str(end_ts),
                    limit=limit
                )
                
                if not klines:
                    logger.warning(f"No data returned for {symbol} at {timeframe} timeframe")
                    return False
                
                df = self._process_klines_data(klines)
                
                success = self.db.store_market_data(df, symbol, timeframe)
                
                if success:
                    # Update local cache
                    cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
                    self.loaded_data[cache_key] = df
                    self.cache.set(symbol, timeframe, start_date, end_date, df)
                    
                    logger.info(f"Successfully downloaded and stored {len(df)} candles")
                    return True
                else:
                    logger.error("Failed to store downloaded market data")
                    return False
                
        except Exception as e:
            logger.error(f"Error downloading market data: {e}", exc_info=True)
            return False
    
    def _process_klines_data(self, klines: List) -> pd.DataFrame:
        """Process raw klines data into a DataFrame."""
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
        
        return df
    
    def _validate_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Validate market data for common issues.
        
        Args:
            data: Market data DataFrame
            symbol: Trading symbol
            timeframe: Timeframe
            
        Raises:
            InvalidDataError: If data validation fails
        """
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise InvalidDataError(
                f"Missing required columns: {', '.join(missing_columns)}",
                {"symbol": symbol, "timeframe": timeframe, "missing_columns": missing_columns}
            )
        
        # Check for NaN values in critical columns
        nan_counts = data[required_columns].isna().sum()
        has_nans = nan_counts.any()
        
        if has_nans:
            # Log the NaN information but continue (don't raise error)
            logger.warning(f"NaN values detected in {symbol} {timeframe} data: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check for duplicate timestamps
        duplicates = data['timestamp'].duplicated()
        if duplicates.any():
            duplicate_count = duplicates.sum()
            logger.warning(f"Found {duplicate_count} duplicate timestamps in {symbol} {timeframe} data")
    
    def generate_test_data(self, symbol: str, timeframe: str, start_date: str, end_date: str, 
                          trend: str = 'random', volatility: float = 0.02,
                          seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic test data for backtesting.
        
        Args:
            symbol: Symbol to generate data for
            timeframe: Timeframe (e.g., '1h')
            start_date: Start date in ISO format
            end_date: End date in ISO format
            trend: Trend type ('up', 'down', 'sideways', 'random')
            volatility: Price volatility (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Synthetic market data
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Parse dates and create timestamp range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Map timeframe to pandas frequency
        freq_map = {
            '1m': 'T',  # minute
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': 'H',
            '4h': '4H',
            '1d': 'D',
            '1w': 'W'
        }
        
        if timeframe not in freq_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        freq = freq_map[timeframe]
        
        # Generate timestamps
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        
        # Generate price data
        n = len(timestamps)
        
        # Base price and trend factor
        base_price = 1000  # Starting price
        
        if trend == 'up':
            trend_factor = np.linspace(0, 1, n) * 0.5  # 50% overall increase
        elif trend == 'down':
            trend_factor = np.linspace(0, -1, n) * 0.3  # 30% overall decrease
        elif trend == 'sideways':
            trend_factor = np.zeros(n)
        else:  # random
            trend_factor = np.cumsum(np.random.normal(0, 0.01, n))
        
        # Calculate daily returns with trend and volatility
        daily_returns = np.random.normal(0.0005 + trend_factor, volatility, n)
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns)
        
        # Calculate price series
        prices = base_price * cumulative_returns
        
        # Generate OHLC data
        open_prices = prices
        close_prices = np.roll(prices, -1)  # Shift to get next price as close
        close_prices[-1] = open_prices[-1] * (1 + np.random.normal(0, volatility))
        
        # Add random noise for high and low
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, volatility/2, n)))
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, volatility/2, n)))
        
        # Generate volume data (correlated with price movement)
        price_changes = np.abs(np.diff(np.append(base_price, prices)))
        volumes = np.random.normal(1000, 300, n) * (1 + price_changes / base_price * 10)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        # Add symbol and timeframe columns
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        
        logger.info(f"Generated synthetic data for {symbol} ({len(df)} candles)")
        
        # Store in database for future use
        success = self.db.store_market_data(df, symbol, timeframe)
        if success:
            logger.info(f"Stored synthetic data in database")
        
        return df
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to market data.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        if df is None or df.empty:
            return df
            
        try:
            # Ensure we have a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Calculate RSI
            df_copy['rsi'] = calculate_rsi(df_copy)
            
            # Calculate Bollinger Bands
            bb_df = calculate_bollinger_bands(df_copy)
            df_copy['upper_band'] = bb_df['upper_band']
            df_copy['middle_band'] = bb_df['sma']
            df_copy['lower_band'] = bb_df['lower_band']
            
            # Calculate MACD
            macd_df = calculate_macd(df_copy)
            df_copy['macd_line'] = macd_df['macd_line']
            df_copy['signal_line'] = macd_df['signal_line']
            df_copy['macd_histogram'] = macd_df['macd_histogram']
            
            # Calculate moving averages
            df_copy['sma_20'] = calculate_sma(df_copy, 20)
            df_copy['sma_50'] = calculate_sma(df_copy, 50)
            df_copy['sma_200'] = calculate_sma(df_copy, 200)
            df_copy['ema_12'] = calculate_ema(df_copy, 12)
            df_copy['ema_26'] = calculate_ema(df_copy, 26)
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            # Return the original DataFrame if indicators fail
            return df
    
    def prepare_multi_timeframe_data(self, symbol: str, timeframes: List[str], 
                                    start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Prepare market data for multiple timeframes with indicators.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes (e.g., ['1h', '4h'])
            start_date: Start date in ISO format
            end_date: End date in ISO format
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of timeframe -> DataFrame with market data
        """
        result = {}
        
        for timeframe in timeframes:
            try:
                # Load market data for this timeframe
                data = self.get_market_data(symbol, timeframe, start_date, end_date)
                
                # Add indicators
                data_with_indicators = self.add_indicators(data)
                
                result[timeframe] = data_with_indicators
                logger.info(f"Prepared {symbol} data for {timeframe} timeframe ({len(data)} candles)")
                
            except MissingDataError:
                logger.warning(f"No data available for {symbol} at {timeframe} timeframe")
            except Exception as e:
                logger.error(f"Error preparing data for {timeframe}: {e}", exc_info=True)
        
        return result 