"""
Core backtesting engine for simulating trading strategies on historical data.
Optimized for performance and robustness.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from bot.backtesting.config.settings import DEFAULT_BACKTEST_SETTINGS, PERFORMANCE_SETTINGS
from bot.backtesting.data.market_data import MarketData
from bot.backtesting.models.trade import Trade, TradeBatch
from bot.backtesting.models.results import EquityPoint, BacktestResult, OptimizationResult
from bot.backtesting.exceptions.base import (
    BacktestError, DataError, MissingDataError, InvalidParameterError, 
    StrategyError, TradeExecutionError
)

logger = logging.getLogger("trading_bot.backtest")


class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data.
    Optimized for performance and error handling.
    """
    
    def __init__(self, symbol: str, timeframes: List[str], start_date: str, 
                 end_date: str, initial_capital: float = None, 
                 commission_rate: float = None, db_path: str = None,
                 position_size_pct: float = None):
        """
        Initialize the backtesting engine with parameters.
        
        Args:
            symbol: Trading symbol to backtest
            timeframes: List of timeframes to include (e.g., ['1m', '5m', '1h'])
            start_date: Start date for backtest (ISO format: 'YYYY-MM-DD')
            end_date: End date for backtest (ISO format: 'YYYY-MM-DD')
            initial_capital: Initial capital to start with
            commission_rate: Commission rate as a decimal (e.g., 0.001 for 0.1%)
            db_path: Optional custom path for the database
            position_size_pct: Percentage of capital to use per trade (0-1)
            
        Raises:
            InvalidParameterError: If parameters are invalid
        """
        # Validate inputs
        self._validate_inputs(symbol, timeframes, start_date, end_date)
        
        # Set parameters (using defaults from config if not provided)
        self.symbol = symbol
        self.timeframes = timeframes
        self.start_date = start_date
        self.end_date = end_date
        
        # Use config defaults if not provided
        self.initial_capital = initial_capital or DEFAULT_BACKTEST_SETTINGS['initial_capital']
        self.commission_rate = commission_rate or DEFAULT_BACKTEST_SETTINGS['commission_rate']
        self.position_size_pct = position_size_pct or DEFAULT_BACKTEST_SETTINGS['position_size_pct']
        
        # Initialize market data manager
        self.data_manager = MarketData(db_path)
        
        # Data structures for backtesting
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_capital = self.initial_capital
        self.position_size = 0.0
        self.trades: List[Trade] = []
        self.equity_curve: List[EquityPoint] = []
        
        # Run ID for this backtest
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load historical data for backtesting
        try:
            self._load_market_data()
        except DataError as e:
            logger.error(f"Error loading market data: {e.message}")
            # Re-raise with more context
            raise DataError(f"Failed to initialize backtest engine: {e.message}", e.details)
    
    def _validate_inputs(self, symbol: str, timeframes: List[str], 
                        start_date: str, end_date: str) -> None:
        """
        Validate input parameters.
        
        Raises:
            InvalidParameterError: If any parameter is invalid
        """
        # Check symbol
        if not symbol or not isinstance(symbol, str):
            raise InvalidParameterError("Symbol must be a non-empty string", {"symbol": symbol})
        
        # Check timeframes
        if not timeframes or not isinstance(timeframes, list) or not all(isinstance(tf, str) for tf in timeframes):
            raise InvalidParameterError("Timeframes must be a non-empty list of strings", {"timeframes": timeframes})
        
        # Validate timeframe format (simple check)
        valid_units = ['m', 'h', 'd', 'w']
        for tf in timeframes:
            if not any(tf.endswith(unit) for unit in valid_units):
                raise InvalidParameterError(f"Invalid timeframe format: {tf}", {"timeframe": tf})
                
        # Check dates
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            if start_dt >= end_dt:
                raise InvalidParameterError(
                    "Start date must be before end date", 
                    {"start_date": start_date, "end_date": end_date}
                )
                
            # Check if dates are too far in the future
            now = pd.to_datetime(datetime.now())
            if end_dt > now:
                logger.warning(f"End date {end_date} is in the future, using current date instead")
        except ValueError as e:
            raise InvalidParameterError(f"Invalid date format: {e}", 
                                       {"start_date": start_date, "end_date": end_date})
    
    def _load_market_data(self) -> None:
        """
        Load historical market data for all requested timeframes.
        
        Raises:
            DataError: If data loading fails
        """
        for timeframe in self.timeframes:
            try:
                data = self.data_manager.get_market_data(
                    symbol=self.symbol,
                    timeframe=timeframe,
                    start_time=self.start_date,
                    end_time=self.end_date
                )
                
                # Add data to market_data dict
                self.market_data[timeframe] = data
                logger.info(f"Loaded {len(data)} candles for {self.symbol} at {timeframe} timeframe")
                
            except MissingDataError as e:
                logger.warning(f"No historical data for {self.symbol} at {timeframe} timeframe: {e.message}")
            except Exception as e:
                logger.error(f"Error loading market data for {timeframe}: {str(e)}")
                # Continue with other timeframes instead of failing completely
        
        # Ensure we have at least the primary timeframe
        if not self.market_data or self.timeframes[0] not in self.market_data:
            raise DataError(
                f"Failed to load market data for primary timeframe {self.timeframes[0]}",
                {"symbol": self.symbol, "timeframe": self.timeframes[0]}
            )
    
    def prepare_data(self) -> None:
        """
        Prepare market data by adding necessary indicators.
        
        Raises:
            DataError: If data preparation fails
        """
        try:
            for timeframe in self.timeframes:
                if timeframe in self.market_data:
                    self.market_data[timeframe] = self.data_manager.add_indicators(self.market_data[timeframe])
                    logger.info(f"Added indicators to {timeframe} data")
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}", exc_info=True)
            raise DataError(f"Failed to prepare data with indicators: {str(e)}")
    
    def run_backtest(self, strategy_func: Callable) -> BacktestResult:
        """
        Run a backtest using the specified strategy function.
        
        Args:
            strategy_func: A function that takes market data and returns a signal ('BUY', 'SELL', 'HOLD')
                The function signature should be: 
                    strategy_func(data_dict: Dict[str, pd.DataFrame], symbol: str) -> str
        
        Returns:
            BacktestResult: Object containing backtest results
            
        Raises:
            BacktestError: If the backtest fails
        """
        if not self.market_data:
            logger.error("No market data available for backtesting")
            raise DataError("No market data available for backtesting")
        
        try:
            # Start timing the backtest
            start_time = time.time()
            
            # Prepare data with indicators
            self.prepare_data()
            
            # Get the primary timeframe data (first in the list)
            primary_tf = self.timeframes[0]
            if primary_tf not in self.market_data:
                raise DataError(f"Primary timeframe {primary_tf} data not available")
            
            primary_data = self.market_data[primary_tf]
            
            # Initialize results
            self.current_capital = self.initial_capital
            self.position_size = 0.0
            self.trades = []
            self.equity_curve = [EquityPoint(
                timestamp=primary_data['timestamp'].iloc[0],
                equity=self.current_capital,
                position_size=0.0
            )]
            
            # Determine minimum candles required for indicators
            min_candles_required = DEFAULT_BACKTEST_SETTINGS['min_candles_required']
            
            # Use vectorized operations if enabled
            use_vectorized = PERFORMANCE_SETTINGS.get('use_vectorized_operations', True)
            
            if use_vectorized and self._can_use_vectorized_backtest(strategy_func):
                # Vectorized backtesting for better performance
                try:
                    self._run_vectorized_backtest(strategy_func)
                except StrategyError as e:
                    # Catch and convert StrategyError to BacktestError with appropriate message
                    logger.error(f"Strategy error in vectorized backtest: {str(e)}")
                    raise BacktestError(f"Failed to run vectorized backtest: {str(e)}")
            else:
                # Traditional candle-by-candle backtesting
                self._run_traditional_backtest(strategy_func, min_candles_required)
            
            # Close any open position at the end of the backtest
            if self.position_size > 0:
                self._execute_trade('SELL', primary_data['timestamp'].iloc[-1], 
                                   primary_data['close'].iloc[-1], self.position_size)
            
            # Calculate end time and duration
            duration = time.time() - start_time
            
            # Create and return results object
            result = self._create_backtest_result(strategy_func)
            
            logger.info(f"Backtest completed in {duration:.2f} seconds with {len(self.trades)} trades")
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}", exc_info=True)
            raise BacktestError(f"Failed to run backtest: {str(e)}")
    
    def _can_use_vectorized_backtest(self, strategy_func: Callable) -> bool:
        """
        Check if vectorized backtesting can be used with this strategy.
        
        Args:
            strategy_func: Strategy function
            
        Returns:
            bool: True if vectorized backtesting can be used
        """
        # Check if the strategy function has a vectorized attribute
        if hasattr(strategy_func, 'vectorized') and strategy_func.vectorized:
            logger.info(f"Using vectorized backtesting for strategy: {getattr(strategy_func, '__name__', 'Unknown')}")
            return True
        
        # Check if the function name contains 'vectorized'
        if 'vectorized' in getattr(strategy_func, '__name__', '').lower():
            logger.info(f"Using vectorized backtesting based on function name: {strategy_func.__name__}")
            return True
        
        # Default to traditional backtesting
        return False
    
    def _run_vectorized_backtest(self, strategy_func: Callable) -> None:
        """
        Run backtest using vectorized operations for better performance.
        
        Args:
            strategy_func: Strategy function that supports vectorized operations
        """
        # Start timing for performance measurement
        start_time = time.time()
        
        # Get the primary timeframe data
        primary_tf = self.timeframes[0]
        primary_data = self.market_data[primary_tf].copy()
        
        # Call the vectorized strategy function which should return signals for all candles
        try:
            # Prepare data dictionary
            data_dict = {tf: self.market_data[tf].copy() for tf in self.timeframes if tf in self.market_data}
            
            # Get signals from the strategy
            signals = strategy_func(data_dict, self.symbol, vectorized=True)
            
            # Check the type of signals
            if not isinstance(signals, (pd.Series, np.ndarray, list)):
                logger.error(f"Vectorized strategy must return a Series, array or list of signals, got {type(signals)}")
                raise StrategyError(f"Invalid return type from vectorized strategy: {type(signals)}")
            
            # Convert to a pandas Series with the same index as the primary data
            if isinstance(signals, (np.ndarray, list)):
                # Check length
                if len(signals) != len(primary_data):
                    logger.error(f"Signal length mismatch: {len(signals)} vs {len(primary_data)}")
                    raise StrategyError(f"Signal length mismatch: {len(signals)} vs {len(primary_data)}")
                
                signals = pd.Series(signals, index=primary_data.index)
            elif isinstance(signals, pd.Series) and len(signals) != len(primary_data):
                logger.error(f"Signal length mismatch: {len(signals)} vs {len(primary_data)}")
                raise StrategyError(f"Signal length mismatch: {len(signals)} vs {len(primary_data)}")
            
            # Process all signals in one go using vectorized operations
            self._process_vectorized_signals(primary_data, signals)
            
            logger.info(f"Vectorized backtest completed in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in vectorized backtest: {str(e)}", exc_info=True)
            raise BacktestError(f"Vectorized backtest failed: {str(e)}")
    
    def _process_vectorized_signals(self, data: pd.DataFrame, signals: pd.Series) -> None:
        """
        Process all signals at once using vectorized operations.
        
        Args:
            data: DataFrame with price data
            signals: Series with signals ('BUY', 'SELL', 'HOLD')
        """
        # Skip warm-up period
        min_candles_required = DEFAULT_BACKTEST_SETTINGS['min_candles_required']
        signals.iloc[:min_candles_required] = 'HOLD'
        
        # Initialize position tracking
        position_open = np.zeros(len(data), dtype=bool)
        entry_price = np.zeros(len(data))
        entry_time = np.empty(len(data), dtype='datetime64[ns]')
        position_size = np.zeros(len(data))
        
        # Track equity and capital
        capital = np.ones(len(data)) * self.initial_capital
        equity = np.ones(len(data)) * self.initial_capital
        current_position = 0.0
        current_capital = self.initial_capital
        
        # Initialize trades list
        trades = []
        
        # Process each signal
        for i in range(1, len(data)):
            timestamp = data['timestamp'].iloc[i]
            price = data['close'].iloc[i]
            
            # Ensure price is valid
            if price <= 0:
                price = 0.01  # Set minimum valid price
                logger.warning(f"Invalid price {data['close'].iloc[i]} detected at {timestamp}, using minimum price of 0.01")
            
            signal = signals.iloc[i]
            
            # Previous state
            was_in_position = current_position > 0
            
            # Process signal
            if signal == 'BUY' and not was_in_position:
                # Calculate position size
                available_capital = current_capital * self.position_size_pct
                
                # Validate available capital
                if available_capital <= 0:
                    logger.warning(f"Available capital {available_capital} is not positive. Skipping BUY.")
                    continue
                
                current_position = available_capital / price
                
                # Validate position size
                if current_position <= 0:
                    logger.warning(f"Calculated position {current_position} is not positive. Skipping BUY.")
                    continue
                
                # Update capital
                trade_value = price * current_position
                commission = trade_value * self.commission_rate
                current_capital -= (trade_value + commission)
                
                # Validate capital update
                if current_capital < 0:
                    logger.warning(f"Capital {current_capital} went negative after trade. Adjusting to minimum.")
                    current_capital = 0.01
                
                # Record entry
                position_open[i] = True
                entry_price[i] = price
                entry_time[i] = timestamp
                position_size[i] = current_position
                
                # Create trade object
                trade = self._create_vectorized_trade('BUY', timestamp, price, current_position, commission)
                trade.entry_point = True
                trade.entry_time = timestamp
                trade.market_indicators = self._get_market_indicators(timestamp)
                trades.append(trade)
                
            elif signal == 'SELL' and was_in_position:
                # Calculate profit/loss
                trade_value = price * current_position
                commission = trade_value * self.commission_rate
                profit_loss = trade_value - commission
                
                # Find the most recent entry point
                entry_idx = np.where(position_open[:i])[0][-1] if any(position_open[:i]) else None
                
                # Update capital
                current_capital += profit_loss
                
                # Validate capital update
                if current_capital <= 0:
                    logger.warning(f"Capital {current_capital} went negative or zero after SELL. Adjusting to minimum.")
                    current_capital = 0.01
                
                # Reset position
                position_open[i] = False
                
                # Create exit trade
                trade = self._create_vectorized_trade('SELL', timestamp, price, current_position, commission)
                
                # Add P/L info if we found an entry
                if entry_idx is not None:
                    entry_px = entry_price[entry_idx]
                    
                    # Ensure entry_px is valid
                    if entry_px <= 0:
                        entry_px = 0.01
                        logger.warning(f"Invalid entry price {entry_price[entry_idx]} detected, using minimum price of 0.01")
                    
                    raw_pl = (price - entry_px) * current_position - commission
                    trade.entry_price = entry_px
                    trade.exit_price = price
                    trade.profit_loss = raw_pl
                    
                    # Calculate ROI safely
                    entry_value = entry_px * current_position
                    if entry_value > 0:
                        trade.roi_pct = (raw_pl / entry_value) * 100
                        
                        # Clamp ROI to reasonable limits
                        if abs(trade.roi_pct) > 1000:
                            logger.warning(f"Extreme ROI value detected: {trade.roi_pct}%. Clamping to ±1000%.")
                            trade.roi_pct = min(max(trade.roi_pct, -1000), 1000)
                    else:
                        trade.roi_pct = 0
                        logger.warning(f"Entry value is zero or negative at index {entry_idx}, setting ROI to 0")
                    
                    trade.entry_time = entry_time[entry_idx]
                    trade.exit_time = timestamp
                    
                    # Calculate holding period
                    if isinstance(timestamp, pd.Timestamp) and isinstance(entry_time[entry_idx], pd.Timestamp):
                        diff = timestamp - entry_time[entry_idx]
                        trade.holding_period_hours = diff.total_seconds() / 3600
                
                # Reset position
                current_position = 0
                
                # Add market indicators
                trade.market_indicators = self._get_market_indicators(timestamp)
                trades.append(trade)
            
            # Update equity curve
            capital[i] = current_capital
            equity[i] = current_capital + (current_position * price)
            position_size[i] = current_position
        
        # Close any open position at the end
        if current_position > 0:
            last_timestamp = data['timestamp'].iloc[-1]
            last_price = data['close'].iloc[-1]
            
            # Ensure price is valid
            if last_price <= 0:
                last_price = 0.01
                logger.warning(f"Invalid closing price detected, using minimum price of 0.01")
            
            # Create SELL trade
            trade = self._create_vectorized_trade('SELL', last_timestamp, last_price, current_position, 
                                                 last_price * current_position * self.commission_rate)
            
            # Find the entry point
            entry_idx = np.where(position_open)[0][-1] if any(position_open) else None
            
            if entry_idx is not None:
                entry_px = entry_price[entry_idx]
                
                # Ensure entry_px is valid
                if entry_px <= 0:
                    entry_px = 0.01
                    logger.warning(f"Invalid final entry price {entry_px} detected, using minimum price of 0.01")
                
                commission = last_price * current_position * self.commission_rate
                raw_pl = (last_price - entry_px) * current_position - commission
                
                trade.entry_price = entry_px
                trade.exit_price = last_price
                trade.profit_loss = raw_pl
                
                # Calculate ROI safely
                entry_value = entry_px * current_position
                if entry_value > 0:
                    trade.roi_pct = (raw_pl / entry_value) * 100
                    
                    # Clamp ROI to reasonable limits
                    if abs(trade.roi_pct) > 1000:
                        logger.warning(f"Extreme final ROI value detected: {trade.roi_pct}%. Clamping to ±1000%.")
                        trade.roi_pct = min(max(trade.roi_pct, -1000), 1000)
                else:
                    trade.roi_pct = 0
                    logger.warning("Final entry value is zero or negative, setting ROI to 0")
                
                trade.entry_time = entry_time[entry_idx]
                trade.exit_time = last_timestamp
                
                # Calculate holding period
                if isinstance(last_timestamp, pd.Timestamp) and isinstance(entry_time[entry_idx], pd.Timestamp):
                    diff = last_timestamp - entry_time[entry_idx]
                    trade.holding_period_hours = diff.total_seconds() / 3600
            
            trade.market_indicators = self._get_market_indicators(last_timestamp)
            trades.append(trade)
            
            # Reset position and update capital
            trade_value = last_price * current_position
            commission = trade_value * self.commission_rate
            current_capital += (trade_value - commission)
            current_position = 0
        
        # Store trades and equity curve
        self.trades = trades
        
        # Convert equity curve to list of EquityPoint objects
        self.equity_curve = [
            EquityPoint(
                timestamp=data['timestamp'].iloc[i],
                equity=equity[i],
                position_size=position_size[i]
            )
            for i in range(len(data))
        ]
        
        # Update final capital
        self.current_capital = current_capital
        self.position_size = current_position
    
    def _create_vectorized_trade(self, side: str, timestamp: datetime, price: float, 
                                quantity: float, commission: float) -> Trade:
        """
        Create a Trade object for vectorized backtesting.
        
        Args:
            side: Trade side ('BUY' or 'SELL')
            timestamp: Trade timestamp
            price: Trade price
            quantity: Trade quantity
            commission: Trade commission
            
        Returns:
            Trade: Trade object
        """
        # Ensure commission is calculated correctly (matches self.commission_rate)
        # This is particularly important for zero commission case
        trade_value = price * quantity
        commission = trade_value * self.commission_rate
        
        return Trade(
            symbol=self.symbol,
            side=side,
            timestamp=timestamp,
            price=price,
            quantity=quantity,
            commission=commission,
            status="FILLED",
            raw_data={
                'timeframe': self.timeframes[0],
                'strategy': getattr(self, 'strategy_name', 'Custom_Strategy'),
                'vectorized': True
            }
        )
    
    def _run_traditional_backtest(self, strategy_func: Callable, min_candles_required: int) -> None:
        """
        Run traditional candle-by-candle backtest.
        
        Args:
            strategy_func: Strategy function
            min_candles_required: Minimum candles required for valid signals
        """
        primary_tf = self.timeframes[0]
        primary_data = self.market_data[primary_tf]
        
        # Loop through each candle (skipping the first few to ensure indicators are valid)
        for i in range(min_candles_required, len(primary_data)):
            current_time = primary_data['timestamp'].iloc[i]
            current_price = primary_data['close'].iloc[i]
            
            # Prepare data for strategy (up to current candle only)
            data_for_strategy = {}
            for tf in self.timeframes:
                if tf in self.market_data:
                    tf_data = self.market_data[tf]
                    # Filter data up to current timestamp
                    mask = tf_data['timestamp'] <= current_time
                    data_for_strategy[tf] = tf_data[mask].copy()
            
            try:
                # Get trading signal from strategy
                signal = strategy_func(data_for_strategy, self.symbol)
                
                # Ensure signal is a string and one of the valid values
                if not isinstance(signal, str) or signal not in ['BUY', 'SELL', 'HOLD']:
                    logger.warning(f"Invalid signal '{signal}' received from strategy, using 'HOLD' instead")
                    signal = 'HOLD'
                
                # Process signal
                self._process_signal(signal, current_time, current_price)
                
                # Update equity curve
                position_value = self.position_size * current_price
                equity = self.current_capital + position_value
                self.equity_curve.append(EquityPoint(
                    timestamp=current_time,
                    equity=equity,
                    position_size=self.position_size
                ))
                
            except Exception as e:
                logger.error(f"Error processing candle at {current_time}: {str(e)}", exc_info=True)
                # Continue with next candle
                
                # Add equity point with unchanged values to maintain continuity
                if self.equity_curve:
                    last_equity = self.equity_curve[-1].equity
                    self.equity_curve.append(EquityPoint(
                        timestamp=current_time,
                        equity=last_equity,
                        position_size=self.position_size
                    ))
    
    def _process_signal(self, signal: str, timestamp: datetime, price: float) -> None:
        """
        Process a trading signal.
        
        Args:
            signal: Trading signal ('BUY', 'SELL', 'HOLD')
            timestamp: Current timestamp
            price: Current price
            
        Raises:
            TradeExecutionError: If trade execution fails
        """
        try:
            if signal == 'BUY' and self.position_size == 0:
                # Calculate position size based on position_size_pct setting
                available_capital = self.current_capital * self.position_size_pct
                position_size = available_capital / price
                
                # Execute the trade
                self._execute_trade('BUY', timestamp, price, position_size)
                
            elif signal == 'SELL' and self.position_size > 0:
                # Sell entire position
                self._execute_trade('SELL', timestamp, price, self.position_size)
                
        except Exception as e:
            logger.error(f"Error processing signal {signal}: {str(e)}", exc_info=True)
            raise TradeExecutionError(f"Failed to process signal: {str(e)}")
    
    def _execute_trade(self, side: str, timestamp: datetime, price: float, quantity: float) -> Trade:
        """
        Execute a simulated trade.
        
        Args:
            side: Trade side ('BUY' or 'SELL')
            timestamp: Trade timestamp
            price: Trade price
            quantity: Trade quantity
            
        Returns:
            Trade: Executed trade object
            
        Raises:
            TradeExecutionError: If trade execution fails
        """
        try:
            # Validate price to prevent division by zero and unrealistic trades
            if price <= 0:
                logger.warning(f"Invalid price {price} detected, using minimum price of 0.01")
                price = 0.01  # Set minimum valid price
            
            # Validate quantity
            if quantity <= 0:
                logger.warning(f"Invalid quantity {quantity} detected. Adjusting to minimum.")
                quantity = 0.01
            
            trade_value = price * quantity
            commission_amount = trade_value * self.commission_rate
            
            trade = Trade(
                symbol=self.symbol,
                side=side,
                timestamp=timestamp,
                price=price,
                quantity=quantity,
                commission=commission_amount,
                status="FILLED",
                raw_data={
                    'timeframe': self.timeframes[0],
                    'strategy': getattr(self, 'strategy_name', 'Custom_Strategy')
                }
            )
            
            if side == 'BUY':
                # Update position and capital
                self.position_size = quantity
                self.current_capital -= (trade_value + commission_amount)
                
                # Validate capital is not negative
                if self.current_capital < 0:
                    logger.warning(f"Capital went negative after BUY: {self.current_capital}. Adjusting to minimum.")
                    self.current_capital = 0.01
                
                # Set entry point flag
                trade.entry_point = True
                trade.entry_time = timestamp
                
                # Add market indicators at entry point (if available)
                trade.market_indicators = self._get_market_indicators(timestamp)
                
                # Record the trade
                self.trades.append(trade)
                
                logger.debug(f"BUY: {quantity} {self.symbol} at {price} (Value: {trade_value:.2f}, Commission: {commission_amount:.2f})")
                
            elif side == 'SELL':
                # Find matching buy trade
                entry_trade = None
                for t in reversed(self.trades):
                    if t.side == 'BUY' and t.entry_point:
                        entry_trade = t
                        break
                
                if entry_trade:
                    # Calculate profit/loss
                    entry_price = entry_trade.price
                    profit_loss = (price - entry_price) * quantity - commission_amount
                    
                    # Calculate ROI safely (prevent division by zero)
                    entry_value = entry_price * quantity
                    if entry_value > 0:
                        roi_pct = (profit_loss / entry_value) * 100
                        
                        # Clamp ROI to reasonable limits
                        if abs(roi_pct) > 1000:
                            logger.warning(f"Extreme ROI value detected: {roi_pct}%. Clamping to ±1000%.")
                            roi_pct = min(max(roi_pct, -1000), 1000)
                    else:
                        logger.warning("Entry value is zero or negative, setting ROI to 0")
                        roi_pct = 0
                    
                    # Calculate holding period
                    if entry_trade.timestamp:
                        trade.entry_time = entry_trade.timestamp
                        trade.exit_time = timestamp
                        if isinstance(timestamp, pd.Timestamp) and isinstance(entry_trade.timestamp, pd.Timestamp):
                            diff = timestamp - entry_trade.timestamp
                            trade.holding_period_hours = diff.total_seconds() / 3600
                    
                    # Set profit/loss info
                    trade.entry_price = entry_price
                    trade.exit_price = price
                    trade.profit_loss = profit_loss
                    trade.roi_pct = roi_pct
                    
                    # Mark the entry trade as closed
                    entry_trade.entry_point = False
                    
                else:
                    # No matching entry found (shouldn't happen in normal operation)
                    logger.warning("No matching BUY entry found for SELL trade")
                    trade.profit_loss = -commission_amount
                    trade.roi_pct = 0
                
                # Add market indicators at exit point
                trade.market_indicators = self._get_market_indicators(timestamp)
                
                # Update position and capital
                self.current_capital += (trade_value - commission_amount)
                # Validate capital is not negative
                if self.current_capital < 0:
                    logger.warning(f"Capital went negative after SELL: {self.current_capital}. Adjusting to minimum.")
                    self.current_capital = 0.01
                
                self.position_size = 0
                
                # Record the trade
                self.trades.append(trade)
                
                logger.debug(f"SELL: {quantity} {self.symbol} at {price} (P/L: {trade.profit_loss:.2f}, ROI: {trade.roi_pct:.2f}%)")
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing {side} trade: {str(e)}", exc_info=True)
            raise TradeExecutionError(f"Failed to execute {side} trade: {str(e)}")
    
    def _get_market_indicators(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Get market indicators at a specific timestamp.
        
        Args:
            timestamp: Timestamp to get indicators for
            
        Returns:
            Dict: Dictionary of market indicators
        """
        indicators = {}
        
        try:
            # Get data from primary timeframe
            primary_tf = self.timeframes[0]
            if primary_tf in self.market_data:
                df = self.market_data[primary_tf]
                
                # Find closest candle
                candle = df[df['timestamp'] <= timestamp].iloc[-1]
                
                # Extract common indicators
                for indicator in ['rsi', 'macd_line', 'macd_histogram', 'signal_line', 
                                'upper_band', 'middle_band', 'lower_band']:
                    if indicator in candle:
                        indicators[indicator] = candle[indicator]
                
                # Add derived indicators
                if all(x in candle for x in ['upper_band', 'lower_band', 'close']):
                    try:
                        # Calculate BB position (0-1 range)
                        upper = candle['upper_band']
                        lower = candle['lower_band']
                        
                        # Prevent division by zero
                        if upper != lower:
                            bb_position = (candle['close'] - lower) / (upper - lower)
                            indicators['bb_position'] = bb_position
                    except (ZeroDivisionError, TypeError):
                        pass
                
                # Add trend indicator
                if 'middle_band' in candle:
                    indicators['trend'] = 'uptrend' if candle['close'] > candle['middle_band'] else 'downtrend'
                
                # Add volatility
                if all(x in candle for x in ['upper_band', 'lower_band', 'middle_band']):
                    try:
                        # Prevent division by zero
                        if candle['middle_band'] != 0:
                            indicators['volatility'] = (candle['upper_band'] - candle['lower_band']) / candle['middle_band']
                    except (ZeroDivisionError, TypeError):
                        pass
        
        except Exception as e:
            logger.warning(f"Error extracting market indicators: {e}")
        
        return indicators
    
    def _create_backtest_result(self, strategy_func: Callable) -> BacktestResult:
        """
        Create a BacktestResult object from the backtest data.
        
        Args:
            strategy_func: Strategy function used in the backtest
            
        Returns:
            BacktestResult: Backtest results
        """
        # Set strategy name for reporting purposes
        strategy_name = getattr(strategy_func, '__name__', 'Custom_Strategy')
        
        # Collect completed trades (SELL trades with profit_loss)
        completed_trades = [t for t in self.trades if t.side == 'SELL' and t.profit_loss is not None]
        
        # Calculate win/loss statistics
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t.profit_loss > 0])
        losing_trades = total_trades - winning_trades
        
        # Calculate final equity safely
        final_equity = self.initial_capital
        if self.equity_curve:
            final_equity = self.equity_curve[-1].equity
            
            # Safety check for unreasonable final equity
            max_reasonable_equity = self.initial_capital * 1000  # 100,000% return is unlikely
            if final_equity > max_reasonable_equity:
                logger.warning(f"Final equity {final_equity} exceeds reasonable limits, capping at 1000x initial capital")
                final_equity = max_reasonable_equity
            elif final_equity <= 0:
                logger.warning(f"Final equity {final_equity} is non-positive, setting to 1% of initial capital")
                final_equity = self.initial_capital * 0.01
        
        # Create result object
        result = BacktestResult(
            symbol=self.symbol,
            strategy_name=strategy_name,
            timeframes=self.timeframes,
            start_date=self.start_date,
            end_date=self.end_date,
            run_id=self.run_id,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            equity_curve=self.equity_curve,
            trades=[vars(t) for t in self.trades]  # Convert Trade objects to dicts
        )
        
        # Calculate additional metrics
        result.calculate_metrics()
        
        # Validate metrics and clamp unrealistic values
        if hasattr(result.metrics, 'total_return_pct'):
            if abs(result.metrics.total_return_pct) > 100000:  # 100,000% is extremely unlikely
                logger.warning(f"Return percentage {result.metrics.total_return_pct}% is extreme, clamping to ±100,000%")
                result.metrics.total_return_pct = min(max(result.metrics.total_return_pct, -100000), 100000)
        
        return result
    
    def optimize_parameters(self, strategy_factory: Callable, param_grid: Dict[str, List[Any]],
                          metric: str = 'sharpe_ratio', ascending: bool = False) -> OptimizationResult:
        """
        Optimize strategy parameters through grid search.
        Uses parallel processing if enabled in settings.
        
        Args:
            strategy_factory: A function that takes parameters and returns a strategy function
            param_grid: Dictionary of parameter names to lists of values to try
            metric: Metric to optimize (e.g., 'sharpe_ratio', 'total_return_pct')
            ascending: Whether lower values are better (True) or higher values (False)
        
        Returns:
            OptimizationResult: Results of the optimization
        """
        # Generate all parameter combinations
        import itertools
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(param_combinations)
        logger.info(f"Running parameter optimization with {total_combinations} combinations")
        
        # Start timing
        start_time = time.time()
        
        # Check if parallel processing is enabled
        use_parallel = PERFORMANCE_SETTINGS.get('use_parallel_processing', True)
        num_processes = min(
            PERFORMANCE_SETTINGS.get('num_processes', multiprocessing.cpu_count()),
            total_combinations  # Don't use more processes than combinations
        )
        
        best_backtest = None
        best_params = None
        all_results = []
        
        if use_parallel and num_processes > 1:
            # Run optimizations in parallel
            logger.info(f"Using parallel processing with {num_processes} processes")
            
            # Define the worker function for each parameter combination
            def _optimize_worker(combo_idx: int):
                combo = param_combinations[combo_idx]
                params = dict(zip(param_keys, combo))
                
                # Generate strategy function with these parameters
                strategy_func = strategy_factory(params)
                
                # Create a fresh backtest engine to avoid state issues
                engine = BacktestEngine(
                    symbol=self.symbol,
                    timeframes=self.timeframes,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    initial_capital=self.initial_capital,
                    commission_rate=self.commission_rate
                )
                
                # Run backtest
                result = engine.run_backtest(strategy_func)
                
                # Return parameters and result
                return (params, result)
            
            # Run in parallel
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [executor.submit(_optimize_worker, i) for i in range(total_combinations)]
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        params, result = future.result()
                        metric_value = getattr(result.metrics, metric, 0)
                        
                        all_results.append({
                            'parameters': params,
                            'result': result,
                            metric: metric_value
                        })
                        
                        # Check if this is the best result so far
                        if best_backtest is None or (
                            (not ascending and metric_value > getattr(best_backtest.metrics, metric, 0)) or
                            (ascending and metric_value < getattr(best_backtest.metrics, metric, 0))
                        ):
                            best_backtest = result
                            best_params = params
                        
                        logger.info(f"Combination {i+1}/{total_combinations}: {params} - {metric}: {metric_value:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error in optimization worker: {e}", exc_info=True)
                        
        else:
            # Run optimizations sequentially
            logger.info("Running optimizations sequentially")
            
            for i, combo in enumerate(param_combinations):
                params = dict(zip(param_keys, combo))
                
                try:
                    # Generate strategy function with these parameters
                    strategy_func = strategy_factory(params)
                    
                    # Run backtest
                    result = self.run_backtest(strategy_func)
                    
                    metric_value = getattr(result.metrics, metric, 0)
                    
                    all_results.append({
                        'parameters': params,
                        'result': result,
                        metric: metric_value
                    })
                    
                    # Check if this is the best result so far
                    if best_backtest is None or (
                        (not ascending and metric_value > getattr(best_backtest.metrics, metric, 0)) or
                        (ascending and metric_value < getattr(best_backtest.metrics, metric, 0))
                    ):
                        best_backtest = result
                        best_params = params
                    
                    logger.info(f"Combination {i+1}/{total_combinations}: {params} - {metric}: {metric_value:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error in optimization: {e}", exc_info=True)
        
        # Calculate optimization time
        optimization_time = time.time() - start_time
        
        # Create optimization result
        if best_backtest and best_params:
            optimization_result = OptimizationResult(
                strategy_name=best_backtest.strategy_name,
                symbol=self.symbol,
                timeframes=self.timeframes,
                parameter_grid=param_grid,
                best_parameters=best_params,
                best_backtest=best_backtest,
                all_results=[{
                    'parameters': r['parameters'],
                    'sharpe_ratio': getattr(r['result'].metrics, 'sharpe_ratio', 0),
                    'total_return_pct': getattr(r['result'].metrics, 'total_return_pct', 0),
                    'max_drawdown_pct': getattr(r['result'].metrics, 'max_drawdown_pct', 0),
                    'win_rate': getattr(r['result'].metrics, 'win_rate', 0),
                    'total_trades': r['result'].total_trades
                } for r in all_results],
                optimization_time_seconds=optimization_time
            )
            
            logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best {metric}: {getattr(best_backtest.metrics, metric, 0):.4f}")
            
            return optimization_result
        else:
            logger.warning("Optimization failed to find valid parameters")
            raise BacktestError("Optimization failed to find valid parameters")


class MultiSymbolBacktester:
    """Run backtests for multiple symbols with the same strategy."""
    
    def __init__(self, symbols: List[str], timeframes: List[str], 
                start_date: str, end_date: str, initial_capital: float = None,
                commission_rate: float = None, db_path: str = None):
        """
        Initialize with list of symbols to test.
        
        Args:
            symbols: List of trading symbols to test
            timeframes: List of timeframes to include
            start_date: Start date for backtest (ISO format)
            end_date: End date for backtest (ISO format)
            initial_capital: Initial capital to start with
            commission_rate: Commission rate as a decimal
            db_path: Optional custom path for the database
        """
        self.symbols = symbols
        self.timeframes = timeframes
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital or DEFAULT_BACKTEST_SETTINGS['initial_capital']
        self.commission_rate = commission_rate or DEFAULT_BACKTEST_SETTINGS['commission_rate']
        self.db_path = db_path
        
        # Results storage
        self.results: Dict[str, Dict[str, BacktestResult]] = {}
    
    def run_for_all_symbols(self, strategy_func: Callable, 
                          use_parallel: bool = None) -> Dict[str, BacktestResult]:
        """
        Run the same strategy for all symbols.
        
        Args:
            strategy_func: Strategy function to use
            use_parallel: Whether to use parallel processing (overrides settings)
            
        Returns:
            Dict[str, BacktestResult]: Results for each symbol
        """
        # Use setting if not specified
        if use_parallel is None:
            use_parallel = PERFORMANCE_SETTINGS.get('use_parallel_processing', True)
        
        num_processes = PERFORMANCE_SETTINGS.get('num_processes', multiprocessing.cpu_count())
        
        strategy_name = getattr(strategy_func, '__name__', 'Custom_Strategy')
        results = {}
        
        logger.info(f"Running {strategy_name} on {len(self.symbols)} symbols")
        
        # Start timing
        start_time = time.time()
        
        if use_parallel and num_processes > 1 and len(self.symbols) > 1:
            # Run in parallel
            logger.info(f"Using parallel processing with {num_processes} processes")
            
            # Define worker function
            def _backtest_symbol(symbol: str) -> Tuple[str, BacktestResult]:
                try:
                    engine = BacktestEngine(
                        symbol=symbol,
                        timeframes=self.timeframes,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        initial_capital=self.initial_capital,
                        commission_rate=self.commission_rate,
                        db_path=self.db_path
                    )
                    
                    result = engine.run_backtest(strategy_func)
                    return (symbol, result)
                except Exception as e:
                    logger.error(f"Error backtesting {symbol}: {e}", exc_info=True)
                    return (symbol, None)
            
            # Run backtests in parallel
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [executor.submit(_backtest_symbol, symbol) for symbol in self.symbols]
                
                for future in as_completed(futures):
                    try:
                        symbol, result = future.result()
                        if result:
                            results[symbol] = result
                    except Exception as e:
                        logger.error(f"Error retrieving backtest result: {e}", exc_info=True)
        else:
            # Run sequentially
            logger.info("Running backtests sequentially")
            
            for symbol in self.symbols:
                try:
                    engine = BacktestEngine(
                        symbol=symbol,
                        timeframes=self.timeframes,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        initial_capital=self.initial_capital,
                        commission_rate=self.commission_rate,
                        db_path=self.db_path
                    )
                    
                    result = engine.run_backtest(strategy_func)
                    results[symbol] = result
                    
                except Exception as e:
                    logger.error(f"Error backtesting {symbol}: {e}", exc_info=True)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Store in results dict
        self.results[strategy_name] = results
        
        logger.info(f"Completed backtests for {len(results)} symbols in {total_time:.2f} seconds")
        
        return results
    
    def compare_symbols(self, strategy_name: str = None) -> pd.DataFrame:
        """
        Compare performance across symbols for a specific strategy.
        
        Args:
            strategy_name: Name of strategy to compare (uses first if None)
            
        Returns:
            pd.DataFrame: Performance comparison
        """
        if not self.results:
            logger.warning("No backtest results available for comparison")
            return pd.DataFrame()
        
        # If strategy_name not specified, use the first one
        if strategy_name is None:
            strategy_name = list(self.results.keys())[0]
        
        if strategy_name not in self.results:
            logger.warning(f"Strategy '{strategy_name}' not found in results")
            return pd.DataFrame()
        
        # Build comparison dataframe
        comparison_data = []
        
        for symbol, result in self.results[strategy_name].items():
            comparison_data.append({
                'symbol': symbol,
                'total_return_pct': result.metrics.total_return_pct,
                'sharpe_ratio': result.metrics.sharpe_ratio,
                'max_drawdown_pct': result.metrics.max_drawdown_pct,
                'win_rate': result.metrics.win_rate,
                'profit_factor': result.metrics.profit_factor,
                'total_trades': result.total_trades
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by Sharpe ratio
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
        
        return comparison_df 