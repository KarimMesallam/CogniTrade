#!/usr/bin/env python
"""
Arithmetic Validation Test for Backtesting Engine

This script creates a controlled dataset with predetermined price movements
and validates that the backtesting engine calculates profits, losses, and
returns correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import tempfile

# Ensure the parent directory is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("arithmetic_test.log")
    ]
)

logger = logging.getLogger("arithmetic_test")

def create_test_data():
    """
    Create a controlled dataset with known price movements.
    """
    # Create date range for 10 days with hourly data
    start_date = datetime(2025, 1, 1)
    dates = pd.date_range(start=start_date, periods=240, freq='1h')  # 10 days * 24 hours
    
    # Create a dataframe with known price pattern:
    # - Starts at 100
    # - Rises to 110 (10% increase)
    # - Drops to 99 (10% decrease)
    # - Rises to 108.9 (10% increase)
    
    # Create a simple price pattern with 4 segments
    segment_length = 60  # 60 hours per segment
    
    # Segment 1: Flat at 100
    prices1 = np.full(segment_length, 100.0)
    
    # Segment 2: Linear rise from 100 to 110
    prices2 = np.linspace(100.0, 110.0, segment_length)
    
    # Segment 3: Linear fall from 110 to 99
    prices3 = np.linspace(110.0, 99.0, segment_length)
    
    # Segment 4: Linear rise from 99 to 108.9
    prices4 = np.linspace(99.0, 108.9, segment_length)
    
    # Combine all segments
    prices = np.concatenate([prices1, prices2, prices3, prices4])
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.001,  # 0.1% above close
        'low': prices * 0.999,   # 0.1% below close
        'close': prices,
        'volume': np.random.normal(1000, 100, len(dates))
    })
    
    # Add symbol and timeframe
    df['symbol'] = 'TEST'
    df['timeframe'] = '1h'
    
    return df

def perfect_timing_strategy(data_dict, symbol):
    """
    A strategy with perfect timing (for testing arithmetic):
    - Buy at exactly 100 (start)
    - Sell at exactly 110 (peak)
    - Buy at exactly 99 (bottom)
    - Sell at exactly 108.9 (end)
    """
    df = data_dict['1h']
    
    # Get current price
    current_time = df.iloc[-1]['timestamp']
    current_price = df.iloc[-1]['close']
    
    # Log price for debugging only at key price points
    if abs(current_price - 100.0) < 0.01 or abs(current_price - 110.0) < 0.01 or \
       abs(current_price - 99.0) < 0.01 or abs(current_price - 108.9) < 0.01:
        logger.info(f"[Strategy] Time: {current_time}, Price: {current_price:.2f}")
    
    # Buy at the beginning (price = 100)
    if abs(current_price - 100.0) < 0.01:
        logger.info(f"[Strategy] BUY signal at price {current_price:.2f}")
        return 'BUY'
    
    # Sell at the first peak (price = 110)
    elif abs(current_price - 110.0) < 0.01:
        logger.info(f"[Strategy] SELL signal at price {current_price:.2f}")
        return 'SELL'
    
    # Buy at the bottom (price = 99)
    elif abs(current_price - 99.0) < 0.01:
        logger.info(f"[Strategy] BUY signal at price {current_price:.2f}")
        return 'BUY'
    
    # Sell at the end (price = 108.9)
    elif abs(current_price - 108.9) < 0.01:
        logger.info(f"[Strategy] SELL signal at price {current_price:.2f}")
        return 'SELL'
    
    # Hold otherwise
    else:
        return 'HOLD'

def calculate_expected_results():
    """
    Calculate the expected results based on our predetermined price movements.
    """
    # Initial capital
    initial_capital = 10000.0
    
    # Commission rate
    commission_rate = 0.001  # 0.1%
    
    # First trade: Buy at 100, Sell at 110
    buy_price_1 = 100.0
    sell_price_1 = 110.0
    position_size_pct = 0.5  # Use 50% of capital per trade
    
    # Calculate position for first trade
    available_capital_1 = initial_capital * position_size_pct
    position_1 = available_capital_1 / buy_price_1
    
    # Calculate commission for first buy
    commission_buy_1 = buy_price_1 * position_1 * commission_rate
    
    # Calculate remaining capital after first buy
    remaining_capital_1 = initial_capital - (buy_price_1 * position_1) - commission_buy_1
    
    # Calculate proceeds from first sell
    sell_value_1 = sell_price_1 * position_1
    commission_sell_1 = sell_value_1 * commission_rate
    proceeds_1 = sell_value_1 - commission_sell_1
    
    # Calculate profit from first trade
    profit_1 = proceeds_1 - (buy_price_1 * position_1)
    
    # Calculate capital after first trade
    capital_after_trade_1 = remaining_capital_1 + proceeds_1
    
    # Second trade: Buy at 99, Sell at 108.9
    buy_price_2 = 99.0
    sell_price_2 = 108.9
    
    # Calculate position for second trade
    available_capital_2 = capital_after_trade_1 * position_size_pct
    position_2 = available_capital_2 / buy_price_2
    
    # Calculate commission for second buy
    commission_buy_2 = buy_price_2 * position_2 * commission_rate
    
    # Calculate remaining capital after second buy
    remaining_capital_2 = capital_after_trade_1 - (buy_price_2 * position_2) - commission_buy_2
    
    # Calculate proceeds from second sell
    sell_value_2 = sell_price_2 * position_2
    commission_sell_2 = sell_value_2 * commission_rate
    proceeds_2 = sell_value_2 - commission_sell_2
    
    # Calculate profit from second trade
    profit_2 = proceeds_2 - (buy_price_2 * position_2)
    
    # Calculate final capital
    final_capital = remaining_capital_2 + proceeds_2
    
    # Calculate overall return percentage
    total_return_pct = (final_capital - initial_capital) / initial_capital * 100
    
    # Return calculated values
    expected = {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return_pct': total_return_pct,
        'trades': [
            {
                'entry_price': buy_price_1,
                'exit_price': sell_price_1,
                'position': position_1,
                'profit': profit_1
            },
            {
                'entry_price': buy_price_2,
                'exit_price': sell_price_2,
                'position': position_2,
                'profit': profit_2
            }
        ]
    }
    
    return expected

def run_arithmetic_test():
    """
    Run the arithmetic validation test.
    """
    from bot.backtesting.core.engine import BacktestEngine
    
    # Create a subclass of BacktestEngine that logs more details about trade execution
    class DiagnosticBacktestEngine(BacktestEngine):
        """A diagnostic version of BacktestEngine that logs additional details."""
        
        def _load_market_data(self) -> None:
            """Override to skip database loading since we'll set the data directly."""
            # Initialize market_data without loading
            self.market_data = {}
            logger.info("Skipping database loading in test engine")
        
        def _process_signal(self, signal, timestamp, price) -> None:
            """Override to add more logging to the signal processing."""
            # Only log BUY/SELL signals to reduce verbosity
            if signal != 'HOLD':
                logger.info(f"[Engine] Processing signal: {signal} at price {price:.2f}")
            
            # Check position before processing
            before_position = self.position_size
            before_capital = self.current_capital
            
            # Call parent method
            super()._process_signal(signal, timestamp, price)
            
            # Check position after processing
            after_position = self.position_size
            after_capital = self.current_capital
            
            # Log what happened, but only if something changed
            if signal == 'BUY' and before_position == 0 and after_position > 0:
                logger.info(f"[Engine] BUY executed: {after_position:.6f} units at {price:.2f}")
                logger.info(f"[Engine] Capital change: {before_capital:.2f} -> {after_capital:.2f}")
            elif signal == 'SELL' and before_position > 0 and after_position == 0:
                logger.info(f"[Engine] SELL executed: {before_position:.6f} units at {price:.2f}")
                logger.info(f"[Engine] Capital change: {before_capital:.2f} -> {after_capital:.2f}")
            elif signal in ['BUY', 'SELL'] and before_position != after_position:
                logger.info(f"[Engine] {signal} partially executed: {before_position:.6f} -> {after_position:.6f}")
                logger.info(f"[Engine] Capital change: {before_capital:.2f} -> {after_capital:.2f}")
            elif signal in ['BUY', 'SELL']:
                logger.warning(f"[Engine] Signal {signal} not executed. Position: {before_position:.6f} -> {after_position:.6f}")
    
    logger.info("Starting arithmetic validation test")
    
    try:
        # Create test data
        logger.info("Creating controlled test dataset")
        test_data = create_test_data()
        
        # Verify the test data contains the expected price points
        price_points = [100.0, 110.0, 99.0, 108.9]
        for price in price_points:
            matches = test_data[abs(test_data['close'] - price) < 0.01]
            if len(matches) == 0:
                logger.warning(f"Test data does not contain price point {price:.2f}")
            else:
                logger.info(f"Test data contains {len(matches)} candles with price ~{price:.2f}")
                if len(matches) == 1:
                    logger.info(f"  At time: {matches.iloc[0]['timestamp']}")
        
        # Calculate expected results
        logger.info("Calculating expected results")
        expected = calculate_expected_results()
        
        # Create a temporary directory for test data
        temp_dir = tempfile.mkdtemp()
        test_db_path = os.path.join(temp_dir, "test_arithmetic.db")
        logger.info(f"Using test path: {test_db_path}")
        
        # Initialize our diagnostic engine
        logger.info("Initializing diagnostic backtest engine")
        engine = DiagnosticBacktestEngine(
            symbol='TEST',
            timeframes=['1h'],
            start_date='2025-01-01',
            end_date='2025-01-11',
            initial_capital=10000.0,
            commission_rate=0.001,
            position_size_pct=0.5,
            db_path=test_db_path
        )
        
        # Directly set the market data in the engine
        engine.market_data = {'1h': test_data}
        
        # Run backtest with our perfect timing strategy
        logger.info("Running backtest with perfect timing strategy")
        result = engine.run_backtest(perfect_timing_strategy)
        
        # Print all trades for debugging
        logger.info("\n==== All Trades ====")
        for i, trade in enumerate(result.trades):
            logger.info(f"Trade {i+1}: {trade['side']} at {trade['price']:.2f}, Quantity: {trade['quantity']:.6f}")
        
        # Examine the state of the engine after the backtest
        logger.info(f"\nFinal position size: {engine.position_size:.6f}")
        logger.info(f"Final capital: ${engine.current_capital:.2f}")
        
        # Compare the actual and expected final results
        logger.info("\n==== Final Results Comparison ====")
        logger.info(f"Initial Capital - Expected: ${expected['initial_capital']:.2f}, Actual: ${result.initial_capital:.2f}")
        logger.info(f"Final Capital - Expected: ${expected['final_capital']:.2f}, Actual: ${result.final_equity:.2f}")
        logger.info(f"Total Return % - Expected: {expected['total_return_pct']:.2f}%, Actual: {result.metrics.total_return_pct:.2f}%")
        
        # Print difference between expected and actual
        logger.info(f"Final Capital Difference: ${result.final_equity - expected['final_capital']:.2f}")
        logger.info(f"Return % Difference: {result.metrics.total_return_pct - expected['total_return_pct']:.2f} percentage points")
        
        return result, expected
        
    except Exception as e:
        logger.error(f"Error running arithmetic validation test: {e}", exc_info=True)
        return None, None

if __name__ == "__main__":
    print("\n=== Backtesting Engine Arithmetic Validation Test ===\n")
    result, expected = run_arithmetic_test()
    
    if result is not None:
        print("\n==== Expected vs Actual Results ====")
        print(f"Initial Capital: ${expected['initial_capital']:.2f} vs ${result.initial_capital:.2f}")
        print(f"Final Capital: ${expected['final_capital']:.2f} vs ${result.final_equity:.2f}")
        print(f"Total Return %: {expected['total_return_pct']:.2f}% vs {result.metrics.total_return_pct:.2f}%")
        
        if abs(result.metrics.total_return_pct - expected['total_return_pct']) < 0.1:
            print("\n✅ ARITHMETIC VALIDATION PASSED ✅")
        else:
            print("\n❌ ARITHMETIC VALIDATION FAILED ❌") 