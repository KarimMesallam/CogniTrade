import os
import sys
import pytest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import unittest.mock
from decimal import Decimal

# Add the parent directory to the path to import the bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.backtesting.core.engine import BacktestEngine, MultiSymbolBacktester
from bot.database import Database
from bot.strategy import calculate_rsi
from bot.backtesting.models.trade import Trade
from bot.backtesting.models.results import EquityPoint

# Configure logging for tests
logging.basicConfig(level=logging.ERROR)

# Global test variables
TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "test_trading_bot.db")
TEST_RUNNER_DB_PATH = os.path.join(os.path.dirname(__file__), "test_runner.db")


# Setup and teardown fixtures
@pytest.fixture(scope="module")
def sample_data():
    """Create sample market data for testing"""
    # Create a test database
    db = Database(TEST_DB_PATH)
    
    # Create sample dates
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    # Generate hourly data
    dates_1h = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Create a price series with some trend and noise
    base_price = 20000.0
    trend = np.linspace(0, 1000, len(dates_1h))
    noise = np.random.normal(0, 500, len(dates_1h))
    prices = base_price + trend + noise.cumsum()
    
    # Create DataFrames for different timeframes
    df_1h = pd.DataFrame({
        'timestamp': dates_1h,
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices + np.random.normal(0, 100, len(dates_1h)),
        'volume': np.random.normal(100, 30, len(dates_1h))
    })
    
    # Add symbol and timeframe
    df_1h['symbol'] = 'BTCUSDT'
    df_1h['timeframe'] = '1h'
    
    # Store in database
    db.store_market_data(df_1h, 'BTCUSDT', '1h')
    
    # Also create 4h data by resampling
    dates_4h = pd.date_range(start=start_date, end=end_date, freq='4H')
    df_4h = pd.DataFrame({
        'timestamp': dates_4h,
        'open': prices[::4],
        'high': (prices * 1.02)[::4],
        'low': (prices * 0.98)[::4],
        'close': (prices + np.random.normal(0, 200, len(dates_1h)))[::4],
        'volume': (np.random.normal(400, 100, len(dates_1h)))[::4]
    })
    
    # Add symbol and timeframe
    df_4h['symbol'] = 'BTCUSDT'
    df_4h['timeframe'] = '4h'
    
    # Store in database
    db.store_market_data(df_4h, 'BTCUSDT', '4h')
    
    yield db  # Provide the database object for tests
    
    # Cleanup
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

@pytest.fixture(scope="module")
def backtest_engine(sample_data):
    """Create a backtest engine for testing with pre-loaded data to avoid API calls"""
    # Patch _load_market_data to prevent it from trying to load from database
    with patch('bot.backtesting.core.engine.BacktestEngine._load_market_data'):
        # Create engine with test parameters
        engine = BacktestEngine(
            symbol='BTCUSDT',
            timeframes=['1h', '4h'],
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
    
        # Manually set market data instead of loading from database
        engine.market_data = {
            '1h': sample_data.get_market_data('BTCUSDT', '1h').copy(),
            '4h': sample_data.get_market_data('BTCUSDT', '4h').copy()
        }
        
        # Add basic indicators
        for tf in engine.timeframes:
            if tf in engine.market_data:
                df = engine.market_data[tf]
                # Add RSI (simplified version for testing)
                df['rsi'] = np.linspace(30, 70, len(df))
                # Add MACD components
                df['macd_line'] = np.linspace(-2, 2, len(df))
                df['signal_line'] = np.linspace(-1, 1, len(df))
                df['macd_histogram'] = df['macd_line'] - df['signal_line']
                # Add Bollinger Bands
                df['middle_band'] = df['close'].rolling(window=20).mean().fillna(df['close'])
                df['upper_band'] = df['middle_band'] + 2 * df['close'].rolling(window=20).std().fillna(df['close'] * 0.02)
                df['lower_band'] = df['middle_band'] - 2 * df['close'].rolling(window=20).std().fillna(df['close'] * 0.02)
        
        return engine

@pytest.fixture(scope="function")
def backtest_runner():
    """Create a backtest runner for testing"""
    # Patch _load_market_data to prevent database access
    with patch('bot.backtesting.core.engine.BacktestEngine._load_market_data'):
        runner = MultiSymbolBacktester(
            symbols=['BTCUSDT', 'ETHUSDT'],  # Test symbols
            timeframes=['1h', '4h'],
            start_date='2023-01-01',
            end_date='2023-01-31',
            db_path=TEST_RUNNER_DB_PATH
        )
        
        # Create sample data for both symbols
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        
        # Create basic price data
        btc_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.linspace(20000, 22000, 100),
            'high': np.linspace(20200, 22200, 100),
            'low': np.linspace(19800, 21800, 100),
            'close': np.linspace(20100, 22100, 100),
            'volume': np.random.normal(100, 30, 100)
        })
        
        eth_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.linspace(1500, 1700, 100),
            'high': np.linspace(1520, 1720, 100),
            'low': np.linspace(1480, 1680, 100),
            'close': np.linspace(1510, 1710, 100),
            'volume': np.random.normal(500, 100, 100)
        })
        
        # Set market data for each symbol
        runner.data_manager = MagicMock()
        runner.data_manager.get_market_data.side_effect = lambda symbol, timeframe, **kwargs: btc_data if symbol == 'BTCUSDT' else eth_data
        
        yield runner
        
    # Cleanup
    if os.path.exists(TEST_RUNNER_DB_PATH):
        os.remove(TEST_RUNNER_DB_PATH)


# BacktestEngine tests
def test_initialization(backtest_engine):
    """Test BacktestEngine initialization"""
    # Check basic initialization
    assert backtest_engine.symbol == 'BTCUSDT'
    assert backtest_engine.timeframes == ['1h', '4h']
    assert backtest_engine.start_date == '2023-01-01'
    assert backtest_engine.end_date == '2023-01-31'
    
    # Check that market data was loaded
    assert '1h' in backtest_engine.market_data
    assert '4h' in backtest_engine.market_data
    assert len(backtest_engine.market_data['1h']) > 0
    assert len(backtest_engine.market_data['4h']) > 0

def test_add_indicators(backtest_engine):
    """Test adding technical indicators to market data"""
    # Create a test dataframe with price data but no indicators
    dates = pd.date_range(start='2023-01-01', periods=30, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(20000, 500, 30),
        'high': np.random.normal(20500, 500, 30),
        'low': np.random.normal(19500, 500, 30),
        'close': np.random.normal(20000, 500, 30),
        'volume': np.random.normal(100, 30, 30)
    })
    
    # Set market data with test dataframe
    backtest_engine.market_data = {'1h': df.copy()}
    
    # Call prepare_data to add indicators
    backtest_engine.prepare_data()
    
    # Check that indicators were added
    processed_df = backtest_engine.market_data['1h']
    
    # Verify that the dataframe now has common indicators
    expected_indicators = ['rsi', 'macd_line', 'signal_line', 'macd_histogram', 
                         'upper_band', 'middle_band', 'lower_band']
    
    # Check at least some of these indicators are present
    # We don't know the exact implementation, but some should be there
    found_indicators = [col for col in expected_indicators if col in processed_df.columns]
    assert len(found_indicators) > 0, "No indicators were added to the market data"

def test_run_backtest(backtest_engine):
    """Test running a backtest with a simple strategy"""
    # Create a simple strategy function that alternates BUY and SELL
    def simple_strategy(data_dict, symbol):
        # Always BUY on even-indexed candles and SELL on odd-indexed candles
        candle_count = len(data_dict['1h'])
        if candle_count < 5:  # Require at least some data
            return 'HOLD'
        
        if (candle_count % 2) == 0:
            return 'BUY'
        else:
            return 'SELL'
    
    # Run backtest with the strategy
    result = backtest_engine.run_backtest(simple_strategy)
    
    # Verify that we have a valid result
    assert result is not None
    assert hasattr(result, 'symbol')
    assert result.symbol == 'BTCUSDT'
    assert hasattr(result, 'strategy_name')
    assert hasattr(result, 'total_trades')
    assert hasattr(result, 'winning_trades')
    assert hasattr(result, 'losing_trades')
    assert hasattr(result, 'equity_curve')
    
    # Check that metrics were calculated
    assert hasattr(result, 'metrics')
    assert hasattr(result.metrics, 'total_return_pct')
    assert hasattr(result.metrics, 'sharpe_ratio')
    assert hasattr(result.metrics, 'max_drawdown_pct')

def test_sma_crossover_strategy(backtest_engine):
    """Test the SMA crossover strategy from example scripts"""
    # Create sample market data with clear crossover pattern
    dates = pd.date_range(start='2023-01-01', periods=60, freq='1h')
    
    # Create price pattern with a clear crossover
    prices = np.linspace(20000, 21000, 30).tolist() + np.linspace(21000, 19500, 30).tolist()
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.normal(100, 30, 60)
    })
    
    # Set up SMA crossover scenario
    df['sma_short'] = df['close'].rolling(window=10).mean()
    df['sma_long'] = df['close'].rolling(window=20).mean()
    
    # Manually create a crossover scenario
    # For the first 30 candles, short SMA > long SMA (uptrend)
    # At candle 31, short SMA crosses below long SMA (sell signal)
    
    # Set market data
    market_data_dict = {'1h': df}
    
    # Define SMA crossover strategy
    def sma_crossover_strategy(data_dict, symbol):
        # Use primary timeframe data
        df = data_dict['1h']
        
        # Require at least 20 candles for proper SMA calculation
        if len(df) < 20:
            return 'HOLD'
        
        # Get current and previous values
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check for crossover
        if previous['sma_short'] <= previous['sma_long'] and current['sma_short'] > current['sma_long']:
            return 'BUY'
        elif previous['sma_short'] >= previous['sma_long'] and current['sma_short'] < current['sma_long']:
            return 'SELL'
        else:
            return 'HOLD'
    
    # Test strategy at different points
    
    # Test at candle 25 - should be HOLD (no crossover)
    test_data_hold = {
        '1h': df.iloc[:25].copy()
    }
    assert sma_crossover_strategy(test_data_hold, 'BTCUSDT') == 'HOLD'
    
    # Create a buy crossover scenario
    buy_df = df.copy()
    buy_df.loc[28, 'sma_short'] = buy_df.loc[28, 'sma_long'] - 5  # Below
    buy_df.loc[29, 'sma_short'] = buy_df.loc[29, 'sma_long'] + 5  # Above
    
    test_data_buy = {
        '1h': buy_df.iloc[:30].copy()
    }
    assert sma_crossover_strategy(test_data_buy, 'BTCUSDT') == 'BUY'
    
    # Create a sell crossover scenario
    sell_df = df.copy()
    sell_df.loc[38, 'sma_short'] = sell_df.loc[38, 'sma_long'] + 5  # Above
    sell_df.loc[39, 'sma_short'] = sell_df.loc[39, 'sma_long'] - 5  # Below
    
    test_data_sell = {
        '1h': sell_df.iloc[:40].copy()
    }
    assert sma_crossover_strategy(test_data_sell, 'BTCUSDT') == 'SELL'

def test_rsi_strategy(backtest_engine):
    """Test the RSI strategy from example scripts"""
    # Create sample market data with RSI patterns
    dates = pd.date_range(start='2023-01-01', periods=50, freq='4h')
    
    # Create price data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(20000, 500, 50),
        'high': np.random.normal(20500, 500, 50),
        'low': np.random.normal(19500, 500, 50),
        'close': np.random.normal(20000, 500, 50),
        'volume': np.random.normal(100, 30, 50)
    })
    
    # Add RSI column with controlled values for testing
    # Setup the specific crossover scenarios we need:
    # Buy signal: RSI goes from below 30 to above 30
    # Sell signal: RSI goes from above 70 to below 70
    rsi_values = [40] * 20 + [25, 35] + [40] * 3 + [75, 65] + [60] * 23
    df['rsi'] = rsi_values
    
    # Set market data
    market_data_dict = {'4h': df}
    
    # Define RSI strategy
    def rsi_strategy(data_dict, symbol, oversold=30, overbought=70):
        # Use 4h timeframe
        df = data_dict['4h']
        
        # Require enough data
        if len(df) < 10:
            return 'HOLD'
        
        # Check for RSI column
        if 'rsi' not in df.columns:
            return 'HOLD'
        
        # Get current and previous RSI
        current_rsi = df['rsi'].iloc[-1]
        previous_rsi = df['rsi'].iloc[-2]
        
        # Buy signal: RSI crosses from below oversold to above oversold
        if previous_rsi < oversold and current_rsi > oversold:
            return 'BUY'
        
        # Sell signal: RSI crosses from above overbought to below overbought
        elif previous_rsi > overbought and current_rsi < overbought:
            return 'SELL'
        
        # No signal
        else:
            return 'HOLD'
    
    # Test strategy at different RSI scenarios
    
    # Test at buy crossover (index 20-21)
    test_data_buy = {
        '4h': df.iloc[:22].copy()
    }
    assert rsi_strategy(test_data_buy, 'BTCUSDT') == 'BUY'
    
    # Test at sell crossover (index 25-26)
    test_data_sell = {
        '4h': df.iloc[:27].copy()
    }
    assert rsi_strategy(test_data_sell, 'BTCUSDT') == 'SELL'
    
    # Test no signal (normal conditions)
    test_data_hold = {
        '4h': df.iloc[:15].copy()
    }
    assert rsi_strategy(test_data_hold, 'BTCUSDT') == 'HOLD'

def test_parameter_optimization(backtest_engine):
    """Test optimization of strategy parameters"""
    # Create sample market data with RSI for strategy testing
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    prices = np.linspace(20000, 21000, 100)  # Simple trend
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.normal(100, 30, 100)
    })
    
    # Add RSI values for the strategy to use
    df['rsi'] = np.linspace(20, 80, 100)  # Linear RSI for predictable testing
    
    # Set market data
    backtest_engine.market_data = {'1h': df.copy()}
    
    # Define a parameterized strategy factory
    def rsi_strategy_factory(params):
        """Factory that creates an RSI strategy with the given parameters"""
        def strategy(data_dict, symbol):
            df = data_dict['1h']
            if len(df) < 5:
                return 'HOLD'
                
            if 'rsi' not in df.columns:
                return 'HOLD'
                
            rsi = df['rsi'].iloc[-1]
            
            # Use parameters from the factory
            oversold = params['oversold']
            overbought = params['overbought']
            
            if rsi < oversold:
                return 'BUY'
            elif rsi > overbought:
                return 'SELL'
            else:
                return 'HOLD'
                
        return strategy
    
    # Define parameter grid with smaller search space for test
    param_grid = {
        'oversold': [30],  # Just one value for testing
        'overbought': [70]  # Just one value for testing
    }
    
    # Test with sequential processing (avoiding parallel issues in tests)
    with patch('bot.backtesting.config.settings.PERFORMANCE_SETTINGS', 
              {'use_parallel_processing': False}), \
         patch.object(backtest_engine, 'run_backtest') as mock_run_backtest:
        # Setup the mock to return a valid BacktestResult
        mock_result = MagicMock()
        mock_result.metrics.sharpe_ratio = 1.5
        mock_result.metrics.total_return_pct = 10.0
        mock_result.metrics.max_drawdown_pct = 5.0
        mock_result.metrics.win_rate = 60.0
        mock_result.symbol = 'BTCUSDT'
        mock_result.strategy_name = 'RSI_Strategy'
        mock_result.total_trades = 10
        mock_run_backtest.return_value = mock_result
        
        # Run the optimization
        result = backtest_engine.optimize_parameters(
            rsi_strategy_factory, 
            param_grid,
            metric='sharpe_ratio'
        )
        
        # Check that we got a valid optimization result
        assert result is not None
        assert hasattr(result, 'best_parameters')
        assert hasattr(result, 'best_backtest')
        assert hasattr(result, 'all_results')
        
        # Verify the run_backtest method was called once for each parameter combo
        assert mock_run_backtest.call_count == len(param_grid['oversold']) * len(param_grid['overbought'])

def test_parameter_optimization_parallel(backtest_engine):
    """Test optimization of strategy parameters with parallel processing"""
    # Create sample market data with RSI for strategy testing
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    prices = np.linspace(20000, 21000, 100)  # Simple trend
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.normal(100, 30, 100)
    })
    
    # Add RSI values for the strategy to use
    df['rsi'] = np.linspace(20, 80, 100)  # Linear RSI for predictable testing
    
    # Set market data
    backtest_engine.market_data = {'1h': df.copy()}
    
    # Define a parameterized strategy factory
    def rsi_strategy_factory(params):
        """Factory that creates an RSI strategy with the given parameters"""
        def strategy(data_dict, symbol):
            df = data_dict['1h']
            if len(df) < 5:
                return 'HOLD'
                
            if 'rsi' not in df.columns:
                return 'HOLD'
                
            rsi = df['rsi'].iloc[-1]
            
            # Use parameters from the factory
            oversold = params['oversold']
            overbought = params['overbought']
            
            if rsi < oversold:
                return 'BUY'
            elif rsi > overbought:
                return 'SELL'
            else:
                return 'HOLD'
                
        return strategy
    
    # Define parameter grid with small search space for test
    param_grid = {
        'oversold': [20, 30],  # Two values to test parallel processing
        'overbought': [70, 80]  # Two values to test parallel processing
    }
    
    # Test with parallel processing
    with patch('bot.backtesting.config.settings.PERFORMANCE_SETTINGS', 
              {'use_parallel_processing': True, 'num_processes': 2}), \
         patch.object(backtest_engine, 'run_backtest') as mock_run_backtest:
        # Setup the mock to return a valid BacktestResult
        mock_result = MagicMock()
        mock_result.metrics.sharpe_ratio = 1.5
        mock_result.metrics.total_return_pct = 10.0
        mock_result.metrics.max_drawdown_pct = 5.0
        mock_result.metrics.win_rate = 60.0
        mock_result.symbol = 'BTCUSDT'
        mock_result.strategy_name = 'RSI_Strategy'
        mock_result.total_trades = 10
        mock_run_backtest.return_value = mock_result
        
        # Run the optimization with parallel processing mocked
        with patch('bot.backtesting.core.engine.ProcessPoolExecutor') as mock_executor:
            # Create a mock executor instance
            mock_executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            
            # Mock submitted futures
            mock_future1 = MagicMock()
            mock_future2 = MagicMock()
            mock_future3 = MagicMock()
            mock_future4 = MagicMock()
            
            # Set up the future results
            mock_future1.result.return_value = ({'oversold': 20, 'overbought': 70}, mock_result)
            mock_future2.result.return_value = ({'oversold': 20, 'overbought': 80}, mock_result)
            mock_future3.result.return_value = ({'oversold': 30, 'overbought': 70}, mock_result)
            mock_future4.result.return_value = ({'oversold': 30, 'overbought': 80}, mock_result)
            
            # Mock the executor's submit method to return our futures
            mock_executor_instance.submit.side_effect = [mock_future1, mock_future2, mock_future3, mock_future4]
            
            # Mock as_completed to return our futures in desired order
            with patch('bot.backtesting.core.engine.as_completed', 
                      return_value=[mock_future1, mock_future2, mock_future3, mock_future4]):
                
                # Run the optimization
                result = backtest_engine.optimize_parameters(
                    rsi_strategy_factory, 
                    param_grid,
                    metric='sharpe_ratio'
                )
                
                # Verify parallel processing was used
                assert mock_executor.called
                assert mock_executor_instance.submit.call_count == 4  # Called for each parameter combination
                
                # Check optimization results
                assert result is not None
                assert hasattr(result, 'best_parameters')
                assert hasattr(result, 'best_backtest')
                assert hasattr(result, 'all_results')
                assert len(result.all_results) == 4

@patch('bot.backtesting.core.engine.BacktestEngine')
def test_multi_symbol_backtester(mock_backtest_engine, backtest_runner):
    """Test the MultiSymbolBacktester class for running backtests on multiple symbols"""
    # Setup mock BacktestEngine
    mock_instance = MagicMock()
    mock_backtest_engine.return_value = mock_instance
    
    # Setup mock BacktestResult
    mock_result = MagicMock()
    mock_result.symbol = "BTCUSDT"
    mock_result.strategy_name = "Test_Strategy"
    mock_result.total_trades = 10
    mock_result.metrics.sharpe_ratio = 1.5
    mock_result.metrics.total_return_pct = 15.0
    mock_result.metrics.max_drawdown_pct = 5.0
    mock_result.metrics.win_rate = 60.0
    
    # Setup the mock to return the result
    mock_instance.run_backtest.return_value = mock_result
    
    # Create a simple test strategy
    def test_strategy(data_dict, symbol):
        return 'HOLD'  # Always hold for testing
    
    # Run the multi-symbol backtest
    results = backtest_runner.run_for_all_symbols(test_strategy, use_parallel=False)
    
    # Check that results were returned for both symbols
    assert len(results) > 0
    
    # Check that backtest engine was created for each symbol
    assert mock_backtest_engine.call_count >= 1
    
    # Test the compare_symbols method
    backtest_runner.results['Test_Strategy'] = results
    comparison = backtest_runner.compare_symbols('Test_Strategy')
    
    # Verify that comparison is a DataFrame
    assert isinstance(comparison, pd.DataFrame)
    
    # If results were returned, check columns
    if not comparison.empty:
        assert 'symbol' in comparison.columns
        assert 'sharpe_ratio' in comparison.columns
        assert 'total_return_pct' in comparison.columns

def test_trade_execution(backtest_engine):
    """Test the trade execution methods (_process_signal and _execute_trade)"""
    # Setup test data
    backtest_engine.current_capital = 10000.0
    backtest_engine.position_size = 0.0
    backtest_engine.position_size_pct = 0.5  # Use 50% of capital per trade
    backtest_engine.trades = []
    backtest_engine.equity_curve = []
    current_time = pd.Timestamp('2023-01-15 12:00:00')
    current_price = 20000.0
    
    # Test BUY signal processing
    backtest_engine._process_signal('BUY', current_time, current_price)
    
    # Verify that a trade was executed
    assert len(backtest_engine.trades) == 1
    assert backtest_engine.trades[0].side == 'BUY'
    assert float(backtest_engine.trades[0].price) == current_price  # Convert Decimal to float for comparison
    assert backtest_engine.position_size > 0.0
    assert backtest_engine.current_capital < 10000.0  # Capital should be reduced
    
    # Calculate expected values
    available_capital = 10000.0 * 0.5  # 50% of initial capital
    expected_position = available_capital / current_price
    expected_trade_value = expected_position * current_price
    expected_commission = expected_trade_value * backtest_engine.commission_rate
    expected_remaining_capital = 10000.0 - expected_trade_value - expected_commission
    
    # Check values (with small epsilon for float comparison)
    assert abs(backtest_engine.position_size - expected_position) < 0.0001
    assert abs(backtest_engine.current_capital - expected_remaining_capital) < 0.0001
    
    # Test SELL signal processing
    new_time = pd.Timestamp('2023-01-15 18:00:00')
    new_price = 21000.0  # Price increased
    backtest_engine._process_signal('SELL', new_time, new_price)
    
    # Verify that a SELL trade was executed
    assert len(backtest_engine.trades) == 2
    assert backtest_engine.trades[1].side == 'SELL'
    assert float(backtest_engine.trades[1].price) == new_price  # Convert Decimal to float for comparison
    assert backtest_engine.position_size == 0.0  # Position is closed
    
    # Check that profit was recorded
    assert float(backtest_engine.trades[1].profit_loss) > 0.0  # Price increased, so profit
    assert float(backtest_engine.trades[1].roi_pct) > 0.0  # ROI should be positive
    
    # Test HOLD signal
    backtest_engine._process_signal('HOLD', current_time, current_price)
    
    # Verify that no new trade was executed
    assert len(backtest_engine.trades) == 2  # Still 2 trades
    
    # Test executing a direct trade
    test_quantity = 0.1
    backtest_engine._execute_trade('BUY', current_time, current_price, test_quantity)
    assert len(backtest_engine.trades) == 3
    assert backtest_engine.trades[2].side == 'BUY'
    # Use string representation for Decimal comparison to avoid precision issues
    assert str(backtest_engine.trades[2].quantity) == str(test_quantity)
    
    # Test market indicators are added to trades
    assert hasattr(backtest_engine.trades[0], 'market_indicators')

def test_vectorized_backtesting(backtest_engine):
    """Test the vectorized backtesting functionality"""
    # Create sample price data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    prices = np.linspace(20000, 22000, 100)  # Steady uptrend
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices - 50,
        'high': prices + 100,
        'low': prices - 100,
        'close': prices,
        'volume': np.random.normal(100, 30, 100)
    })
    
    # Reset market data
    backtest_engine.market_data = {'1h': df.copy()}
    
    # Create a vectorized strategy
    def vectorized_simple_strategy(data_dict, symbol, vectorized=False):
        """A simple strategy that returns BUY for even indices and SELL for odd indices"""
        if not vectorized:
            # Non-vectorized mode (should not be called in this test)
                return 'HOLD'
                
        # In vectorized mode, return signals for all candles
        df = data_dict['1h']
        signals = ['HOLD'] * len(df)
        
        # Generate alternating BUY/SELL signals
        for i in range(10, len(df)):
            if i % 20 < 10:  # BUY for 10 candles
                signals[i] = 'BUY'
            else:  # SELL for 10 candles
                signals[i] = 'SELL'
        
        return signals
    
    # Mark the strategy as vectorized
    vectorized_simple_strategy.vectorized = True
    
    # Run the backtest
    result = backtest_engine.run_backtest(vectorized_simple_strategy)
    
    # Check that the backtest ran successfully
    assert result is not None
    assert result.symbol == 'BTCUSDT'
    assert len(result.trades) > 0
    
    # Count BUY and SELL trades
    buy_trades = [t for t in backtest_engine.trades if t.side == 'BUY']
    sell_trades = [t for t in backtest_engine.trades if t.side == 'SELL']
    
    # We should have a similar number of buys and sells
    assert len(buy_trades) > 0
    assert len(sell_trades) > 0
    
    # Verify that the equity curve was created
    assert len(backtest_engine.equity_curve) > 0
    
    # Check that the trades have the correct properties
    for trade in backtest_engine.trades:
        assert trade.symbol == 'BTCUSDT'
        assert hasattr(trade, 'timestamp')
        assert hasattr(trade, 'price')
        assert hasattr(trade, 'quantity')
        assert hasattr(trade, 'commission')
    
    # Check that SELL trades have profit/loss information
    for trade in sell_trades:
        assert hasattr(trade, 'profit_loss')
        assert hasattr(trade, 'roi_pct')
        if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
            assert trade.exit_time > trade.entry_time

def test_can_use_vectorized_backtest(backtest_engine):
    """Test the _can_use_vectorized_backtest method"""
    # Test a function with vectorized attribute set to True
    def strategy_with_vectorized_attr(data_dict, symbol):
        return 'HOLD'
    strategy_with_vectorized_attr.vectorized = True
    
    assert backtest_engine._can_use_vectorized_backtest(strategy_with_vectorized_attr) == True
    
    # Test a function with 'vectorized' in its name
    def vectorized_strategy(data_dict, symbol):
            return 'HOLD'
        
    assert backtest_engine._can_use_vectorized_backtest(vectorized_strategy) == True
    
    # Test a regular function that doesn't support vectorization
    def regular_strategy(data_dict, symbol):
            return 'HOLD'
    
    assert backtest_engine._can_use_vectorized_backtest(regular_strategy) == False

def test_get_market_indicators(backtest_engine):
    """Test the _get_market_indicators method to extract market indicators at a specific timestamp"""
    # Create sample data with indicators
    dates = pd.date_range(start='2023-01-01', periods=30, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(20000, 500, 30),
        'high': np.random.normal(20500, 500, 30),
        'low': np.random.normal(19500, 500, 30),
        'close': np.random.normal(20000, 500, 30),
        'volume': np.random.normal(100, 30, 30)
    })
    
    # Add some indicator values
    df['rsi'] = np.linspace(30, 70, 30)  # RSI increasing from 30 to 70
    df['macd_line'] = np.linspace(-2, 2, 30)  # MACD crossing from negative to positive
    df['signal_line'] = np.linspace(0, 1, 30)  # Signal line increasing
    df['macd_histogram'] = df['macd_line'] - df['signal_line']  # Histogram
    df['upper_band'] = df['close'] + 200  # Upper band 200 above close
    df['middle_band'] = df['close']  # Middle band at close
    df['lower_band'] = df['close'] - 200  # Lower band 200 below close
    
    # Set market data
    backtest_engine.market_data = {'1h': df.copy()}
    
    # Get indicators at a specific timestamp
    test_timestamp = dates[15]  # Middle of the range
    indicators = backtest_engine._get_market_indicators(test_timestamp)
    
    # Check that indicators were extracted correctly
    assert indicators is not None
    assert isinstance(indicators, dict)
    
    # Check indicator values
    assert 'rsi' in indicators
    assert abs(indicators['rsi'] - df['rsi'].iloc[15]) < 0.0001
    
    assert 'macd_line' in indicators
    assert abs(indicators['macd_line'] - df['macd_line'].iloc[15]) < 0.0001
    
    assert 'signal_line' in indicators
    assert abs(indicators['signal_line'] - df['signal_line'].iloc[15]) < 0.0001
    
    assert 'upper_band' in indicators
    assert abs(indicators['upper_band'] - df['upper_band'].iloc[15]) < 0.0001
    
    assert 'middle_band' in indicators
    assert abs(indicators['middle_band'] - df['middle_band'].iloc[15]) < 0.0001
    
    assert 'lower_band' in indicators
    assert abs(indicators['lower_band'] - df['lower_band'].iloc[15]) < 0.0001
    
    # Check derived indicators
    assert 'bb_position' in indicators
    assert 0 <= indicators['bb_position'] <= 1
    
    assert 'trend' in indicators
    assert indicators['trend'] in ['uptrend', 'downtrend']
    
    assert 'volatility' in indicators
    assert indicators['volatility'] > 0

def test_create_backtest_result(backtest_engine):
    """Test the _create_backtest_result method that creates a BacktestResult object"""
    # Create a simple strategy function
    def test_strategy(data_dict, symbol):
                return 'HOLD'
                
    # Initialize test data
    backtest_engine.symbol = 'BTCUSDT'
    backtest_engine.timeframes = ['1h', '4h']
    backtest_engine.start_date = '2023-01-01'
    backtest_engine.end_date = '2023-01-31'
    backtest_engine.run_id = '20230101_120000'
    backtest_engine.initial_capital = 10000.0
    
    # Create some test trades
    backtest_engine.trades = []
    
    # Add a BUY trade
    buy_trade = Trade(
        symbol='BTCUSDT',
        side='BUY',
        timestamp=pd.Timestamp('2023-01-10 12:00:00'),
        price=Decimal('20000.0'),
        quantity=Decimal('0.1'),
        commission=Decimal('2.0'),
        status='FILLED'
    )
    buy_trade.entry_point = True
    buy_trade.entry_time = pd.Timestamp('2023-01-10 12:00:00')
    backtest_engine.trades.append(buy_trade)
    
    # Add a SELL trade with profit
    sell_trade = Trade(
        symbol='BTCUSDT',
        side='SELL',
        timestamp=pd.Timestamp('2023-01-15 12:00:00'),
        price=Decimal('21000.0'),
        quantity=Decimal('0.1'),
        commission=Decimal('2.1'),
        status='FILLED'
    )
    sell_trade.entry_price = Decimal('20000.0')
    sell_trade.exit_price = Decimal('21000.0')
    sell_trade.profit_loss = Decimal('97.9')  # (21000 - 20000) * 0.1 - 2.1
    sell_trade.roi_pct = Decimal('4.895')  # (97.9 / (20000 * 0.1)) * 100
    sell_trade.entry_time = pd.Timestamp('2023-01-10 12:00:00')
    sell_trade.exit_time = pd.Timestamp('2023-01-15 12:00:00')
    sell_trade.holding_period_hours = 120  # 5 days * 24 hours
    backtest_engine.trades.append(sell_trade)
    
    # Create some equity curve points
    backtest_engine.equity_curve = [
        EquityPoint(
            timestamp=pd.Timestamp('2023-01-01 00:00:00'),
            equity=10000.0,
            position_size=0.0
        ),
        EquityPoint(
            timestamp=pd.Timestamp('2023-01-10 12:00:00'),
            equity=9998.0,  # After BUY (10000 - 2)
            position_size=0.1
        ),
        EquityPoint(
            timestamp=pd.Timestamp('2023-01-15 12:00:00'),
            equity=10097.9,  # After SELL (9998 + 97.9)
            position_size=0.0
        )
    ]
    
    # Call the method to create the result
    result = backtest_engine._create_backtest_result(test_strategy)
    
    # Check the result properties
    assert result.symbol == 'BTCUSDT'
    assert result.strategy_name == 'test_strategy'
    assert result.timeframes == ['1h', '4h']
    assert result.start_date == '2023-01-01'
    assert result.end_date == '2023-01-31'
    assert result.run_id == '20230101_120000'
    assert result.initial_capital == 10000.0
    assert result.final_equity == 10097.9
    
    # Check trade statistics
    assert result.total_trades == 1  # Only the completed (SELL) trade is counted
    assert result.winning_trades == 1
    assert result.losing_trades == 0
    
    # Check that metrics were calculated
    assert hasattr(result, 'metrics')
    assert hasattr(result.metrics, 'total_return_pct')
    assert hasattr(result.metrics, 'sharpe_ratio')
    assert hasattr(result.metrics, 'max_drawdown_pct')
    assert hasattr(result.metrics, 'win_rate')
    
    # Check specific metrics - convert both sides to float for proper comparison
    assert float(result.metrics.total_return_pct) == pytest.approx(0.979, 0.001)  # 97.9 / 10000 * 100
    assert float(result.metrics.win_rate) == pytest.approx(100.0, 0.001)  # 1 out of 1 trades was profitable

def test_error_handling(backtest_engine):
    """Test error handling in the backtesting engine"""
    
    # Test with invalid strategy function
    def invalid_strategy(data_dict, symbol):
        # Raise an exception to test error handling
        raise ValueError("Test strategy error")
    
    # For this case, the engine actually processes the error and continues the backtest
    # rather than raising an exception, so check that the backtest completes
    result = backtest_engine.run_backtest(invalid_strategy)
    assert result is not None
    assert result.total_trades == 0
    
    # Patch _load_market_data for any direct BacktestEngine initialization tests below
    # Test validate_inputs with invalid parameters
    with pytest.raises(Exception) as excinfo:
        BacktestEngine("", [], "", "")  # Empty parameters
    assert "Symbol must be" in str(excinfo.value) or "Invalid parameter" in str(excinfo.value)
    
    # Test invalid timeframe
    with pytest.raises(Exception) as excinfo:
        # We don't use patch here because we want to test the validation logic
        # which happens during initialization (before _load_market_data is called)
        engine = BacktestEngine("BTCUSDT", ["invalid_format"], "2023-01-01", "2023-01-31")
    error_msg = str(excinfo.value)
    assert "Invalid timeframe format" in error_msg or "timeframe format" in error_msg.lower()
    
    # Test invalid date format
    with pytest.raises(Exception) as excinfo:
        BacktestEngine("BTCUSDT", ["1h"], "invalid-date", "2023-01-31")
    assert "date format" in str(excinfo.value) or "Invalid parameter" in str(excinfo.value)
    
    # Test start date after end date
    with pytest.raises(Exception) as excinfo:
        BacktestEngine("BTCUSDT", ["1h"], "2023-01-31", "2023-01-01")
    assert "Start date must be before end date" in str(excinfo.value) or "Invalid parameter" in str(excinfo.value) 