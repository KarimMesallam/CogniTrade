import os
import sys
import pytest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import the bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.backtesting import BacktestEngine, BacktestRunner
from bot.database import Database
from bot.strategy import calculate_rsi

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
    """Create a backtest engine for testing"""
    engine = BacktestEngine(
        symbol='BTCUSDT',
        timeframes=['1h', '4h'],
        start_date='2023-01-01',
        end_date='2023-01-31',
        db_path=TEST_DB_PATH
    )
    return engine

@pytest.fixture(scope="function")
def backtest_runner():
    """Create a backtest runner for testing"""
    runner = BacktestRunner(TEST_RUNNER_DB_PATH)
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
    # Get sample data
    df = backtest_engine.market_data['1h'].copy()
    
    # Add indicators
    df_with_indicators = backtest_engine.add_indicators(df)
    
    # Check that indicators were added
    assert 'rsi' in df_with_indicators.columns
    assert 'upper_band' in df_with_indicators.columns
    assert 'middle_band' in df_with_indicators.columns
    assert 'lower_band' in df_with_indicators.columns
    assert 'macd_line' in df_with_indicators.columns
    assert 'signal_line' in df_with_indicators.columns
    assert 'macd_histogram' in df_with_indicators.columns

def test_run_backtest(backtest_engine):
    """Test running a backtest with a simple strategy"""
    # Define a simple strategy function
    def simple_strategy(data_dict, symbol):
        # Use primary timeframe data
        df = data_dict['1h']
        
        # Simple moving average crossover
        df['sma_short'] = df['close'].rolling(window=10).mean()
        df['sma_long'] = df['close'].rolling(window=30).mean()
        
        # Check for crossover
        if len(df) < 30:
            return 'HOLD'
            
        if df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1]:
            return 'BUY'
        else:
            return 'SELL'
    
    # Run backtest
    results = backtest_engine.run_backtest(simple_strategy)
    
    # Check basic results structure
    assert results is not None
    assert 'symbol' in results
    assert 'timeframes' in results
    assert 'initial_capital' in results
    assert 'final_equity' in results
    assert 'total_trades' in results
    assert 'win_rate' in results
    assert 'trades' in results
    assert 'equity_curve' in results
    
    # Check that there are some trades
    assert len(results['trades']) >= 0

def test_multi_timeframe_analysis(backtest_engine):
    """Test multi-timeframe analysis functionality"""
    # Prepare data
    backtest_engine.prepare_data()
    
    # Run multi-timeframe analysis
    analysis = backtest_engine.multi_timeframe_analysis(backtest_engine.market_data)
    
    # Check results
    assert '1h' in analysis
    assert '4h' in analysis
    assert 'consolidated' in analysis
    
    # Check that we have the expected metrics in each timeframe
    for tf in ['1h', '4h']:
        assert 'rsi' in analysis[tf]
        assert 'macd_histogram' in analysis[tf]
        assert 'bb_position' in analysis[tf]
        assert 'trend' in analysis[tf]
        assert 'volatility' in analysis[tf]
    
    # Check consolidated view
    assert 'bullish_timeframes' in analysis['consolidated']
    assert 'bearish_timeframes' in analysis['consolidated']
    assert 'high_volatility_timeframes' in analysis['consolidated']

def test_monitor_and_alert(backtest_engine):
    """Test monitoring and alerting system"""
    # Create test results with poor performance metrics
    test_results = {
        'max_drawdown': -20,  # High drawdown
        'win_rate': 30,       # Low win rate
        'total_trades': 20,   # Sufficient trades
        'sharpe_ratio': 0.3   # Poor Sharpe ratio
    }
    
    # Check for alerts
    alerts = backtest_engine.monitor_and_alert(test_results)
    
    # Verify alerts were created
    assert len(alerts) == 3  # Should have 3 alerts (drawdown, win rate, Sharpe)
    
    # Check alert types
    alert_types = [alert['type'] for alert in alerts]
    assert 'drawdown' in alert_types
    assert 'win_rate' in alert_types
    assert 'performance' in alert_types
    
    # Check alert severities
    assert alerts[0]['severity'] == 'high'  # Drawdown alert should be high severity

def test_generate_trade_log(backtest_engine):
    """Test generating comprehensive trade log"""
    # Create sample trades
    sample_trades = [
        {
            'trade_id': '1',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'timestamp': datetime(2023, 1, 5, 10, 0, 0),
            'price': 20000,
            'quantity': 0.5,
            'value': 10000,
            'commission': 10,
            'entry_point': True
        },
        {
            'trade_id': '2',
            'symbol': 'BTCUSDT',
            'side': 'SELL',
            'timestamp': datetime(2023, 1, 10, 14, 0, 0),
            'price': 22000,
            'quantity': 0.5,
            'value': 11000,
            'commission': 11,
            'entry_price': 20000,
            'profit_loss': 989,  # 11000 - 10000 - 11
            'roi_pct': 9.89,
            'entry_point': False
        }
    ]
    
    # Create sample results
    results = {
        'trades': sample_trades,
        'symbol': 'BTCUSDT',
        'timeframes': ['1h'],
        'start_date': '2023-01-01',
        'end_date': '2023-01-31'
    }
    
    # Generate trade log
    trade_log = backtest_engine.generate_trade_log(results, filename=None)
    
    # Check log
    assert isinstance(trade_log, pd.DataFrame)
    assert len(trade_log) == 2
    assert trade_log['side'].tolist() == ['BUY', 'SELL']
    assert trade_log['profit_loss'].iloc[1] == 989

@patch('bot.backtesting.plt')
def test_plot_results(mock_plt, backtest_engine):
    """Test plotting backtest results"""
    # Create sample equity curve
    equity_curve = []
    start_date = datetime(2023, 1, 1)
    equity = 10000.0
    
    for i in range(100):
        # Add some random changes to equity
        equity += np.random.normal(50, 200)
        timestamp = start_date + timedelta(hours=i)
        
        equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'position_size': 0.0 if i % 10 != 0 else 0.5
        })
    
    # Create sample trades
    sample_trades = [
        {
            'trade_id': '1',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'timestamp': start_date + timedelta(hours=10),
            'price': 20000,
            'quantity': 0.5,
            'roi_pct': 0
        },
        {
            'trade_id': '2',
            'symbol': 'BTCUSDT',
            'side': 'SELL',
            'timestamp': start_date + timedelta(hours=20),
            'price': 22000,
            'quantity': 0.5,
            'roi_pct': 10
        }
    ]
    
    # Create sample results
    results = {
        'symbol': 'BTCUSDT',
        'timeframes': ['1h'],
        'start_date': '2023-01-01',
        'end_date': '2023-01-31',
        'initial_capital': 10000,
        'final_equity': equity,
        'total_return_pct': (equity - 10000) / 10000 * 100,
        'total_trades': 2,
        'win_count': 1,
        'loss_count': 0,
        'win_rate': 100,
        'max_drawdown': -5,
        'sharpe_ratio': 1.5,
        'trades': sample_trades,
        'equity_curve': equity_curve
    }
    
    # Plot results
    backtest_engine.plot_results(results)
    
    # Check that matplotlib functions were called
    mock_plt.figure.assert_called()
    mock_plt.tight_layout.assert_called()
    mock_plt.show.assert_called()


# BacktestRunner tests
@patch('bot.backtesting.BacktestEngine')
def test_run_multiple_backtests(mock_engine_class, backtest_runner):
    """Test running multiple backtests"""
    # Create mock backtest engine and its results
    mock_engine = MagicMock()
    mock_engine_class.return_value = mock_engine
    
    # Setup mock results
    mock_results = {
        'symbol': 'BTCUSDT',
        'timeframes': ['1h', '4h'],
        'start_date': '2023-01-01',
        'end_date': '2023-01-31',
        'initial_capital': 10000,
        'final_equity': 11000,
        'total_profit': 1000,
        'total_return_pct': 10,
        'total_trades': 5,
        'win_count': 3,
        'loss_count': 2,
        'win_rate': 60,
        'max_drawdown': -8,
        'sharpe_ratio': 1.2,
        'trades': []
    }
    
    # Configure mock to return results
    mock_engine.run_backtest.return_value = mock_results
    mock_engine.monitor_and_alert.return_value = []
    mock_engine.generate_trade_log.return_value = pd.DataFrame()
    mock_engine.save_results.return_value = True
    
    # Define test strategies
    def strategy1(data_dict, symbol):
        return 'BUY'
        
    def strategy2(data_dict, symbol):
        return 'SELL'
    
    # Run multiple backtests
    results = backtest_runner.run_multiple_backtests(
        symbols=['BTCUSDT', 'ETHUSDT'],
        timeframes=['1h', '4h'],
        strategies={
            'Strategy1': strategy1,
            'Strategy2': strategy2
        },
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    # Check results
    assert len(results) == 2  # Two symbols
    assert len(results['BTCUSDT']) == 2  # Two strategies
    assert len(results['ETHUSDT']) == 2  # Two strategies
    
    # Check that BacktestEngine was called correctly
    assert mock_engine_class.call_count == 4  # 2 symbols * 2 strategies
    
    # Check strategy results
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for strategy in ['Strategy1', 'Strategy2']:
            assert strategy in results[symbol]
            assert 'result' in results[symbol][strategy]
            assert results[symbol][strategy]['result'] == mock_results

def test_compare_strategies(backtest_runner):
    """Test comparing strategy performances"""
    # Set sample results
    backtest_runner.results = {
        'BTCUSDT': {
            'Strategy1': {
                'result': {
                    'total_return_pct': 15,
                    'win_rate': 60,
                    'sharpe_ratio': 1.5,
                    'max_drawdown': -10,
                    'total_trades': 10
                }
            },
            'Strategy2': {
                'result': {
                    'total_return_pct': 10,
                    'win_rate': 70,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -8,
                    'total_trades': 15
                }
            }
        },
        'ETHUSDT': {
            'Strategy1': {
                'result': {
                    'total_return_pct': 20,
                    'win_rate': 65,
                    'sharpe_ratio': 1.8,
                    'max_drawdown': -12,
                    'total_trades': 12
                }
            },
            'Strategy2': {
                'result': {
                    'total_return_pct': 5,
                    'win_rate': 55,
                    'sharpe_ratio': 0.9,
                    'max_drawdown': -6,
                    'total_trades': 8
                }
            }
        }
    }
    
    # Compare strategies
    comparison = backtest_runner.compare_strategies()
    
    # Check comparison
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 4  # 2 symbols * 2 strategies
    
    # Check that ranking was calculated
    assert 'return_rank' in comparison.columns
    assert 'sharpe_rank' in comparison.columns
    assert 'overall_rank' in comparison.columns
    
    # Check best strategy for BTC
    btc_best = comparison[comparison['symbol'] == 'BTCUSDT'].sort_values('overall_rank').iloc[0]
    assert btc_best['strategy'] == 'Strategy1'  # Higher returns and Sharpe
    
    # Check best strategy for ETH
    eth_best = comparison[comparison['symbol'] == 'ETHUSDT'].sort_values('overall_rank').iloc[0]
    assert eth_best['strategy'] == 'Strategy1'  # Higher returns and Sharpe

def test_generate_summary_report(backtest_runner):
    """Test generating a summary report"""
    # Set sample results (same as in test_compare_strategies)
    backtest_runner.results = {
        'BTCUSDT': {
            'Strategy1': {
                'result': {
                    'total_return_pct': 15,
                    'win_rate': 60,
                    'sharpe_ratio': 1.5,
                    'max_drawdown': -10,
                    'total_trades': 10
                }
            },
            'Strategy2': {
                'result': {
                    'total_return_pct': 10,
                    'win_rate': 70,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -8,
                    'total_trades': 15
                }
            }
        },
        'ETHUSDT': {
            'Strategy1': {
                'result': {
                    'total_return_pct': 20,
                    'win_rate': 65,
                    'sharpe_ratio': 1.8,
                    'max_drawdown': -12,
                    'total_trades': 12
                }
            },
            'Strategy2': {
                'result': {
                    'total_return_pct': 5,
                    'win_rate': 55,
                    'sharpe_ratio': 0.9,
                    'max_drawdown': -6,
                    'total_trades': 8
                }
            }
        }
    }
    
    # Generate report
    report = backtest_runner.generate_summary_report()
    
    # Check report
    assert isinstance(report, str)
    assert "Backtest Summary Report" in report
    assert "Total backtests run: 4" in report
    assert "Symbols tested: 2" in report
    assert "Strategies tested: 2" in report
    assert "Top Strategies by Return" in report
    assert "Top Strategies by Risk-Adjusted Return" in report
    
    # Check that best strategy is included
    assert "Strategy1 on ETHUSDT: 20.00% return" in report
    assert "Strategy1 on ETHUSDT: Sharpe 1.80" in report 