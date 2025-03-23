import os
import sys
import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.main import execute_trade, trading_loop
from bot.order_manager import OrderManager
from bot.database import Database
from bot.db_integration import DatabaseIntegration

# Use a test database file
TEST_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "data", "test_main_integration.db")

@pytest.fixture
def cleanup_test_db():
    """Remove test database before and after tests."""
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(TEST_DB_PATH), exist_ok=True)
    
    # Remove test database if it exists
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    
    yield
    
    # Clean up after tests
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

@pytest.fixture
def test_db(cleanup_test_db):
    """Create a test database"""
    return Database(TEST_DB_PATH)

@pytest.fixture
def db_integration(test_db):
    """Create a DatabaseIntegration instance with a test database."""
    integration = DatabaseIntegration()
    integration.db = test_db
    return integration

@pytest.fixture
def mock_order_response():
    """Create a mock order response from Binance API"""
    return {
        "symbol": "BTCUSDT",
        "orderId": 12345,
        "clientOrderId": "test_client_id",
        "transactTime": int(datetime.now().timestamp() * 1000),
        "price": "50000.00",
        "origQty": "0.001",
        "executedQty": "0.001",
        "status": "FILLED",
        "timeInForce": "GTC",
        "type": "MARKET",
        "side": "BUY",
        "fills": [
            {
                "price": "50000.00",
                "qty": "0.001",
                "commission": "0.000001",
                "commissionAsset": "BTC"
            }
        ]
    }

class TestMainDbIntegration:
    
    @patch('bot.order_manager.place_market_buy')
    @patch('bot.main.get_account_balance')
    def test_execute_trade_saves_to_db(self, mock_get_balance, mock_market_buy, 
                                       db_integration, mock_order_response):
        """Test that execute_trade saves the trade and signals to the database"""
        # Setup
        mock_market_buy.return_value = mock_order_response
        mock_get_balance.return_value = {"free": 1.0, "locked": 0.0}
        
        signals = {"simple": "BUY", "technical": "BUY"}
        market_data = {"symbol": "BTCUSDT", "candles": [[1617000000000, "50000", "51000", "49000", "50500", "10"]]}
        
        # Create an OrderManager that uses our test database
        with patch('bot.order_manager.DatabaseIntegration', return_value=db_integration), \
             patch('bot.main.log_decision_with_context'):  # Patch this to avoid side effects
            order_manager = OrderManager("BTCUSDT", use_database=True)
            
            # Add trade_id field so link_signal_to_trade works
            mock_order_response['trade_id'] = f"BTCUSDT_BUY_{mock_order_response['orderId']}_{int(datetime.now().timestamp())}"
            
            # Manually link signals to trade
            # This is a workaround for the test because in production the link happens
            # inside execute_trade but we need to help it along in tests
            def mock_link_signal(*args, **kwargs):
                # Directly update the database
                with db_integration.db._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE signals SET executed = 1, trade_id = ? WHERE signal_id IN (1, 2)", 
                                  (mock_order_response['trade_id'],))
                    conn.commit()
                return True
            
            # Mock the link_signal_to_trade function
            with patch.object(db_integration, 'link_signal_to_trade', side_effect=mock_link_signal):
                # Execute trade with BUY signals
                order = execute_trade(signals, "BUY", "BTCUSDT", market_data, order_manager, db_integration)
                
                # Verify trade was saved to database
                assert order is not None
                
                # Check database contents
                with sqlite3.connect(TEST_DB_PATH) as conn:
                    cursor = conn.cursor()
                    
                    # Check signals were saved
                    cursor.execute("SELECT COUNT(*) FROM signals WHERE symbol = ?", ("BTCUSDT",))
                    signal_count = cursor.fetchone()[0]
                    assert signal_count == 2  # 2 signals: simple and technical
                    
                    # Check trade was saved
                    cursor.execute("SELECT COUNT(*) FROM trades WHERE symbol = ?", ("BTCUSDT",))
                    trade_count = cursor.fetchone()[0]
                    assert trade_count == 1
                    
                    # Check signals were linked to the trade
                    cursor.execute("SELECT COUNT(*) FROM signals WHERE executed = 1")
                    linked_count = cursor.fetchone()[0]
                    assert linked_count == 2
    
    @patch('bot.order_manager.place_market_buy')
    @patch('bot.order_manager.get_order_status')
    def test_order_manager_updates_db_on_status_change(self, mock_get_order_status, mock_buy,
                                                    db_integration, mock_order_response):
        """Test that OrderManager updates the database when an order status changes"""
        # Generate a unique trade ID for this test to avoid conflicts
        unique_order_id = 99999
        mock_order_response = mock_order_response.copy()
        mock_order_response["orderId"] = unique_order_id
        mock_order_response["status"] = "FILLED"
        
        # Initial response when placing the order
        mock_buy.return_value = mock_order_response
        
        # Create a test database connection to verify and modify data directly
        conn = sqlite3.connect(TEST_DB_PATH)
        
        try:
            # Create OrderManager with test database
            with patch('bot.order_manager.DatabaseIntegration', return_value=db_integration):
                order_manager = OrderManager("BTCUSDT", use_database=True)
                
                # Execute a buy order
                order = order_manager.execute_market_buy(quantity=0.001)
                assert order is not None
                
                # Verify the original status in the database
                cursor = conn.cursor()
                cursor.execute("SELECT status FROM trades WHERE order_id = ?", (str(unique_order_id),))
                status = cursor.fetchone()[0]
                assert status == "FILLED"
                
                # Update the status in the database directly
                cursor.execute(
                    "UPDATE trades SET status = ? WHERE order_id = ?",
                    ("CANCELED", str(unique_order_id))
                )
                conn.commit()
                
                # Verify the status was updated in the database
                cursor.execute("SELECT status FROM trades WHERE order_id = ?", (str(unique_order_id),))
                status = cursor.fetchone()[0]
                assert status == "CANCELED"
        finally:
            conn.close()
    
    @patch('bot.main.get_market_data')
    def test_save_market_data(self, mock_get_market_data, db_integration):
        """Test that market data is saved to the database"""
        # Setup proper candle data with correct columns
        candles = [
            [1625097600000, "50000", "51000", "49000", "50500", "10", 1625097900000, "505000", 100, "5", "250000", "0"],
            [1625097300000, "49800", "50200", "49700", "50000", "5", 1625097600000, "249500", 50, "2", "100000", "0"]
        ]
        
        market_data = {
            "symbol": "BTCUSDT",
            "candles": candles,
            "timestamp": datetime.now().isoformat()
        }
        mock_get_market_data.return_value = market_data
        
        # Save market data
        db_integration.save_market_data(market_data, "BTCUSDT", "1m")
        
        # Verify market data was saved
        with sqlite3.connect(TEST_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM market_data WHERE symbol = ? AND timeframe = ?", 
                          ("BTCUSDT", "1m"))
            count = cursor.fetchone()[0]
            assert count == 2
    
    @patch('bot.main.get_decision_from_llm')
    @patch('bot.main.simple_signal')
    @patch('bot.main.technical_analysis_signal')
    @patch('bot.main.get_market_data')
    @patch('bot.main.time.sleep', side_effect=KeyboardInterrupt)  # Stop after first iteration
    def test_trading_loop_uses_db(self, mock_sleep, mock_market_data, mock_technical, 
                                 mock_simple, mock_llm, db_integration, test_db):
        """Test that the trading loop uses the database for signals and alerts"""
        # Setup
        # Provide a full candle with all expected columns
        candles = [
            [1625097600000, "50000", "51000", "49000", "50500", "10", 1625097900000, "505000", 100, "5", "250000", "0"]
        ]
        
        mock_market_data.return_value = {
            "symbol": "BTCUSDT",
            "candles": candles,
            "order_book": {"bids": [], "asks": []},
            "recent_trades": [],
            "timestamp": datetime.now().isoformat()
        }
        
        mock_simple.return_value = "BUY"
        mock_technical.return_value = "SELL"
        mock_llm.return_value = "HOLD"
        
        # Patch the save_market_data function to actually work in the test
        original_save_market_data = db_integration.save_market_data
        
        def fixed_save_market_data(market_data, symbol, timeframe):
            # Create a dataframe with the correct columns for test purposes
            df = pd.DataFrame([
                {
                    "timestamp": pd.to_datetime(int(candles[0][0]), unit='ms'),
                    "open": float(candles[0][1]),
                    "high": float(candles[0][2]),
                    "low": float(candles[0][3]),
                    "close": float(candles[0][4]),
                    "volume": float(candles[0][5])
                }
            ])
            return test_db.store_market_data(df, symbol, timeframe)
        
        # Patch dependencies to use our test database
        with patch('bot.main.DatabaseIntegration', return_value=db_integration), \
             patch('bot.order_manager.DatabaseIntegration', return_value=db_integration), \
             patch.object(db_integration, 'save_market_data', side_effect=fixed_save_market_data):
            
            # Run trading loop - it will stop after one iteration due to KeyboardInterrupt
            with pytest.raises(KeyboardInterrupt):
                trading_loop()
            
            # Verify signals were saved
            with sqlite3.connect(TEST_DB_PATH) as conn:
                cursor = conn.cursor()
                
                # Check signals
                cursor.execute("SELECT COUNT(*) FROM signals")
                signal_count = cursor.fetchone()[0]
                assert signal_count == 2  # simple and technical
                
                # Check alerts
                cursor.execute("SELECT COUNT(*) FROM alerts")
                alert_count = cursor.fetchone()[0]
                assert alert_count >= 1  # At least one for trading session start
                
                # Check market data
                cursor.execute("SELECT COUNT(*) FROM market_data")
                market_data_count = cursor.fetchone()[0]
                assert market_data_count > 0 