import os
import sys
import pytest
import pandas as pd
import sqlite3
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
import json

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.db_integration import DatabaseIntegration
from bot.database import Database

# Use a test database file
TEST_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "data", "test_trading_bot.db")

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
def mock_db():
    """Create a mock Database object."""
    mock = MagicMock(spec=Database)
    mock.insert_signal.return_value = 123
    mock.insert_trade.return_value = True
    mock.update_trade.return_value = True
    mock.store_market_data.return_value = True
    mock.add_alert.return_value = 456
    return mock

@pytest.fixture
def db_integration(cleanup_test_db):
    """Create a DatabaseIntegration instance with a test database."""
    # Create a Database instance with the test path
    test_db = Database(TEST_DB_PATH)
    
    # Now create the DatabaseIntegration instance and manually set its db attribute
    integration = DatabaseIntegration()
    integration.db = test_db
    
    return integration

class TestDatabaseIntegration:
    
    def test_init(self, cleanup_test_db):
        """Test initialization of DatabaseIntegration."""
        with patch('bot.database.DB_PATH', TEST_DB_PATH):
            integration = DatabaseIntegration()
            assert integration.db is not None
    
    def test_init_error(self):
        """Test handling of initialization errors."""
        with patch('bot.database.Database.__init__', side_effect=Exception("Test error")):
            with pytest.raises(Exception):
                DatabaseIntegration()
    
    def test_save_signal(self, db_integration):
        """Test saving a signal to the database."""
        signal_id = db_integration.save_signal(
            symbol="BTCUSDT",
            timeframe="1h",
            strategy="test_strategy",
            signal="BUY",
            indicators={"rsi": 30},
            llm_decision="BUY"
        )
        
        assert signal_id > 0
        
        # Verify the signal was saved
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM signals WHERE signal_id = ?", (signal_id,))
        signal = cursor.fetchone()
        conn.close()
        
        assert signal is not None
        assert signal[1] == "BTCUSDT"  # symbol
        assert signal[2] == "1h"  # timeframe
        assert signal[3] == "test_strategy"  # strategy
        assert signal[4] == "BUY"  # signal
    
    def test_save_signal_error(self, db_integration):
        """Test error handling when saving a signal fails."""
        with patch.object(db_integration.db, 'insert_signal', side_effect=Exception("Test error")):
            signal_id = db_integration.save_signal(
                symbol="BTCUSDT",
                timeframe="1h",
                strategy="test_strategy",
                signal="BUY"
            )
            assert signal_id == -1
    
    def test_save_trade(self, db_integration):
        """Test saving a trade to the database."""
        trade_data = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.001,
            "price": 50000.0,
            "status": "FILLED"
        }
        
        success = db_integration.save_trade(trade_data)
        assert success is True
        
        # Verify the trade was saved
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE symbol = ?", ("BTCUSDT",))
        trade = cursor.fetchone()
        conn.close()
        
        assert trade is not None
        assert trade[2] == "BUY"  # side
        assert float(trade[3]) == 0.001  # quantity
        assert float(trade[4]) == 50000.0  # price
    
    def test_save_trade_with_id(self, db_integration):
        """Test saving a trade with a predefined ID."""
        trade_id = "test_trade_id_123"
        trade_data = {
            "trade_id": trade_id,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.001,
            "price": 50000.0,
            "status": "FILLED"
        }
        
        success = db_integration.save_trade(trade_data)
        assert success is True
        
        # Verify the trade was saved with the correct ID
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
        trade = cursor.fetchone()
        conn.close()
        
        assert trade is not None
        assert trade[0] == trade_id  # trade_id
    
    def test_save_trade_error(self, db_integration):
        """Test error handling when saving a trade fails."""
        with patch.object(db_integration.db, 'insert_trade', side_effect=Exception("Test error")):
            success = db_integration.save_trade({
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "price": 50000.0,
                "status": "FILLED"
            })
            assert success is False
    
    def test_update_trade(self, db_integration):
        """Test updating a trade in the database."""
        # First, insert a trade
        trade_id = "update_test_trade_id"
        db_integration.save_trade({
            "trade_id": trade_id,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.001,
            "price": 50000.0,
            "status": "NEW"
        })
        
        # Now update it
        update_data = {
            "status": "FILLED",
            "price": 50100.0,
            "notes": "Test update"
        }
        
        success = db_integration.update_trade(trade_id, update_data)
        assert success is True
        
        # Verify the trade was updated
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT status, price, notes FROM trades WHERE trade_id = ?", (trade_id,))
        trade = cursor.fetchone()
        conn.close()
        
        assert trade is not None
        assert trade[0] == "FILLED"  # status
        assert float(trade[1]) == 50100.0  # price
        assert trade[2] == "Test update"  # notes
    
    def test_update_trade_error(self, db_integration):
        """Test error handling when updating a trade fails."""
        with patch.object(db_integration.db, 'update_trade', side_effect=Exception("Test error")):
            success = db_integration.update_trade("test_id", {"status": "FILLED"})
            assert success is False
    
    def test_link_signal_to_trade(self, db_integration):
        """Test linking a signal to a trade."""
        # First, insert a signal and a trade
        signal_id = db_integration.save_signal(
            symbol="BTCUSDT",
            timeframe="1h",
            strategy="test_strategy",
            signal="BUY"
        )
        
        trade_id = "link_test_trade_id"
        db_integration.save_trade({
            "trade_id": trade_id,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.001,
            "price": 50000.0,
            "status": "FILLED"
        })
        
        # Now link them
        success = db_integration.link_signal_to_trade(signal_id, trade_id)
        assert success is True
        
        # Verify the signal was linked
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT executed, trade_id FROM signals WHERE signal_id = ?", (signal_id,))
        signal = cursor.fetchone()
        conn.close()
        
        assert signal is not None
        assert signal[0] == 1  # executed
        assert signal[1] == trade_id  # trade_id
    
    def test_link_signal_to_trade_error(self, db_integration):
        """Test error handling when linking a signal to a trade fails."""
        with patch.object(db_integration, '_get_connection', side_effect=Exception("Test error")):
            success = db_integration.link_signal_to_trade(123, "test_id")
            assert success is False
    
    def test_save_market_data(self, db_integration):
        """Test saving market data to the database."""
        # Create sample market data
        candles = [
            [1625097600000, "50000", "51000", "49000", "50500", "10", 1625097900000, "505000", 100, "5", "250000", "0"],
            [1625097300000, "49800", "50200", "49700", "50000", "5", 1625097600000, "249500", 50, "2", "100000", "0"]
        ]
        
        market_data = {
            "symbol": "BTCUSDT",
            "candles": candles,
            "timestamp": datetime.now().isoformat()
        }
        
        success = db_integration.save_market_data(market_data, "BTCUSDT", "1h")
        assert success is True
        
        # Verify the market data was saved
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_data WHERE symbol = ? AND timeframe = ?", 
                      ("BTCUSDT", "1h"))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 2  # Two candles should be saved
    
    def test_save_market_data_with_dataframe(self, db_integration):
        """Test saving market data as a DataFrame."""
        # Create sample DataFrame
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "open": [50000.0, 50500.0],
            "high": [51000.0, 51500.0],
            "low": [49000.0, 49500.0],
            "close": [50500.0, 51000.0],
            "volume": [10.0, 15.0]
        })
        
        success = db_integration.save_market_data(df, "BTCUSDT", "1d")
        assert success is True
        
        # Verify the market data was saved
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_data WHERE symbol = ? AND timeframe = ?", 
                      ("BTCUSDT", "1d"))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 2  # Two days of data should be saved
    
    def test_save_market_data_error(self, db_integration):
        """Test error handling when saving market data fails."""
        with patch.object(db_integration.db, 'store_market_data', side_effect=Exception("Test error")):
            success = db_integration.save_market_data({"candles": []}, "BTCUSDT", "1h")
            assert success is False
    
    def test_add_system_alert(self, db_integration):
        """Test adding a system alert."""
        alert_id = db_integration.add_system_alert(
            message="Test alert",
            alert_type="info",
            severity="low",
            data={"test": "data"}
        )
        
        assert alert_id > 0
        
        # Verify the alert was saved
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
        alert = cursor.fetchone()
        conn.close()
        
        assert alert is not None
        assert alert[1] == "info"  # type
        assert alert[2] == "low"  # severity
        assert alert[3] == "Test alert"  # message
    
    def test_add_system_alert_error(self, db_integration):
        """Test error handling when adding an alert fails."""
        with patch.object(db_integration.db, 'add_alert', side_effect=Exception("Test error")):
            alert_id = db_integration.add_system_alert("Test alert")
            assert alert_id == -1

    # Mock tests
    def test_save_signal_with_mock(self, mock_db):
        """Test saving a signal using a mock database."""
        with patch('bot.db_integration.Database', return_value=mock_db):
            integration = DatabaseIntegration()
            signal_id = integration.save_signal(
                symbol="BTCUSDT",
                timeframe="1h",
                strategy="test_strategy",
                signal="BUY"
            )
            
            assert signal_id == 123
            mock_db.insert_signal.assert_called_once()
    
    def test_save_trade_with_mock(self, mock_db):
        """Test saving a trade using a mock database."""
        with patch('bot.db_integration.Database', return_value=mock_db):
            integration = DatabaseIntegration()
            success = integration.save_trade({
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "price": 50000.0,
                "status": "FILLED"
            })
            
            assert success is True
            mock_db.insert_trade.assert_called_once()
    
    def test_update_trade_with_mock(self, mock_db):
        """Test updating a trade using a mock database."""
        with patch('bot.db_integration.Database', return_value=mock_db):
            integration = DatabaseIntegration()
            success = integration.update_trade("test_id", {"status": "FILLED"})
            
            assert success is True
            mock_db.update_trade.assert_called_once()
    
    def test_add_system_alert_with_mock(self, mock_db):
        """Test adding a system alert using a mock database."""
        with patch('bot.db_integration.Database', return_value=mock_db):
            integration = DatabaseIntegration()
            alert_id = integration.add_system_alert("Test alert")
            
            assert alert_id == 456
            mock_db.add_alert.assert_called_once() 