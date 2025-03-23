import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.order_manager import OrderManager

@pytest.fixture
def mock_order_data():
    """Fixture to provide standard Binance order response data."""
    return {
        "symbol": "BTCUSDT",
        "orderId": 123456,
        "clientOrderId": "test_client_order_id",
        "transactTime": 1617000000000,
        "price": "50000.00",
        "origQty": "0.001",
        "executedQty": "0.001",
        "status": "FILLED",
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

@pytest.fixture
def order_manager():
    """Fixture to create an OrderManager instance."""
    with patch('bot.order_manager.os.path.exists', return_value=False):
        manager = OrderManager("BTCUSDT", 1.0)
    return manager

class TestOrderManager:
    
    def test_init(self):
        """Test OrderManager initialization."""
        with patch('bot.order_manager.os.path.exists', return_value=False):
            manager = OrderManager("BTCUSDT", 1.0)
        
        assert manager.symbol == "BTCUSDT"
        assert manager.risk_percentage == 1.0
        assert manager.active_orders == {}
        assert manager.order_history == []
    
    def test_load_order_history(self):
        """Test loading order history from file."""
        mock_history = [{"orderId": 1, "status": "FILLED"}]
        
        # Mock os.path.exists to return True and open to return mock data
        with patch('bot.order_manager.os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_history))):
            manager = OrderManager("BTCUSDT", 1.0)
            
        assert manager.order_history == mock_history
    
    def test_save_order_history(self):
        """Test saving order history to file."""
        manager = OrderManager("BTCUSDT", 1.0)
        manager.order_history = [{"orderId": 1, "status": "FILLED"}]
        
        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            manager._save_order_history()
        
        # We only check that open was called once with the correct arguments
        # json.dump may call write() multiple times internally
        mock_file.assert_called_once()
        # Verify it's writing json data (don't check exact number of writes)
        assert mock_file().write.call_count > 0
    
    def test_log_order(self, order_manager, mock_order_data):
        """Test logging an order to history."""
        # Test with a filled order
        order_manager._log_order(mock_order_data, "BUY", "FILLED")
        
        assert len(order_manager.order_history) == 1
        assert order_manager.order_history[0]["order_id"] == mock_order_data["orderId"]
        assert order_manager.order_history[0]["status"] == "FILLED"
        assert order_manager.order_history[0]["action"] == "BUY"
        assert mock_order_data["orderId"] not in order_manager.active_orders
        
        # Test with a new order
        mock_order_data["status"] = "NEW"
        order_manager._log_order(mock_order_data, "BUY", "NEW")
        
        assert len(order_manager.order_history) == 2
        assert mock_order_data["orderId"] in order_manager.active_orders
    
    @patch('bot.order_manager.place_market_buy')
    @patch('bot.order_manager.calculate_order_quantity')
    def test_execute_market_buy(self, mock_calculate, mock_market_buy, order_manager, mock_order_data):
        """Test executing a market buy order."""
        mock_calculate.return_value = 0.001
        mock_market_buy.return_value = mock_order_data
        
        # Test with quantity provided
        result = order_manager.execute_market_buy(quantity=0.001)
        assert result == mock_order_data
        mock_market_buy.assert_called_with("BTCUSDT", 0.001)
        
        # Test with quote amount provided
        mock_market_buy.reset_mock()
        result = order_manager.execute_market_buy(quote_amount=50)
        assert result == mock_order_data
        mock_calculate.assert_called_with("BTCUSDT", 50)
        mock_market_buy.assert_called_with("BTCUSDT", 0.001)
        
        # Test failure case
        mock_market_buy.return_value = None
        result = order_manager.execute_market_buy(quantity=0.001)
        assert result is None
    
    @patch('bot.order_manager.place_market_sell')
    def test_execute_market_sell(self, mock_market_sell, order_manager, mock_order_data):
        """Test executing a market sell order."""
        mock_market_sell.return_value = mock_order_data
        
        result = order_manager.execute_market_sell(0.001)
        assert result == mock_order_data
        mock_market_sell.assert_called_with("BTCUSDT", 0.001)
        
        # Test failure case
        mock_market_sell.return_value = None
        result = order_manager.execute_market_sell(0.001)
        assert result is None
    
    @patch('bot.order_manager.place_limit_buy')
    def test_execute_limit_buy(self, mock_limit_buy, order_manager, mock_order_data):
        """Test executing a limit buy order."""
        mock_limit_buy.return_value = mock_order_data
        
        result = order_manager.execute_limit_buy(0.001, 50000)
        assert result == mock_order_data
        mock_limit_buy.assert_called_with("BTCUSDT", 0.001, 50000)
        
        # Test failure case
        mock_limit_buy.return_value = None
        result = order_manager.execute_limit_buy(0.001, 50000)
        assert result is None
    
    @patch('bot.order_manager.place_limit_sell')
    def test_execute_limit_sell(self, mock_limit_sell, order_manager, mock_order_data):
        """Test executing a limit sell order."""
        mock_limit_sell.return_value = mock_order_data
        
        result = order_manager.execute_limit_sell(0.001, 50000)
        assert result == mock_order_data
        mock_limit_sell.assert_called_with("BTCUSDT", 0.001, 50000)
        
        # Test failure case
        mock_limit_sell.return_value = None
        result = order_manager.execute_limit_sell(0.001, 50000)
        assert result is None
    
    @patch('bot.order_manager.get_open_orders')
    @patch('bot.order_manager.cancel_order')
    def test_cancel_all_orders(self, mock_cancel, mock_open_orders, order_manager, mock_order_data):
        """Test canceling all open orders."""
        mock_open_orders.return_value = [
            {"orderId": 123, "symbol": "BTCUSDT"},
            {"orderId": 456, "symbol": "BTCUSDT"}
        ]
        mock_cancel.return_value = mock_order_data
        
        results = order_manager.cancel_all_orders()
        assert len(results) == 2
        assert mock_cancel.call_count == 2
        
        # Test partial failure
        mock_cancel.reset_mock()
        mock_cancel.side_effect = [mock_order_data, None]
        results = order_manager.cancel_all_orders()
        assert len(results) == 1
        assert mock_cancel.call_count == 2
    
    @patch('bot.order_manager.get_order_status')
    def test_update_order_statuses(self, mock_status, order_manager, mock_order_data):
        """Test updating status of active orders."""
        # Add an active order to track
        order_manager.active_orders = {
            123: {"order_id": 123, "status": "NEW", "action": "BUY"}
        }
        
        # Mock the updated status
        updated_order = mock_order_data.copy()
        updated_order["orderId"] = 123
        updated_order["status"] = "FILLED"
        mock_status.return_value = updated_order
        
        results = order_manager.update_order_statuses()
        assert 123 in results
        assert 123 not in order_manager.active_orders  # Should be removed when filled
        assert len(order_manager.order_history) == 1
        
        # Test with order that doesn't exist
        mock_status.return_value = None
        order_manager.active_orders = {789: {"order_id": 789, "status": "NEW", "action": "BUY"}}
        results = order_manager.update_order_statuses()
        assert results == {}
    
    def test_get_active_orders(self, order_manager):
        """Test getting active orders."""
        order_manager.active_orders = {"test": "data"}
        assert order_manager.get_active_orders() == {"test": "data"}
    
    def test_get_order_history(self, order_manager):
        """Test getting order history with and without limit."""
        order_manager.order_history = [1, 2, 3, 4, 5]
        
        # Test without limit
        assert order_manager.get_order_history() == [1, 2, 3, 4, 5]
        
        # Test with limit
        assert order_manager.get_order_history(limit=2) == [4, 5] 