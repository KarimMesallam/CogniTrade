import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file before importing modules that use env variables
load_dotenv()

from bot.binance_api import (
    synchronize_time, get_recent_closes, get_account_balance,
    get_symbol_info, calculate_order_quantity, place_market_buy,
    place_market_sell, place_limit_buy, place_limit_sell,
    get_open_orders, cancel_order, get_order_status
)
from bot.config import SYMBOL

@pytest.fixture(scope="module")
def time_offset():
    """Get time offset from Binance server for all tests."""
    offset = synchronize_time()
    print(f"Time offset with Binance server: {offset}ms")
    return offset

def test_server_connection(time_offset):
    """Test that we can connect to the Binance server and get a time offset."""
    # The synchronize_time function will raise an exception if the connection fails
    assert time_offset is not None

def test_get_recent_closes():
    """Test that we can get recent closing prices."""
    closes = get_recent_closes(SYMBOL, '1h', 5)
    assert isinstance(closes, list)
    assert len(closes) <= 5  # Could be less if the symbol is new
    if closes:
        assert isinstance(closes[0], float)

@patch('bot.binance_api.client')
def test_get_account_balance(mock_client):
    """Test that we can get account balances."""
    # Mock the API response instead of making a real call
    mock_response = {
        'balances': [
            {'asset': 'BTC', 'free': '1.0', 'locked': '0.0'},
            {'asset': 'ETH', 'free': '5.0', 'locked': '0.0'},
            {'asset': 'USDT', 'free': '1000.0', 'locked': '50.0'}
        ]
    }
    mock_client.get_account.return_value = mock_response
    
    # Test getting all balances
    balances = get_account_balance()
    assert balances is not None
    assert isinstance(balances, dict)
    assert 'BTC' in balances
    assert 'USDT' in balances
    
    # Test getting a specific asset balance
    usdt_balance = get_account_balance('USDT')
    assert usdt_balance is not None
    assert isinstance(usdt_balance, dict)
    assert 'free' in usdt_balance
    assert usdt_balance['free'] == 1000.0

def test_get_symbol_info():
    """Test that we can get symbol information."""
    symbol_info = get_symbol_info(SYMBOL)
    assert symbol_info is not None
    assert symbol_info['symbol'] == SYMBOL

def test_calculate_order_quantity():
    """Test that we can calculate the order quantity."""
    # Test with a small amount
    quantity = calculate_order_quantity(SYMBOL, 10.0)
    # This could return None if the amount is too small for the symbol's minimum
    if quantity:
        assert isinstance(quantity, float)
        assert quantity > 0

# Add new tests below

@patch('bot.binance_api.client')
def test_calculate_order_quantity_with_mock(mock_client):
    """Test order quantity calculation with mocked responses."""
    # Mock the necessary API responses
    mock_client.get_symbol_ticker.return_value = {"symbol": SYMBOL, "price": "50000.00"}
    mock_client.get_symbol_info.return_value = {
        "symbol": SYMBOL,
        "filters": [
            {
                "filterType": "LOT_SIZE",
                "minQty": "0.00010000",
                "maxQty": "9000.00000000",
                "stepSize": "0.00010000"
            }
        ]
    }
    
    # Test normal calculation
    quantity = calculate_order_quantity(SYMBOL, 100.0)
    assert quantity is not None
    assert quantity == 0.002  # 100 / 50000 = 0.002
    
    # Test minimum quantity enforcement
    quantity = calculate_order_quantity(SYMBOL, 1.0)
    assert quantity is None  # 1 / 50000 = 0.00002 which is below minQty
    
    # Test with missing LOT_SIZE filter
    mock_client.get_symbol_info.return_value = {"symbol": SYMBOL, "filters": []}
    quantity = calculate_order_quantity(SYMBOL, 100.0)
    assert quantity is None

@patch('bot.binance_api.client')
def test_place_market_buy(mock_client):
    """Test placing a market buy order."""
    mock_response = {
        "symbol": SYMBOL,
        "orderId": 123456,
        "clientOrderId": "test123",
        "transactTime": 1617000000000,
        "price": "0.00000000",
        "origQty": "0.001",
        "executedQty": "0.001",
        "status": "FILLED",
        "timeInForce": "GTC",
        "type": "MARKET",
        "side": "BUY"
    }
    mock_client.order_market_buy.return_value = mock_response
    
    # Test successful order
    order = place_market_buy(SYMBOL, 0.001)
    assert order == mock_response
    mock_client.order_market_buy.assert_called_once()
    
    # Test exception handling
    mock_client.order_market_buy.side_effect = Exception("Test error")
    order = place_market_buy(SYMBOL, 0.001)
    assert order is None

@patch('bot.binance_api.client')
def test_place_market_sell(mock_client):
    """Test placing a market sell order."""
    mock_response = {
        "symbol": SYMBOL,
        "orderId": 123456,
        "clientOrderId": "test123",
        "transactTime": 1617000000000,
        "price": "0.00000000",
        "origQty": "0.001",
        "executedQty": "0.001",
        "status": "FILLED",
        "timeInForce": "GTC",
        "type": "MARKET",
        "side": "SELL"
    }
    mock_client.order_market_sell.return_value = mock_response
    
    # Test successful order
    order = place_market_sell(SYMBOL, 0.001)
    assert order == mock_response
    mock_client.order_market_sell.assert_called_once()
    
    # Test exception handling
    mock_client.order_market_sell.side_effect = Exception("Test error")
    order = place_market_sell(SYMBOL, 0.001)
    assert order is None

@patch('bot.binance_api.client')
def test_place_limit_buy(mock_client):
    """Test placing a limit buy order."""
    mock_response = {
        "symbol": SYMBOL,
        "orderId": 123456,
        "clientOrderId": "test123",
        "transactTime": 1617000000000,
        "price": "50000.00",
        "origQty": "0.001",
        "executedQty": "0.000",
        "status": "NEW",
        "timeInForce": "GTC",
        "type": "LIMIT",
        "side": "BUY"
    }
    mock_client.order_limit_buy.return_value = mock_response
    
    # Test successful order
    order = place_limit_buy(SYMBOL, 0.001, 50000)
    assert order == mock_response
    mock_client.order_limit_buy.assert_called_once()
    
    # Test exception handling
    mock_client.order_limit_buy.side_effect = Exception("Test error")
    order = place_limit_buy(SYMBOL, 0.001, 50000)
    assert order is None

@patch('bot.binance_api.client')
def test_place_limit_sell(mock_client):
    """Test placing a limit sell order."""
    mock_response = {
        "symbol": SYMBOL,
        "orderId": 123456,
        "clientOrderId": "test123",
        "transactTime": 1617000000000,
        "price": "50000.00",
        "origQty": "0.001",
        "executedQty": "0.000",
        "status": "NEW",
        "timeInForce": "GTC",
        "type": "LIMIT",
        "side": "SELL"
    }
    mock_client.order_limit_sell.return_value = mock_response
    
    # Test successful order
    order = place_limit_sell(SYMBOL, 0.001, 50000)
    assert order == mock_response
    mock_client.order_limit_sell.assert_called_once()
    
    # Test exception handling
    mock_client.order_limit_sell.side_effect = Exception("Test error")
    order = place_limit_sell(SYMBOL, 0.001, 50000)
    assert order is None

@patch('bot.binance_api.client')
def test_get_open_orders(mock_client):
    """Test getting open orders."""
    mock_response = [
        {
            "symbol": SYMBOL,
            "orderId": 123456,
            "clientOrderId": "test123",
            "price": "50000.00",
            "origQty": "0.001",
            "executedQty": "0.000",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY",
            "time": 1617000000000
        }
    ]
    mock_client.get_open_orders.return_value = mock_response
    
    # Test getting all open orders
    orders = get_open_orders()
    assert orders == mock_response
    mock_client.get_open_orders.assert_called_once()
    
    # Test getting open orders for a specific symbol
    mock_client.get_open_orders.reset_mock()
    orders = get_open_orders(SYMBOL)
    assert orders == mock_response
    mock_client.get_open_orders.assert_called_once()
    
    # Test exception handling
    mock_client.get_open_orders.side_effect = Exception("Test error")
    orders = get_open_orders()
    assert orders == []

@patch('bot.binance_api.client')
def test_cancel_order(mock_client):
    """Test canceling an order."""
    mock_response = {
        "symbol": SYMBOL,
        "orderId": 123456,
        "clientOrderId": "test123",
        "price": "50000.00",
        "origQty": "0.001",
        "executedQty": "0.000",
        "status": "CANCELED",
        "timeInForce": "GTC",
        "type": "LIMIT",
        "side": "BUY"
    }
    mock_client.cancel_order.return_value = mock_response
    
    # Test successful cancellation
    result = cancel_order(SYMBOL, 123456)
    assert result == mock_response
    mock_client.cancel_order.assert_called_once()
    
    # Test exception handling
    mock_client.cancel_order.side_effect = Exception("Test error")
    result = cancel_order(SYMBOL, 123456)
    assert result is None

@patch('bot.binance_api.client')
def test_get_order_status(mock_client):
    """Test getting order status."""
    mock_response = {
        "symbol": SYMBOL,
        "orderId": 123456,
        "clientOrderId": "test123",
        "price": "50000.00",
        "origQty": "0.001",
        "executedQty": "0.001",
        "status": "FILLED",
        "timeInForce": "GTC",
        "type": "MARKET",
        "side": "BUY",
        "time": 1617000000000
    }
    mock_client.get_order.return_value = mock_response
    
    # Test getting order status
    order = get_order_status(SYMBOL, 123456)
    assert order == mock_response
    mock_client.get_order.assert_called_once()
    
    # Test exception handling
    mock_client.get_order.side_effect = Exception("Test error")
    order = get_order_status(SYMBOL, 123456)
    assert order is None
