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
    get_symbol_info, calculate_order_quantity
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
