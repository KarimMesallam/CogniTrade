import os
import sys
import pytest
import time
from unittest.mock import patch, MagicMock, call
from binance.exceptions import BinanceAPIException, BinanceRequestException
from dotenv import load_dotenv

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file before importing modules that use env variables
load_dotenv()

from bot.binance_api import (
    synchronize_time, get_recent_closes, get_account_balance,
    get_symbol_info, calculate_order_quantity, place_market_buy,
    place_market_sell, place_limit_buy, place_limit_sell,
    get_open_orders, cancel_order, get_order_status, time_offset,
    validate_order_filters, reset_exchange_circuit_breaker,
    get_futures_mark_price, get_futures_position_qty,
    calculate_futures_order_quantity, place_futures_market_order
)
import bot.binance_api as binance_api
from bot.config import SYMBOL


@pytest.fixture(autouse=True)
def isolate_exchange_resilience(monkeypatch):
    """Reset circuit breaker and disable retries by default in unit tests."""
    reset_exchange_circuit_breaker()
    monkeypatch.setattr(binance_api, "EXCHANGE_MAX_RETRIES", 0)
    monkeypatch.setattr(binance_api, "EXCHANGE_RETRY_BACKOFF_SECONDS", 0.0)
    yield
    reset_exchange_circuit_breaker()

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


@patch('bot.binance_api.client')
def test_get_futures_mark_price(mock_client):
    """Test retrieving futures mark price."""
    mock_client.futures_mark_price.return_value = {"symbol": SYMBOL, "markPrice": "50000.12"}
    mark_price = get_futures_mark_price(SYMBOL)
    assert mark_price == 50000.12


@patch('bot.binance_api.client')
def test_get_futures_position_qty(mock_client):
    """Test retrieving signed futures position quantity."""
    mock_client.futures_position_information.return_value = [{"symbol": SYMBOL, "positionAmt": "-0.015"}]
    position_qty = get_futures_position_qty(SYMBOL)
    assert position_qty == -0.015


@patch('bot.binance_api.client')
@patch('bot.binance_api.get_futures_mark_price')
def test_calculate_futures_order_quantity_with_mock(mock_mark_price, mock_client):
    """Futures quantity calculation should apply leverage and lot-size rounding."""
    mock_mark_price.return_value = 50000.0
    mock_client.futures_exchange_info.return_value = {
        "symbols": [
            {
                "symbol": SYMBOL,
                "filters": [
                    {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"}
                ],
            }
        ]
    }

    qty = calculate_futures_order_quantity(SYMBOL, quote_amount=25.0, leverage=2.0)
    assert qty == 0.001


@patch('bot.binance_api.client')
def test_place_futures_market_order(mock_client):
    """Futures market order should support leverage setup and reduce-only mode."""
    mock_client.futures_change_leverage.return_value = {"leverage": 3}
    mock_client.futures_create_order.return_value = {
        "orderId": 123,
        "symbol": SYMBOL,
        "side": "SELL",
        "type": "MARKET",
        "origQty": "0.001",
        "status": "FILLED",
    }

    order = place_futures_market_order(
        symbol=SYMBOL,
        side="SELL",
        quantity=0.001,
        reduce_only=False,
        leverage=3.0,
    )

    assert order is not None
    mock_client.futures_change_leverage.assert_called_once()
    mock_client.futures_create_order.assert_called_once()

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
def test_validate_order_filters(mock_client):
    """Validate success and rejection paths for Binance pre-trade filters."""
    mock_client.get_symbol_info.return_value = {
        "symbol": SYMBOL,
        "baseAsset": "BTC",
        "filters": [
            {"filterType": "LOT_SIZE", "minQty": "0.00010000", "maxQty": "100.00000000", "stepSize": "0.00010000"},
            {"filterType": "PRICE_FILTER", "minPrice": "100.00", "maxPrice": "1000000.00", "tickSize": "0.10"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "10.00", "applyToMarket": True},
        ],
    }
    mock_client.get_symbol_ticker.return_value = {"symbol": SYMBOL, "price": "50000.00"}

    is_valid, reason = validate_order_filters(
        symbol=SYMBOL,
        side="BUY",
        order_type="LIMIT",
        quantity=0.001,
        price=50000.0,
    )
    assert is_valid is True
    assert reason is None

    is_valid, reason = validate_order_filters(
        symbol=SYMBOL,
        side="BUY",
        order_type="LIMIT",
        quantity=0.001,
        price=50000.05,
    )
    assert is_valid is False
    assert "tickSize" in reason

    is_valid, reason = validate_order_filters(
        symbol=SYMBOL,
        side="BUY",
        order_type="MARKET",
        quantity=0.0001,
    )
    assert is_valid is False
    assert "MIN_NOTIONAL" in reason or "Notional" in reason

@patch('bot.binance_api.validate_order_filters', return_value=(True, None))
@patch('bot.binance_api.client')
def test_place_market_buy(mock_client, mock_validate):
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

@patch('bot.binance_api.validate_order_filters', return_value=(True, None))
@patch('bot.binance_api.client')
def test_place_market_sell(mock_client, mock_validate):
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

@patch('bot.binance_api.validate_order_filters', return_value=(True, None))
@patch('bot.binance_api.client')
def test_place_limit_buy(mock_client, mock_validate):
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

@patch('bot.binance_api.validate_order_filters', return_value=(True, None))
@patch('bot.binance_api.client')
def test_place_limit_sell(mock_client, mock_validate):
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


@patch('bot.binance_api.validate_order_filters', return_value=(False, "step size mismatch"))
@patch('bot.binance_api.client')
def test_place_limit_buy_blocks_invalid_filter(mock_client, mock_validate):
    """Order placement should be blocked before exchange call when validation fails."""
    order = place_limit_buy(SYMBOL, 0.00123, 50000)
    assert order is None
    mock_client.order_limit_buy.assert_not_called()

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

# New tests for specific Binance exceptions and edge cases

@patch('bot.binance_api.validate_order_filters', return_value=(True, None))
@patch('bot.binance_api.client')
def test_binance_api_exception_handling(mock_client, mock_validate):
    """Test handling of specific Binance API exceptions."""
    # Test BinanceAPIException in get_account_balance
    mock_client.get_account.side_effect = BinanceAPIException(
        status_code=400,
        text='{"code": -1022, "msg": "Signature for this request is not valid."}',
        response=None
    )
    result = get_account_balance()
    assert result is None
    
    # Test BinanceRequestException in get_symbol_info
    mock_client.get_symbol_info.side_effect = BinanceRequestException(message="Connection error")
    result = get_symbol_info(SYMBOL)
    assert result is None
    
    # Test with real Binance exceptions in order operations
    api_exception = BinanceAPIException(
        status_code=400,
        text='{"code": -2010, "msg": "Account has insufficient balance for requested action."}',
        response=None
    )
    
    # Test place_market_buy with API exception
    mock_client.order_market_buy.side_effect = api_exception
    result = place_market_buy(SYMBOL, 0.001)
    assert result is None
    
    # Test place_market_sell with API exception
    mock_client.order_market_sell.side_effect = api_exception
    result = place_market_sell(SYMBOL, 0.001)
    assert result is None
    
    # Test cancel_order with API exception
    mock_client.cancel_order.side_effect = api_exception
    result = cancel_order(SYMBOL, 123456)
    assert result is None

@patch('bot.binance_api.client')
@patch('bot.binance_api.time')
def test_synchronize_time_detailed(mock_time, mock_client):
    """Test time synchronization in detail."""
    # Test successful time synchronization
    mock_time.time.return_value = 1617000000.0  # Mock local time
    mock_client.get_server_time.return_value = {"serverTime": 1617000100000}  # Server time with 100s difference
    
    offset = synchronize_time()
    assert offset == 100000  # 100s difference in milliseconds
    assert mock_client.timestamp_offset == 100000
    
    # Test exception handling in synchronize_time
    mock_client.get_server_time.side_effect = BinanceAPIException(
        status_code=500,
        text='{"code": -1000, "msg": "An unknown error occurred while processing the request."}',
        response=None
    )
    offset = synchronize_time()
    assert offset == 0  # Should return 0 on error
    
    # Test with a connection error
    mock_client.get_server_time.side_effect = BinanceRequestException(message="Connection refused")
    offset = synchronize_time()
    assert offset == 0  # Should return 0 on connection error

@patch('bot.binance_api.client')
def test_edge_cases(mock_client):
    """Test edge cases and additional error scenarios."""
    # Test get_recent_closes with API exception
    mock_client.get_klines.side_effect = BinanceAPIException(
        status_code=400,
        text='{"code": -1121, "msg": "Invalid symbol."}',
        response=None
    )
    result = get_recent_closes("INVALID", "1h")
    assert result == []
    
    # Test get_account_balance with malformed response
    mock_client.get_account.return_value = {}  # Missing 'balances' key
    result = get_account_balance()
    assert result is None
    
    mock_client.get_account.return_value = {'balances': []}  # Empty balances
    result = get_account_balance('BTC')
    assert result is None
    
    # Test calculate_order_quantity with invalid inputs
    # Test when price ticker is missing
    mock_client.get_symbol_ticker.return_value = {}
    result = calculate_order_quantity(SYMBOL, 100.0)
    assert result is None
    
    # Test with malformed symbol info
    mock_client.get_symbol_ticker.return_value = {"symbol": SYMBOL, "price": "50000.00"}
    mock_client.get_symbol_info.return_value = None
    result = calculate_order_quantity(SYMBOL, 100.0)
    assert result is None


@patch('bot.binance_api.time.sleep', return_value=None)
@patch('bot.binance_api.client')
def test_exchange_retry_and_circuit_breaker(mock_client, mock_sleep):
    """Exchange calls should retry on transient errors and then open circuit on repeated failures."""
    reset_exchange_circuit_breaker()
    original_threshold = binance_api.EXCHANGE_CIRCUIT_BREAKER.failure_threshold
    original_cooldown = binance_api.EXCHANGE_CIRCUIT_BREAKER.cooldown_seconds

    try:
        # Retry path: one transient failure then success.
        with patch('bot.binance_api.EXCHANGE_MAX_RETRIES', 1):
            mock_client.get_symbol_info.side_effect = [
                BinanceRequestException(message="temporary issue"),
                {"symbol": SYMBOL, "filters": []},
            ]
            symbol_info = get_symbol_info(SYMBOL)
            assert symbol_info is not None
            assert symbol_info["symbol"] == SYMBOL
            assert mock_client.get_symbol_info.call_count == 2

        # Circuit-breaker path: open after first failure and block subsequent call.
        reset_exchange_circuit_breaker()
        binance_api.EXCHANGE_CIRCUIT_BREAKER.failure_threshold = 1
        binance_api.EXCHANGE_CIRCUIT_BREAKER.cooldown_seconds = 999.0
        with patch('bot.binance_api.EXCHANGE_MAX_RETRIES', 0):
            mock_client.get_symbol_info.reset_mock()
            mock_client.get_symbol_info.side_effect = BinanceRequestException(message="down")
            assert get_symbol_info(SYMBOL) is None
            assert get_symbol_info(SYMBOL) is None
            assert mock_client.get_symbol_info.call_count == 1
    finally:
        binance_api.EXCHANGE_CIRCUIT_BREAKER.failure_threshold = original_threshold
        binance_api.EXCHANGE_CIRCUIT_BREAKER.cooldown_seconds = original_cooldown
        reset_exchange_circuit_breaker()
