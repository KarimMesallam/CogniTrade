import os
import sys
import json
import uuid
import pytest
from unittest.mock import patch, mock_open
from datetime import datetime, timedelta

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
        manager = OrderManager(
            "BTCUSDT",
            1.0,
            use_database=False,
            max_order_notional_usd=0,
            max_position_exposure_usd=0
        )
    return manager

class TestOrderManager:
    
    def test_init(self):
        """Test OrderManager initialization."""
        with patch('bot.order_manager.os.path.exists', return_value=False):
            manager = OrderManager(
                "BTCUSDT",
                1.0,
                use_database=False,
                max_order_notional_usd=0,
                max_position_exposure_usd=0
            )
        
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
            manager = OrderManager(
                "BTCUSDT",
                1.0,
                use_database=False,
                max_order_notional_usd=0,
                max_position_exposure_usd=0
            )
            
        assert manager.order_history == mock_history
    
    def test_save_order_history(self):
        """Test saving order history to file."""
        manager = OrderManager(
            "BTCUSDT",
            1.0,
            use_database=False,
            max_order_notional_usd=0,
            max_position_exposure_usd=0
        )
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
    @patch('bot.order_manager.get_account_balance')
    @patch('bot.order_manager.client')
    @patch('bot.order_manager.validate_order_filters', return_value=(True, None))
    def test_execute_market_buy(
        self,
        mock_validate_filters,
        mock_client,
        mock_get_account_balance,
        mock_calculate,
        mock_market_buy,
        order_manager,
        mock_order_data
    ):
        """Test executing a market buy order."""
        mock_client.get_symbol_ticker.return_value = {"price": "50000"}
        mock_get_account_balance.return_value = {"free": 0.0, "locked": 0.0, "total": 0.0}
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
    @patch('bot.order_manager.client')
    @patch('bot.order_manager.validate_order_filters', return_value=(True, None))
    def test_execute_market_sell(
        self,
        mock_validate_filters,
        mock_client,
        mock_market_sell,
        order_manager,
        mock_order_data
    ):
        """Test executing a market sell order."""
        mock_client.get_symbol_ticker.return_value = {"price": "50000"}
        mock_market_sell.return_value = mock_order_data
        
        result = order_manager.execute_market_sell(0.001)
        assert result == mock_order_data
        mock_market_sell.assert_called_with("BTCUSDT", 0.001)
        
        # Test failure case
        mock_market_sell.return_value = None
        result = order_manager.execute_market_sell(0.001)
        assert result is None
    
    @patch('bot.order_manager.place_limit_buy')
    @patch('bot.order_manager.get_account_balance')
    @patch('bot.order_manager.validate_order_filters', return_value=(True, None))
    def test_execute_limit_buy(
        self,
        mock_validate_filters,
        mock_get_account_balance,
        mock_limit_buy,
        order_manager,
        mock_order_data
    ):
        """Test executing a limit buy order."""
        mock_get_account_balance.return_value = {"free": 0.0, "locked": 0.0, "total": 0.0}
        mock_limit_buy.return_value = mock_order_data
        
        result = order_manager.execute_limit_buy(0.001, 50000)
        assert result == mock_order_data
        mock_limit_buy.assert_called_with("BTCUSDT", 0.001, 50000)
        
        # Test failure case
        mock_limit_buy.return_value = None
        result = order_manager.execute_limit_buy(0.001, 50000)
        assert result is None
    
    @patch('bot.order_manager.place_limit_sell')
    @patch('bot.order_manager.validate_order_filters', return_value=(True, None))
    def test_execute_limit_sell(self, mock_validate_filters, mock_limit_sell, order_manager, mock_order_data):
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
        # Disable database integration for this test
        order_manager.use_database = False
        
        # Add an active order to track
        order_manager.active_orders = {
            123: {
                "order_id": 123, 
                "status": "NEW", 
                "action": "BUY"
            }
        }
        
        # Mock the updated status
        updated_order = mock_order_data.copy()
        updated_order["orderId"] = 123
        updated_order["status"] = "FILLED"
        mock_status.return_value = updated_order
        
        # Run the method
        results = order_manager.update_order_statuses()
        
        # Verify the results
        assert 123 in results
        assert 123 not in order_manager.active_orders  # Should be removed when filled
        assert len(order_manager.order_history) >= 1
    
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

    @patch('bot.order_manager.place_market_buy')
    @patch('bot.order_manager.get_account_balance')
    @patch('bot.order_manager.client')
    def test_market_buy_rejected_when_notional_exceeds_max(
        self,
        mock_client,
        mock_get_balance,
        mock_market_buy
    ):
        """Buy should be blocked when order notional exceeds configured max."""
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            max_order_notional_usd=10.0,
            max_position_exposure_usd=1000.0
        )
        mock_client.get_symbol_ticker.return_value = {"price": "50000"}
        mock_get_balance.return_value = {"free": 0.0, "locked": 0.0, "total": 0.0}

        result = manager.execute_market_buy(quantity=0.001)  # ~50 USDT notional

        assert result is None
        assert "Order notional" in manager.last_reject_reason
        mock_market_buy.assert_not_called()

    @patch('bot.order_manager.place_market_buy')
    @patch('bot.order_manager.get_account_balance')
    @patch('bot.order_manager.client')
    def test_market_buy_rejected_when_exposure_exceeds_max(
        self,
        mock_client,
        mock_get_balance,
        mock_market_buy
    ):
        """Buy should be blocked when projected exposure exceeds configured max."""
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            max_order_notional_usd=1000.0,
            max_position_exposure_usd=60.0
        )
        mock_client.get_symbol_ticker.return_value = {"price": "50000"}
        mock_get_balance.return_value = {"free": 0.001, "locked": 0.0, "total": 0.001}

        result = manager.execute_market_buy(quantity=0.001)  # projected exposure ~100 USDT

        assert result is None
        assert "Projected position exposure" in manager.last_reject_reason
        mock_market_buy.assert_not_called()

    @patch('bot.order_manager.place_market_buy')
    @patch('bot.order_manager.validate_order_filters', return_value=(False, "step size mismatch"))
    @patch('bot.order_manager.get_account_balance')
    @patch('bot.order_manager.client')
    def test_market_buy_rejected_when_binance_filters_fail(
        self,
        mock_client,
        mock_get_balance,
        mock_validate_filters,
        mock_market_buy
    ):
        """Buy should be blocked when Binance filter validation fails."""
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            max_order_notional_usd=1000.0,
            max_position_exposure_usd=1000.0
        )
        mock_client.get_symbol_ticker.return_value = {"price": "50000"}
        mock_get_balance.return_value = {"free": 0.0, "locked": 0.0, "total": 0.0}

        result = manager.execute_market_buy(quantity=0.001)

        assert result is None
        assert "Binance filter validation failed" in manager.last_reject_reason
        mock_market_buy.assert_not_called()

    @patch('bot.order_manager.place_market_sell')
    @patch('bot.order_manager.client')
    def test_market_sell_rejected_when_notional_exceeds_max(self, mock_client, mock_market_sell):
        """Sell should be blocked when order notional exceeds configured max."""
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            max_order_notional_usd=10.0,
            max_position_exposure_usd=0.0
        )
        mock_client.get_symbol_ticker.return_value = {"price": "50000"}

        result = manager.execute_market_sell(0.001)  # ~50 USDT notional

        assert result is None
        assert "Order notional" in manager.last_reject_reason
        mock_market_sell.assert_not_called()

    def test_trade_id_is_deterministic_uuid_per_order_id(self, order_manager, mock_order_data):
        """Repeated logs for same Binance order should keep a stable UUID trade_id."""
        order_new = mock_order_data.copy()
        order_new["status"] = "NEW"
        order_filled = mock_order_data.copy()
        order_filled["status"] = "FILLED"

        order_manager._log_order(order_new, "BUY", "NEW")
        first_trade_id = order_new["trade_id"]
        uuid.UUID(first_trade_id)

        order_manager._log_order(order_filled, "BUY", "FILLED")
        second_trade_id = order_filled["trade_id"]

        assert first_trade_id == second_trade_id

    @patch('bot.order_manager.place_futures_market_order')
    @patch('bot.order_manager.get_futures_position_qty')
    @patch('bot.order_manager.get_futures_mark_price')
    @patch('bot.order_manager.calculate_futures_order_quantity')
    def test_execute_market_short_success(
        self,
        mock_calc_futures_qty,
        mock_futures_price,
        mock_futures_position_qty,
        mock_place_futures_order,
        mock_order_data,
    ):
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            trade_mode="FUTURES",
            enable_futures_shorts=True,
            max_order_notional_usd=1000.0,
            max_position_exposure_usd=1000.0,
            max_short_notional_usd=500.0,
            default_futures_leverage=2.0,
            max_short_leverage=5.0,
            min_short_liquidation_buffer_pct=10.0,
        )
        mock_calc_futures_qty.return_value = 0.001
        mock_futures_price.return_value = 50000.0
        mock_futures_position_qty.return_value = 0.0
        mock_place_futures_order.return_value = {
            **mock_order_data,
            "side": "SELL",
            "origQty": "0.001",
            "price": "50000.0",
        }

        result = manager.execute_market_short(quote_amount=25.0, leverage=2.0)

        assert result is not None
        mock_place_futures_order.assert_called_once_with(
            symbol="BTCUSDT",
            side="SELL",
            quantity=0.001,
            reduce_only=False,
            leverage=2.0,
        )

    @patch('bot.order_manager.place_futures_market_order')
    @patch('bot.order_manager.get_futures_position_qty', return_value=0.0)
    @patch('bot.order_manager.get_futures_mark_price', return_value=50000.0)
    def test_market_short_rejected_when_leverage_exceeds_max(
        self,
        mock_futures_price,
        mock_futures_position_qty,
        mock_place_futures_order,
    ):
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            trade_mode="FUTURES",
            enable_futures_shorts=True,
            max_order_notional_usd=1000.0,
            max_position_exposure_usd=1000.0,
            max_short_notional_usd=1000.0,
            max_short_leverage=2.0,
            min_short_liquidation_buffer_pct=5.0,
        )

        result = manager.execute_market_short(quantity=0.001, leverage=5.0)

        assert result is None
        assert "exceeds max" in manager.last_reject_reason
        mock_place_futures_order.assert_not_called()

    @patch('bot.order_manager.place_futures_market_order')
    @patch('bot.order_manager.get_futures_position_qty', return_value=0.0)
    @patch('bot.order_manager.get_futures_mark_price', return_value=50000.0)
    def test_market_short_rejected_when_liquidation_buffer_too_small(
        self,
        mock_futures_price,
        mock_futures_position_qty,
        mock_place_futures_order,
    ):
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            trade_mode="FUTURES",
            enable_futures_shorts=True,
            max_order_notional_usd=1000.0,
            max_position_exposure_usd=1000.0,
            max_short_notional_usd=1000.0,
            max_short_leverage=10.0,
            min_short_liquidation_buffer_pct=30.0,
        )

        result = manager.execute_market_short(quantity=0.001, leverage=5.0)  # 100/5 = 20%

        assert result is None
        assert "liquidation buffer" in manager.last_reject_reason.lower()
        mock_place_futures_order.assert_not_called()

    @patch('bot.order_manager.place_futures_market_order')
    @patch('bot.order_manager.get_futures_position_qty', return_value=0.0)
    @patch('bot.order_manager.get_futures_mark_price', return_value=50000.0)
    def test_market_short_rejected_when_notional_exceeds_short_cap(
        self,
        mock_futures_price,
        mock_futures_position_qty,
        mock_place_futures_order,
    ):
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            trade_mode="FUTURES",
            enable_futures_shorts=True,
            max_order_notional_usd=1000.0,
            max_position_exposure_usd=1000.0,
            max_short_notional_usd=20.0,
            max_short_leverage=5.0,
            min_short_liquidation_buffer_pct=5.0,
        )

        result = manager.execute_market_short(quantity=0.001, leverage=1.0)  # ~50 USDT notional

        assert result is None
        assert "Short order notional" in manager.last_reject_reason
        mock_place_futures_order.assert_not_called()

    @patch('bot.order_manager.place_futures_market_order')
    @patch('bot.order_manager.get_futures_position_qty', return_value=-0.02)
    @patch('bot.order_manager.get_futures_mark_price', return_value=50000.0)
    def test_execute_market_cover_uses_reduce_only_buy(
        self,
        mock_futures_price,
        mock_futures_position_qty,
        mock_place_futures_order,
        mock_order_data,
    ):
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            trade_mode="FUTURES",
            enable_futures_shorts=True,
            max_order_notional_usd=2000.0,
            max_position_exposure_usd=2000.0,
            max_short_notional_usd=2000.0,
        )
        mock_place_futures_order.return_value = {
            **mock_order_data,
            "side": "BUY",
            "origQty": "0.02",
            "price": "50000.0",
        }

        result = manager.execute_market_cover(quantity=0.05)

        assert result is not None
        mock_place_futures_order.assert_called_once_with(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.02,
            reduce_only=True,
        )

    def test_get_recent_turnover_notional_includes_recent_filled_orders(self):
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            max_order_notional_usd=0,
            max_position_exposure_usd=0,
        )
        now = datetime.now()
        manager.order_history = [
            {
                "timestamp": (now - timedelta(minutes=10)).isoformat(),
                "status": "FILLED",
                "quantity": "0.10",
                "price": "50000",
                "fills": [],
            },
            {
                "timestamp": (now - timedelta(minutes=5)).isoformat(),
                "status": "FILLED",
                "quantity": "0.05",
                "price": "51000",
                "fills": [],
            },
            {
                "timestamp": (now - timedelta(hours=2)).isoformat(),
                "status": "FILLED",
                "quantity": "0.10",
                "price": "52000",
                "fills": [],
            },
            {
                "timestamp": now.isoformat(),
                "status": "NEW",
                "quantity": "0.20",
                "price": "53000",
                "fills": [],
            },
        ]

        recent_turnover = manager.get_recent_turnover_notional(window_seconds=3600)
        # 0.10*50000 + 0.05*51000 = 7550
        assert float(recent_turnover) == pytest.approx(7550.0, rel=1e-6)

    def test_exceeds_turnover_limit_sets_reject_reason_when_cap_breached(self):
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            max_order_notional_usd=0,
            max_position_exposure_usd=0,
        )
        now = datetime.now()
        manager.order_history = [
            {
                "timestamp": (now - timedelta(minutes=3)).isoformat(),
                "status": "FILLED",
                "quantity": "0.10",
                "price": "50000",
                "fills": [],
            }
        ]

        exceeds = manager.exceeds_turnover_limit(
            projected_notional=1000.0,
            capital_base_usd=5000.0,
            max_turnover_ratio=1.0,
            window_seconds=3600,
        )

        assert exceeds is True
        assert "Turnover cap exceeded" in manager.last_reject_reason

    def test_exceeds_turnover_limit_returns_false_when_within_cap(self):
        manager = OrderManager(
            "BTCUSDT",
            use_database=False,
            max_order_notional_usd=0,
            max_position_exposure_usd=0,
        )
        now = datetime.now()
        manager.order_history = [
            {
                "timestamp": (now - timedelta(minutes=3)).isoformat(),
                "status": "FILLED",
                "quantity": "0.02",
                "price": "50000",
                "fills": [],
            }
        ]

        exceeds = manager.exceeds_turnover_limit(
            projected_notional=500.0,
            capital_base_usd=5000.0,
            max_turnover_ratio=1.0,
            window_seconds=3600,
        )

        assert exceeds is False
