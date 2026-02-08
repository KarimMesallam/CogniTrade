import os
import sys
from unittest.mock import patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.order_manager import OrderManager
from bot.reconciliation import ExchangeReconciler


def test_exchange_reconciler_detects_stale_and_missing_orders():
    reconciler = ExchangeReconciler(symbol="BTCUSDT", trade_mode="SPOT")
    report = reconciler.reconcile_orders(
        local_active_orders={
            1: {"order_id": 1, "status": "NEW"},
            2: {"order_id": 2, "status": "NEW"},
        },
        exchange_open_orders=[
            {"orderId": 2, "status": "NEW"},
            {"orderId": 3, "status": "NEW"},
        ],
    )

    assert report.stale_local_order_ids == ["1"]
    assert report.missing_local_order_ids == ["3"]
    assert report.synced_local_order_ids == ["2"]


def test_exchange_reconciler_position_mismatch_detection():
    reconciler = ExchangeReconciler(symbol="BTCUSDT", trade_mode="FUTURES")
    mismatch, delta = reconciler.reconcile_position(
        local_position_qty=0.1000,
        exchange_position_qty=0.1002,
        tolerance=0.00005,
    )
    assert mismatch is True
    assert abs(delta - 0.0002) < 1e-8


@patch("bot.order_manager.os.path.exists", return_value=False)
@patch("bot.order_manager.get_open_orders")
def test_order_manager_recover_state_from_exchange(mock_get_open_orders, _mock_exists):
    mock_get_open_orders.return_value = [
        {
            "orderId": 11,
            "clientOrderId": "abc",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "0.001",
            "price": "50000",
            "status": "NEW",
        }
    ]
    manager = OrderManager("BTCUSDT", use_database=False, max_order_notional_usd=0, max_position_exposure_usd=0)

    report = manager.recover_state_from_exchange()

    assert report["recovered"] == 1
    assert 11 in manager.active_orders
    assert manager.active_orders[11]["status"] == "NEW"
    mock_get_open_orders.assert_called_once_with("BTCUSDT", trade_mode="SPOT")


@patch("bot.order_manager.os.path.exists", return_value=False)
@patch("bot.order_manager.get_order_status")
@patch("bot.order_manager.get_open_orders")
def test_order_manager_reconcile_with_exchange_syncs_state(
    mock_get_open_orders,
    mock_get_order_status,
    _mock_exists,
):
    manager = OrderManager("BTCUSDT", use_database=False, max_order_notional_usd=0, max_position_exposure_usd=0)
    manager.active_orders = {
        100: {"order_id": 100, "action": "BUY", "status": "NEW"},
        200: {"order_id": 200, "action": "BUY", "status": "NEW"},
    }

    # Exchange has order 200 (synced) and order 300 (missing locally).
    mock_get_open_orders.return_value = [
        {
            "orderId": 200,
            "clientOrderId": "sync",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "0.001",
            "price": "50000",
            "status": "NEW",
        },
        {
            "orderId": 300,
            "clientOrderId": "new",
            "side": "SELL",
            "type": "LIMIT",
            "origQty": "0.002",
            "price": "50500",
            "status": "NEW",
        },
    ]
    mock_get_order_status.return_value = {
        "orderId": 100,
        "status": "FILLED",
        "side": "BUY",
        "origQty": "0.001",
        "price": "50000",
    }

    with patch.object(manager, "_log_order") as mock_log, patch.object(
        manager,
        "_get_current_position_qty",
        return_value=0.0,
    ):
        report = manager.reconcile_with_exchange(position_tolerance=0.0001)

    assert "100" in report["stale_local_order_ids"]
    assert "300" in report["missing_local_order_ids"]
    assert 300 in manager.active_orders
    mock_log.assert_called_once()
    mock_get_order_status.assert_called_once_with("BTCUSDT", 100, trade_mode="SPOT")

