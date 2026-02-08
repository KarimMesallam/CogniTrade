#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api.main as api_main
from bot import database as database_module
from bot.backtesting.config import settings as backtest_settings
from bot.backtesting.data import market_data as market_data_module


@pytest.fixture(scope="function")
def integration_db(tmp_path, monkeypatch):
    """
    Use a real sqlite database on a temporary path for endpoint integration tests.
    """
    db_path = tmp_path / "api_integration.db"
    previous_cache_enabled = backtest_settings.DATABASE_SETTINGS.get("cache_enabled", True)

    class IsolatedDatabase(database_module.Database):
        def __init__(self, _db_path=None):
            super().__init__(str(db_path))

    monkeypatch.setattr(api_main.database, "Database", IsolatedDatabase)
    monkeypatch.setattr(market_data_module, "Database", IsolatedDatabase)
    backtest_settings.DATABASE_SETTINGS["cache_enabled"] = False

    # Keep global API db handle aligned with isolated database class.
    api_main.db = IsolatedDatabase()
    db = IsolatedDatabase()
    yield db

    backtest_settings.DATABASE_SETTINGS["cache_enabled"] = previous_cache_enabled


@pytest.fixture(scope="function")
def client(integration_db):
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_contract(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload.get("version"), str)


def test_database_endpoint_contracts_with_real_db(client, integration_db):
    trade_id = "integration-trade-1"
    trade_ok = integration_db.insert_trade({
        "trade_id": trade_id,
        "symbol": "INTBTCUSDT",
        "side": "BUY",
        "quantity": 0.01,
        "price": 42000.0,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "FILLED",
        "strategy": "integration_sma",
        "raw_data": {"source": "integration-test"},
    })
    assert trade_ok is True

    signal_id = integration_db.insert_signal({
        "symbol": "INTBTCUSDT",
        "timeframe": "1h",
        "strategy": "integration_sma",
        "signal": "BUY",
        "timestamp": datetime.utcnow().isoformat(),
        "indicators": {"rsi": 42.0},
        "llm_decision": "buy",
        "price": 42000.0,
    })
    assert signal_id > 0

    trades_response = client.get("/database/trades?symbol=INTBTCUSDT&limit=10")
    assert trades_response.status_code == 200
    trades_payload = trades_response.json()
    assert trades_payload["status"] == "success"
    assert isinstance(trades_payload["trades"], list)
    assert len(trades_payload["trades"]) == 1
    assert trades_payload["trades"][0]["trade_id"] == trade_id
    assert trades_payload["trades"][0]["symbol"] == "INTBTCUSDT"

    signals_response = client.get("/database/signals?symbol=INTBTCUSDT&strategy=integration_sma&limit=10")
    assert signals_response.status_code == 200
    signals_payload = signals_response.json()
    assert signals_payload["status"] == "success"
    assert isinstance(signals_payload["signals"], list)
    assert len(signals_payload["signals"]) == 1
    assert signals_payload["signals"][0]["signal_id"] == signal_id
    assert signals_payload["signals"][0]["symbol"] == "INTBTCUSDT"


def test_database_endpoint_validation_failures(client):
    invalid_limit_response = client.get("/database/trades?limit=0")
    assert invalid_limit_response.status_code == 422

    invalid_offset_response = client.get("/database/signals?offset=-1")
    assert invalid_offset_response.status_code == 422


def test_llm_decision_rule_based_contract(client):
    request_data = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "market_data": {
            "price": 40000,
            "indicators": {
                "rsi": 35,
                "macd": {"line": 2, "signal": 1, "histogram": 1}
            }
        },
        "context": "Integration test context",
        "strategy_signals": {"sma": "buy", "rsi": "buy"}
    }

    response = client.post("/llm/decision", json=request_data)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["decision"] in {"buy", "sell", "hold"}
    assert isinstance(payload["confidence"], (int, float))
    assert isinstance(payload["reasoning"], str)


def test_backtest_unknown_strategy_failure_contract(client):
    payload = {
        "symbol": "INTUNKNOWNUSDT",
        "timeframes": ["1h"],
        "start_date": "2024-01-01 00:00:00",
        "end_date": "2024-01-10 00:00:00",
        "initial_capital": 10000.0,
        "commission": 0.001,
        "strategy_name": "unknown_strategy",
        "strategy_params": {}
    }

    response = client.post("/backtest/run", json=payload)
    assert response.status_code == 400
    assert "Unknown strategy" in response.json().get("detail", "")


def test_backtest_no_market_data_failure_contract(client):
    payload = {
        "symbol": "INTNODATAUSDT",
        "timeframes": ["1h"],
        "start_date": "2024-01-01 00:00:00",
        "end_date": "2024-01-10 00:00:00",
        "initial_capital": 10000.0,
        "commission": 0.001,
        "strategy_name": "sma_crossover",
        "strategy_params": {"short_period": 10, "long_period": 50}
    }

    response = client.post("/backtest/run", json=payload)
    assert response.status_code == 500
    assert "Error running backtest" in response.json().get("detail", "")


def test_backtest_runs_with_real_seeded_data(client, integration_db):
    symbol = "INTSEEDUSDT"
    timeframe = "1h"
    start_dt = datetime(2024, 1, 1, 0, 0, 0)
    periods = 240

    timestamps = [start_dt + timedelta(hours=i) for i in range(periods)]
    close_prices = [30000 + i * 3 + (i % 8) for i in range(periods)]
    market_data = pd.DataFrame({
        "timestamp": timestamps,
        "open": [price - 5 for price in close_prices],
        "high": [price + 15 for price in close_prices],
        "low": [price - 15 for price in close_prices],
        "close": close_prices,
        "volume": [1000 + i for i in range(periods)],
    })

    assert integration_db.store_market_data(market_data, symbol, timeframe) is True

    payload = {
        "symbol": symbol,
        "timeframes": [timeframe],
        "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": timestamps[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "initial_capital": 10000.0,
        "commission": 0.001,
        "strategy_name": "sma_crossover",
        "strategy_params": {"short_period": 10, "long_period": 50}
    }

    response = client.post("/backtest/run", json=payload)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert isinstance(payload["results"], dict)
    for key in [
        "final_value",
        "profit_loss",
        "profit_loss_percent",
        "sharpe_ratio",
        "max_drawdown",
        "trades",
    ]:
        assert key in payload["results"]
