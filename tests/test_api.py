#!/usr/bin/env python3
import os
import sys
import pytest
import json
from decimal import Decimal
from types import SimpleNamespace
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API and bot modules
from api.main import app
from bot import binance_api, strategy, order_manager, database, config, llm_manager
from bot.backtesting import run_backtest, generate_report, optimize_strategy
from bot.backtesting.models.results import BacktestResult, PerformanceMetrics
import api.main as api_main

# Create test client - updated for newer FastAPI/starlette versions
client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_trading_state():
    """Ensure trading bot global state is isolated between API tests."""
    api_main.trading_bot = None
    api_main.trading_task = None
    if hasattr(api_main, "observability"):
        api_main.observability.reset()
    if hasattr(api_main, "api_security"):
        api_main.api_security.update_config(config.get_api_security_config())
        api_main.api_security.reset_runtime_state()
    if hasattr(api_main, "rollout_gate"):
        api_main.rollout_gate.load_state({})
    yield
    api_main.trading_bot = None
    api_main.trading_task = None
    if hasattr(api_main, "observability"):
        api_main.observability.reset()
    if hasattr(api_main, "api_security"):
        api_main.api_security.update_config(config.get_api_security_config())
        api_main.api_security.reset_runtime_state()
    if hasattr(api_main, "rollout_gate"):
        api_main.rollout_gate.load_state({})

# Fixture for mocking the database
@pytest.fixture
def mock_db():
    with patch('api.main.database.Database') as mock:
        db_instance = MagicMock()
        mock.return_value = db_instance
        yield db_instance

# Fixture for mocking binance API client
@pytest.fixture
def mock_binance_client():
    with patch('bot.binance_api.get_client') as mock:
        client_mock = MagicMock()
        mock.return_value = client_mock
        yield client_mock

# Fixture for mocking background tasks
@pytest.fixture
def mock_background_tasks():
    with patch('fastapi.BackgroundTasks.add_task') as mock:
        yield mock

# Fixture for mocking LLM manager
@pytest.fixture
def mock_llm_manager():
    with patch('bot.llm_manager.LLMManager') as mock:
        manager_mock = MagicMock()
        mock.return_value = manager_mock
        yield manager_mock

# Test API health check
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "version" in response.json()


def test_cors_defaults_do_not_use_wildcard_origin():
    cors_middlewares = [entry for entry in app.user_middleware if entry.cls.__name__ == "CORSMiddleware"]
    assert cors_middlewares
    allow_origins = cors_middlewares[0].kwargs.get("allow_origins", [])
    assert "*" not in allow_origins


def test_request_id_header_is_set_and_propagated():
    generated = client.get("/health")
    assert generated.status_code == 200
    assert "X-Request-ID" in generated.headers
    assert generated.headers["X-Request-ID"]

    provided = client.get("/health", headers={"X-Request-ID": "req-123"})
    assert provided.status_code == 200
    assert provided.headers["X-Request-ID"] == "req-123"


def test_telemetry_dashboard_and_events_endpoints():
    client.get("/health")
    client.get("/health")

    dashboard_response = client.get("/observability/dashboard?window_minutes=60")
    assert dashboard_response.status_code == 200
    assert dashboard_response.json()["status"] == "success"
    dashboard = dashboard_response.json()["dashboard"]
    assert dashboard["trace_count"] >= 1.0
    assert dashboard["latency"]["count"] >= 1.0

    events_response = client.get("/observability/events?limit=10")
    assert events_response.status_code == 200
    assert events_response.json()["status"] == "success"
    events = events_response.json()["events"]
    assert len(events) > 0
    assert "event_type" in events[-1]


def test_rollout_quality_endpoint():
    summary = {
        "walk_forward_folds": 4,
        "walk_forward_summary": {
            "sharpe_ratio": {"mean": 0.3},
            "calmar_ratio": {"mean": 0.2},
            "max_drawdown_pct": {"mean": -12.0},
        },
        "regime_slices": {
            "BEAR": {"sample_count": 30.0, "win_rate": 0.5},
            "SIDEWAYS": {"sample_count": 25.0, "win_rate": 0.48},
        },
    }
    response = client.post("/rollout/quality/evaluate", json={"validation_summary": summary})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert "quality_gate" in payload
    assert payload["quality_gate"]["passed"] is True


@patch("api.main._persist_rollout_gate_state")
def test_rollout_stage_endpoints(mock_persist):
    rollout_id = "api-rollout-demo"
    shadow = client.post(
        "/rollout/evaluate/shadow",
        json={"rollout_id": rollout_id, "sample_count": 100, "error_rate": 0.02},
    )
    assert shadow.status_code == 200
    assert shadow.json()["decision"]["approved"] is True

    canary = client.post(
        "/rollout/evaluate/canary",
        json={
            "rollout_id": rollout_id,
            "sample_count": 60,
            "error_rate": 0.03,
            "drawdown_pct": 2.0,
            "total_return_pct": 1.2,
            "latency_p95_ms": 700.0,
        },
    )
    assert canary.status_code == 200
    assert canary.json()["decision"]["approved"] is True

    production = client.post(
        "/rollout/evaluate/production",
        json={
            "rollout_id": rollout_id,
            "sample_count": 60,
            "error_rate": 0.03,
            "drawdown_pct": 2.0,
            "total_return_pct": 1.2,
            "latency_p95_ms": 700.0,
            "metadata": {"quality_gate": {"passed": True, "reasons": []}},
        },
    )
    assert production.status_code == 200
    assert production.json()["decision"]["approved"] is True

    status = client.get(f"/rollout/status/{rollout_id}")
    assert status.status_code == 200
    assert status.json()["rollout"]["production_passed"] is True
    assert mock_persist.called


@patch("api.main._persist_rollout_gate_state")
def test_rollout_production_rejects_without_quality_gate(mock_persist):
    rollout_id = "api-rollout-quality-missing"
    client.post(
        "/rollout/evaluate/shadow",
        json={"rollout_id": rollout_id, "sample_count": 100, "error_rate": 0.02},
    )
    client.post(
        "/rollout/evaluate/canary",
        json={
            "rollout_id": rollout_id,
            "sample_count": 60,
            "error_rate": 0.03,
            "drawdown_pct": 2.0,
            "total_return_pct": 1.2,
            "latency_p95_ms": 700.0,
        },
    )
    production = client.post(
        "/rollout/evaluate/production",
        json={
            "rollout_id": rollout_id,
            "sample_count": 60,
            "error_rate": 0.03,
            "drawdown_pct": 2.0,
            "total_return_pct": 1.2,
            "latency_p95_ms": 700.0,
        },
    )
    assert production.status_code == 200
    payload = production.json()
    assert payload["decision"]["approved"] is False
    assert "missing_quality_gate_evidence" in payload["decision"]["reasons"]


def test_api_auth_blocks_missing_key_when_enabled():
    api_main.api_security.update_config(
        {
            "auth_enabled": True,
            "allow_public_health": False,
            "read_api_keys": ["read-key"],
            "admin_api_keys": ["admin-key"],
            "rate_limit_enabled": False,
            "rate_limit_requests": 100,
            "rate_limit_window_seconds": 60,
        }
    )

    response = client.get("/observability/events?limit=1")
    assert response.status_code == 401
    assert "detail" in response.json()


def test_api_auth_read_key_cannot_access_write_endpoint(mock_background_tasks):
    api_main.api_security.update_config(
        {
            "auth_enabled": True,
            "allow_public_health": True,
            "read_api_keys": ["read-key"],
            "admin_api_keys": ["admin-key"],
            "rate_limit_enabled": False,
            "rate_limit_requests": 100,
            "rate_limit_window_seconds": 60,
        }
    )

    trading_config = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "trade_amount": 100.0,
        "strategies": [{"name": "sma_crossover", "params": {"short_period": 10, "long_period": 50}, "active": True}],
    }

    response = client.post("/trading/start", json=trading_config, headers={"X-API-Key": "read-key"})
    assert response.status_code == 403
    assert "detail" in response.json()


def test_api_auth_admin_key_can_access_write_endpoint(mock_background_tasks):
    api_main.api_security.update_config(
        {
            "auth_enabled": True,
            "allow_public_health": True,
            "read_api_keys": ["read-key"],
            "admin_api_keys": ["admin-key"],
            "rate_limit_enabled": False,
            "rate_limit_requests": 100,
            "rate_limit_window_seconds": 60,
        }
    )

    trading_config = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "trade_amount": 100.0,
        "strategies": [{"name": "sma_crossover", "params": {"short_period": 10, "long_period": 50}, "active": True}],
    }

    response = client.post("/trading/start", json=trading_config, headers={"X-API-Key": "admin-key"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_api_rate_limit_blocks_excess_requests():
    api_main.api_security.update_config(
        {
            "auth_enabled": False,
            "allow_public_health": True,
            "read_api_keys": [],
            "admin_api_keys": [],
            "rate_limit_enabled": True,
            "rate_limit_requests": 2,
            "rate_limit_window_seconds": 60,
        }
    )

    first = client.get("/health")
    second = client.get("/health")
    third = client.get("/health")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "Retry-After" in third.headers

# Test start trading endpoint
def test_start_trading(mock_background_tasks):
    trading_config = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "trade_amount": 100.0,
        "strategies": [
            {
                "name": "sma_crossover",
                "params": {
                    "short_period": 10,
                    "long_period": 50
                },
                "active": True
            }
        ]
    }
    
    response = client.post("/trading/start", json=trading_config)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["message"] == "Trading bot started"
    mock_background_tasks.assert_called_once()


def test_start_trading_rejects_when_already_running():
    trading_config = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "trade_amount": 100.0,
        "strategies": [
            {
                "name": "sma_crossover",
                "params": {"short_period": 10, "long_period": 50},
                "active": True
            }
        ]
    }

    first = client.post("/trading/start", json=trading_config)
    second = client.post("/trading/start", json=trading_config)

    assert first.status_code == 200
    assert second.status_code == 400
    assert second.json()["detail"] == "Trading bot is already running"

# Test stop trading endpoint
def test_stop_trading():
    # First we need to start the bot
    trading_config = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "trade_amount": 100.0,
        "strategies": [
            {
                "name": "sma_crossover",
                "params": {
                    "short_period": 10,
                    "long_period": 50
                },
                "active": True
            }
        ]
    }
    
    client.post("/trading/start", json=trading_config)
    
    # Then we can stop it
    response = client.post("/trading/stop")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["message"] == "Trading bot stopped"


def test_stop_trading_when_not_running_returns_400():
    response = client.post("/trading/stop")
    assert response.status_code == 400
    assert response.json()["detail"] == "Trading bot is not running"

# Test get account info endpoint
def test_get_account_info(mock_binance_client):
    # Mock the account info response
    mock_binance_client.get_account.return_value = {
        "balances": [
            {"asset": "BTC", "free": "0.5", "locked": "0.0"},
            {"asset": "ETH", "free": "5.0", "locked": "0.0"},
            {"asset": "USDT", "free": "10000.0", "locked": "0.0"},
            {"asset": "BNB", "free": "0.0", "locked": "0.0"}  # Should be filtered out
        ]
    }
    
    response = client.get("/account/info")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(response.json()["balances"]) == 3  # BNB with 0 balance should be filtered out
    
    # Verify the exception handling by making the mock raise an exception
    mock_binance_client.get_account.side_effect = Exception("API error")
    response = client.get("/account/info")
    
    assert response.status_code == 502
    assert "detail" in response.json()

# Test get symbols endpoint
def test_get_symbols(mock_binance_client):
    # Mock the exchange info response
    mock_binance_client.get_exchange_info.return_value = {
        "symbols": [
            {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "XRPUSDT", "baseAsset": "XRP", "quoteAsset": "USDT", "status": "HALT"}  # Should be filtered out
        ]
    }
    
    response = client.get("/market/symbols")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(response.json()["symbols"]) == 2  # XRPUSDT should be filtered out
    
    # Verify the exception handling
    mock_binance_client.get_exchange_info.side_effect = Exception("API error")
    response = client.get("/market/symbols")
    
    assert response.status_code == 502
    assert "detail" in response.json()

# Test get market data endpoint
def test_get_market_data(mock_binance_client):
    # Mock the klines response
    mock_binance_client.get_klines.return_value = [
        [1625097600000, "35000.0", "36000.0", "34500.0", "35500.0", "100.0", 1625097899999, "3500000.0", 100, "50.0", "1750000.0", "0.0"],
        [1625097900000, "35500.0", "36500.0", "35000.0", "36000.0", "120.0", 1625098199999, "4200000.0", 120, "60.0", "2100000.0", "0.0"]
    ]
    
    response = client.get("/market/data/BTCUSDT/1h?limit=2")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(response.json()["candles"]) == 2
    
    # Verify the data transformation
    candle = response.json()["candles"][0]
    assert "time" in candle
    assert "open" in candle
    assert "high" in candle
    assert "low" in candle
    assert "close" in candle
    assert "volume" in candle
    
    # Verify the exception handling
    mock_binance_client.get_klines.side_effect = Exception("API error")
    response = client.get("/market/data/BTCUSDT/1h?limit=2")
    
    assert response.status_code == 502
    assert "detail" in response.json()

# Test get strategies endpoint
def test_get_strategies():
    response = client.get("/strategies")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "strategies" in response.json()
    assert len(response.json()["strategies"]) > 0
    
    # Verify the structure of a strategy
    strategy = response.json()["strategies"][0]
    assert "name" in strategy
    assert "display_name" in strategy
    assert "description" in strategy
    assert "parameters" in strategy
    
    # Verify that LLM strategy is included
    llm_strategy = None
    strategy_names = set()
    for s in response.json()["strategies"]:
        strategy_names.add(s["name"])
        if s["name"] == "llm_strategy":
            llm_strategy = s
    
    assert llm_strategy is not None
    assert "llm_model" in llm_strategy["parameters"]
    assert "base_strategy" in llm_strategy["parameters"]
    for required in {"ema_crossover", "donchian_breakout", "bear_rally_short", "regime_switch_adaptive"}:
        assert required in strategy_names

    base_options = set(llm_strategy["parameters"]["base_strategy"]["options"])
    for required in {"ema_crossover", "donchian_breakout", "bear_rally_short", "regime_switch_adaptive"}:
        assert required in base_options

# Test run backtest endpoint
@patch('api.main.run_backtest')
def test_run_backtest(mock_run_backtest):
    # Create a proper mock for BacktestResult and its metrics
    mock_metrics = MagicMock(spec=PerformanceMetrics)
    mock_metrics.total_return_pct = 20.0
    mock_metrics.sharpe_ratio = 1.5
    mock_metrics.max_drawdown_pct = -10.0
    mock_metrics.win_rate = 60.0
    mock_metrics.avg_win = 3.0
    mock_metrics.avg_loss = 2.0
    mock_metrics.profit_factor = 1.5
    
    mock_result = MagicMock(spec=BacktestResult)
    mock_result.symbol = "BTCUSDT"
    mock_result.timeframes = ["1h", "4h"]
    mock_result.start_date = "2023-01-01"
    mock_result.end_date = "2023-01-31"
    mock_result.initial_capital = 10000.0
    mock_result.final_equity = 12000.0
    mock_result.total_trades = 2
    mock_result.metrics = mock_metrics
    
    # Configure the mock to return our BacktestResult
    mock_run_backtest.return_value = mock_result
    
    backtest_config = {
        "symbol": "BTCUSDT",
        "timeframes": ["1h", "4h"],
        "start_date": "2023-01-01",
        "end_date": "2023-01-31",
        "initial_capital": 10000.0,
        "commission": 0.001,
        "strategy_name": "sma_crossover",
        "strategy_params": {
            "short_period": 10,
            "long_period": 50
        }
    }
    
    response = client.post("/backtest/run", json=backtest_config)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "results" in response.json()
    
    results = response.json()["results"]
    # Verify we're getting proper values, not random ones
    assert "profit_loss_percent" in results
    assert "sharpe_ratio" in results
    assert isinstance(results["profit_loss_percent"], (int, float))
    assert isinstance(results["sharpe_ratio"], (int, float))
    
    # Test with RSI strategy
    backtest_config["strategy_name"] = "rsi"
    backtest_config["strategy_params"] = {
        "period": 14,
        "overbought": 70,
        "oversold": 30
    }
    
    response = client.post("/backtest/run", json=backtest_config)
    assert response.status_code == 200

    # Test with EMA crossover strategy
    backtest_config["strategy_name"] = "ema_crossover"
    backtest_config["strategy_params"] = {
        "fast_period": 12,
        "slow_period": 50
    }
    response = client.post("/backtest/run", json=backtest_config)
    assert response.status_code == 200

    # Test with Donchian breakout strategy
    backtest_config["strategy_name"] = "donchian_breakout"
    backtest_config["strategy_params"] = {
        "lookback_period": 20,
        "breakout_buffer_bps": 0.0
    }
    response = client.post("/backtest/run", json=backtest_config)
    assert response.status_code == 200

    # Test with bear short strategy
    backtest_config["strategy_name"] = "bear_rally_short"
    backtest_config["strategy_params"] = {
        "fast_period": 8,
        "slow_period": 30,
        "rsi_period": 14,
        "rsi_overbought": 65,
        "rsi_oversold": 35
    }
    response = client.post("/backtest/run", json=backtest_config)
    assert response.status_code == 200

    # Test with regime switch adaptive strategy
    backtest_config["strategy_name"] = "regime_switch_adaptive"
    backtest_config["strategy_params"] = {
        "regime_lookback_candles": 50,
        "trend_threshold_pct": 0.02,
        "sideways_threshold_pct": 0.01,
        "high_volatility_threshold_pct": 0.015,
        "bull_short_period": 8,
        "bull_long_period": 50,
        "bear_fast_period": 8,
        "bear_slow_period": 30,
        "bear_rsi_period": 14,
        "bear_rsi_overbought": 60,
        "bear_rsi_oversold": 30,
        "sideways_rsi_period": 21,
        "sideways_rsi_overbought": 65,
        "sideways_rsi_oversold": 35,
        "high_vol_lookback_period": 20,
        "high_vol_breakout_buffer_bps": 5.0
    }
    response = client.post("/backtest/run", json=backtest_config)
    assert response.status_code == 200

    # Test regime-switch strategy validation
    backtest_config["strategy_name"] = "regime_switch_adaptive"
    backtest_config["strategy_params"] = {
        "trend_threshold_pct": 0.01,
        "sideways_threshold_pct": 0.02
    }
    response = client.post("/backtest/run", json=backtest_config)
    assert response.status_code == 400
    assert "Invalid regime-switch parameters" in response.json().get("detail", "")
    
    # Test with unknown strategy
    backtest_config["strategy_name"] = "unknown_strategy"
    response = client.post("/backtest/run", json=backtest_config)
    # API should now return an error for unknown strategy
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "Unknown strategy" in response.json()["detail"]


@patch("api.main.db")
@patch("api.main.run_backtest")
def test_run_backtest_supports_short_execution_and_validation(
    mock_run_backtest,
    mock_db,
):
    mock_metrics = MagicMock(spec=PerformanceMetrics)
    mock_metrics.total_return_pct = 5.0
    mock_metrics.sharpe_ratio = 0.4
    mock_metrics.max_drawdown_pct = -8.0
    mock_metrics.win_rate = 55.0
    mock_metrics.avg_win = 2.5
    mock_metrics.avg_loss = 1.8
    mock_metrics.profit_factor = 1.2

    mock_result = MagicMock(spec=BacktestResult)
    mock_result.final_equity = 10500.0
    mock_result.total_trades = 3
    mock_result.metrics = mock_metrics
    mock_result.equity_curve = []
    mock_run_backtest.return_value = mock_result
    mock_db.get_market_data.return_value = pd.DataFrame()

    payload = {
        "symbol": "BTCUSDT",
        "timeframes": ["1h"],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "initial_capital": 10000.0,
        "commission": 0.001,
        "strategy_name": "sma_crossover",
        "strategy_params": {"short_period": 10, "long_period": 50},
        "trade_mode": "FUTURES",
        "execution_simulation": {"enabled": True, "spread_bps": 5.0},
        "run_walk_forward_validation": True,
        "include_regime_slices": True,
    }

    response = client.post("/backtest/run", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "validation" in body["results"]
    assert "quality_gate" in body["results"]

    kwargs = mock_run_backtest.call_args.kwargs
    assert kwargs["allow_short_positions"] is True
    assert kwargs["execution_simulation"]["enabled"] is True


@patch("api.main.MarketRegimeDetector")
@patch("api.main.db")
@patch("api.main.run_backtest")
def test_run_backtest_regime_slices_ignore_flat_returns(
    mock_run_backtest,
    mock_db,
    mock_regime_detector,
):
    mock_metrics = MagicMock(spec=PerformanceMetrics)
    mock_metrics.total_return_pct = 1.0
    mock_metrics.sharpe_ratio = 0.4
    mock_metrics.max_drawdown_pct = -5.0
    mock_metrics.win_rate = 50.0
    mock_metrics.avg_win = 1.0
    mock_metrics.avg_loss = 1.0
    mock_metrics.profit_factor = 1.0

    timestamps = pd.date_range(start="2025-01-01", periods=7, freq="h")
    equities = [100.0, 100.0, 101.0, 101.0, 99.0, 99.0, 100.0]
    equity_curve = [
        SimpleNamespace(timestamp=ts.to_pydatetime(), equity=eq)
        for ts, eq in zip(timestamps, equities)
    ]

    mock_result = MagicMock(spec=BacktestResult)
    mock_result.final_equity = 10000.0
    mock_result.total_trades = 3
    mock_result.metrics = mock_metrics
    mock_result.equity_curve = equity_curve
    mock_run_backtest.return_value = mock_result

    market_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * len(timestamps),
            "high": [101.0] * len(timestamps),
            "low": [99.0] * len(timestamps),
            "close": [100.0, 100.0, 101.0, 101.0, 100.0, 100.0, 101.0],
            "volume": [1.0] * len(timestamps),
        }
    )
    mock_db.get_market_data.return_value = market_df

    detector_instance = MagicMock()
    detector_instance.detect_from_closes.side_effect = [
        SimpleNamespace(regime="BULL", confidence=1.0) for _ in range(len(timestamps))
    ]
    mock_regime_detector.return_value = detector_instance

    payload = {
        "symbol": "BTCUSDT",
        "timeframes": ["1h"],
        "start_date": "2025-01-01",
        "end_date": "2025-01-02",
        "initial_capital": 10000.0,
        "commission": 0.001,
        "strategy_name": "sma_crossover",
        "strategy_params": {"short_period": 2, "long_period": 3},
        "run_walk_forward_validation": False,
        "include_regime_slices": True,
    }

    response = client.post("/backtest/run", json=payload)
    assert response.status_code == 200

    regime_slices = response.json()["results"]["validation"]["regime_slices"]
    bull = regime_slices["BULL"]

    # Only non-zero returns should be counted for regime win-rate evaluation.
    assert bull["sample_count"] == 3.0
    assert bull["win_rate"] == pytest.approx(2.0 / 3.0, rel=1e-6)


@patch("api.main.run_backtest")
def test_run_backtest_sanitizes_non_finite_metric_values(mock_run_backtest):
    mock_metrics = MagicMock(spec=PerformanceMetrics)
    mock_metrics.total_return_pct = Decimal("10.5")
    mock_metrics.sharpe_ratio = Decimal("1.25")
    mock_metrics.max_drawdown_pct = Decimal("NaN")
    mock_metrics.win_rate = Decimal("60")
    mock_metrics.avg_win = Decimal("3.0")
    mock_metrics.avg_loss = Decimal("0")
    mock_metrics.profit_factor = Decimal("Infinity")

    mock_result = MagicMock(spec=BacktestResult)
    mock_result.final_equity = 11000.0
    mock_result.total_trades = 4
    mock_result.metrics = mock_metrics
    mock_result.equity_curve = []
    mock_run_backtest.return_value = mock_result

    payload = {
        "symbol": "BTCUSDT",
        "timeframes": ["1h"],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "initial_capital": 10000.0,
        "commission": 0.001,
        "strategy_name": "sma_crossover",
        "strategy_params": {"short_period": 10, "long_period": 50},
    }

    response = client.post("/backtest/run", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]
    assert results["profit_loss_percent"] == 10.5
    assert results["sharpe_ratio"] == 1.25
    assert results["max_drawdown"] is None
    assert results["profit_factor"] is None


def test_run_backtest_rejects_unknown_execution_simulation_fields():
    api_main.api_security.update_config(
        {
            "auth_enabled": False,
            "allow_public_health": True,
            "read_api_keys": [],
            "admin_api_keys": [],
            "rate_limit_enabled": False,
            "rate_limit_requests": 100,
            "rate_limit_window_seconds": 60,
        }
    )

    payload = {
        "symbol": "BTCUSDT",
        "timeframes": ["1h"],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "initial_capital": 10000.0,
        "commission": 0.001,
        "strategy_name": "sma_crossover",
        "strategy_params": {"short_period": 10, "long_period": 50},
        "execution_simulation": {"enabled": True, "latency_ms": 250},
    }

    response = client.post("/backtest/run", json=payload)
    assert response.status_code == 422
    assert "latency_ms" in response.text

# Test get order history endpoint
def test_get_order_history():
    response = client.get("/orders/history?symbol=BTCUSDT&limit=10")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "orders" in response.json()
    assert len(response.json()["orders"]) <= 10
    
    # Verify the structure of an order
    if response.json()["orders"]:
        order = response.json()["orders"][0]
        assert "id" in order
        assert "time" in order
        assert "symbol" in order
        assert "side" in order
        assert "quantity" in order
        assert "price" in order
        assert "status" in order

# Test get signal history endpoint
def test_get_signal_history():
    response = client.get("/signals/history?symbol=BTCUSDT&limit=10")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "signals" in response.json()
    assert len(response.json()["signals"]) <= 10
    
    # Verify the structure of a signal
    if response.json()["signals"]:
        signal = response.json()["signals"][0]
        assert "id" in signal
        assert "time" in signal
        assert "symbol" in signal
        assert "timeframe" in signal
        assert "strategy" in signal
        assert "signal" in signal
        assert "strength" in signal
        assert "price" in signal

# Test LLM integration endpoint with rule-based fallback
def test_llm_decision_rule_based(mock_llm_manager):
    # Mock the rule-based decision
    mock_llm_manager.make_rule_based_decision.return_value = {
        "decision": "buy",
        "confidence": 0.8,
        "reasoning": "Strong buy signal from technical indicators"
    }
    mock_llm_manager._make_rule_based_decision.side_effect = AssertionError(
        "Private fallback should not be used by API"
    )
    
    # Request data
    request_data = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "market_data": {
            "price": 40000,
            "volume": 100,
            "indicators": {
                "rsi": 30,
                "macd": {"line": 10, "signal": 5, "histogram": 5}
            }
        },
        "context": "BTC has been consolidating",
        "strategy_signals": {
            "sma_crossover": "buy",
            "rsi": "buy"
        }
    }
    
    response = client.post("/llm/decision", json=request_data)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["decision"] == "buy"
    assert response.json()["confidence"] == 0.8
    assert "reasoning" in response.json()
    
    # Verify the right method was called
    mock_llm_manager.make_rule_based_decision.assert_called_once()
    mock_llm_manager._make_rule_based_decision.assert_not_called()
    mock_llm_manager.make_llm_decision.assert_not_called()

# Test LLM integration endpoint with actual LLM
def test_llm_decision_with_llm(mock_llm_manager):
    # Mock the LLM decision
    mock_llm_manager.make_llm_decision.return_value = {
        "decision": "sell",
        "confidence": 0.7,
        "reasoning": "Market conditions suggest a local top."
    }
    
    # Request data with LLM model specified
    request_data = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "market_data": {
            "llm_model": "gpt4",  # Specify LLM model
            "price": 40000,
            "volume": 100,
            "indicators": {
                "rsi": 75,
                "macd": {"line": -5, "signal": 0, "histogram": -5}
            }
        },
        "context": "BTC has risen 20% in the last week",
        "strategy_signals": {
            "sma_crossover": "sell",
            "rsi": "sell"
        }
    }
    
    response = client.post("/llm/decision", json=request_data)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["decision"] == "sell"
    assert response.json()["confidence"] == 0.7
    assert "reasoning" in response.json()
    
    # Verify the right method was called
    mock_llm_manager.make_llm_decision.assert_called_once()
    mock_llm_manager.make_rule_based_decision.assert_not_called()

# Test error handling in LLM integration
def test_llm_decision_error_handling(mock_llm_manager):
    # Make the mock raise an exception
    mock_llm_manager.make_rule_based_decision.side_effect = Exception("LLM error")
    
    # Request data
    request_data = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "market_data": {
            "price": 40000,
            "volume": 100
        },
        "context": "Market analysis",
        "strategy_signals": {}
    }
    
    response = client.post("/llm/decision", json=request_data)
    
    assert response.status_code == 500
    assert "detail" in response.json()

def test_llm_decision_invalid_payload(mock_llm_manager):
    # Force an invalid manager response shape
    mock_llm_manager.make_rule_based_decision.return_value = {"decision": "buy"}

    request_data = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "market_data": {"price": 40000},
        "context": "Market analysis",
        "strategy_signals": {}
    }

    response = client.post("/llm/decision", json=request_data)
    assert response.status_code == 500
    assert "detail" in response.json()

# Test database trades endpoint
def test_database_trades(mock_db):
    # Mock the database response
    mock_db.get_trade_records.return_value = [
        {
            "id": 1,
            "symbol": "BTCUSDT",
            "entry_time": "2023-03-01T12:00:00",
            "exit_time": "2023-03-02T14:30:00",
            "entry_price": 40000.0,
            "exit_price": 41000.0,
            "quantity": 0.1,
            "profit_loss": 100.0,
            "profit_loss_percent": 2.5,
            "strategy": "sma_crossover"
        }
    ]
    
    response = client.get("/database/trades?limit=10")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "trades" in response.json()
    
    # Test filtering by symbol
    mock_db.get_trade_records.return_value = [
        {
            "id": 1,
            "symbol": "ETHUSDT",
            "entry_time": "2023-03-01T12:00:00",
            "exit_time": "2023-03-02T14:30:00",
            "entry_price": 2000.0,
            "exit_price": 2100.0,
            "quantity": 1.0,
            "profit_loss": 100.0,
            "profit_loss_percent": 5.0,
            "strategy": "sma_crossover"
        }
    ]
    
    response = client.get("/database/trades?symbol=ETHUSDT&limit=10")
    
    assert response.status_code == 200
    assert "trades" in response.json()
    mock_db.get_trade_records.assert_called_with(symbol="ETHUSDT", limit=10, offset=0)
    
    # Test error handling
    mock_db.get_trade_records.side_effect = Exception("Database error")
    response = client.get("/database/trades?limit=10")
    
    assert response.status_code == 500
    assert "detail" in response.json()

# Test database signals endpoint
def test_database_signals(mock_db):
    # Mock the database response
    mock_db.get_signal_records.return_value = [
        {
            "id": 1,
            "symbol": "BTCUSDT",
            "timestamp": "2023-03-01T12:00:00",
            "strategy": "sma_crossover",
            "timeframe": "1h",
            "signal": "buy",
            "strength": 0.8,
            "price": 40000.0
        }
    ]
    
    response = client.get("/database/signals?limit=10")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "signals" in response.json()
    
    # Test filtering by symbol and strategy
    response = client.get("/database/signals?symbol=BTCUSDT&strategy=rsi&limit=10")
    
    assert response.status_code == 200
    assert "signals" in response.json()
    mock_db.get_signal_records.assert_called_with(symbol="BTCUSDT", strategy="rsi", limit=10, offset=0)
    
    # Test error handling
    mock_db.get_signal_records.side_effect = Exception("Database error")
    response = client.get("/database/signals?limit=10")
    
    assert response.status_code == 500
    assert "detail" in response.json()

def test_database_pagination_validation():
    trades_response = client.get("/database/trades?limit=0")
    assert trades_response.status_code == 422

    signals_response = client.get("/database/signals?offset=-1")
    assert signals_response.status_code == 422

# Test starting the bot with LLM strategy
def test_start_with_llm_strategy(mock_background_tasks):
    trading_config = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "trade_amount": 100.0,
        "strategies": [
            {
                "name": "llm_strategy",
                "params": {
                    "base_strategy": "sma_crossover",
                    "llm_model": "gpt4"
                },
                "active": True
            }
        ]
    }
    
    response = client.post("/trading/start", json=trading_config)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_background_tasks.assert_called_once()

if __name__ == "__main__":
    pytest.main() 
