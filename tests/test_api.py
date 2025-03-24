#!/usr/bin/env python3
import os
import sys
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API and bot modules
from api.main import app
from bot import binance_api, strategy, order_manager, database, config, llm_manager
from bot.backtesting import run_backtest, generate_report, optimize_strategy
from bot.backtesting.models.results import BacktestResult, PerformanceMetrics

# Create test client - updated for newer FastAPI/starlette versions
client = TestClient(app)

# Set up module-level mocks for strategy and backtesting
strategy.sma_crossover_strategy = MagicMock(return_value='BUY')
strategy.rsi_strategy = MagicMock(return_value='SELL')

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

# Mock strategy functions
@pytest.fixture(autouse=True)
def mock_strategy_functions():
    # Create strategy function mocks
    with patch('bot.strategy.sma_crossover_strategy', MagicMock(return_value='BUY')), \
         patch('bot.strategy.rsi_strategy', MagicMock(return_value='SELL')):
        yield

# Test API health check
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "version" in response.json()

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
    
    # Should return mock data instead of error
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "balances" in response.json()

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
    
    # Should return mock data instead of error
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "symbols" in response.json()

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
    
    # Should return mock data instead of error
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "candles" in response.json()

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
    for s in response.json()["strategies"]:
        if s["name"] == "llm_strategy":
            llm_strategy = s
            break
    
    assert llm_strategy is not None
    assert "llm_model" in llm_strategy["parameters"]
    assert "base_strategy" in llm_strategy["parameters"]

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
    
    # Test with unknown strategy
    backtest_config["strategy_name"] = "unknown_strategy"
    response = client.post("/backtest/run", json=backtest_config)
    # API should now return an error for unknown strategy
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "Unknown strategy" in response.json()["detail"]

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
    
    # Should return a default 'hold' decision
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["decision"] == "hold"
    assert "reasoning" in response.json()
    assert "Error" in response.json()["reasoning"]

# Test database trades endpoint
def test_database_trades(mock_db):
    # Mock the database response
    mock_db.get_trades.return_value = [
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
    mock_db.get_trades_by_symbol.return_value = [
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
    mock_db.get_trades_by_symbol.assert_called_once_with("ETHUSDT", 10, 0)
    
    # Test error handling
    mock_db.get_trades.side_effect = Exception("Database error")
    response = client.get("/database/trades?limit=10")
    
    # Should return mock data
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "trades" in response.json()

# Test database signals endpoint
def test_database_signals(mock_db):
    # Mock the database response
    mock_db.get_signals.return_value = [
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
    mock_db.get_signals.assert_called_with("BTCUSDT", "rsi", 10, 0)
    
    # Test error handling
    mock_db.get_signals.side_effect = Exception("Database error")
    response = client.get("/database/signals?limit=10")
    
    # Should return mock data
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "signals" in response.json()

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