import os
import sys
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd

# Add project root to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.backtesting.core.engine import BacktestEngine
from bot.backtesting.models.execution import (
    ExecutionSimulationConfig,
    ExecutionSimulator,
)


def _build_market_data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(start="2024-01-01", periods=rows, freq="1h")
    closes = np.linspace(100.0, 120.0, rows)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": closes * 1.002,
        "low": closes * 0.998,
        "close": closes,
        "volume": np.full(rows, 10.0),
    })


def test_execution_simulator_applies_spread_slippage_latency():
    simulator = ExecutionSimulator(
        ExecutionSimulationConfig(
            enabled=True,
            spread_bps=10.0,
            slippage_bps=5.0,
            latency_bps=2.0,
        )
    )

    buy_fill = simulator.simulate_fill(
        side="BUY",
        reference_price=Decimal("100"),
        requested_quantity=Decimal("2"),
        candle_volume=Decimal("100"),
    )
    sell_fill = simulator.simulate_fill(
        side="SELL",
        reference_price=Decimal("100"),
        requested_quantity=Decimal("2"),
        candle_volume=Decimal("100"),
    )

    assert buy_fill.executed_price > Decimal("100")
    assert sell_fill.executed_price < Decimal("100")
    assert buy_fill.spread_cost > 0
    assert buy_fill.slippage_cost > 0
    assert buy_fill.latency_cost > 0


def test_execution_simulator_partial_fill_and_funding():
    simulator = ExecutionSimulator(
        ExecutionSimulationConfig(
            enabled=True,
            max_volume_participation=0.1,
            min_partial_fill_ratio=0.2,
            funding_rate_per_8h=0.001,
        )
    )

    fill = simulator.simulate_fill(
        side="BUY",
        reference_price=Decimal("100"),
        requested_quantity=Decimal("5"),
        candle_volume=Decimal("10"),
    )
    assert fill.fill_ratio == Decimal("0.2")
    assert fill.executed_quantity == Decimal("1.0")

    long_funding = simulator.estimate_funding_cost(
        notional=Decimal("1000"),
        holding_hours=16,
        position_side="LONG",
    )
    short_funding = simulator.estimate_funding_cost(
        notional=Decimal("1000"),
        holding_hours=16,
        position_side="SHORT",
    )
    assert long_funding == Decimal("2.000")
    assert short_funding == Decimal("-2.000")


def test_backtest_engine_execution_simulation_affects_trade_fields():
    with patch("bot.backtesting.core.engine.BacktestEngine._load_market_data"):
        engine = BacktestEngine(
            symbol="BTCUSDT",
            timeframes=["1h"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            execution_simulation={
                "enabled": True,
                "spread_bps": 8.0,
                "slippage_bps": 4.0,
                "latency_bps": 3.0,
                "max_volume_participation": 0.2,
                "min_partial_fill_ratio": 0.1,
                "funding_rate_per_8h": 0.001,
            },
        )

    df = _build_market_data()
    engine.market_data = {"1h": df}

    entry_time = df["timestamp"].iloc[40]
    exit_time = df["timestamp"].iloc[56]

    buy_trade = engine._execute_trade("BUY", entry_time, price=110.0, quantity=5.0)
    assert buy_trade.requested_quantity == Decimal("5.0")
    assert buy_trade.quantity < buy_trade.requested_quantity
    assert buy_trade.fill_ratio < Decimal("1")
    assert buy_trade.spread_cost > 0
    assert buy_trade.slippage_cost > 0
    assert buy_trade.latency_cost > 0

    sell_trade = engine._execute_trade("SELL", exit_time, price=115.0, quantity=engine.position_size)
    assert sell_trade.funding_cost > 0
    assert sell_trade.profit_loss is not None
    assert engine.position_size >= 0


def test_execution_simulation_forces_traditional_backtest_path():
    with patch("bot.backtesting.core.engine.BacktestEngine._load_market_data"):
        engine = BacktestEngine(
            symbol="BTCUSDT",
            timeframes=["1h"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            execution_simulation={"enabled": True},
        )

    df = _build_market_data()
    engine.market_data = {"1h": df}

    def vectorized_strategy(data_dict, symbol, vectorized=False):
        if vectorized:
            return ["HOLD"] * len(data_dict["1h"])
        return "HOLD"

    vectorized_strategy.vectorized = True

    with patch.object(engine, "prepare_data"), \
         patch.object(engine, "_run_traditional_backtest") as mock_traditional, \
         patch.object(engine, "_run_vectorized_backtest") as mock_vectorized:
        engine.run_backtest(vectorized_strategy)

    mock_traditional.assert_called_once()
    mock_vectorized.assert_not_called()
