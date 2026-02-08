import os
import sys

import numpy as np
import pandas as pd

# Add project root to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.main import build_portfolio_targets, optimize_portfolio_from_returns
from bot.portfolio import PortfolioConstraints, PortfolioOptimizer


def _sample_returns() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    btc = np.random.normal(0.0010, 0.020, n)
    eth = np.random.normal(0.0012, 0.025, n)
    sol = np.random.normal(0.0014, 0.035, n)
    return pd.DataFrame({"BTCUSDT": btc, "ETHUSDT": eth, "SOLUSDT": sol})


def test_risk_parity_optimizer_returns_valid_weights():
    returns = _sample_returns()
    optimizer = PortfolioOptimizer(
        PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.7,
            max_leverage=1.0,
        )
    )

    result = optimizer.optimize(returns, method="risk_parity")
    weights = result.weights

    assert set(weights.keys()) == set(returns.columns)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(0.0 <= weight <= 0.7 + 1e-9 for weight in weights.values())
    assert result.expected_volatility_annual >= 0
    assert np.isfinite(result.sharpe_ratio)
    assert abs(sum(result.risk_contributions.values()) - 1.0) < 1e-6


def test_mean_variance_optimizer_respects_weight_cap():
    returns = _sample_returns()
    optimizer = PortfolioOptimizer(
        PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.5,
            max_leverage=1.0,
        )
    )

    result = optimizer.optimize(returns, method="mean_variance")
    assert all(weight <= 0.5 + 1e-9 for weight in result.weights.values())
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6


def test_allocate_units_respects_min_notional_threshold():
    optimizer = PortfolioOptimizer()
    allocations = optimizer.allocate_units(
        total_capital=1000.0,
        weights={"BTCUSDT": 0.7, "ETHUSDT": 0.3},
        prices={"BTCUSDT": 50000.0, "ETHUSDT": 2500.0},
        min_notional=400.0,
    )

    assert allocations["BTCUSDT"]["notional"] == 700.0
    assert allocations["BTCUSDT"]["units"] > 0
    assert allocations["ETHUSDT"]["notional"] == 300.0
    assert allocations["ETHUSDT"]["units"] == 0.0


def test_main_helpers_optimize_and_build_targets():
    returns = _sample_returns()
    asset_returns = {
        "BTCUSDT": returns["BTCUSDT"].tolist(),
        "ETHUSDT": returns["ETHUSDT"].tolist(),
    }

    allocation = optimize_portfolio_from_returns(
        asset_returns=asset_returns,
        method="risk_parity",
        constraints={"max_weight": 0.8, "max_leverage": 1.0},
    )
    targets = build_portfolio_targets(
        total_capital=10000.0,
        allocation=allocation,
        latest_prices={"BTCUSDT": 50000.0, "ETHUSDT": 2500.0},
        min_notional=100.0,
    )

    assert set(targets.keys()) == {"BTCUSDT", "ETHUSDT"}
    total_notional = sum(target["notional"] for target in targets.values())
    assert abs(total_notional - 10000.0) < 1e-5
    assert all(target["units"] >= 0 for target in targets.values())
