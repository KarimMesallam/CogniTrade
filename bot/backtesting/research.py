from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

import pandas as pd

from bot.backtesting.models.results import BacktestResult
from research.registry import ExperimentRecord, ExperimentRegistry, fingerprint_dataframe


def dataset_fingerprint_from_market_data(
    market_data: Dict[str, pd.DataFrame],
    *,
    primary_timeframe: Optional[str] = None,
) -> str:
    """
    Build deterministic dataset fingerprint from backtest market-data payload.
    """
    if not market_data:
        raise ValueError("market_data is required")

    if primary_timeframe and primary_timeframe in market_data:
        frame = market_data[primary_timeframe]
    else:
        first_key = sorted(market_data.keys())[0]
        frame = market_data[first_key]
    return fingerprint_dataframe(frame)


def _num(value: Any) -> float:
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except Exception:
        return 0.0


def backtest_metrics_payload(result: BacktestResult) -> Dict[str, float]:
    """
    Convert BacktestResult into numeric metric payload for experiment tracking.
    """
    return {
        "total_return_pct": _num(result.metrics.total_return_pct),
        "sharpe_ratio": _num(result.metrics.sharpe_ratio),
        "max_drawdown_pct": _num(result.metrics.max_drawdown_pct),
        "win_rate": _num(result.metrics.win_rate),
        "profit_factor": _num(result.metrics.profit_factor),
        "total_trades": float(result.total_trades),
        "final_equity": _num(result.final_equity),
        "initial_capital": _num(result.initial_capital),
    }


def register_backtest_experiment(
    *,
    registry: ExperimentRegistry,
    experiment_name: str,
    result: BacktestResult,
    dataset_fingerprint: str,
    strategy_params: Optional[Dict[str, Any]] = None,
    validation_summary: Optional[Dict[str, Any]] = None,
    execution_config: Optional[Dict[str, Any]] = None,
) -> ExperimentRecord:
    """
    Register a backtest result in the research experiment registry.
    """
    metrics = backtest_metrics_payload(result)
    return registry.register_experiment(
        experiment_name=experiment_name,
        strategy_name=result.strategy_name,
        dataset_fingerprint=dataset_fingerprint,
        params=strategy_params or {},
        metrics=metrics,
        validation_summary=validation_summary or {},
        execution_config=execution_config or {},
    )
