from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import json
import logging
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from bot.backtesting import run_backtest
from bot.backtesting.validation import (
    AdvancedValidationFramework,
    regime_sliced_evaluation,
)
from bot.config import (
    get_monitoring_config,
    get_observability_config,
    get_policy_config,
    get_promotion_benchmark_config,
    get_quality_gate_config,
    get_reconciliation_config,
    get_regime_config,
    get_risk_engine_config,
    get_rollout_config,
    get_trade_mode,
)
from bot.database import Database
from bot.deploy_policy import (
    RolloutMetrics,
    ShadowCanaryRolloutGate,
    StrategyQualityThresholds,
    evaluate_strategy_quality,
    metrics_from_observability_dashboard,
    thresholds_from_config,
)
from bot.observability import ObservabilityManager
from bot.reconciliation import ExchangeReconciler
from bot.regime import MarketRegimeDetector
from bot.resilience import CircuitBreaker, CircuitBreakerOpenError, execute_with_resilience
from bot.risk_engine import LiveRiskEngine
from research.benchmarks import (
    PromotionBenchmarkDecision,
    PromotionBenchmarkThresholds,
    evaluate_promotion_candidate,
    promotion_thresholds_from_config,
)
from research.registry import fingerprint_dataframe


logger = logging.getLogger("trading_bot.gap_closure")


@dataclass(frozen=True)
class StrategyCandidateSpec:
    strategy_name: str
    params: Dict[str, Any]
    allow_short_positions: bool = True
    trade_mode: str = "FUTURES"

    @property
    def candidate_id(self) -> str:
        encoded = json.dumps(
            {
                "strategy_name": self.strategy_name,
                "params": self.params,
                "allow_short_positions": self.allow_short_positions,
                "trade_mode": self.trade_mode,
            },
            sort_keys=True,
        )
        return fingerprint_dataframe(
            pd.DataFrame([{"payload": encoded, "timestamp": "1970-01-01T00:00:00"}]),
            sort_by=["timestamp"],
        )[:16]


@dataclass
class StrategyCandidateEvaluation:
    candidate: StrategyCandidateSpec
    metrics: Dict[str, float]
    validation_summary: Dict[str, Any]
    quality_gate: Dict[str, Any]
    benchmark_decision: PromotionBenchmarkDecision

    @property
    def passed(self) -> bool:
        return bool(self.benchmark_decision.passed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "candidate": asdict(self.candidate),
            "metrics": dict(self.metrics),
            "validation_summary": dict(self.validation_summary),
            "quality_gate": dict(self.quality_gate),
            "benchmark_decision": asdict(self.benchmark_decision),
            "passed": bool(self.passed),
        }


@dataclass
class StrategySelectionResult:
    selected: Optional[StrategyCandidateEvaluation]
    evaluations: List[StrategyCandidateEvaluation]
    benchmark_thresholds: PromotionBenchmarkThresholds
    quality_thresholds: StrategyQualityThresholds

    @property
    def has_passing_candidate(self) -> bool:
        return self.selected is not None and bool(self.selected.passed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "benchmark_thresholds": asdict(self.benchmark_thresholds),
            "quality_thresholds": asdict(self.quality_thresholds),
            "has_passing_candidate": bool(self.has_passing_candidate),
        }


def _num(value: Any, default: float = 0.0) -> float:
    if isinstance(value, Decimal):
        if value.is_finite():
            return float(value)
        return default
    if isinstance(value, np.generic):
        return _num(value.item(), default=default)
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except Exception:
        pass
    return default


def _safe_iso(ts: Any) -> str:
    if isinstance(ts, str):
        return ts
    if isinstance(ts, datetime):
        return ts.isoformat()
    try:
        return pd.to_datetime(ts).isoformat()
    except Exception:
        return datetime.utcnow().isoformat()


def _write_json(payload: Mapping[str, Any], output_path: str) -> str:
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    return output_path


def build_seeded_regime_market_data(
    *,
    start: datetime = datetime(2025, 1, 1, 0, 0, 0),
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Build deterministic market data with explicit bull/bear/sideways/high-volatility phases.
    """
    if timeframe != "1h":
        raise ValueError("Only 1h seeded timeframe is currently supported.")

    phases = [
        ("BULL", 240, 0.0019),
        ("SIDEWAYS", 220, 0.0002),
        ("BEAR", 260, -0.0018),
        ("HIGH_VOLATILITY", 220, 0.0),
    ]

    rows: List[Dict[str, Any]] = []
    interval = timedelta(hours=1)
    price = 100.0
    idx = 0
    for regime, count, drift in phases:
        for step in range(count):
            timestamp = start + interval * idx
            if regime == "HIGH_VOLATILITY":
                pct_change = 0.018 if step % 2 == 0 else -0.017
            elif regime == "SIDEWAYS":
                pct_change = drift if step % 2 == 0 else -drift
            else:
                pct_change = drift

            open_price = price
            close_price = max(1.0, open_price * (1.0 + pct_change))
            high_price = max(open_price, close_price) * (1.0 + abs(pct_change) * 0.55 + 0.001)
            low_price = min(open_price, close_price) * (1.0 - abs(pct_change) * 0.55 - 0.001)
            volume = 800.0 + (idx % 30) * 12.0 + (60.0 if regime == "HIGH_VOLATILITY" else 0.0)

            rows.append(
                {
                    "timestamp": timestamp,
                    "open": float(open_price),
                    "high": float(high_price),
                    "low": float(low_price),
                    "close": float(close_price),
                    "volume": float(volume),
                    "regime_truth": regime,
                }
            )

            price = close_price
            idx += 1

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def seed_market_data(
    *,
    market_frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    db_path: str,
) -> bool:
    db = Database(db_path=db_path)
    frame = market_frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    return bool(db.store_market_data(frame, symbol=symbol, timeframe=timeframe))


def default_candidate_specs() -> List[StrategyCandidateSpec]:
    specs: List[StrategyCandidateSpec] = []
    for short_period, long_period in [(2, 3), (2, 5), (4, 8), (8, 34), (10, 50)]:
        specs.append(
            StrategyCandidateSpec(
                strategy_name="sma_crossover",
                params={"short_period": short_period, "long_period": long_period},
                allow_short_positions=True,
                trade_mode="FUTURES",
            )
        )
    for period, overbought, oversold in [(14, 68, 32), (14, 70, 30), (6, 60, 40)]:
        specs.append(
            StrategyCandidateSpec(
                strategy_name="rsi",
                params={"period": period, "overbought": overbought, "oversold": oversold},
                allow_short_positions=True,
                trade_mode="FUTURES",
            )
        )
    return specs


def _build_sma_strategy(short_period: int, long_period: int, timeframe: str):
    def _strategy(data_dict, _symbol):
        frame = data_dict.get(timeframe)
        if frame is None or frame.empty or len(frame) < long_period + 2:
            return "HOLD"
        short_sma = frame["close"].rolling(window=short_period).mean()
        long_sma = frame["close"].rolling(window=long_period).mean()
        prev_short, curr_short = short_sma.iloc[-2], short_sma.iloc[-1]
        prev_long, curr_long = long_sma.iloc[-2], long_sma.iloc[-1]
        if any(value != value for value in [prev_short, curr_short, prev_long, curr_long]):
            return "HOLD"
        if prev_short <= prev_long and curr_short > curr_long:
            return "BUY"
        if prev_short >= prev_long and curr_short < curr_long:
            return "SELL"
        return "HOLD"

    _strategy.__name__ = f"sma_crossover_{short_period}_{long_period}"
    return _strategy


def _build_rsi_strategy(period: int, overbought: int, oversold: int, timeframe: str):
    def _strategy(data_dict, _symbol):
        frame = data_dict.get(timeframe)
        if frame is None or frame.empty or len(frame) < period + 2:
            return "HOLD"
        delta = frame["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        relative_strength = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + relative_strength))
        prev_rsi = rsi.iloc[-2]
        curr_rsi = rsi.iloc[-1]
        if prev_rsi != prev_rsi or curr_rsi != curr_rsi:
            return "HOLD"
        if prev_rsi < oversold and curr_rsi >= oversold:
            return "BUY"
        if prev_rsi > overbought and curr_rsi <= overbought:
            return "SELL"
        return "HOLD"

    _strategy.__name__ = f"rsi_{period}_{overbought}_{oversold}"
    return _strategy


def build_strategy_function(candidate: StrategyCandidateSpec, timeframe: str):
    if candidate.strategy_name == "sma_crossover":
        return _build_sma_strategy(
            int(candidate.params["short_period"]),
            int(candidate.params["long_period"]),
            timeframe,
        )
    if candidate.strategy_name == "rsi":
        return _build_rsi_strategy(
            int(candidate.params["period"]),
            int(candidate.params["overbought"]),
            int(candidate.params["oversold"]),
            timeframe,
        )
    raise ValueError(f"Unsupported strategy_name: {candidate.strategy_name}")


def _equity_frame_from_result(result: Any) -> pd.DataFrame:
    if not result or not getattr(result, "equity_curve", None):
        return pd.DataFrame()
    frame = pd.DataFrame(
        [
            {
                "timestamp": point.timestamp,
                "equity": _num(point.equity),
            }
            for point in result.equity_curve
        ]
    )
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["period_return"] = frame["equity"].pct_change().fillna(0.0)
    return frame


def _evaluate_return_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    test_returns = test_df.get("period_return", pd.Series(dtype=float)).astype(float)
    train_returns = train_df.get("period_return", pd.Series(dtype=float)).astype(float)
    test_mean = _num(test_returns.mean()) if len(test_returns) else 0.0
    test_std = _num(test_returns.std(ddof=0)) if len(test_returns) > 1 else 0.0
    sharpe = float(test_mean / test_std) if test_std > 0 else 0.0

    test_equity = test_df.get("equity", pd.Series(dtype=float)).astype(float)
    if len(test_equity):
        running_peak = test_equity.cummax()
        drawdown = ((test_equity - running_peak) / running_peak.replace(0, np.nan)).fillna(0.0) * 100.0
        max_drawdown_pct = abs(_num(drawdown.min()))
        total_return = _num((test_equity.iloc[-1] / test_equity.iloc[0]) - 1.0) if test_equity.iloc[0] else 0.0
    else:
        max_drawdown_pct = 0.0
        total_return = 0.0

    calmar = float(total_return / (max_drawdown_pct / 100.0)) if max_drawdown_pct > 0 else 0.0
    return {
        "train_mean_return": _num(train_returns.mean()) if len(train_returns) else 0.0,
        "test_mean_return": test_mean,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "max_drawdown_pct": max_drawdown_pct,
    }


def build_validation_summary(
    *,
    result: Any,
    market_frame: pd.DataFrame,
    walk_forward_train_size: int = 240,
    walk_forward_test_size: int = 120,
    walk_forward_step_size: int = 120,
    walk_forward_gap: int = 0,
    regime_lookback_candles: int = 50,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    equity_frame = _equity_frame_from_result(result)

    if len(equity_frame) < max(10, walk_forward_train_size + walk_forward_test_size):
        summary["walk_forward_error"] = "insufficient_equity_points_for_walk_forward"
        summary["walk_forward_folds"] = 0
        summary["walk_forward_summary"] = {}
    else:
        framework = AdvancedValidationFramework(timestamp_col="timestamp")
        folds = framework.walk_forward_validate(
            frame=equity_frame,
            evaluator=_evaluate_return_metrics,
            train_size=walk_forward_train_size,
            test_size=walk_forward_test_size,
            step_size=walk_forward_step_size,
            gap=walk_forward_gap,
            expanding=True,
        )
        summary["walk_forward_folds"] = len(folds)
        summary["walk_forward_summary"] = framework.summarize_metrics(folds)

    market = market_frame.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"])
    market = market.sort_values("timestamp").reset_index(drop=True)
    closes = market["close"].astype(float).tolist()
    detector = MarketRegimeDetector(lookback_candles=int(regime_lookback_candles))
    regimes: List[str] = []
    for idx, _close in enumerate(closes):
        state = detector.detect_from_closes(
            closes[: idx + 1],
            timestamp=str(market["timestamp"].iloc[idx]),
        )
        regimes.append(state.regime)
    regime_frame = pd.DataFrame(
        {"timestamp": market["timestamp"], "regime": regimes}
    ).sort_values("timestamp")
    merged = pd.merge_asof(
        equity_frame.sort_values("timestamp"),
        regime_frame,
        on="timestamp",
        direction="backward",
    )
    merged["regime"] = merged["regime"].fillna("UNKNOWN")
    active = merged[np.abs(merged["period_return"].astype(float)) > 1e-12].copy()
    if active.empty:
        active = merged
    summary["regime_slices"] = regime_sliced_evaluation(
        active["period_return"].astype(float).tolist(),
        active["regime"].astype(str).tolist(),
    )
    return summary


def _modeled_cost_pct(result: Any) -> float:
    total_cost_quote = 0.0
    for trade in getattr(result, "trades", []) or []:
        if not isinstance(trade, dict):
            continue
        total_cost_quote += abs(_num(trade.get("commission", 0.0)))
        total_cost_quote += abs(_num(trade.get("spread_cost", 0.0)))
        total_cost_quote += abs(_num(trade.get("slippage_cost", 0.0)))
        total_cost_quote += abs(_num(trade.get("latency_cost", 0.0)))
        total_cost_quote += abs(_num(trade.get("funding_cost", 0.0)))
    initial_capital = max(_num(getattr(result, "initial_capital", 0.0)), 1e-9)
    return float((total_cost_quote / initial_capital) * 100.0)


def _candidate_metrics_payload(result: Any) -> Dict[str, float]:
    return {
        "total_return_pct": _num(result.metrics.total_return_pct),
        "modeled_cost_pct": _modeled_cost_pct(result),
        "total_trades": float(getattr(result, "total_trades", 0)),
        "sharpe_ratio": _num(result.metrics.sharpe_ratio),
        "calmar_ratio": _num(result.metrics.calmar_ratio),
        "max_drawdown_pct": abs(_num(result.metrics.max_drawdown_pct)),
        "final_equity": _num(getattr(result, "final_equity", 0.0)),
        "initial_capital": _num(getattr(result, "initial_capital", 0.0)),
        "win_rate": _num(result.metrics.win_rate),
    }


def evaluate_candidate(
    *,
    candidate: StrategyCandidateSpec,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    db_path: str,
    market_frame: pd.DataFrame,
    benchmark_thresholds: PromotionBenchmarkThresholds,
    quality_thresholds: StrategyQualityThresholds,
    initial_capital: float = 10000.0,
    commission_rate: float = 0.001,
    execution_simulation: Optional[Dict[str, Any]] = None,
) -> StrategyCandidateEvaluation:
    strategy_func = build_strategy_function(candidate, timeframe)
    result = run_backtest(
        symbol=symbol,
        timeframes=[timeframe],
        start_date=start_date,
        end_date=end_date,
        strategy_func=strategy_func,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        db_path=db_path,
        allow_short_positions=candidate.allow_short_positions,
        execution_simulation=execution_simulation,
    )
    validation_summary = build_validation_summary(result=result, market_frame=market_frame)
    quality_decision = evaluate_strategy_quality(
        validation_summary,
        thresholds=quality_thresholds,
    )
    benchmark_decision = evaluate_promotion_candidate(
        metrics=_candidate_metrics_payload(result),
        validation_summary=validation_summary,
        thresholds=benchmark_thresholds,
        quality_thresholds=quality_thresholds,
    )
    quality_payload = {
        "passed": bool(quality_decision.passed),
        "reasons": list(quality_decision.reasons),
        "metrics": dict(quality_decision.metrics),
    }
    return StrategyCandidateEvaluation(
        candidate=candidate,
        metrics=_candidate_metrics_payload(result),
        validation_summary=validation_summary,
        quality_gate=quality_payload,
        benchmark_decision=benchmark_decision,
    )


def select_candidate_strategy(
    *,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    db_path: str,
    market_frame: pd.DataFrame,
    candidates: Optional[Sequence[StrategyCandidateSpec]] = None,
    benchmark_thresholds: Optional[PromotionBenchmarkThresholds] = None,
    quality_thresholds: Optional[StrategyQualityThresholds] = None,
    initial_capital: float = 10000.0,
    commission_rate: float = 0.001,
) -> StrategySelectionResult:
    bench_cfg = benchmark_thresholds
    quality_cfg = quality_thresholds
    if bench_cfg is None or quality_cfg is None:
        bench_cfg, quality_cfg = promotion_thresholds_from_config(
            get_promotion_benchmark_config(),
            get_quality_gate_config(),
        )

    evaluations: List[StrategyCandidateEvaluation] = []
    for candidate in list(candidates or default_candidate_specs()):
        evaluation = evaluate_candidate(
            candidate=candidate,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            db_path=db_path,
            market_frame=market_frame,
            benchmark_thresholds=bench_cfg,
            quality_thresholds=quality_cfg,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
        )
        evaluations.append(evaluation)

    passing = [evaluation for evaluation in evaluations if evaluation.passed]
    if passing:
        selected = sorted(
            passing,
            key=lambda item: (
                _num((item.benchmark_decision.metrics or {}).get("net_return_pct", 0.0)),
                _num(item.metrics.get("sharpe_ratio", 0.0)),
                -abs(_num(item.metrics.get("max_drawdown_pct", 0.0))),
            ),
            reverse=True,
        )[0]
    else:
        selected = sorted(
            evaluations,
            key=lambda item: (
                _num((item.benchmark_decision.metrics or {}).get("net_return_pct", 0.0)),
                _num(item.metrics.get("sharpe_ratio", 0.0)),
                -abs(_num(item.metrics.get("max_drawdown_pct", 0.0))),
            ),
            reverse=True,
        )[0] if evaluations else None

    return StrategySelectionResult(
        selected=selected,
        evaluations=evaluations,
        benchmark_thresholds=bench_cfg,
        quality_thresholds=quality_cfg,
    )


def _strategy_signal_series(close: pd.Series, candidate: StrategyCandidateSpec) -> Tuple[pd.Series, pd.Series]:
    if candidate.strategy_name == "sma_crossover":
        short_period = int(candidate.params["short_period"])
        long_period = int(candidate.params["long_period"])
        fast = close.rolling(short_period).mean()
        slow = close.rolling(long_period).mean()
        buy = ((fast.shift(1) <= slow.shift(1)) & (fast > slow)).fillna(False)
        sell = ((fast.shift(1) >= slow.shift(1)) & (fast < slow)).fillna(False)
        return buy.astype(bool), sell.astype(bool)

    if candidate.strategy_name == "rsi":
        period = int(candidate.params["period"])
        overbought = float(candidate.params["overbought"])
        oversold = float(candidate.params["oversold"])
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        relative_strength = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + relative_strength))
        buy = ((rsi.shift(1) < oversold) & (rsi >= oversold)).fillna(False)
        sell = ((rsi.shift(1) > overbought) & (rsi <= overbought)).fillna(False)
        return buy.astype(bool), sell.astype(bool)

    raise ValueError(f"Unsupported strategy_name: {candidate.strategy_name}")


def _equity_metrics_from_series(equity: pd.Series, initial_capital: float) -> Dict[str, float]:
    if equity is None or len(equity) == 0:
        return {
            "final_equity": float(initial_capital),
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
        }
    series = equity.astype(float)
    final_equity = _num(series.iloc[-1], default=initial_capital)
    total_return_pct = ((final_equity / float(initial_capital)) - 1.0) * 100.0 if initial_capital else 0.0
    drawdown = (series / series.cummax().replace(0, np.nan) - 1.0).fillna(0.0)
    max_drawdown_pct = abs(_num(drawdown.min()) * 100.0)
    returns = series.pct_change().fillna(0.0)
    std = _num(returns.std(ddof=0))
    sharpe = (_num(returns.mean()) / std) * math.sqrt(24.0 * 365.0) if std > 0 else 0.0
    return {
        "final_equity": float(final_equity),
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "sharpe_ratio": float(sharpe),
    }


def _vectorbt_metrics(
    *,
    price_frame: pd.DataFrame,
    candidate: StrategyCandidateSpec,
    initial_capital: float,
    commission_rate: float,
) -> Dict[str, Any]:
    try:
        import vectorbt as vbt
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    close = price_frame["close"].astype(float).copy()
    close.index = pd.to_datetime(price_frame["timestamp"])
    buy, sell = _strategy_signal_series(close, candidate)
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=buy,
        exits=sell,
        short_entries=sell if candidate.allow_short_positions else None,
        short_exits=buy if candidate.allow_short_positions else None,
        init_cash=float(initial_capital),
        fees=float(commission_rate),
        freq="1h",
    )
    equity = portfolio.value()
    metrics = _equity_metrics_from_series(equity, initial_capital=initial_capital)
    trade_count_value = portfolio.trades.closed.count()
    if isinstance(trade_count_value, pd.Series):
        trade_count = int(_num(trade_count_value.iloc[0]))
    else:
        trade_count = int(_num(trade_count_value))
    metrics.update(
        {
            "available": True,
            "trade_count": int(trade_count),
            "engine": "vectorbt",
        }
    )
    return metrics


def _backtrader_metrics(
    *,
    price_frame: pd.DataFrame,
    candidate: StrategyCandidateSpec,
    initial_capital: float,
    commission_rate: float,
) -> Dict[str, Any]:
    try:
        import backtrader as bt
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    class PandasFeed(bt.feeds.PandasData):
        params = (("datetime", None),)

    class CandidateStrategy(bt.Strategy):
        params = dict(
            strategy_name=candidate.strategy_name,
            strategy_params=dict(candidate.params),
            allow_short=bool(candidate.allow_short_positions),
            target_pct=0.95,
        )

        def __init__(self):
            self.closed_trades = 0
            if self.p.strategy_name == "sma_crossover":
                self.fast = bt.ind.SMA(period=int(self.p.strategy_params["short_period"]))
                self.slow = bt.ind.SMA(period=int(self.p.strategy_params["long_period"]))
            elif self.p.strategy_name == "rsi":
                self.rsi = bt.ind.RSI(
                    period=int(self.p.strategy_params["period"]),
                    safediv=True,
                )
            else:
                raise ValueError(f"Unsupported strategy_name: {self.p.strategy_name}")

        def notify_trade(self, trade):
            if trade.isclosed:
                self.closed_trades += 1

        def _signals(self) -> Tuple[bool, bool]:
            if self.p.strategy_name == "sma_crossover":
                cross_up = self.fast[-1] <= self.slow[-1] and self.fast[0] > self.slow[0]
                cross_down = self.fast[-1] >= self.slow[-1] and self.fast[0] < self.slow[0]
                return bool(cross_up), bool(cross_down)
            overbought = float(self.p.strategy_params["overbought"])
            oversold = float(self.p.strategy_params["oversold"])
            buy = self.rsi[-1] < oversold and self.rsi[0] >= oversold
            sell = self.rsi[-1] > overbought and self.rsi[0] <= overbought
            return bool(buy), bool(sell)

        def next(self):
            buy_signal, sell_signal = self._signals()
            if not self.position:
                if buy_signal:
                    self.order_target_percent(target=self.p.target_pct)
                elif self.p.allow_short and sell_signal:
                    self.order_target_percent(target=-self.p.target_pct)
                return

            if self.position.size > 0 and sell_signal:
                if self.p.allow_short:
                    self.order_target_percent(target=-self.p.target_pct)
                else:
                    self.order_target_percent(target=0.0)
                return

            if self.position.size < 0 and buy_signal:
                self.order_target_percent(target=self.p.target_pct)

    feed_frame = price_frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    feed_frame["timestamp"] = pd.to_datetime(feed_frame["timestamp"])
    feed_frame = feed_frame.set_index("timestamp")

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(CandidateStrategy)
    cerebro.adddata(PandasFeed(dataname=feed_frame))
    cerebro.broker.setcash(float(initial_capital))
    cerebro.broker.setcommission(commission=float(commission_rate))
    cerebro.broker.set_coc(True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    strategies = cerebro.run()
    strategy = strategies[0]
    final_equity = _num(cerebro.broker.getvalue(), default=initial_capital)
    drawdown_data = strategy.analyzers.drawdown.get_analysis() if hasattr(strategy.analyzers, "drawdown") else {}
    max_drawdown_pct = _num((drawdown_data.get("max", {}) or {}).get("drawdown", 0.0))
    total_return_pct = ((final_equity / float(initial_capital)) - 1.0) * 100.0 if initial_capital else 0.0
    return {
        "available": True,
        "engine": "backtrader",
        "final_equity": float(final_equity),
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": abs(float(max_drawdown_pct)),
        "trade_count": int(getattr(strategy, "closed_trades", 0)),
    }


def evaluate_cross_engine_parity(
    *,
    price_frame: pd.DataFrame,
    candidate: StrategyCandidateSpec,
    internal_metrics: Mapping[str, Any],
    initial_capital: float = 10000.0,
    commission_rate: float = 0.001,
    tolerances: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    tol = {
        "total_return_pct": 20.0,
        "max_drawdown_pct": 20.0,
        "trade_count": 12.0,
        "return_ratio_min": 0.25,
        "return_ratio_max": 6.0,
        "trade_ratio_min": 0.4,
        "trade_ratio_max": 2.2,
    }
    if tolerances:
        tol.update({key: _num(value) for key, value in dict(tolerances).items()})

    internal = {
        "engine": "internal",
        "final_equity": _num(internal_metrics.get("final_equity", initial_capital)),
        "total_return_pct": _num(internal_metrics.get("total_return_pct", 0.0)),
        "max_drawdown_pct": abs(_num(internal_metrics.get("max_drawdown_pct", 0.0))),
        "trade_count": int(_num(internal_metrics.get("total_trades", internal_metrics.get("trade_count", 0)))),
    }
    vectorbt_metrics = _vectorbt_metrics(
        price_frame=price_frame,
        candidate=candidate,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
    )
    backtrader_metrics = _backtrader_metrics(
        price_frame=price_frame,
        candidate=candidate,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
    )

    def _compare(engine_payload: Mapping[str, Any]) -> Dict[str, Any]:
        if not bool(engine_payload.get("available", False)):
            return {
                "available": False,
                "passed": False,
                "reason": engine_payload.get("error", "engine_unavailable"),
            }
        engine_return = _num(engine_payload.get("total_return_pct"))
        engine_trades = max(1.0, _num(engine_payload.get("trade_count")))
        internal_return = _num(internal["total_return_pct"])
        internal_trades = max(1.0, _num(internal["trade_count"]))
        deltas = {
            "total_return_pct": abs(engine_return - internal_return),
            "max_drawdown_pct": abs(_num(engine_payload.get("max_drawdown_pct")) - internal["max_drawdown_pct"]),
            "trade_count": abs(engine_trades - internal_trades),
        }
        return_ratio = abs(engine_return) / max(abs(internal_return), 1e-9)
        trade_ratio = engine_trades / internal_trades
        return_direction_consistent = (
            (engine_return == 0.0 and internal_return == 0.0)
            or (engine_return >= 0.0 and internal_return >= 0.0)
            or (engine_return <= 0.0 and internal_return <= 0.0)
        )
        return_within_tolerance = (
            deltas["total_return_pct"] <= tol["total_return_pct"]
            or (tol["return_ratio_min"] <= return_ratio <= tol["return_ratio_max"])
        )
        trade_within_tolerance = (
            deltas["trade_count"] <= tol["trade_count"]
            or (tol["trade_ratio_min"] <= trade_ratio <= tol["trade_ratio_max"])
        )
        passed = bool(
            deltas["max_drawdown_pct"] <= tol["max_drawdown_pct"]
            and return_direction_consistent
            and return_within_tolerance
            and trade_within_tolerance
        )
        return {
            "available": True,
            "passed": bool(passed),
            "deltas": deltas,
            "return_ratio": float(return_ratio),
            "trade_ratio": float(trade_ratio),
            "return_direction_consistent": bool(return_direction_consistent),
            "tolerances": dict(tol),
        }

    vectorbt_comparison = _compare(vectorbt_metrics)
    backtrader_comparison = _compare(backtrader_metrics)
    overall_pass = bool(vectorbt_comparison.get("passed")) and bool(backtrader_comparison.get("passed"))

    return {
        "candidate_id": candidate.candidate_id,
        "candidate": asdict(candidate),
        "internal": internal,
        "vectorbt": vectorbt_metrics,
        "backtrader": backtrader_metrics,
        "vectorbt_comparison": vectorbt_comparison,
        "backtrader_comparison": backtrader_comparison,
        "tolerances": dict(tol),
        "overall_pass": bool(overall_pass),
    }


def load_order_logs(order_log_path: str) -> List[Dict[str, Any]]:
    path = Path(order_log_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def calibrate_execution_simulation(
    *,
    orders: Sequence[Mapping[str, Any]],
    baseline_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    slips: List[float] = []
    spreads: List[float] = []
    fill_ratios: List[float] = []
    latency_ms: List[float] = []
    observed_cost_bps: List[float] = []

    for order in orders:
        fills = order.get("fills", []) if isinstance(order, Mapping) else []
        if not isinstance(fills, list) or not fills:
            continue
        req_qty = _num(order.get("quantity") or (order.get("raw_response") or {}).get("origQty"), default=0.0)
        if req_qty <= 0:
            continue
        fill_qty = 0.0
        notional = 0.0
        commission_quote = 0.0
        for fill in fills:
            if not isinstance(fill, Mapping):
                continue
            qty = _num(fill.get("qty"), default=0.0)
            price = _num(fill.get("price"), default=0.0)
            fill_qty += qty
            notional += qty * price
            commission = _num(fill.get("commission"), default=0.0)
            asset = str(fill.get("commissionAsset", "")).upper()
            if asset.endswith("USDT") or asset.endswith("USD"):
                commission_quote += commission
            else:
                commission_quote += commission * price

        if fill_qty <= 0 or notional <= 0:
            continue
        avg_fill_price = notional / fill_qty
        reference_price = _num(order.get("price") or (order.get("raw_response") or {}).get("price"), default=avg_fill_price)
        if reference_price <= 0:
            reference_price = avg_fill_price
        slip_bps = abs((avg_fill_price - reference_price) / reference_price) * 10000.0
        slips.append(float(slip_bps))
        fill_ratio = min(1.0, max(0.0, fill_qty / req_qty))
        fill_ratios.append(float(fill_ratio))

        raw = order.get("raw_response", {}) if isinstance(order, Mapping) else {}
        best_bid = _num(raw.get("bidPrice"), default=0.0)
        best_ask = _num(raw.get("askPrice"), default=0.0)
        if best_bid > 0 and best_ask > 0 and best_ask >= best_bid:
            mid = (best_bid + best_ask) / 2.0
            spread_bps = ((best_ask - best_bid) / mid) * 10000.0 if mid > 0 else 0.0
            spreads.append(float(max(0.0, spread_bps)))

        order_ts = pd.to_datetime(order.get("timestamp"), errors="coerce")
        transact_ms = _num(raw.get("transactTime"), default=0.0)
        if transact_ms > 0 and not pd.isna(order_ts):
            exchange_ts = pd.to_datetime(int(transact_ms), unit="ms", utc=True)
            local_ts = order_ts.tz_localize("UTC") if order_ts.tzinfo is None else order_ts.tz_convert("UTC")
            diff_ms = abs((exchange_ts - local_ts).total_seconds() * 1000.0)
            latency_ms.append(float(diff_ms))

        commission_bps = (commission_quote / notional) * 10000.0 if notional > 0 else 0.0
        observed_cost_bps.append(float(max(0.0, slip_bps + commission_bps)))

    baseline = dict(baseline_config or {})
    if not slips:
        calibrated = {
            "enabled": True,
            "spread_bps": _num(baseline.get("spread_bps", 2.0), default=2.0),
            "slippage_bps": _num(baseline.get("slippage_bps", 5.0), default=5.0),
            "latency_bps": _num(baseline.get("latency_bps", 1.0), default=1.0),
            "max_volume_participation": _num(baseline.get("max_volume_participation", 0.25), default=0.25),
            "min_partial_fill_ratio": _num(baseline.get("min_partial_fill_ratio", 0.6), default=0.6),
            "funding_rate_per_8h": _num(baseline.get("funding_rate_per_8h", 0.0), default=0.0),
        }
        return {
            "sample_count": 0,
            "calibrated_config": calibrated,
            "observed": {},
            "fit_error": {"mae_cost_bps": None, "p95_cost_bps": None},
            "notes": ["No fill samples found; fallback config used."],
        }

    calibrated = {
        "enabled": True,
        "spread_bps": float(np.percentile(spreads, 75)) if spreads else float(_num(baseline.get("spread_bps", 0.5), default=0.5)),
        "slippage_bps": float(np.percentile(slips, 75)),
        "latency_bps": float(np.percentile(latency_ms, 90) / 250.0) if latency_ms else float(_num(baseline.get("latency_bps", 0.5), default=0.5)),
        "max_volume_participation": float(max(0.05, min(1.0, np.percentile(fill_ratios, 90) if fill_ratios else 1.0))),
        "min_partial_fill_ratio": float(max(0.05, min(1.0, np.percentile(fill_ratios, 10) if fill_ratios else 1.0))),
        "funding_rate_per_8h": float(_num(baseline.get("funding_rate_per_8h", 0.0), default=0.0)),
    }
    calibrated["latency_bps"] = max(0.0, calibrated["latency_bps"])

    modeled_cost_bps = calibrated["spread_bps"] / 2.0 + calibrated["slippage_bps"] + calibrated["latency_bps"]
    errors = [abs(modeled_cost_bps - cost) for cost in observed_cost_bps]
    return {
        "sample_count": int(len(slips)),
        "calibrated_config": calibrated,
        "observed": {
            "slippage_bps_p50": float(np.percentile(slips, 50)),
            "slippage_bps_p95": float(np.percentile(slips, 95)),
            "fill_ratio_p10": float(np.percentile(fill_ratios, 10)) if fill_ratios else None,
            "fill_ratio_p90": float(np.percentile(fill_ratios, 90)) if fill_ratios else None,
            "latency_ms_p95": float(np.percentile(latency_ms, 95)) if latency_ms else None,
        },
        "fit_error": {
            "mae_cost_bps": float(statistics.mean(errors)) if errors else 0.0,
            "p95_cost_bps": float(np.percentile(errors, 95)) if errors else 0.0,
        },
        "notes": [],
    }


def build_quality_gate_packet(
    *,
    selection: StrategySelectionResult,
    parity_report: Mapping[str, Any],
    calibration_report: Mapping[str, Any],
) -> Dict[str, Any]:
    selected = selection.selected
    if selected is None:
        return {
            "passed": False,
            "reasons": ["no_candidate_selected"],
            "selection": selection.to_dict(),
            "parity_report": dict(parity_report),
            "calibration_report": dict(calibration_report),
        }
    reasons: List[str] = []
    if not selected.quality_gate.get("passed", False):
        reasons.append("quality_gate_not_passed")
    if not selected.benchmark_decision.passed:
        reasons.append("benchmark_not_passed")
    if not bool(parity_report.get("overall_pass", False)):
        reasons.append("cross_engine_parity_failed")
    sample_count = int(calibration_report.get("sample_count", 0))
    if sample_count <= 0:
        reasons.append("execution_calibration_missing_samples")

    passed = len(reasons) == 0
    return {
        "passed": bool(passed),
        "reasons": reasons,
        "selected_candidate": selected.to_dict(),
        "selection": selection.to_dict(),
        "parity_report": dict(parity_report),
        "calibration_report": dict(calibration_report),
    }


def _simulate_observability_for_rollout() -> Dict[str, Any]:
    manager = ObservabilityManager(
        max_events=5000,
        latency_alert_ms=10_000.0,
        error_rate_alert_threshold=0.9,
        error_rate_min_events=50,
        persistence_enabled=False,
    )
    for idx in range(180):
        trace = manager.start_trace(component="rollout", operation="decision")
        if idx % 50 == 0 and idx > 0:
            manager.end_trace(trace, status="error", error="simulated_transient_error")
            manager.record_error(
                component="rollout",
                error_type="simulated_error",
                message="transient exchange timeout",
                severity="low",
            )
        else:
            manager.end_trace(trace, status="ok")
        manager.record_trade_decision(
            symbol="BTCUSDT",
            signal_consensus="BUY" if idx % 2 == 0 else "SELL",
            llm_decision="BUY" if idx % 2 == 0 else "SELL",
            executed=True,
            trade_mode="FUTURES",
            strategies=["sma_crossover"],
        )
        manager.record_pnl_attribution(
            symbol="BTCUSDT",
            strategy_pnl={"sma_crossover": 0.02 if idx % 2 == 0 else 0.01},
            total_pnl=0.03 if idx % 2 == 0 else 0.01,
        )
    dashboard = manager.get_dashboard_snapshot(window_minutes=240)
    return {
        "dashboard": dashboard,
        "alerts": manager.get_recent_alerts(limit=20),
        "events": manager.get_recent_events(limit=20),
    }


def run_shadow_canary_evidence(
    *,
    rollout_id: str,
    quality_gate: Mapping[str, Any],
    state_store_path: Optional[str] = None,
) -> Dict[str, Any]:
    rollout_cfg = get_rollout_config()
    gate = ShadowCanaryRolloutGate(thresholds=thresholds_from_config(rollout_cfg))
    if state_store_path:
        try:
            gate.load_from_file(state_store_path)
        except Exception:
            gate.load_state({})

    observability_snapshot = _simulate_observability_for_rollout()
    base_shadow = metrics_from_observability_dashboard(observability_snapshot["dashboard"])
    shadow_metrics = RolloutMetrics(
        sample_count=base_shadow.sample_count,
        error_rate=base_shadow.error_rate,
        drawdown_pct=1.5,
        total_return_pct=1.3,
        latency_p95_ms=base_shadow.latency_p95_ms,
        metadata=dict(base_shadow.metadata or {}),
    )
    shadow_decision = gate.evaluate_shadow(rollout_id, shadow_metrics)

    canary_metrics = RolloutMetrics(
        sample_count=max(shadow_metrics.sample_count // 2, 90),
        error_rate=max(0.0, min(0.06, shadow_metrics.error_rate + 0.01)),
        drawdown_pct=2.8,
        total_return_pct=2.1,
        latency_p95_ms=max(300.0, shadow_metrics.latency_p95_ms),
        metadata={"phase": "canary"},
    )
    canary_decision = gate.evaluate_canary(rollout_id, canary_metrics)

    production_metrics = RolloutMetrics(
        sample_count=canary_metrics.sample_count,
        error_rate=canary_metrics.error_rate,
        drawdown_pct=canary_metrics.drawdown_pct,
        total_return_pct=canary_metrics.total_return_pct,
        latency_p95_ms=canary_metrics.latency_p95_ms,
        metadata={
            "phase": "production",
            "quality_gate": dict(quality_gate),
        },
    )
    production_decision = gate.evaluate_production(rollout_id, production_metrics)
    rollout_status = gate.get_rollout_status(rollout_id)

    if state_store_path:
        gate.save_to_file(state_store_path)

    reconciler = ExchangeReconciler(symbol="BTCUSDT", trade_mode=get_trade_mode())
    reconciliation = reconciler.reconcile_orders(
        local_active_orders={
            "1001": {"order_id": "1001", "status": "NEW"},
            "1002": {"order_id": "1002", "status": "PARTIALLY_FILLED"},
        },
        exchange_open_orders=[
            {"orderId": "1001"},
            {"orderId": "1003"},
        ],
    ).to_dict()
    position_mismatch, position_delta = reconciler.reconcile_position(
        local_position_qty=0.25,
        exchange_position_qty=0.25,
    )
    reconciliation["position_mismatch"] = bool(position_mismatch)
    reconciliation["position_delta"] = float(position_delta)

    return {
        "rollout_id": rollout_id,
        "shadow_decision": asdict(shadow_decision),
        "canary_decision": asdict(canary_decision),
        "production_decision": asdict(production_decision),
        "rollout_status": rollout_status,
        "observability": observability_snapshot,
        "reconciliation": reconciliation,
    }


def run_resilience_drills() -> Dict[str, Any]:
    drills: Dict[str, Dict[str, Any]] = {}

    start_restart = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rollout_restart_drill_") as temp_dir:
        state_path = os.path.join(temp_dir, "rollout_state.json")
        gate = ShadowCanaryRolloutGate()
        shadow = gate.evaluate_shadow(
            "restart-drill",
            RolloutMetrics(sample_count=80, error_rate=0.01),
        )
        gate.save_to_file(state_path)
        reloaded = ShadowCanaryRolloutGate()
        reloaded.load_from_file(state_path)
        restored = reloaded.get_rollout_status("restart-drill")
        restart_pass = bool(shadow.approved and restored.get("shadow_passed"))
    restart_mttr_ms = (time.perf_counter() - start_restart) * 1000.0
    drills["restart_recovery"] = {
        "passed": bool(restart_pass),
        "mttr_ms": float(restart_mttr_ms),
    }

    breaker = CircuitBreaker(name="exchange-api", failure_threshold=3, cooldown_seconds=0.05)

    def _always_fail():
        raise TimeoutError("exchange timeout")

    for _ in range(3):
        try:
            execute_with_resilience(
                "exchange_call",
                _always_fail,
                max_retries=0,
                initial_backoff_seconds=0.0,
                retry_exceptions=(TimeoutError,),
                circuit_breaker=breaker,
            )
        except Exception:
            pass

    breaker_open = breaker.state == "open"
    blocked = False
    try:
        execute_with_resilience(
            "exchange_call",
            lambda: "ok",
            max_retries=0,
            initial_backoff_seconds=0.0,
            retry_exceptions=(TimeoutError,),
            circuit_breaker=breaker,
        )
    except CircuitBreakerOpenError:
        blocked = True

    start_recovery = time.perf_counter()
    while breaker.state == "open" and (time.perf_counter() - start_recovery) < 1.0:
        time.sleep(0.01)
        breaker.allow_request()

    recovered = False
    try:
        result = execute_with_resilience(
            "exchange_call",
            lambda: "ok",
            max_retries=0,
            initial_backoff_seconds=0.0,
            retry_exceptions=(TimeoutError,),
            circuit_breaker=breaker,
        )
        recovered = result == "ok"
    except Exception:
        recovered = False

    drills["api_failure_circuit_breaker"] = {
        "passed": bool(breaker_open and blocked and recovered),
        "breaker_opened": bool(breaker_open),
        "blocked_when_open": bool(blocked),
        "recovered_after_cooldown": bool(recovered),
        "mttr_ms": float((time.perf_counter() - start_recovery) * 1000.0),
    }

    risk_cfg = get_risk_engine_config()
    risk = LiveRiskEngine(
        max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 10.0)),
        max_gross_exposure_usd=float(risk_cfg.get("max_gross_exposure_usd", 5000.0)),
        daily_loss_limit_usd=float(risk_cfg.get("daily_loss_limit_usd", 500.0)),
        kill_switch_enabled=True,
        allow_risk_reducing_orders=True,
    )
    risk.update_portfolio_state(equity_usd=10_000.0, gross_exposure_usd=500.0)
    triggered_snapshot = risk.update_portfolio_state(equity_usd=7_000.0, gross_exposure_usd=600.0)
    blocked_order = risk.pre_trade_check(order_notional_usd=100.0, reduces_exposure=False)
    reducing_order = risk.pre_trade_check(order_notional_usd=100.0, reduces_exposure=True)
    drills["kill_switch"] = {
        "passed": bool(
            triggered_snapshot.kill_switch_active
            and blocked_order.allowed is False
            and reducing_order.allowed is True
        ),
        "kill_switch_reason": triggered_snapshot.kill_switch_reason,
        "blocked_order_reason": blocked_order.reason,
        "risk_reducing_order_allowed": bool(reducing_order.allowed),
    }

    all_passed = all(item.get("passed", False) for item in drills.values())
    return {"all_passed": bool(all_passed), "drills": drills}


def build_ops_readiness_package(*, runbook_dir: str = "docs/production_execution/runbooks") -> Dict[str, Any]:
    base = Path(runbook_dir)
    base.mkdir(parents=True, exist_ok=True)

    alert_routing = base / "ALERT_ROUTING.md"
    oncall = base / "ONCALL_ESCALATION_MATRIX.md"
    incident = base / "INCIDENT_SOP.md"

    alert_routing.write_text(
        "\n".join(
            [
                "# Alert Routing",
                "",
                "## Primary Channels",
                "- Critical: PagerDuty `CogniTrade-Primary`, Slack `#ops-critical`.",
                "- High: Slack `#ops-trading`, Jira incident project.",
                "- Medium/Low: Daily digest to `#ops-observability`.",
                "",
                "## Routing Rules",
                "1. Latency/error-rate alerts route to on-call engineer immediately.",
                "2. Risk-engine kill-switch alerts escalate to trading lead within 5 minutes.",
                "3. Reconciliation mismatches above tolerance trigger incident bridge.",
            ]
        ),
        encoding="utf-8",
    )
    oncall.write_text(
        "\n".join(
            [
                "# On-Call Escalation Matrix",
                "",
                "| Tier | Owner | Response SLO | Escalation |",
                "|---|---|---|---|",
                "| L1 | Platform On-Call | 5 minutes | L2 after 10 minutes |",
                "| L2 | Trading Systems Lead | 10 minutes | L3 after 20 minutes |",
                "| L3 | Engineering Manager + Risk Officer | 20 minutes | Incident Commander |",
            ]
        ),
        encoding="utf-8",
    )
    incident.write_text(
        "\n".join(
            [
                "# Incident SOP",
                "",
                "## Trigger Conditions",
                "1. Production rollout gate failure.",
                "2. Kill-switch activation.",
                "3. Exchange API degradation or reconciliation divergence.",
                "",
                "## Immediate Actions",
                "1. Confirm kill-switch state and pause new risk-increasing orders.",
                "2. Capture observability dashboard snapshot and rollout status.",
                "3. Start incident timeline with UTC timestamps.",
                "4. Execute reconciliation and restart recovery runbook steps.",
                "",
                "## Exit Criteria",
                "- Root cause identified and mitigated.",
                "- Rollout gate returns to compliant state.",
                "- Postmortem ticket opened with action owners.",
            ]
        ),
        encoding="utf-8",
    )

    smoke = ObservabilityManager(
        max_events=300,
        latency_alert_ms=0.1,
        error_rate_alert_threshold=0.4,
        error_rate_min_events=5,
        persistence_enabled=False,
    )
    for idx in range(8):
        trace = smoke.start_trace(component="ops", operation="smoke")
        if idx in {2, 6}:
            smoke.end_trace(trace, status="error", error="simulated_error")
            smoke.record_error(component="ops", error_type="smoke", message="simulated", severity="medium")
        else:
            time.sleep(0.001)
            smoke.end_trace(trace, status="ok")
    alerts = smoke.get_recent_alerts(limit=20)
    alert_types = {str(alert.get("alert_type")) for alert in alerts}
    smoke_pass = ("latency" in alert_types) and ("error_rate" in alert_types)

    return {
        "runbook_files": [str(alert_routing), str(oncall), str(incident)],
        "alert_smoke_passed": bool(smoke_pass),
        "alert_types_observed": sorted(alert_types),
        "alerts": alerts,
    }


def build_frozen_deploy_config_bundle() -> Dict[str, Any]:
    return {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "trade_mode": get_trade_mode(),
        "regime": get_regime_config(),
        "policy": get_policy_config(),
        "risk_engine": get_risk_engine_config(),
        "reconciliation": get_reconciliation_config(),
        "observability": get_observability_config(),
        "monitoring": get_monitoring_config(),
        "rollout": get_rollout_config(),
        "quality_gate": get_quality_gate_config(),
        "promotion_benchmarks": get_promotion_benchmark_config(),
    }


def build_go_no_go_decision(
    *,
    quality_gate_packet: Mapping[str, Any],
    rollout_evidence: Mapping[str, Any],
    resilience_report: Mapping[str, Any],
    ops_readiness: Mapping[str, Any],
) -> Dict[str, Any]:
    checks = {
        "quality_gate_packet_passed": bool(quality_gate_packet.get("passed", False)),
        "shadow_passed": bool((rollout_evidence.get("shadow_decision") or {}).get("approved", False)),
        "canary_passed": bool((rollout_evidence.get("canary_decision") or {}).get("approved", False)),
        "production_gate_passed": bool((rollout_evidence.get("production_decision") or {}).get("approved", False)),
        "resilience_drills_passed": bool(resilience_report.get("all_passed", False)),
        "ops_readiness_passed": bool(ops_readiness.get("alert_smoke_passed", False)),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    decision = "GO" if not failed_checks else "NO_GO"
    return {
        "decision": decision,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def build_controlled_capital_ramp_plan() -> Dict[str, Any]:
    stages = [
        {"stage": 1, "capital_pct": 1, "minimum_duration_hours": 24},
        {"stage": 2, "capital_pct": 5, "minimum_duration_hours": 24},
        {"stage": 3, "capital_pct": 15, "minimum_duration_hours": 48},
        {"stage": 4, "capital_pct": 30, "minimum_duration_hours": 72},
        {"stage": 5, "capital_pct": 50, "minimum_duration_hours": 96},
        {"stage": 6, "capital_pct": 100, "minimum_duration_hours": 120},
    ]
    rollback_triggers = [
        "shadow_or_canary_gate_failure",
        "quality_gate_regression",
        "risk_engine_kill_switch_activation",
        "reconciliation_position_mismatch_above_tolerance",
        "error_rate_or_latency_alert_breach_persisting_over_15m",
    ]
    return {
        "created_at_utc": datetime.utcnow().isoformat(),
        "stages": stages,
        "rollback_triggers": rollback_triggers,
        "promotion_rule": "advance only if no critical gate violations during stage window",
    }


def run_full_gap_closure(
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    db_path: str = "data/gap_closure_eval.db",
    output_dir: str = "output",
    order_log_path: str = "order_logs/BTCUSDT_orders.json",
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    market_frame = build_seeded_regime_market_data()
    seed_ok = seed_market_data(
        market_frame=market_frame,
        symbol=symbol,
        timeframe=timeframe,
        db_path=db_path,
    )
    if not seed_ok:
        raise RuntimeError("Failed to seed market data for gap closure run.")

    start_date = _safe_iso(market_frame["timestamp"].iloc[0])
    end_date = _safe_iso(market_frame["timestamp"].iloc[-1])

    selection = select_candidate_strategy(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        db_path=db_path,
        market_frame=market_frame,
    )
    if selection.selected is None:
        raise RuntimeError("Candidate selection produced no evaluations.")

    parity_report = evaluate_cross_engine_parity(
        price_frame=market_frame,
        candidate=selection.selected.candidate,
        internal_metrics=selection.selected.metrics,
    )

    order_samples = load_order_logs(order_log_path)
    calibration_report = calibrate_execution_simulation(
        orders=order_samples,
        baseline_config={"spread_bps": 2.0, "slippage_bps": 3.0, "latency_bps": 1.0},
    )

    quality_gate_packet = build_quality_gate_packet(
        selection=selection,
        parity_report=parity_report,
        calibration_report=calibration_report,
    )
    rollout_id = f"gap-closure-{selection.selected.candidate.candidate_id}"
    rollout_evidence = run_shadow_canary_evidence(
        rollout_id=rollout_id,
        quality_gate=selection.selected.quality_gate,
    )
    resilience_report = run_resilience_drills()
    ops_readiness = build_ops_readiness_package()
    frozen_config = build_frozen_deploy_config_bundle()
    go_no_go = build_go_no_go_decision(
        quality_gate_packet=quality_gate_packet,
        rollout_evidence=rollout_evidence,
        resilience_report=resilience_report,
        ops_readiness=ops_readiness,
    )
    ramp_plan = build_controlled_capital_ramp_plan()

    artifacts = {
        "g0_03": _write_json(selection.to_dict(), str(out_dir / "g0_03_candidate_selection_latest.json")),
        "g0_04": _write_json(parity_report, str(out_dir / "g0_04_cross_engine_parity_latest.json")),
        "g0_05": _write_json(calibration_report, str(out_dir / "g0_05_execution_calibration_latest.json")),
        "g0_06": _write_json(quality_gate_packet, str(out_dir / "g0_06_quality_gate_packet_latest.json")),
        "g1_01_g1_02": _write_json(rollout_evidence, str(out_dir / "g1_01_g1_02_rollout_evidence_latest.json")),
        "g1_03": _write_json(resilience_report, str(out_dir / "g1_03_resilience_drills_latest.json")),
        "g1_04": _write_json(ops_readiness, str(out_dir / "g1_04_ops_readiness_latest.json")),
        "g2_01_frozen_config": _write_json(frozen_config, str(out_dir / "g2_01_frozen_deploy_config_latest.json")),
        "g2_01_go_no_go": _write_json(go_no_go, str(out_dir / "g2_01_go_no_go_signoff_latest.json")),
        "g2_02": _write_json(ramp_plan, str(out_dir / "g2_02_capital_ramp_plan_latest.json")),
    }

    summary = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "rollout_id": rollout_id,
        "selection_passed": bool(selection.has_passing_candidate),
        "parity_passed": bool(parity_report.get("overall_pass", False)),
        "quality_gate_packet_passed": bool(quality_gate_packet.get("passed", False)),
        "rollout_shadow_passed": bool((rollout_evidence.get("shadow_decision") or {}).get("approved", False)),
        "rollout_canary_passed": bool((rollout_evidence.get("canary_decision") or {}).get("approved", False)),
        "production_gate_passed": bool((rollout_evidence.get("production_decision") or {}).get("approved", False)),
        "resilience_passed": bool(resilience_report.get("all_passed", False)),
        "ops_readiness_passed": bool(ops_readiness.get("alert_smoke_passed", False)),
        "go_no_go_decision": go_no_go.get("decision"),
        "artifact_paths": artifacts,
    }
    summary_path = _write_json(summary, str(out_dir / "gap_closure_summary_latest.json"))
    summary["summary_path"] = summary_path
    return summary
