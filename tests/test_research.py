import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.backtesting.models.results import BacktestResult, PerformanceMetrics
from bot.backtesting.research import register_backtest_experiment
from bot.deploy_policy import StrategyQualityThresholds
from research.benchmarks import (
    PromotionBenchmarkThresholds,
    build_promotion_benchmark_spec,
    evaluate_promotion_candidate,
    promotion_thresholds_from_config,
)
from research.registry import ExperimentRegistry, fingerprint_dataframe


TEST_RESEARCH_DB = os.path.join(os.path.dirname(__file__), "test_research.db")


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="H"),
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [10, 11, 12, 13, 14],
        }
    )


def _result(strategy_name: str, sharpe: float, total_return: float) -> BacktestResult:
    metrics = PerformanceMetrics(
        total_return_pct=total_return,
        sharpe_ratio=sharpe,
        max_drawdown_pct=-4.0,
        win_rate=60.0,
        profit_factor=1.4,
    )
    return BacktestResult(
        symbol="BTCUSDT",
        strategy_name=strategy_name,
        timeframes=["1h"],
        start_date="2026-01-01",
        end_date="2026-01-31",
        initial_capital=1000.0,
        final_equity=1100.0,
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        metrics=metrics,
    )


def setup_function(_):
    if os.path.exists(TEST_RESEARCH_DB):
        os.remove(TEST_RESEARCH_DB)


def teardown_function(_):
    if os.path.exists(TEST_RESEARCH_DB):
        os.remove(TEST_RESEARCH_DB)


def test_dataset_fingerprint_is_deterministic():
    frame = _sample_frame()
    f1 = fingerprint_dataframe(frame)
    f2 = fingerprint_dataframe(frame.copy())
    assert f1 == f2

    changed = frame.copy()
    changed.loc[0, "close"] = 999.0
    f3 = fingerprint_dataframe(changed)
    assert f3 != f1


def test_dataset_registration_is_idempotent_by_fingerprint():
    registry = ExperimentRegistry(TEST_RESEARCH_DB)
    frame = _sample_frame()

    first = registry.register_dataset(name="btc_1h", dataframe=frame, metadata={"source": "unit"})
    second = registry.register_dataset(name="btc_1h_repeat", dataframe=frame.copy(), metadata={"source": "unit"})

    assert first.dataset_id == second.dataset_id
    assert first.fingerprint == second.fingerprint


def test_experiment_registration_is_idempotent_for_same_key():
    registry = ExperimentRegistry(TEST_RESEARCH_DB)
    dataset = registry.register_dataset(name="btc_1h", dataframe=_sample_frame())

    first = registry.register_experiment(
        experiment_name="exp-a",
        strategy_name="sma",
        dataset_fingerprint=dataset.fingerprint,
        params={"a": 1},
        metrics={"sharpe_ratio": 1.2, "total_return_pct": 8.0},
    )
    second = registry.register_experiment(
        experiment_name="exp-a",
        strategy_name="sma",
        dataset_fingerprint=dataset.fingerprint,
        params={"a": 1},
        metrics={"sharpe_ratio": 9.9},
    )

    assert first.experiment_id == second.experiment_id


def test_leaderboard_orders_by_metric_desc():
    registry = ExperimentRegistry(TEST_RESEARCH_DB)
    dataset = registry.register_dataset(name="btc_1h", dataframe=_sample_frame())

    registry.register_experiment(
        experiment_name="exp-low",
        strategy_name="sma",
        dataset_fingerprint=dataset.fingerprint,
        params={"p": 1},
        metrics={"sharpe_ratio": 0.5},
    )
    registry.register_experiment(
        experiment_name="exp-high",
        strategy_name="sma",
        dataset_fingerprint=dataset.fingerprint,
        params={"p": 2},
        metrics={"sharpe_ratio": 1.8},
    )

    leaderboard = registry.get_leaderboard(metric="sharpe_ratio", strategy_name="sma", limit=2)
    assert len(leaderboard) == 2
    assert leaderboard[0]["metric_value"] >= leaderboard[1]["metric_value"]
    assert leaderboard[0]["experiment_name"] == "exp-high"


def test_reproducibility_check_detects_drift():
    registry = ExperimentRegistry(TEST_RESEARCH_DB)
    dataset = registry.register_dataset(name="btc_1h", dataframe=_sample_frame())
    record = registry.register_experiment(
        experiment_name="exp-check",
        strategy_name="sma",
        dataset_fingerprint=dataset.fingerprint,
        params={"p": 5},
        metrics={"sharpe_ratio": 1.0, "total_return_pct": 5.0},
    )

    match = registry.check_reproducibility(
        experiment_id=record.experiment_id,
        metrics={"sharpe_ratio": 1.0, "total_return_pct": 5.0},
        tolerance=1e-12,
    )
    assert match["reproducible"] is True

    mismatch = registry.check_reproducibility(
        experiment_id=record.experiment_id,
        metrics={"sharpe_ratio": 1.0, "total_return_pct": 7.5},
        tolerance=1e-12,
    )
    assert mismatch["reproducible"] is False
    assert mismatch["deltas"]["total_return_pct"] > 0


def test_register_backtest_experiment_helper_tracks_metrics():
    registry = ExperimentRegistry(TEST_RESEARCH_DB)
    dataset = registry.register_dataset(name="btc_1h", dataframe=_sample_frame())

    record = register_backtest_experiment(
        registry=registry,
        experiment_name="wf_validation_v1",
        result=_result("sma_strategy", sharpe=1.6, total_return=12.0),
        dataset_fingerprint=dataset.fingerprint,
        strategy_params={"short": 10, "long": 50},
        validation_summary={"folds": 5},
        execution_config={"slippage_bps": 3},
    )

    assert record.experiment_name == "wf_validation_v1"
    assert record.strategy_name == "sma_strategy"
    assert float(record.metrics["sharpe_ratio"]) == 1.6
    assert float(record.metrics["total_return_pct"]) == 12.0


def test_build_promotion_benchmark_spec_has_deterministic_fingerprint():
    thresholds = PromotionBenchmarkThresholds(
        min_total_trades=30,
        min_net_return_pct=2.0,
        min_sharpe_ratio=0.3,
        min_calmar_ratio=0.2,
        max_drawdown_pct=20.0,
    )
    quality = StrategyQualityThresholds(
        min_walk_forward_folds=4,
        min_sharpe_ratio=0.25,
        min_calmar_ratio=0.15,
        max_drawdown_pct=18.0,
        min_regime_samples=25,
        min_regime_win_rate=0.5,
        min_regimes_passing=2,
    )
    first = build_promotion_benchmark_spec(thresholds=thresholds, quality_thresholds=quality)
    second = build_promotion_benchmark_spec(thresholds=thresholds, quality_thresholds=quality)
    assert first["spec_fingerprint"] == second["spec_fingerprint"]


def test_evaluate_promotion_candidate_enforces_activity_and_net_return_floors():
    thresholds = PromotionBenchmarkThresholds(
        min_total_trades=20,
        min_net_return_pct=3.0,
        min_sharpe_ratio=0.4,
        min_calmar_ratio=0.2,
        max_drawdown_pct=20.0,
        require_quality_gate=False,
    )
    passed = evaluate_promotion_candidate(
        metrics={
            "total_return_pct": 8.0,
            "modeled_cost_pct": 2.5,
            "total_trades": 42,
            "sharpe_ratio": 0.7,
            "calmar_ratio": 0.35,
            "max_drawdown_pct": -12.0,
        },
        thresholds=thresholds,
    )
    assert passed.passed is True
    assert passed.reasons == []

    failed = evaluate_promotion_candidate(
        metrics={
            "total_return_pct": 2.0,
            "modeled_cost_pct": 1.5,
            "total_trades": 7,
            "sharpe_ratio": 0.2,
            "calmar_ratio": 0.1,
            "max_drawdown_pct": -28.0,
        },
        thresholds=thresholds,
    )
    assert failed.passed is False
    assert "benchmark_trade_activity_below_floor" in failed.reasons
    assert "benchmark_net_return_below_floor" in failed.reasons
    assert "benchmark_sharpe_below_floor" in failed.reasons
    assert "benchmark_calmar_below_floor" in failed.reasons
    assert "benchmark_drawdown_above_cap" in failed.reasons


def test_evaluate_promotion_candidate_includes_quality_gate_when_enabled():
    thresholds = PromotionBenchmarkThresholds(require_quality_gate=True)
    decision = evaluate_promotion_candidate(
        metrics={
            "net_return_pct": 10.0,
            "total_trades": 55,
            "sharpe_ratio": 1.0,
            "calmar_ratio": 0.6,
            "max_drawdown_pct": -10.0,
        },
        validation_summary={
            "walk_forward_folds": 1,
            "walk_forward_summary": {},
            "regime_slices": {},
        },
        thresholds=thresholds,
    )
    assert decision.passed is False
    assert "benchmark_quality_gate_failed" in decision.reasons
    assert decision.quality_gate is not None
    assert decision.quality_gate["passed"] is False


def test_promotion_thresholds_from_config_maps_expected_values():
    benchmark_cfg = {
        "min_total_trades": 33,
        "min_net_return_pct": 4.5,
        "min_sharpe_ratio": 0.55,
        "min_calmar_ratio": 0.25,
        "max_drawdown_pct": 19.0,
        "require_quality_gate": False,
    }
    quality_cfg = {
        "min_walk_forward_folds": 4,
        "min_sharpe_ratio": 0.30,
        "min_calmar_ratio": 0.20,
        "max_drawdown_pct": 17.0,
        "min_regime_samples": 18,
        "min_regime_win_rate": 0.5,
        "min_regimes_passing": 3,
    }
    thresholds, quality = promotion_thresholds_from_config(benchmark_cfg, quality_cfg)
    assert thresholds.min_total_trades == 33
    assert thresholds.min_net_return_pct == 4.5
    assert thresholds.require_quality_gate is False
    assert quality.min_walk_forward_folds == 4
    assert quality.min_regime_samples == 18
