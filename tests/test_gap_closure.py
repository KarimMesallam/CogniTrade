import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.deploy_policy import StrategyQualityThresholds
from research.benchmarks import PromotionBenchmarkThresholds
from research.gap_closure import (
    StrategyCandidateSpec,
    build_go_no_go_decision,
    build_ops_readiness_package,
    build_seeded_regime_market_data,
    build_quality_gate_packet,
    calibrate_execution_simulation,
    evaluate_candidate,
    evaluate_cross_engine_parity,
    run_resilience_drills,
    run_shadow_canary_evidence,
    seed_market_data,
    select_candidate_strategy,
)


def _seeded_db(tmp_path, symbol="GAPTESTUSDT", timeframe="1h"):
    frame = build_seeded_regime_market_data()
    db_path = str(tmp_path / "gap_closure_test.db")
    assert seed_market_data(market_frame=frame, symbol=symbol, timeframe=timeframe, db_path=db_path) is True
    start = pd.to_datetime(frame["timestamp"].iloc[0]).isoformat()
    end = pd.to_datetime(frame["timestamp"].iloc[-1]).isoformat()
    return frame, db_path, start, end


def test_select_candidate_strategy_returns_evaluations(tmp_path):
    frame, db_path, start, end = _seeded_db(tmp_path)
    candidates = [
        StrategyCandidateSpec(
            strategy_name="sma_crossover",
            params={"short_period": 10, "long_period": 50},
            allow_short_positions=True,
        ),
        StrategyCandidateSpec(
            strategy_name="rsi",
            params={"period": 14, "overbought": 70, "oversold": 30},
            allow_short_positions=True,
        ),
    ]
    selection = select_candidate_strategy(
        symbol="GAPTESTUSDT",
        timeframe="1h",
        start_date=start,
        end_date=end,
        db_path=db_path,
        market_frame=frame,
        candidates=candidates,
        benchmark_thresholds=PromotionBenchmarkThresholds(
            min_total_trades=1,
            min_net_return_pct=-100.0,
            min_sharpe_ratio=-10.0,
            min_calmar_ratio=-10.0,
            max_drawdown_pct=100.0,
            require_quality_gate=False,
        ),
        quality_thresholds=StrategyQualityThresholds(
            min_walk_forward_folds=0,
            min_sharpe_ratio=-10.0,
            min_calmar_ratio=-10.0,
            max_drawdown_pct=100.0,
            min_regime_samples=1,
            min_regime_win_rate=0.0,
            min_regimes_passing=0,
        ),
    )
    assert selection.selected is not None
    assert len(selection.evaluations) == 2
    assert "total_return_pct" in selection.selected.metrics


def test_cross_engine_parity_report_has_required_keys(tmp_path):
    frame, db_path, start, end = _seeded_db(tmp_path)
    candidate = StrategyCandidateSpec(
        strategy_name="sma_crossover",
        params={"short_period": 10, "long_period": 50},
        allow_short_positions=True,
    )
    evaluation = evaluate_candidate(
        candidate=candidate,
        symbol="GAPTESTUSDT",
        timeframe="1h",
        start_date=start,
        end_date=end,
        db_path=db_path,
        market_frame=frame,
        benchmark_thresholds=PromotionBenchmarkThresholds(require_quality_gate=False),
        quality_thresholds=StrategyQualityThresholds(
            min_walk_forward_folds=0,
            min_regimes_passing=0,
            min_regime_samples=1,
            min_regime_win_rate=0.0,
            min_sharpe_ratio=-10.0,
            min_calmar_ratio=-10.0,
            max_drawdown_pct=100.0,
        ),
    )
    report = evaluate_cross_engine_parity(
        price_frame=frame,
        candidate=candidate,
        internal_metrics=evaluation.metrics,
        tolerances={"total_return_pct": 1000.0, "max_drawdown_pct": 1000.0, "trade_count": 1000.0},
    )
    assert "vectorbt_comparison" in report
    assert "backtrader_comparison" in report
    assert report["vectorbt"].get("available") is True
    assert report["backtrader"].get("available") is True


def test_execution_calibration_uses_fill_samples():
    orders = [
        {
            "timestamp": "2026-02-11T00:00:00",
            "price": "50000.0",
            "quantity": "0.01",
            "fills": [
                {"price": "50005.0", "qty": "0.006", "commission": "0.1", "commissionAsset": "USDT"},
                {"price": "50006.0", "qty": "0.004", "commission": "0.08", "commissionAsset": "USDT"},
            ],
            "raw_response": {"origQty": "0.01", "transactTime": 1760000000000},
        }
    ]
    report = calibrate_execution_simulation(orders=orders)
    assert report["sample_count"] == 1
    assert report["calibrated_config"]["enabled"] is True
    assert report["calibrated_config"]["slippage_bps"] >= 0.0


def test_quality_packet_and_go_no_go_decision(tmp_path):
    frame, db_path, start, end = _seeded_db(tmp_path)
    candidate = StrategyCandidateSpec(
        strategy_name="sma_crossover",
        params={"short_period": 10, "long_period": 50},
        allow_short_positions=True,
    )
    selection = select_candidate_strategy(
        symbol="GAPTESTUSDT",
        timeframe="1h",
        start_date=start,
        end_date=end,
        db_path=db_path,
        market_frame=frame,
        candidates=[candidate],
        benchmark_thresholds=PromotionBenchmarkThresholds(
            min_total_trades=1,
            min_net_return_pct=-100.0,
            min_sharpe_ratio=-10.0,
            min_calmar_ratio=-10.0,
            max_drawdown_pct=100.0,
            require_quality_gate=False,
        ),
        quality_thresholds=StrategyQualityThresholds(
            min_walk_forward_folds=0,
            min_regimes_passing=0,
            min_regime_samples=1,
            min_regime_win_rate=0.0,
            min_sharpe_ratio=-10.0,
            min_calmar_ratio=-10.0,
            max_drawdown_pct=100.0,
        ),
    )
    parity = {
        "overall_pass": True,
    }
    calibration = {"sample_count": 3}
    packet = build_quality_gate_packet(
        selection=selection,
        parity_report=parity,
        calibration_report=calibration,
    )
    assert packet["passed"] is True

    rollout = run_shadow_canary_evidence(
        rollout_id="pytest-gap-rollout",
        quality_gate={"passed": True, "reasons": []},
        state_store_path=str(tmp_path / "rollout_state.json"),
    )
    resilience = run_resilience_drills()
    ops = build_ops_readiness_package(runbook_dir=str(tmp_path / "runbooks"))
    decision = build_go_no_go_decision(
        quality_gate_packet=packet,
        rollout_evidence=rollout,
        resilience_report=resilience,
        ops_readiness=ops,
    )
    assert decision["decision"] in {"GO", "NO_GO"}
    assert isinstance(decision["checks"], dict)

