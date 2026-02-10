import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.deploy_policy import (
    RolloutMetrics,
    RolloutThresholds,
    ShadowCanaryRolloutGate,
    StrategyQualityThresholds,
    evaluate_strategy_quality,
    metrics_from_observability_dashboard,
    thresholds_from_config,
)


def _gate() -> ShadowCanaryRolloutGate:
    return ShadowCanaryRolloutGate(
        thresholds=RolloutThresholds(
            min_shadow_samples=10,
            max_shadow_error_rate=0.20,
            min_canary_samples=8,
            max_canary_error_rate=0.15,
            max_canary_drawdown_pct=5.0,
            min_canary_total_return_pct=-0.5,
            max_canary_latency_p95_ms=2000.0,
            require_quality_gate=False,
        )
    )


def test_canary_blocked_without_shadow_approval():
    gate = _gate()
    decision = gate.evaluate_canary(
        "rollout-1",
        RolloutMetrics(sample_count=12, error_rate=0.05, drawdown_pct=2.0, total_return_pct=1.0),
    )

    assert decision.approved is False
    assert "shadow_stage_not_approved" in decision.reasons
    status = gate.get_rollout_status("rollout-1")
    assert status["canary_passed"] is False


def test_shadow_failure_prevents_progression():
    gate = _gate()
    shadow = gate.evaluate_shadow(
        "rollout-2",
        RolloutMetrics(sample_count=4, error_rate=0.30),
    )
    assert shadow.approved is False
    assert "shadow_sample_count_below_minimum" in shadow.reasons
    assert "shadow_error_rate_above_threshold" in shadow.reasons

    canary = gate.evaluate_canary(
        "rollout-2",
        RolloutMetrics(sample_count=20, error_rate=0.01, drawdown_pct=1.0, total_return_pct=2.0),
    )
    assert canary.approved is False
    assert "shadow_stage_not_approved" in canary.reasons


def test_shadow_and_canary_pass_enable_production_promotion():
    gate = _gate()
    shadow = gate.evaluate_shadow(
        "rollout-3",
        RolloutMetrics(sample_count=20, error_rate=0.05),
    )
    assert shadow.approved is True

    canary = gate.evaluate_canary(
        "rollout-3",
        RolloutMetrics(
            sample_count=15,
            error_rate=0.10,
            drawdown_pct=2.0,
            total_return_pct=1.0,
            latency_p95_ms=500.0,
        ),
    )
    assert canary.approved is True

    prod = gate.evaluate_production(
        "rollout-3",
        RolloutMetrics(
            sample_count=15,
            error_rate=0.08,
            drawdown_pct=1.5,
            total_return_pct=1.2,
            latency_p95_ms=700.0,
        ),
    )
    assert prod.approved is True
    assert prod.next_required_stage is None

    status = gate.get_rollout_status("rollout-3")
    assert status["shadow_passed"] is True
    assert status["canary_passed"] is True
    assert status["production_passed"] is True
    assert len(status["history"]) == 3


def test_production_rejected_when_canary_metrics_degrade():
    gate = _gate()
    gate.evaluate_shadow("rollout-4", RolloutMetrics(sample_count=20, error_rate=0.01))
    gate.evaluate_canary(
        "rollout-4",
        RolloutMetrics(sample_count=12, error_rate=0.10, drawdown_pct=2.0, total_return_pct=1.0),
    )
    prod = gate.evaluate_production(
        "rollout-4",
        RolloutMetrics(sample_count=12, error_rate=0.25, drawdown_pct=9.0, total_return_pct=-2.0),
    )

    assert prod.approved is False
    assert "canary_error_rate_above_threshold" in prod.reasons
    assert "canary_drawdown_above_threshold" in prod.reasons
    assert "canary_return_below_threshold" in prod.reasons


def test_metrics_from_observability_dashboard_mapping():
    dashboard = {
        "window_minutes": 60.0,
        "errors": {"error_rate": 0.07},
        "latency": {"p95_ms": 450.0},
        "trade_decisions": {"count": 30.0},
        "pnl_attribution": {"total_pnl": 1.8},
        "drawdown_pct": 2.5,
    }

    metrics = metrics_from_observability_dashboard(dashboard)
    assert metrics.sample_count == 30
    assert metrics.error_rate == 0.07
    assert metrics.latency_p95_ms == 450.0
    assert metrics.total_return_pct == 1.8
    assert metrics.drawdown_pct == 2.5


def test_production_requires_quality_gate_evidence_when_enabled():
    gate = ShadowCanaryRolloutGate(
        thresholds=RolloutThresholds(
            min_shadow_samples=5,
            max_shadow_error_rate=0.20,
            min_canary_samples=5,
            max_canary_error_rate=0.15,
            max_canary_drawdown_pct=5.0,
            min_canary_total_return_pct=-0.5,
            max_canary_latency_p95_ms=2000.0,
            require_quality_gate=True,
        )
    )

    gate.evaluate_shadow("rollout-quality", RolloutMetrics(sample_count=10, error_rate=0.01))
    gate.evaluate_canary(
        "rollout-quality",
        RolloutMetrics(sample_count=10, error_rate=0.01, drawdown_pct=1.0, total_return_pct=0.5),
    )
    prod = gate.evaluate_production(
        "rollout-quality",
        RolloutMetrics(sample_count=10, error_rate=0.01, drawdown_pct=1.0, total_return_pct=0.5),
    )
    assert prod.approved is False
    assert "missing_quality_gate_evidence" in prod.reasons

    prod_ok = gate.evaluate_production(
        "rollout-quality",
        RolloutMetrics(
            sample_count=10,
            error_rate=0.01,
            drawdown_pct=1.0,
            total_return_pct=0.5,
            metadata={"quality_gate": {"passed": True, "reasons": []}},
        ),
    )
    assert prod_ok.approved is True


def test_rollout_state_roundtrip_to_file(tmp_path):
    gate = _gate()
    gate.evaluate_shadow("rollout-file", RolloutMetrics(sample_count=12, error_rate=0.01))
    state_path = tmp_path / "rollout_state.json"
    gate.save_to_file(str(state_path))

    loaded = _gate()
    loaded.load_from_file(str(state_path))
    status = loaded.get_rollout_status("rollout-file")
    assert status["shadow_passed"] is True
    assert len(status["history"]) == 1


def test_rollout_state_save_is_atomic_and_json_parseable(tmp_path):
    gate = _gate()
    gate.evaluate_shadow("rollout-atomic", RolloutMetrics(sample_count=12, error_rate=0.01))
    state_path = tmp_path / "rollout_state.json"
    gate.save_to_file(str(state_path))

    # Re-saving should atomically replace without leaving partial tmp files behind.
    gate.evaluate_canary(
        "rollout-atomic",
        RolloutMetrics(sample_count=10, error_rate=0.05, drawdown_pct=1.0, total_return_pct=0.2),
    )
    gate.save_to_file(str(state_path))

    loaded = _gate()
    loaded.load_from_file(str(state_path))
    status = loaded.get_rollout_status("rollout-atomic")
    assert status["shadow_passed"] is True
    assert status["canary_passed"] is True
    assert len(list(tmp_path.glob(".rollout_state_*.tmp"))) == 0


def test_rollout_state_load_recovers_from_corrupt_json(tmp_path):
    state_path = tmp_path / "rollout_state_corrupt.json"
    state_path.write_text("{invalid-json", encoding="utf-8")

    gate = _gate()
    gate.load_from_file(str(state_path))

    # Corrupt files are quarantined and state falls back safely to empty.
    status = gate.get_rollout_status("missing-rollout")
    assert status["shadow_passed"] is False
    backups = list(tmp_path.glob("rollout_state_corrupt.json.corrupt.*"))
    assert len(backups) == 1
    assert not state_path.exists()


def test_strategy_quality_evaluation_thresholds():
    summary = {
        "walk_forward_folds": 4,
        "walk_forward_summary": {
            "sharpe_ratio": {"mean": 0.25},
            "calmar_ratio": {"mean": 0.15},
            "max_drawdown_pct": {"mean": -10.0},
        },
        "regime_slices": {
            "BEAR": {"sample_count": 30.0, "win_rate": 0.5},
            "SIDEWAYS": {"sample_count": 25.0, "win_rate": 0.48},
        },
    }
    decision = evaluate_strategy_quality(summary, StrategyQualityThresholds())
    assert decision.passed is True

    weak_summary = {
        "walk_forward_folds": 1,
        "walk_forward_summary": {"sharpe_ratio": {"mean": -0.1}, "calmar_ratio": {"mean": 0.0}},
        "regime_slices": {"BEAR": {"sample_count": 10.0, "win_rate": 0.2}},
    }
    weak = evaluate_strategy_quality(weak_summary, StrategyQualityThresholds())
    assert weak.passed is False
    assert "quality_walk_forward_folds_below_minimum" in weak.reasons
    assert "quality_sharpe_below_threshold" in weak.reasons


def test_thresholds_from_config_mapping():
    cfg = {
        "min_shadow_samples": 7,
        "max_shadow_error_rate": 0.11,
        "require_quality_gate": False,
    }
    thresholds = thresholds_from_config(cfg)
    assert thresholds.min_shadow_samples == 7
    assert thresholds.max_shadow_error_rate == 0.11
    assert thresholds.require_quality_gate is False
