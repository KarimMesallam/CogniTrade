from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from typing import Any, Dict, Mapping, Optional, Tuple

from bot.deploy_policy import (
    StrategyQualityDecision,
    StrategyQualityThresholds,
    evaluate_strategy_quality,
)


@dataclass(frozen=True)
class PromotionBenchmarkThresholds:
    """Promotion criteria including activity and net-return floors."""

    min_total_trades: int = 25
    min_net_return_pct: float = 1.0
    min_sharpe_ratio: float = 0.20
    min_calmar_ratio: float = 0.10
    max_drawdown_pct: float = 25.0
    require_quality_gate: bool = True


@dataclass(frozen=True)
class PromotionBenchmarkDecision:
    """Pass/fail result for promotion benchmarks."""

    passed: bool
    reasons: list[str]
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    quality_gate: Optional[Dict[str, Any]]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_quality_thresholds(raw: Optional[Mapping[str, Any]]) -> StrategyQualityThresholds:
    config = dict(raw or {})
    return StrategyQualityThresholds(
        min_walk_forward_folds=int(config.get("min_walk_forward_folds", 3)),
        min_sharpe_ratio=float(config.get("min_sharpe_ratio", 0.20)),
        min_calmar_ratio=float(config.get("min_calmar_ratio", 0.10)),
        max_drawdown_pct=float(config.get("max_drawdown_pct", 25.0)),
        min_regime_samples=int(config.get("min_regime_samples", 20)),
        min_regime_win_rate=float(config.get("min_regime_win_rate", 0.45)),
        min_regimes_passing=int(config.get("min_regimes_passing", 2)),
    )


def promotion_thresholds_from_config(
    benchmark_config: Optional[Mapping[str, Any]] = None,
    quality_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[PromotionBenchmarkThresholds, StrategyQualityThresholds]:
    """
    Build benchmark and quality thresholds from config dictionaries.
    """
    bench = dict(benchmark_config or {})
    thresholds = PromotionBenchmarkThresholds(
        min_total_trades=int(bench.get("min_total_trades", 25)),
        min_net_return_pct=float(bench.get("min_net_return_pct", 1.0)),
        min_sharpe_ratio=float(bench.get("min_sharpe_ratio", 0.20)),
        min_calmar_ratio=float(bench.get("min_calmar_ratio", 0.10)),
        max_drawdown_pct=float(bench.get("max_drawdown_pct", 25.0)),
        require_quality_gate=bool(bench.get("require_quality_gate", True)),
    )
    return thresholds, _coerce_quality_thresholds(quality_config)


def _quality_gate_payload(decision: StrategyQualityDecision) -> Dict[str, Any]:
    return {
        "passed": bool(decision.passed),
        "reasons": list(decision.reasons),
        "metrics": dict(decision.metrics),
    }


def evaluate_promotion_candidate(
    *,
    metrics: Mapping[str, Any],
    validation_summary: Optional[Mapping[str, Any]] = None,
    thresholds: Optional[PromotionBenchmarkThresholds] = None,
    quality_thresholds: Optional[StrategyQualityThresholds] = None,
) -> PromotionBenchmarkDecision:
    """
    Evaluate whether a strategy candidate satisfies promotion benchmarks.
    """
    cfg = thresholds or PromotionBenchmarkThresholds()
    summary = dict(validation_summary or {})

    gross_return_pct = _num(metrics.get("gross_return_pct", metrics.get("total_return_pct", 0.0)))
    modeled_cost_pct = _num(
        metrics.get(
            "modeled_cost_pct",
            metrics.get(
                "execution_cost_pct",
                _num(metrics.get("fees_pct", 0.0)) + _num(metrics.get("slippage_pct", 0.0)),
            ),
        )
    )
    net_return_pct = _num(metrics.get("net_return_pct", gross_return_pct - modeled_cost_pct))
    total_trades = _num(metrics.get("total_trades", 0.0))
    sharpe_ratio = _num(metrics.get("sharpe_ratio", 0.0))
    calmar_ratio = _num(metrics.get("calmar_ratio", 0.0))
    drawdown_abs = abs(_num(metrics.get("max_drawdown_pct", 0.0)))

    reasons: list[str] = []
    if total_trades < float(cfg.min_total_trades):
        reasons.append("benchmark_trade_activity_below_floor")
    if net_return_pct < float(cfg.min_net_return_pct):
        reasons.append("benchmark_net_return_below_floor")
    if sharpe_ratio < float(cfg.min_sharpe_ratio):
        reasons.append("benchmark_sharpe_below_floor")
    if calmar_ratio < float(cfg.min_calmar_ratio):
        reasons.append("benchmark_calmar_below_floor")
    if drawdown_abs > float(cfg.max_drawdown_pct):
        reasons.append("benchmark_drawdown_above_cap")

    quality_gate_payload: Optional[Dict[str, Any]] = None
    if cfg.require_quality_gate:
        quality_decision = evaluate_strategy_quality(
            dict(summary),
            thresholds=quality_thresholds or StrategyQualityThresholds(),
        )
        quality_gate_payload = _quality_gate_payload(quality_decision)
        if not quality_decision.passed:
            reasons.append("benchmark_quality_gate_failed")

    return PromotionBenchmarkDecision(
        passed=len(reasons) == 0,
        reasons=reasons,
        metrics={
            "gross_return_pct": float(gross_return_pct),
            "modeled_cost_pct": float(modeled_cost_pct),
            "net_return_pct": float(net_return_pct),
            "total_trades": float(total_trades),
            "sharpe_ratio": float(sharpe_ratio),
            "calmar_ratio": float(calmar_ratio),
            "max_drawdown_abs_pct": float(drawdown_abs),
        },
        thresholds={
            "min_total_trades": float(cfg.min_total_trades),
            "min_net_return_pct": float(cfg.min_net_return_pct),
            "min_sharpe_ratio": float(cfg.min_sharpe_ratio),
            "min_calmar_ratio": float(cfg.min_calmar_ratio),
            "max_drawdown_pct": float(cfg.max_drawdown_pct),
        },
        quality_gate=quality_gate_payload,
    )


def build_promotion_benchmark_spec(
    *,
    thresholds: Optional[PromotionBenchmarkThresholds] = None,
    quality_thresholds: Optional[StrategyQualityThresholds] = None,
) -> Dict[str, Any]:
    """
    Build deterministic benchmark specification payload for promotion decisions.
    """
    benchmark_thresholds = thresholds or PromotionBenchmarkThresholds()
    quality_cfg = quality_thresholds or StrategyQualityThresholds()

    stable_payload = {
        "spec_version": "g0_benchmark_v1",
        "benchmark_thresholds": asdict(benchmark_thresholds),
        "quality_gate_thresholds": asdict(quality_cfg),
        "reason_codes": [
            "benchmark_trade_activity_below_floor",
            "benchmark_net_return_below_floor",
            "benchmark_sharpe_below_floor",
            "benchmark_calmar_below_floor",
            "benchmark_drawdown_above_cap",
            "benchmark_quality_gate_failed",
        ],
    }
    spec_fingerprint = _sha256(_stable_json(stable_payload))
    payload = dict(stable_payload)
    payload["spec_fingerprint"] = spec_fingerprint
    payload["generated_at"] = datetime.utcnow().isoformat()
    return payload


def export_promotion_benchmark_spec(spec: Mapping[str, Any], output_path: str) -> str:
    """Persist benchmark specification JSON and return path."""
    path = str(output_path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(spec), handle, indent=2, sort_keys=True)
    return path
