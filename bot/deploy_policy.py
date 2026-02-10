from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import tempfile
from typing import Any, Dict, List, Literal, Optional


RolloutStage = Literal["shadow", "canary", "production"]


@dataclass(frozen=True)
class RolloutMetrics:
    """Evidence metrics used to evaluate rollout gates."""

    sample_count: int
    error_rate: float
    drawdown_pct: float = 0.0
    total_return_pct: float = 0.0
    latency_p95_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutDecision:
    """Decision outcome for a rollout stage gate."""

    rollout_id: str
    stage: RolloutStage
    approved: bool
    reasons: List[str]
    next_required_stage: Optional[RolloutStage]
    timestamp: str


@dataclass
class RolloutThresholds:
    """Threshold policy for shadow/canary/production promotion."""

    min_shadow_samples: int = 50
    max_shadow_error_rate: float = 0.20
    min_canary_samples: int = 30
    max_canary_error_rate: float = 0.15
    max_canary_drawdown_pct: float = 8.0
    min_canary_total_return_pct: float = -1.0
    max_canary_latency_p95_ms: float = 3000.0
    require_quality_gate: bool = True


@dataclass(frozen=True)
class StrategyQualityThresholds:
    """Research-quality thresholds required before production promotion."""

    min_walk_forward_folds: int = 3
    min_sharpe_ratio: float = 0.20
    min_calmar_ratio: float = 0.10
    max_drawdown_pct: float = 25.0
    min_regime_samples: int = 20
    min_regime_win_rate: float = 0.45
    min_regimes_passing: int = 2


@dataclass(frozen=True)
class StrategyQualityDecision:
    """Outcome of strategy-quality validation."""

    passed: bool
    reasons: List[str]
    metrics: Dict[str, float]


def evaluate_strategy_quality(
    validation_summary: Dict[str, Any],
    thresholds: Optional[StrategyQualityThresholds] = None,
) -> StrategyQualityDecision:
    """
    Validate strategy robustness evidence from walk-forward and regime-sliced evaluation.
    """
    cfg = thresholds or StrategyQualityThresholds()
    summary = dict(validation_summary or {})
    reasons: List[str] = []

    walk_forward_folds = int(summary.get("walk_forward_folds", 0))
    walk_forward_metrics = summary.get("walk_forward_summary", {}) or {}
    regime_slices = summary.get("regime_slices", {}) or {}

    sharpe_mean = float(
        (walk_forward_metrics.get("sharpe_ratio", {}) or {}).get("mean", 0.0)
    )
    calmar_mean = float(
        (walk_forward_metrics.get("calmar_ratio", {}) or {}).get("mean", 0.0)
    )
    drawdown_mean_raw = float(
        (walk_forward_metrics.get("max_drawdown_pct", {}) or {}).get("mean", 0.0)
    )
    drawdown_abs = abs(drawdown_mean_raw)

    if walk_forward_folds < int(cfg.min_walk_forward_folds):
        reasons.append("quality_walk_forward_folds_below_minimum")
    if sharpe_mean < float(cfg.min_sharpe_ratio):
        reasons.append("quality_sharpe_below_threshold")
    if calmar_mean < float(cfg.min_calmar_ratio):
        reasons.append("quality_calmar_below_threshold")
    if drawdown_abs > float(cfg.max_drawdown_pct):
        reasons.append("quality_drawdown_above_threshold")

    passing_regimes = 0
    eligible_regimes = 0
    for _regime, metrics in regime_slices.items():
        sample_count = float((metrics or {}).get("sample_count", 0.0))
        win_rate = float((metrics or {}).get("win_rate", 0.0))
        if sample_count >= float(cfg.min_regime_samples):
            eligible_regimes += 1
            if win_rate >= float(cfg.min_regime_win_rate):
                passing_regimes += 1

    if passing_regimes < int(cfg.min_regimes_passing):
        reasons.append("quality_regime_consistency_below_threshold")

    decision_metrics = {
        "walk_forward_folds": float(walk_forward_folds),
        "sharpe_ratio_mean": float(sharpe_mean),
        "calmar_ratio_mean": float(calmar_mean),
        "max_drawdown_abs_mean": float(drawdown_abs),
        "eligible_regimes": float(eligible_regimes),
        "passing_regimes": float(passing_regimes),
    }

    return StrategyQualityDecision(
        passed=len(reasons) == 0,
        reasons=reasons,
        metrics=decision_metrics,
    )


class ShadowCanaryRolloutGate:
    """
    Enforce mandatory rollout sequencing:
    shadow -> canary -> production
    """

    def __init__(self, thresholds: Optional[RolloutThresholds] = None):
        self.thresholds = thresholds or RolloutThresholds()
        self._state: Dict[str, Dict[str, Any]] = {}

    def export_state(self) -> Dict[str, Dict[str, Any]]:
        """Export current rollout state to a JSON-serializable mapping."""
        return json.loads(json.dumps(self._state))

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load rollout state mapping."""
        self._state = json.loads(json.dumps(state or {}))

    def load_from_file(self, file_path: str) -> None:
        """Load rollout state from a JSON file if it exists."""
        path = str(file_path)
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError, ValueError):
            # Gracefully recover from corrupted state files and retain evidence.
            self.load_state({})
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
            backup_path = f"{path}.corrupt.{timestamp}"
            try:
                os.replace(path, backup_path)
            except OSError:
                pass
            return
        self.load_state(payload if isinstance(payload, dict) else {})

    def save_to_file(self, file_path: str) -> None:
        """Persist rollout state to a JSON file."""
        path = str(file_path)
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        target_dir = directory or "."
        temp_path: Optional[str] = None

        # Atomic write prevents partial/corrupt files under interrupted writes.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_dir,
            prefix=".rollout_state_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(self.export_state(), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.replace(temp_path, path)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    def _rollout_state(self, rollout_id: str) -> Dict[str, Any]:
        return self._state.setdefault(
            rollout_id,
            {
                "shadow_passed": False,
                "canary_passed": False,
                "production_passed": False,
                "history": [],
            },
        )

    def _record(
        self,
        *,
        rollout_id: str,
        stage: RolloutStage,
        approved: bool,
        reasons: List[str],
        metrics: RolloutMetrics,
    ) -> RolloutDecision:
        state = self._rollout_state(rollout_id)
        timestamp = self._now()

        if stage == "shadow" and approved:
            state["shadow_passed"] = True
        elif stage == "canary" and approved:
            state["canary_passed"] = True
        elif stage == "production" and approved:
            state["production_passed"] = True

        next_required_stage: Optional[RolloutStage] = None
        if not state["shadow_passed"]:
            next_required_stage = "shadow"
        elif not state["canary_passed"]:
            next_required_stage = "canary"
        elif not state["production_passed"]:
            next_required_stage = "production"

        decision = RolloutDecision(
            rollout_id=rollout_id,
            stage=stage,
            approved=approved,
            reasons=list(reasons),
            next_required_stage=next_required_stage,
            timestamp=timestamp,
        )
        state["history"].append(
            {
                "timestamp": timestamp,
                "stage": stage,
                "approved": approved,
                "reasons": list(reasons),
                "metrics": {
                    "sample_count": int(metrics.sample_count),
                    "error_rate": float(metrics.error_rate),
                    "drawdown_pct": float(metrics.drawdown_pct),
                    "total_return_pct": float(metrics.total_return_pct),
                    "latency_p95_ms": float(metrics.latency_p95_ms),
                    "metadata": dict(metrics.metadata),
                },
            }
        )
        return decision

    def _validate_shadow(self, metrics: RolloutMetrics) -> List[str]:
        reasons: List[str] = []
        if int(metrics.sample_count) < int(self.thresholds.min_shadow_samples):
            reasons.append("shadow_sample_count_below_minimum")
        if float(metrics.error_rate) > float(self.thresholds.max_shadow_error_rate):
            reasons.append("shadow_error_rate_above_threshold")
        return reasons

    def _validate_canary(self, metrics: RolloutMetrics) -> List[str]:
        reasons: List[str] = []
        if int(metrics.sample_count) < int(self.thresholds.min_canary_samples):
            reasons.append("canary_sample_count_below_minimum")
        if float(metrics.error_rate) > float(self.thresholds.max_canary_error_rate):
            reasons.append("canary_error_rate_above_threshold")
        if float(metrics.drawdown_pct) > float(self.thresholds.max_canary_drawdown_pct):
            reasons.append("canary_drawdown_above_threshold")
        if float(metrics.total_return_pct) < float(self.thresholds.min_canary_total_return_pct):
            reasons.append("canary_return_below_threshold")
        if float(metrics.latency_p95_ms) > float(self.thresholds.max_canary_latency_p95_ms):
            reasons.append("canary_latency_p95_above_threshold")
        return reasons

    def evaluate_shadow(self, rollout_id: str, metrics: RolloutMetrics) -> RolloutDecision:
        reasons = self._validate_shadow(metrics)
        approved = len(reasons) == 0
        return self._record(
            rollout_id=rollout_id,
            stage="shadow",
            approved=approved,
            reasons=reasons,
            metrics=metrics,
        )

    def evaluate_canary(self, rollout_id: str, metrics: RolloutMetrics) -> RolloutDecision:
        state = self._rollout_state(rollout_id)
        reasons: List[str] = []
        if not state["shadow_passed"]:
            reasons.append("shadow_stage_not_approved")
        reasons.extend(self._validate_canary(metrics))
        approved = len(reasons) == 0
        return self._record(
            rollout_id=rollout_id,
            stage="canary",
            approved=approved,
            reasons=reasons,
            metrics=metrics,
        )

    def evaluate_production(self, rollout_id: str, metrics: RolloutMetrics) -> RolloutDecision:
        state = self._rollout_state(rollout_id)
        reasons: List[str] = []
        if not state["shadow_passed"]:
            reasons.append("shadow_stage_not_approved")
        if not state["canary_passed"]:
            reasons.append("canary_stage_not_approved")
        # Production re-checks canary-grade safety thresholds before promotion.
        reasons.extend(self._validate_canary(metrics))
        quality_gate = (metrics.metadata or {}).get("quality_gate") if isinstance(metrics.metadata, dict) else None
        if self.thresholds.require_quality_gate:
            if not isinstance(quality_gate, dict):
                reasons.append("missing_quality_gate_evidence")
            elif not bool(quality_gate.get("passed", False)):
                reasons.append("quality_gate_not_passed")
                for reason in quality_gate.get("reasons", []) or []:
                    reasons.append(f"quality_gate:{reason}")
        approved = len(reasons) == 0
        return self._record(
            rollout_id=rollout_id,
            stage="production",
            approved=approved,
            reasons=reasons,
            metrics=metrics,
        )

    def get_rollout_status(self, rollout_id: str) -> Dict[str, Any]:
        state = self._rollout_state(rollout_id)
        return {
            "rollout_id": rollout_id,
            "shadow_passed": bool(state["shadow_passed"]),
            "canary_passed": bool(state["canary_passed"]),
            "production_passed": bool(state["production_passed"]),
            "history": list(state["history"]),
        }


def metrics_from_observability_dashboard(dashboard: Dict[str, Any]) -> RolloutMetrics:
    """
    Build rollout metrics from observability snapshot payload.
    """
    errors = dashboard.get("errors", {}) if isinstance(dashboard, dict) else {}
    latency = dashboard.get("latency", {}) if isinstance(dashboard, dict) else {}
    trade_decisions = dashboard.get("trade_decisions", {}) if isinstance(dashboard, dict) else {}
    pnl = dashboard.get("pnl_attribution", {}) if isinstance(dashboard, dict) else {}

    return RolloutMetrics(
        sample_count=int(trade_decisions.get("count", 0)),
        error_rate=float(errors.get("error_rate", 0.0)),
        drawdown_pct=float(dashboard.get("drawdown_pct", 0.0)),
        total_return_pct=float(pnl.get("total_pnl", 0.0)),
        latency_p95_ms=float(latency.get("p95_ms", 0.0)),
        metadata={"window_minutes": dashboard.get("window_minutes", 0)},
    )


def thresholds_from_config(config: Dict[str, Any]) -> RolloutThresholds:
    """Build rollout thresholds from config mapping."""
    cfg = dict(config or {})
    return RolloutThresholds(
        min_shadow_samples=int(cfg.get("min_shadow_samples", 50)),
        max_shadow_error_rate=float(cfg.get("max_shadow_error_rate", 0.20)),
        min_canary_samples=int(cfg.get("min_canary_samples", 30)),
        max_canary_error_rate=float(cfg.get("max_canary_error_rate", 0.15)),
        max_canary_drawdown_pct=float(cfg.get("max_canary_drawdown_pct", 8.0)),
        min_canary_total_return_pct=float(cfg.get("min_canary_total_return_pct", -1.0)),
        max_canary_latency_p95_ms=float(cfg.get("max_canary_latency_p95_ms", 3000.0)),
        require_quality_gate=bool(cfg.get("require_quality_gate", True)),
    )
