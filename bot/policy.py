"""
Regime-aware strategy policy and switch controls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set

from bot.regime import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_HIGH_VOL,
    REGIME_SIDEWAYS,
    REGIME_UNKNOWN,
    RegimeState,
)


@dataclass(frozen=True)
class PolicyDecision:
    """Result of applying regime policy + switch controls."""

    detected_regime: str
    active_regime: str
    confidence: float
    signals: Dict[str, str]
    enabled_strategies: List[str]
    strategy_weight_overrides: Dict[str, float]
    size_multiplier: float
    switch_applied: bool = False
    switch_reason: str = ""
    shadow_mode: bool = False
    candidate_regime: str = ""
    candidate_count: int = 0
    turnover_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegimePolicyEngine:
    """Apply regime-based strategy routing and safe switching controls."""

    def __init__(self, strategy_config: Dict[str, Dict[str, Any]], policy_config: Optional[Dict[str, Any]] = None):
        self.strategy_config = strategy_config or {}
        self.policy_config = policy_config or {}
        self.enabled = bool(self.policy_config.get("enabled", True))
        self.default_regime = str(self.policy_config.get("default_regime", REGIME_SIDEWAYS)).upper()
        self.default_size_multiplier = float(self.policy_config.get("default_size_multiplier", 1.0))
        self.min_switch_confidence = float(self.policy_config.get("min_switch_confidence", 0.55))
        self.switch_hysteresis_confirmations = max(
            1,
            int(self.policy_config.get("switch_hysteresis_confirmations", 2)),
        )
        self.switch_cooldown_seconds = max(
            0,
            int(self.policy_config.get("switch_cooldown_seconds", 900)),
        )
        self.max_strategy_turnover_ratio = float(self.policy_config.get("max_strategy_turnover_ratio", 1.0))
        self.switch_shadow_mode = bool(self.policy_config.get("switch_shadow_mode", False))
        self.regime_policy_map = self._build_regime_policy_map(self.policy_config.get("regimes", {}))

        self.active_regime: Optional[str] = None
        self.active_strategies: Set[str] = set(self._default_enabled_strategies())
        self._last_switch_at: Optional[datetime] = None
        self._candidate_regime: Optional[str] = None
        self._candidate_count: int = 0

    @staticmethod
    def _parse_now(now: Optional[Any]) -> datetime:
        if isinstance(now, datetime):
            return now
        if isinstance(now, str):
            try:
                return datetime.fromisoformat(now)
            except ValueError:
                pass
        return datetime.utcnow()

    @staticmethod
    def _normalize_regime(regime: Optional[str], default_regime: str) -> str:
        candidate = str(regime or "").upper()
        if candidate in {REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS, REGIME_HIGH_VOL}:
            return candidate
        return default_regime

    @staticmethod
    def calculate_strategy_turnover_ratio(current: Iterable[str], target: Iterable[str]) -> float:
        """Compute changed-strategy ratio between current and target sets."""
        current_set = set(current)
        target_set = set(target)
        universe = current_set | target_set
        if not universe:
            return 0.0
        return float(len(current_set ^ target_set) / len(universe))

    def _default_enabled_strategies(self) -> List[str]:
        return [
            strategy_name
            for strategy_name, cfg in self.strategy_config.items()
            if bool(cfg.get("enabled", False))
        ]

    def _build_regime_policy_map(self, custom_regimes: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        defaults = {
            REGIME_BULL: {
                "enabled_strategies": None,  # Use globally enabled strategies.
                "weight_multipliers": {"technical": 1.2},
                "size_multiplier": 1.0,
            },
            REGIME_BEAR: {
                "enabled_strategies": ["technical", "custom"],
                "weight_multipliers": {"technical": 1.3, "custom": 1.1},
                "size_multiplier": 0.75,
            },
            REGIME_SIDEWAYS: {
                "enabled_strategies": None,
                "weight_multipliers": {"simple": 0.9, "technical": 1.0, "custom": 1.0},
                "size_multiplier": 0.65,
            },
            REGIME_HIGH_VOL: {
                "enabled_strategies": ["technical", "custom"],
                "weight_multipliers": {"technical": 1.4, "custom": 1.2},
                "size_multiplier": 0.5,
            },
        }

        merged = dict(defaults)
        for regime_name, entry in (custom_regimes or {}).items():
            key = str(regime_name).upper()
            if key not in merged:
                merged[key] = {}
            merged[key] = {**merged[key], **(entry or {})}
        return merged

    def _resolve_regime_policy(self, regime: str) -> Dict[str, Any]:
        resolved = dict(self.regime_policy_map.get(regime, {}))
        resolved.setdefault("enabled_strategies", None)
        resolved.setdefault("weight_multipliers", {})
        resolved.setdefault("size_multiplier", self.default_size_multiplier)
        return resolved

    def _resolve_enabled_strategies(self, regime_policy: Dict[str, Any]) -> List[str]:
        globally_enabled = set(self._default_enabled_strategies())
        configured = regime_policy.get("enabled_strategies")
        if configured in (None, "", []):
            return sorted(globally_enabled)

        candidate = {str(item) for item in configured}
        return sorted(globally_enabled & candidate)

    def _resolve_weight_overrides(self, enabled_strategies: List[str], regime_policy: Dict[str, Any]) -> Dict[str, float]:
        multipliers = regime_policy.get("weight_multipliers", {}) or {}
        overrides: Dict[str, float] = {}
        for strategy_name in enabled_strategies:
            base_weight = float(self.strategy_config.get(strategy_name, {}).get("weight", 1.0))
            multiplier = float(multipliers.get(strategy_name, 1.0))
            overrides[strategy_name] = max(0.0, base_weight * multiplier)
        return overrides

    def _maybe_switch_regime(self, target_regime: str, confidence: float, now: datetime) -> Dict[str, Any]:
        if self.active_regime is None:
            self.active_regime = target_regime
            self._candidate_regime = None
            self._candidate_count = 0
            return {
                "applied": False,
                "reason": "initialized_active_regime",
                "candidate_regime": "",
                "candidate_count": 0,
                "turnover_ratio": 0.0,
            }

        if target_regime == self.active_regime:
            self._candidate_regime = None
            self._candidate_count = 0
            return {
                "applied": False,
                "reason": "",
                "candidate_regime": "",
                "candidate_count": 0,
                "turnover_ratio": 0.0,
            }

        if confidence < self.min_switch_confidence:
            return {
                "applied": False,
                "reason": f"confidence_below_min:{confidence:.2f}<{self.min_switch_confidence:.2f}",
                "candidate_regime": "",
                "candidate_count": 0,
                "turnover_ratio": 0.0,
            }

        if self._candidate_regime == target_regime:
            self._candidate_count += 1
        else:
            self._candidate_regime = target_regime
            self._candidate_count = 1

        if self._candidate_count < self.switch_hysteresis_confirmations:
            return {
                "applied": False,
                "reason": (
                    f"hysteresis_wait:{self._candidate_count}/"
                    f"{self.switch_hysteresis_confirmations}"
                ),
                "candidate_regime": str(self._candidate_regime),
                "candidate_count": self._candidate_count,
                "turnover_ratio": 0.0,
            }

        if self._last_switch_at is not None:
            elapsed = now - self._last_switch_at
            cooldown = timedelta(seconds=self.switch_cooldown_seconds)
            if elapsed < cooldown:
                remaining = int((cooldown - elapsed).total_seconds())
                return {
                    "applied": False,
                    "reason": f"cooldown_active:{remaining}s_remaining",
                    "candidate_regime": str(self._candidate_regime),
                    "candidate_count": self._candidate_count,
                    "turnover_ratio": 0.0,
                }

        target_policy = self._resolve_regime_policy(target_regime)
        target_strategies = set(self._resolve_enabled_strategies(target_policy))
        turnover_ratio = self.calculate_strategy_turnover_ratio(self.active_strategies, target_strategies)
        if self.max_strategy_turnover_ratio >= 0 and turnover_ratio > self.max_strategy_turnover_ratio:
            return {
                "applied": False,
                "reason": (
                    f"turnover_cap_exceeded:{turnover_ratio:.2f}>"
                    f"{self.max_strategy_turnover_ratio:.2f}"
                ),
                "candidate_regime": str(self._candidate_regime),
                "candidate_count": self._candidate_count,
                "turnover_ratio": turnover_ratio,
            }

        if self.switch_shadow_mode:
            return {
                "applied": False,
                "reason": "shadow_mode_switch_blocked",
                "candidate_regime": str(self._candidate_regime),
                "candidate_count": self._candidate_count,
                "turnover_ratio": turnover_ratio,
            }

        self.active_regime = target_regime
        self.active_strategies = target_strategies
        self._last_switch_at = now
        self._candidate_regime = None
        self._candidate_count = 0
        return {
            "applied": True,
            "reason": "switch_applied",
            "candidate_regime": "",
            "candidate_count": 0,
            "turnover_ratio": turnover_ratio,
        }

    def evaluate(
        self,
        signals: Dict[str, str],
        regime_state: Optional[RegimeState],
        now: Optional[Any] = None,
    ) -> PolicyDecision:
        """Apply policy controls to incoming strategy signals."""
        safe_signals = dict(signals or {})
        detected_regime = self._normalize_regime(
            regime_state.regime if regime_state else REGIME_UNKNOWN,
            default_regime=self.default_regime,
        )
        confidence = float(regime_state.confidence if regime_state else 0.0)

        if not self.enabled:
            enabled = sorted(self._default_enabled_strategies())
            filtered = {name: safe_signals[name] for name in enabled if name in safe_signals}
            overrides = self._resolve_weight_overrides(enabled, {"weight_multipliers": {}})
            return PolicyDecision(
                detected_regime=detected_regime,
                active_regime=self.active_regime or detected_regime,
                confidence=confidence,
                signals=filtered,
                enabled_strategies=enabled,
                strategy_weight_overrides=overrides,
                size_multiplier=1.0,
                switch_applied=False,
                switch_reason="policy_disabled",
                shadow_mode=False,
                metadata={"policy_enabled": False},
            )

        now_dt = self._parse_now(now)
        switch_result = self._maybe_switch_regime(detected_regime, confidence, now_dt)
        active_regime = self.active_regime or detected_regime
        active_policy = self._resolve_regime_policy(active_regime)
        enabled = self._resolve_enabled_strategies(active_policy)
        self.active_strategies = set(enabled)
        filtered = {
            strategy_name: signal
            for strategy_name, signal in safe_signals.items()
            if strategy_name in self.active_strategies
        }
        weight_overrides = self._resolve_weight_overrides(enabled, active_policy)
        size_multiplier = max(0.0, float(active_policy.get("size_multiplier", self.default_size_multiplier)))

        return PolicyDecision(
            detected_regime=detected_regime,
            active_regime=active_regime,
            confidence=confidence,
            signals=filtered,
            enabled_strategies=enabled,
            strategy_weight_overrides=weight_overrides,
            size_multiplier=size_multiplier,
            switch_applied=bool(switch_result.get("applied", False)),
            switch_reason=str(switch_result.get("reason", "")),
            shadow_mode=self.switch_shadow_mode,
            candidate_regime=str(switch_result.get("candidate_regime", "")),
            candidate_count=int(switch_result.get("candidate_count", 0)),
            turnover_ratio=float(switch_result.get("turnover_ratio", 0.0)),
            metadata={
                "policy_enabled": True,
                "switch_hysteresis_confirmations": self.switch_hysteresis_confirmations,
                "switch_cooldown_seconds": self.switch_cooldown_seconds,
                "max_strategy_turnover_ratio": self.max_strategy_turnover_ratio,
            },
        )
