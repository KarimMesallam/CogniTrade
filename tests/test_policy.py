import os
import sys
from datetime import datetime, timedelta

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.policy import RegimePolicyEngine
from bot.regime import RegimeState, REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS


def _base_strategy_config():
    return {
        "simple": {"enabled": True, "weight": 1.0},
        "technical": {"enabled": True, "weight": 2.0},
        "custom": {"enabled": True, "weight": 1.0},
    }


def _regime_state(regime: str, confidence: float = 0.8) -> RegimeState:
    return RegimeState(
        regime=regime,
        confidence=confidence,
        trend_pct=0.01,
        realized_volatility_pct=0.005,
        lookback_candles=50,
        timestamp="2026-02-08T12:00:00",
        details={},
    )


def test_policy_routes_signals_and_weights_by_regime():
    engine = RegimePolicyEngine(
        strategy_config=_base_strategy_config(),
        policy_config={
            "enabled": True,
            "default_regime": REGIME_SIDEWAYS,
            "switch_hysteresis_confirmations": 1,
            "switch_cooldown_seconds": 0,
            "regimes": {
                REGIME_BULL: {
                    "enabled_strategies": ["simple", "technical"],
                    "weight_multipliers": {"simple": 1.5, "technical": 1.0},
                    "size_multiplier": 1.1,
                },
                REGIME_BEAR: {
                    "enabled_strategies": ["technical"],
                    "weight_multipliers": {"technical": 1.25},
                    "size_multiplier": 0.6,
                },
            },
        },
    )

    signals = {"simple": "BUY", "technical": "SELL", "custom": "BUY"}
    decision = engine.evaluate(signals, _regime_state(REGIME_BEAR))

    assert decision.active_regime == REGIME_BEAR
    assert list(decision.signals.keys()) == ["technical"]
    assert decision.strategy_weight_overrides["technical"] == 2.5
    assert decision.size_multiplier == 0.6


def test_policy_switch_hysteresis_requires_confirmations():
    engine = RegimePolicyEngine(
        strategy_config=_base_strategy_config(),
        policy_config={
            "enabled": True,
            "default_regime": REGIME_SIDEWAYS,
            "switch_hysteresis_confirmations": 2,
            "switch_cooldown_seconds": 0,
        },
    )
    signals = {"simple": "BUY", "technical": "BUY", "custom": "HOLD"}

    first = engine.evaluate(signals, _regime_state(REGIME_BULL))
    assert first.active_regime == REGIME_BULL

    second = engine.evaluate(signals, _regime_state(REGIME_BEAR))
    assert second.active_regime == REGIME_BULL
    assert "hysteresis" in second.switch_reason

    third = engine.evaluate(signals, _regime_state(REGIME_BEAR))
    assert third.active_regime == REGIME_BEAR
    assert third.switch_applied is True


def test_policy_switch_respects_cooldown():
    engine = RegimePolicyEngine(
        strategy_config=_base_strategy_config(),
        policy_config={
            "enabled": True,
            "default_regime": REGIME_SIDEWAYS,
            "switch_hysteresis_confirmations": 1,
            "switch_cooldown_seconds": 300,
        },
    )
    signals = {"simple": "BUY", "technical": "BUY", "custom": "BUY"}
    now = datetime(2026, 2, 8, 12, 0, 0)

    engine.evaluate(signals, _regime_state(REGIME_BULL), now=now)
    engine.evaluate(signals, _regime_state(REGIME_BEAR), now=now + timedelta(seconds=1))
    blocked = engine.evaluate(signals, _regime_state(REGIME_BULL), now=now + timedelta(seconds=10))

    assert blocked.active_regime == REGIME_BEAR
    assert "cooldown" in blocked.switch_reason


def test_policy_switch_shadow_mode_blocks_apply():
    engine = RegimePolicyEngine(
        strategy_config=_base_strategy_config(),
        policy_config={
            "enabled": True,
            "default_regime": REGIME_SIDEWAYS,
            "switch_hysteresis_confirmations": 1,
            "switch_cooldown_seconds": 0,
            "switch_shadow_mode": True,
        },
    )
    signals = {"simple": "BUY", "technical": "BUY", "custom": "BUY"}

    engine.evaluate(signals, _regime_state(REGIME_BULL))
    shadow_result = engine.evaluate(signals, _regime_state(REGIME_BEAR))

    assert shadow_result.active_regime == REGIME_BULL
    assert shadow_result.switch_applied is False
    assert "shadow_mode" in shadow_result.switch_reason


def test_policy_switch_blocked_by_turnover_cap():
    engine = RegimePolicyEngine(
        strategy_config=_base_strategy_config(),
        policy_config={
            "enabled": True,
            "default_regime": REGIME_SIDEWAYS,
            "switch_hysteresis_confirmations": 1,
            "switch_cooldown_seconds": 0,
            "max_strategy_turnover_ratio": 0.3,
            "regimes": {
                REGIME_BULL: {"enabled_strategies": ["simple", "technical"]},
                REGIME_BEAR: {"enabled_strategies": ["custom"]},
            },
        },
    )
    signals = {"simple": "BUY", "technical": "BUY", "custom": "BUY"}

    engine.evaluate(signals, _regime_state(REGIME_BULL))
    blocked = engine.evaluate(signals, _regime_state(REGIME_BEAR))

    assert blocked.active_regime == REGIME_BULL
    assert "turnover_cap_exceeded" in blocked.switch_reason
    assert blocked.turnover_ratio > 0.3
