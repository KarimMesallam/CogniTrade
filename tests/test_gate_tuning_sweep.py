from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_gate_tuning_sweep.py"
    spec = importlib.util.spec_from_file_location("run_gate_tuning_sweep", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _has_bear_candidate(candidates, *, fast, slow, rsi_period, overbought, oversold):
    target = {
        "fast_period": fast,
        "slow_period": slow,
        "rsi_period": rsi_period,
        "rsi_overbought": overbought,
        "rsi_oversold": oversold,
    }
    for candidate in candidates:
        if candidate.get("strategy_name") != "bear_rally_short":
            continue
        if candidate.get("strategy_params") == target:
            return True
    return False


def test_sweep_includes_trade_activity_bear_candidates():
    module = _load_module()
    candidates = module._build_candidates()

    assert _has_bear_candidate(
        candidates,
        fast=10,
        slow=20,
        rsi_period=7,
        overbought=60,
        oversold=40,
    )
    assert _has_bear_candidate(
        candidates,
        fast=10,
        slow=20,
        rsi_period=7,
        overbought=60,
        oversold=25,
    )
