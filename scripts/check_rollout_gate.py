#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot import config as bot_config
from bot.deploy_policy import (
    RolloutMetrics,
    ShadowCanaryRolloutGate,
    StrategyQualityThresholds,
    evaluate_strategy_quality,
    thresholds_from_config,
)


def _load_json_payload(raw_or_path: str) -> Dict[str, Any]:
    if not raw_or_path:
        return {}
    candidate = Path(raw_or_path)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(raw_or_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate and enforce rollout gates.")
    parser.add_argument("--rollout-id", required=True, help="Stable rollout identifier.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("shadow", "canary", "production"),
        help="Rollout stage to evaluate.",
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="JSON object or file path containing rollout metrics.",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional rollout state path override (defaults to config rollout.state_store_path).",
    )
    parser.add_argument(
        "--quality-summary",
        default="",
        help="Optional JSON object/path for strategy quality validation summary.",
    )

    args = parser.parse_args()

    rollout_cfg = bot_config.get_rollout_config()
    quality_cfg = bot_config.get_quality_gate_config()
    state_path = args.state_path or rollout_cfg.get("state_store_path", "data/rollout_gate_state.json")

    gate = ShadowCanaryRolloutGate(thresholds=thresholds_from_config(rollout_cfg))
    gate.load_from_file(state_path)

    metrics_payload = _load_json_payload(args.metrics)
    metadata = dict(metrics_payload.get("metadata", {}) or {})

    if args.quality_summary:
        quality_payload = _load_json_payload(args.quality_summary)
        quality_decision = evaluate_strategy_quality(
            quality_payload,
            thresholds=StrategyQualityThresholds(
                min_walk_forward_folds=int(quality_cfg.get("min_walk_forward_folds", 3)),
                min_sharpe_ratio=float(quality_cfg.get("min_sharpe_ratio", 0.20)),
                min_calmar_ratio=float(quality_cfg.get("min_calmar_ratio", 0.10)),
                max_drawdown_pct=float(quality_cfg.get("max_drawdown_pct", 25.0)),
                min_regime_samples=int(quality_cfg.get("min_regime_samples", 20)),
                min_regime_win_rate=float(quality_cfg.get("min_regime_win_rate", 0.45)),
                min_regimes_passing=int(quality_cfg.get("min_regimes_passing", 2)),
            ),
        )
        metadata["quality_gate"] = {
            "passed": quality_decision.passed,
            "reasons": quality_decision.reasons,
            "metrics": quality_decision.metrics,
        }

    metrics = RolloutMetrics(
        sample_count=int(metrics_payload.get("sample_count", 0)),
        error_rate=float(metrics_payload.get("error_rate", 0.0)),
        drawdown_pct=float(metrics_payload.get("drawdown_pct", 0.0)),
        total_return_pct=float(metrics_payload.get("total_return_pct", 0.0)),
        latency_p95_ms=float(metrics_payload.get("latency_p95_ms", 0.0)),
        metadata=metadata,
    )

    if args.stage == "shadow":
        decision = gate.evaluate_shadow(args.rollout_id, metrics)
    elif args.stage == "canary":
        decision = gate.evaluate_canary(args.rollout_id, metrics)
    else:
        decision = gate.evaluate_production(args.rollout_id, metrics)

    gate.save_to_file(state_path)
    output = {
        "rollout_id": decision.rollout_id,
        "stage": decision.stage,
        "approved": decision.approved,
        "reasons": decision.reasons,
        "next_required_stage": decision.next_required_stage,
        "state_path": state_path,
    }
    print(json.dumps(output, indent=2))
    return 0 if decision.approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
