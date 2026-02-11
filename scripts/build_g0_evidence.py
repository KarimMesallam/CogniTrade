#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
import json
import os
from typing import Any, Dict, List

from bot.config import get_promotion_benchmark_config, get_quality_gate_config
from bot.data_pipeline import (
    RegimeDatasetBuildConfig,
    build_regime_evaluation_datasets,
    export_regime_dataset_manifest,
)
from bot.regime import MarketRegimeDetector
from research import (
    ExperimentRegistry,
    build_promotion_benchmark_spec,
    evaluate_promotion_candidate,
    export_promotion_benchmark_spec,
    promotion_thresholds_from_config,
)


def _build_regime_candles() -> List[List[Any]]:
    """
    Build deterministic synthetic candles with explicit bull/bear/sideways/high-vol phases.
    """
    phases = [
        ("bull", 90, 0.0020),
        ("sideways", 90, 0.0002),
        ("bear", 90, -0.0020),
        ("high_vol", 90, 0.0),
    ]

    start = datetime(2025, 1, 1, 0, 0, 0)
    interval = timedelta(hours=1)
    price = 100.0
    candles: List[List[Any]] = []

    idx = 0
    for phase_name, count, base_change in phases:
        for step in range(count):
            open_time = start + (interval * idx)
            close_time = open_time + interval - timedelta(milliseconds=1)

            if phase_name == "high_vol":
                pct_change = 0.030 if step % 2 == 0 else -0.028
            elif phase_name == "sideways":
                pct_change = base_change if step % 2 == 0 else -base_change
            else:
                pct_change = base_change

            open_price = price
            close_price = max(1.0, open_price * (1.0 + pct_change))
            high_price = max(open_price, close_price) * (1.0 + abs(pct_change) * 0.35 + 0.001)
            low_price = min(open_price, close_price) * (1.0 - abs(pct_change) * 0.35 - 0.001)
            volume = 1000.0 + (idx % 20) * 10.0

            candles.append(
                [
                    int(open_time.timestamp() * 1000),
                    f"{open_price:.8f}",
                    f"{high_price:.8f}",
                    f"{low_price:.8f}",
                    f"{close_price:.8f}",
                    f"{volume:.8f}",
                    int(close_time.timestamp() * 1000),
                    "0",
                    0,
                    "0",
                    "0",
                    "0",
                ]
            )
            price = close_price
            idx += 1
    return candles


def _register_dataset_frames(result) -> Dict[str, str]:
    registry = ExperimentRegistry(db_path="research/experiments.db")
    dataset_ids: Dict[str, str] = {}

    all_record = registry.register_dataset(
        name="g0_eval_all_regimes",
        dataframe=result.combined_frame,
        metadata={"task": "G0-01", "scope": "combined"},
        sort_by=["decision_timestamp"],
    )
    dataset_ids["ALL"] = all_record.dataset_id

    for regime, frame in sorted(result.per_regime_frames.items()):
        if frame.empty:
            continue
        dataset = registry.register_dataset(
            name=f"g0_eval_{regime.lower()}",
            dataframe=frame,
            metadata={"task": "G0-01", "scope": "regime", "regime": regime},
            sort_by=["decision_timestamp"],
        )
        dataset_ids[regime] = dataset.dataset_id
    return dataset_ids


def main() -> int:
    candles = _build_regime_candles()

    detector = MarketRegimeDetector(
        lookback_candles=30,
        trend_threshold_pct=0.02,
        sideways_threshold_pct=0.01,
        high_volatility_threshold_pct=0.015,
    )
    config = RegimeDatasetBuildConfig(
        symbol="BTCUSDT",
        timeframe="1h",
        min_history_candles=35,
        decision_stride=3,
        train_ratio=0.70,
        include_unknown_regime=False,
        feature_source="seeded_regime_candles",
    )
    result = build_regime_evaluation_datasets(
        candles=candles,
        config=config,
        detector=detector,
    )
    manifest = dict(result.manifest)
    manifest["registry_dataset_ids"] = _register_dataset_frames(result)
    manifest_path = export_regime_dataset_manifest(
        manifest,
        "output/g0_01_regime_dataset_manifest_latest.json",
    )

    benchmark_cfg = get_promotion_benchmark_config()
    quality_cfg = get_quality_gate_config()
    benchmark_thresholds, quality_thresholds = promotion_thresholds_from_config(
        benchmark_cfg,
        quality_cfg,
    )
    benchmark_spec = build_promotion_benchmark_spec(
        thresholds=benchmark_thresholds,
        quality_thresholds=quality_thresholds,
    )
    benchmark_path = export_promotion_benchmark_spec(
        benchmark_spec,
        "output/g0_02_benchmark_spec_latest.json",
    )

    sample_metrics = {
        "total_return_pct": 9.5,
        "modeled_cost_pct": 2.3,
        "total_trades": 64,
        "sharpe_ratio": 0.84,
        "calmar_ratio": 0.52,
        "max_drawdown_pct": -14.1,
    }
    sample_validation_summary = {
        "walk_forward_folds": 4,
        "walk_forward_summary": {
            "sharpe_ratio": {"mean": 0.71},
            "calmar_ratio": {"mean": 0.41},
            "max_drawdown_pct": {"mean": -11.2},
        },
        "regime_slices": {
            "BULL": {"sample_count": 40, "win_rate": 0.58},
            "BEAR": {"sample_count": 35, "win_rate": 0.50},
            "SIDEWAYS": {"sample_count": 30, "win_rate": 0.55},
        },
    }
    sample_decision = evaluate_promotion_candidate(
        metrics=sample_metrics,
        validation_summary=sample_validation_summary,
        thresholds=benchmark_thresholds,
        quality_thresholds=quality_thresholds,
    )
    summary_output = {
        "generated_at": datetime.utcnow().isoformat(),
        "benchmark_spec_path": benchmark_path,
        "benchmark_spec_fingerprint": benchmark_spec.get("spec_fingerprint"),
        "sample_metrics": sample_metrics,
        "sample_decision": asdict(sample_decision),
    }
    with open("output/g0_02_benchmark_summary_latest.json", "w", encoding="utf-8") as handle:
        json.dump(summary_output, handle, indent=2, sort_keys=True)

    print(
        json.dumps(
            {
                "g0_01_manifest_path": manifest_path,
                "g0_02_benchmark_spec_path": benchmark_path,
                "g0_02_benchmark_summary_path": "output/g0_02_benchmark_summary_latest.json",
                "combined_rows": len(result.combined_frame),
                "regimes_present": manifest.get("representative_regimes_present", []),
                "manifest_fingerprint": manifest.get("manifest_fingerprint"),
                "benchmark_spec_fingerprint": benchmark_spec.get("spec_fingerprint"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
