#!/usr/bin/env python3
"""
Run a gate-oriented strategy sweep against the local API backtest endpoint.

This script optimizes for passing quality/promotion gates, not raw return only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

from bot.config import get_promotion_benchmark_config, get_quality_gate_config
from research.benchmarks import evaluate_promotion_candidate, promotion_thresholds_from_config


def _fetch_4h_range(db_path: Path, symbol: str, timeframe: str) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        frame = pd.read_sql_query(
            """
            SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts, COUNT(*) AS n
            FROM market_data
            WHERE symbol=? AND timeframe=?
            """,
            conn,
            params=(symbol, timeframe),
        )
    finally:
        conn.close()

    row = frame.iloc[0]
    if pd.isna(row["min_ts"]) or pd.isna(row["max_ts"]):
        raise RuntimeError(f"No data for {symbol} {timeframe} in {db_path}")

    return {
        "start_date": pd.to_datetime(row["min_ts"]).strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(row["max_ts"]).strftime("%Y-%m-%d"),
        "bar_count": int(row["n"]),
    }


def _build_candidates() -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for short_p in [2, 3, 5, 8, 10, 12]:
        for long_p in [20, 30, 50, 80, 120]:
            if short_p < long_p:
                candidates.append(
                    {
                        "label": f"sma_{short_p}_{long_p}",
                        "strategy_name": "sma_crossover",
                        "strategy_params": {"short_period": short_p, "long_period": long_p},
                        "trade_mode": "SPOT",
                        "allow_short_positions": False,
                    }
                )

    for fast_p in [3, 5, 8, 12, 20]:
        for slow_p in [30, 50, 80, 120]:
            if fast_p < slow_p:
                candidates.append(
                    {
                        "label": f"ema_{fast_p}_{slow_p}",
                        "strategy_name": "ema_crossover",
                        "strategy_params": {"fast_period": fast_p, "slow_period": slow_p},
                        "trade_mode": "SPOT",
                        "allow_short_positions": False,
                    }
                )

    for lookback in [10, 15, 20, 30, 40, 60]:
        for buffer_bps in [0, 3, 5, 8, 10]:
            candidates.append(
                {
                    "label": f"donchian_{lookback}_{buffer_bps}",
                    "strategy_name": "donchian_breakout",
                    "strategy_params": {
                        "lookback_period": lookback,
                        "breakout_buffer_bps": buffer_bps,
                    },
                    "trade_mode": "FUTURES",
                    "allow_short_positions": True,
                }
            )

    # Include faster bear-rally variants to satisfy activity floor on 4h windows.
    for fast_p, slow_p in [
        (5, 20),
        (8, 20),
        (8, 30),
        (10, 20),
        (10, 25),
        (10, 30),
        (12, 30),
        (12, 50),
        (12, 80),
    ]:
        for overbought, oversold in [
            (55, 25),
            (55, 30),
            (60, 25),
            (60, 30),
            (60, 40),
            (65, 35),
            (70, 30),
        ]:
            for rsi_period in [7, 10, 14]:
                candidates.append(
                    {
                        "label": f"bear_{fast_p}_{slow_p}_{rsi_period}_{overbought}_{oversold}",
                        "strategy_name": "bear_rally_short",
                        "strategy_params": {
                            "fast_period": fast_p,
                            "slow_period": slow_p,
                            "rsi_period": rsi_period,
                            "rsi_overbought": overbought,
                            "rsi_oversold": oversold,
                        },
                        "trade_mode": "FUTURES",
                        "allow_short_positions": True,
                    }
                )

    for period in [10, 14, 21]:
        for overbought, oversold in [(65, 35), (70, 30), (75, 25)]:
            candidates.append(
                {
                    "label": f"rsi_{period}_{overbought}_{oversold}",
                    "strategy_name": "rsi",
                    "strategy_params": {
                        "period": period,
                        "overbought": overbought,
                        "oversold": oversold,
                    },
                    "trade_mode": "SPOT",
                    "allow_short_positions": False,
                }
            )

    # Regime-switch adaptive meta-strategy candidates.
    for trend in [0.015, 0.02, 0.03]:
        for high_vol in [0.012, 0.015, 0.02]:
            for bull_short, bull_long in [(5, 30), (8, 50), (10, 80)]:
                for bear_fast, bear_slow in [(5, 20), (8, 30), (12, 50)]:
                    for donchian_lb in [10, 20, 30]:
                        for min_conf in [0.45, 0.55, 0.65]:
                            for momentum_candles in [3, 4, 6]:
                                label = (
                                    "regime_switch_"
                                    f"t{trend}_hv{high_vol}_"
                                    f"b{bull_short}_{bull_long}_"
                                    f"s{bear_fast}_{bear_slow}_d{donchian_lb}_"
                                    f"c{int(min_conf * 100)}_m{momentum_candles}"
                                )
                                candidates.append(
                                    {
                                        "label": label,
                                        "strategy_name": "regime_switch_adaptive",
                                        "strategy_params": {
                                            "regime_lookback_candles": 50,
                                            "trend_threshold_pct": trend,
                                            "sideways_threshold_pct": max(0.005, trend / 2.0),
                                            "high_volatility_threshold_pct": high_vol,
                                            "min_regime_confidence": min_conf,
                                            "momentum_confirmation_candles": momentum_candles,
                                            "bull_short_period": bull_short,
                                            "bull_long_period": bull_long,
                                            "bear_fast_period": bear_fast,
                                            "bear_slow_period": bear_slow,
                                            "bear_rsi_period": 14,
                                            "bear_rsi_overbought": 60,
                                            "bear_rsi_oversold": 30,
                                            "sideways_rsi_period": 21,
                                            "sideways_rsi_overbought": 65,
                                            "sideways_rsi_oversold": 35,
                                            "high_vol_lookback_period": donchian_lb,
                                            "high_vol_breakout_buffer_bps": 5.0,
                                        },
                                        "trade_mode": "FUTURES",
                                        "allow_short_positions": True,
                                    }
                                )

    return candidates


def _quality_gap_distance(entry: Dict[str, Any], quality_cfg) -> Dict[str, Any]:
    q = entry.get("quality_gate_metrics", {})
    gaps = {
        "walk_forward_folds_gap": max(
            0.0,
            float(quality_cfg.min_walk_forward_folds) - float(q.get("walk_forward_folds", 0.0)),
        ),
        "sharpe_mean_gap": max(
            0.0,
            float(quality_cfg.min_sharpe_ratio) - float(q.get("sharpe_ratio_mean", 0.0)),
        ),
        "calmar_mean_gap": max(
            0.0,
            float(quality_cfg.min_calmar_ratio) - float(q.get("calmar_ratio_mean", 0.0)),
        ),
        "regime_pass_gap": max(
            0.0,
            float(quality_cfg.min_regimes_passing) - float(q.get("passing_regimes", 0.0)),
        ),
    }
    return {
        "quality_gap_breakdown": gaps,
        "quality_gap_distance": float(sum(gaps.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strategy tuning sweep for gate pass")
    parser.add_argument("--api-url", default="http://localhost:8000/backtest/run")
    parser.add_argument("--api-key", default="ctr-admin-local-2026")
    parser.add_argument("--db-path", default="data/trading_bot.db")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--output", default="output/gate_tuning_sweep_latest.json")
    parser.add_argument("--limit", type=int, default=0, help="Optional candidate cap for quick runs")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    date_range = _fetch_4h_range(db_path, args.symbol, args.timeframe)

    benchmark_thresholds, quality_thresholds = promotion_thresholds_from_config(
        get_promotion_benchmark_config(),
        get_quality_gate_config(),
    )

    candidates = _build_candidates()
    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]

    headers = {"Content-Type": "application/json", "X-API-Key": args.api_key}

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for candidate in candidates:
        payload = {
            "symbol": args.symbol,
            "timeframes": [args.timeframe],
            "start_date": date_range["start_date"],
            "end_date": date_range["end_date"],
            "initial_capital": 10000,
            "commission": 0.001,
            "strategy_name": candidate["strategy_name"],
            "strategy_params": candidate["strategy_params"],
            "trade_mode": candidate["trade_mode"],
            "allow_short_positions": candidate["allow_short_positions"],
            "run_walk_forward_validation": True,
            "include_regime_slices": True,
        }

        try:
            response = requests.post(args.api_url, headers=headers, json=payload, timeout=240)
        except Exception as exc:
            errors.append({"label": candidate["label"], "error": str(exc)})
            continue

        if not response.ok:
            errors.append(
                {
                    "label": candidate["label"],
                    "status_code": response.status_code,
                    "body": response.text[:500],
                }
            )
            continue

        body = response.json().get("results", {})
        validation_summary = body.get("validation", {}) or {}
        quality_gate = body.get("quality_gate", {}) or {}

        total_return_pct = float(body.get("profit_loss_percent", 0.0) or 0.0)
        max_drawdown_pct = abs(float(body.get("max_drawdown", 0.0) or 0.0))
        estimated_calmar_ratio = (total_return_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0

        promo = evaluate_promotion_candidate(
            metrics={
                "gross_return_pct": total_return_pct,
                "net_return_pct": total_return_pct,
                "modeled_cost_pct": 0.0,
                "total_trades": float(body.get("trades", 0.0) or 0.0),
                "sharpe_ratio": float(body.get("sharpe_ratio", 0.0) or 0.0),
                "calmar_ratio": float(estimated_calmar_ratio),
                "max_drawdown_pct": max_drawdown_pct,
            },
            validation_summary=validation_summary,
            thresholds=benchmark_thresholds,
            quality_thresholds=quality_thresholds,
        )

        qmetrics = quality_gate.get("metrics", {}) if isinstance(quality_gate, dict) else {}
        row = {
            "label": candidate["label"],
            "strategy_name": candidate["strategy_name"],
            "strategy_params": candidate["strategy_params"],
            "trade_mode": candidate["trade_mode"],
            "allow_short_positions": candidate["allow_short_positions"],
            "start_date": date_range["start_date"],
            "end_date": date_range["end_date"],
            "bars": date_range["bar_count"],
            "profit_loss_percent": total_return_pct,
            "sharpe_ratio": float(body.get("sharpe_ratio", 0.0) or 0.0),
            "estimated_calmar_ratio": estimated_calmar_ratio,
            "max_drawdown_pct": max_drawdown_pct,
            "trades": int(body.get("trades", 0) or 0),
            "quality_gate_passed": bool(quality_gate.get("passed", False)),
            "quality_gate_reasons": list(quality_gate.get("reasons", [])),
            "quality_gate_metrics": {
                "walk_forward_folds": float(qmetrics.get("walk_forward_folds", 0.0) or 0.0),
                "sharpe_ratio_mean": float(qmetrics.get("sharpe_ratio_mean", 0.0) or 0.0),
                "calmar_ratio_mean": float(qmetrics.get("calmar_ratio_mean", 0.0) or 0.0),
                "eligible_regimes": float(qmetrics.get("eligible_regimes", 0.0) or 0.0),
                "passing_regimes": float(qmetrics.get("passing_regimes", 0.0) or 0.0),
            },
            "promotion_passed": bool(promo.passed),
            "promotion_reasons": list(promo.reasons),
            "promotion_metrics": dict(promo.metrics),
        }
        row.update(_quality_gap_distance(row, quality_thresholds))
        results.append(row)

    ranked = sorted(
        results,
        key=lambda r: (
            0 if r["promotion_passed"] else 1,
            0 if r["quality_gate_passed"] else 1,
            r["quality_gap_distance"],
            -r["profit_loss_percent"],
        ),
    )

    summary = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "dataset": {
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "start_date": date_range["start_date"],
            "end_date": date_range["end_date"],
            "bar_count": date_range["bar_count"],
        },
        "thresholds": {
            "promotion": {
                "min_total_trades": benchmark_thresholds.min_total_trades,
                "min_net_return_pct": benchmark_thresholds.min_net_return_pct,
                "min_sharpe_ratio": benchmark_thresholds.min_sharpe_ratio,
                "min_calmar_ratio": benchmark_thresholds.min_calmar_ratio,
                "max_drawdown_pct": benchmark_thresholds.max_drawdown_pct,
                "require_quality_gate": benchmark_thresholds.require_quality_gate,
            },
            "quality": {
                "min_walk_forward_folds": quality_thresholds.min_walk_forward_folds,
                "min_sharpe_ratio": quality_thresholds.min_sharpe_ratio,
                "min_calmar_ratio": quality_thresholds.min_calmar_ratio,
                "max_drawdown_pct": quality_thresholds.max_drawdown_pct,
                "min_regime_samples": quality_thresholds.min_regime_samples,
                "min_regime_win_rate": quality_thresholds.min_regime_win_rate,
                "min_regimes_passing": quality_thresholds.min_regimes_passing,
            },
        },
        "search": {
            "candidate_count": len(candidates),
            "evaluated_count": len(results),
            "error_count": len(errors),
            "quality_pass_count": sum(1 for r in results if r["quality_gate_passed"]),
            "promotion_pass_count": sum(1 for r in results if r["promotion_passed"]),
        },
        "top_candidates": ranked[:25],
        "all_candidates": ranked,
        "errors": errors,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(out_path),
                "candidate_count": len(candidates),
                "evaluated_count": len(results),
                "error_count": len(errors),
                "quality_pass_count": summary["search"]["quality_pass_count"],
                "promotion_pass_count": summary["search"]["promotion_pass_count"],
                "best_candidate": ranked[0]["label"] if ranked else None,
                "best_candidate_quality_distance": ranked[0]["quality_gap_distance"] if ranked else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
