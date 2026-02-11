from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

from bot.data_pipeline.point_in_time import (
    build_asof_feature_rows,
    find_lookahead_violations,
    normalize_candles,
    validate_timestamp_integrity,
)
from bot.regime import (
    MarketRegimeDetector,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_HIGH_VOL,
    REGIME_UNKNOWN,
    REGIME_SIDEWAYS,
)
from research.registry import fingerprint_dataframe


REPRESENTATIVE_REGIMES: Tuple[str, ...] = (
    REGIME_BULL,
    REGIME_BEAR,
    REGIME_SIDEWAYS,
    REGIME_HIGH_VOL,
)


@dataclass(frozen=True)
class RegimeDatasetBuildConfig:
    """Configuration for building representative PIT evaluation datasets."""

    symbol: str
    timeframe: str
    min_history_candles: int = 60
    decision_stride: int = 1
    train_ratio: float = 0.70
    include_unknown_regime: bool = False
    feature_source: str = "exchange_ohlcv"
    dataset_version: str = "g0_regime_pit_v1"


@dataclass(frozen=True)
class RegimeDatasetBuildResult:
    """Result payload containing built frames and deterministic manifest."""

    combined_frame: pd.DataFrame
    per_regime_frames: Mapping[str, pd.DataFrame]
    manifest: Dict[str, Any]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_train_validation(frame: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    ratio = max(0.1, min(0.95, float(train_ratio)))
    if len(frame) < 2:
        return frame.copy(), frame.iloc[0:0].copy()
    split_index = int(len(frame) * ratio)
    split_index = max(1, min(split_index, len(frame) - 1))
    return frame.iloc[:split_index].reset_index(drop=True), frame.iloc[split_index:].reset_index(drop=True)


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    include_columns = sorted(frame.columns.tolist())
    sort_fields = [field for field in ["decision_timestamp", "regime"] if field in include_columns]
    return fingerprint_dataframe(frame, sort_by=sort_fields, include_columns=include_columns)


def _build_manifest(
    *,
    config: RegimeDatasetBuildConfig,
    combined_frame: pd.DataFrame,
    per_regime_frames: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    regime_manifests: Dict[str, Dict[str, Any]] = {}
    for regime, frame in sorted(per_regime_frames.items()):
        train_frame, validation_frame = _split_train_validation(frame, config.train_ratio)
        regime_manifests[regime] = {
            "row_count": int(len(frame)),
            "train_row_count": int(len(train_frame)),
            "validation_row_count": int(len(validation_frame)),
            "start_timestamp": str(frame["decision_timestamp"].iloc[0]) if not frame.empty else None,
            "end_timestamp": str(frame["decision_timestamp"].iloc[-1]) if not frame.empty else None,
            "dataset_fingerprint": _frame_fingerprint(frame),
            "train_fingerprint": _frame_fingerprint(train_frame),
            "validation_fingerprint": _frame_fingerprint(validation_frame),
            "avg_confidence": float(frame["regime_confidence"].mean()) if not frame.empty else 0.0,
        }

    representative_present = [
        regime
        for regime in REPRESENTATIVE_REGIMES
        if int(regime_manifests.get(regime, {}).get("row_count", 0)) > 0
    ]
    missing_representative = [
        regime for regime in REPRESENTATIVE_REGIMES if regime not in representative_present
    ]

    manifest_base = {
        "dataset_version": config.dataset_version,
        "generated_at": datetime.utcnow().isoformat(),
        "symbol": config.symbol,
        "timeframe": config.timeframe,
        "min_history_candles": int(config.min_history_candles),
        "decision_stride": int(config.decision_stride),
        "total_decision_rows": int(len(combined_frame)),
        "combined_dataset_fingerprint": _frame_fingerprint(combined_frame),
        "pit": {
            "timestamp_integrity_passed": True,
            "lookahead_violation_count": 0,
        },
        "representative_regimes_required": list(REPRESENTATIVE_REGIMES),
        "representative_regimes_present": representative_present,
        "missing_representative_regimes": missing_representative,
        "regimes": regime_manifests,
    }

    signature_payload = dict(manifest_base)
    signature_payload.pop("generated_at", None)
    manifest_base["manifest_fingerprint"] = _sha256(_stable_json(signature_payload))
    return manifest_base


def build_regime_evaluation_datasets(
    *,
    candles: Sequence[Any],
    config: RegimeDatasetBuildConfig,
    detector: Optional[MarketRegimeDetector] = None,
) -> RegimeDatasetBuildResult:
    """
    Build representative, deterministic evaluation datasets partitioned by market regime.

    Each decision row is built from PIT-safe features as-of the decision timestamp and
    regime state derived from historical closes only.
    """
    if int(config.min_history_candles) < 3:
        raise ValueError("min_history_candles must be >= 3")
    if int(config.decision_stride) <= 0:
        raise ValueError("decision_stride must be > 0")

    normalized = normalize_candles(candles)
    timestamps_ok, issues = validate_timestamp_integrity(normalized, require_monotonic_timestamps=True)
    if not timestamps_ok:
        raise ValueError(f"Timestamp integrity validation failed: {'; '.join(issues)}")
    if len(normalized) < int(config.min_history_candles):
        raise ValueError("Not enough candles to build regime datasets")

    active_detector = detector or MarketRegimeDetector()
    closes = [candle.close for candle in normalized]
    rows: list[Dict[str, Any]] = []

    for idx in range(int(config.min_history_candles) - 1, len(normalized), int(config.decision_stride)):
        decision_timestamp = normalized[idx].close_time.isoformat()
        snapshot_rows = build_asof_feature_rows(
            symbol=config.symbol,
            timeframe=config.timeframe,
            candles=candles,
            decision_timestamp=decision_timestamp,
            source=config.feature_source,
            require_monotonic_timestamps=False,
        )
        violations = find_lookahead_violations(snapshot_rows)
        if violations:
            raise ValueError(
                f"Detected PIT leakage while building regime datasets at {decision_timestamp}"
            )

        feature_values = {
            str(item["feature_name"]): float(item["feature_value"])
            for item in snapshot_rows
        }
        state = active_detector.detect_from_closes(
            closes[: idx + 1],
            timestamp=decision_timestamp,
        )
        if not config.include_unknown_regime and state.regime == REGIME_UNKNOWN:
            continue

        rows.append(
            {
                "symbol": config.symbol,
                "timeframe": config.timeframe,
                "decision_timestamp": decision_timestamp,
                "decision_index": int(idx),
                "regime": state.regime,
                "regime_confidence": float(state.confidence),
                "trend_pct": float(state.trend_pct),
                "volatility_pct": float(state.realized_volatility_pct),
                "regime_lookback_candles": int(state.lookback_candles),
                "close_last": float(feature_values.get("close_last", 0.0)),
                "return_1": float(feature_values.get("return_1", 0.0)),
                "return_5": float(feature_values.get("return_5", 0.0)),
                "realized_volatility_10": float(feature_values.get("realized_volatility_10", 0.0)),
            }
        )

    if not rows:
        raise ValueError("No decision rows generated for regime dataset build")

    combined_frame = pd.DataFrame(rows).sort_values("decision_timestamp").reset_index(drop=True)
    per_regime_frames: Dict[str, pd.DataFrame] = {}
    for regime, frame in combined_frame.groupby("regime", dropna=False):
        per_regime_frames[str(regime)] = frame.reset_index(drop=True)

    manifest = _build_manifest(
        config=config,
        combined_frame=combined_frame,
        per_regime_frames=per_regime_frames,
    )
    return RegimeDatasetBuildResult(
        combined_frame=combined_frame,
        per_regime_frames=per_regime_frames,
        manifest=manifest,
    )


def export_regime_dataset_manifest(manifest: Mapping[str, Any], output_path: str) -> str:
    """Persist regime dataset manifest to JSON and return the path."""
    path = str(output_path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True)
    return path
