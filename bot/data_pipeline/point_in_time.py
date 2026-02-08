import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class NormalizedCandle:
    """Normalized OHLCV candle with explicit open/close timestamps."""

    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source_index: int


def _to_naive_utc(value: Any) -> datetime:
    """Parse timestamp-like values into naive UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 10_000_000_000:
            # Milliseconds.
            numeric /= 1000.0
        return datetime.utcfromtimestamp(numeric)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError("Empty timestamp string")
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)

    raise ValueError(f"Unsupported timestamp value: {value!r}")


def normalize_candles(candles: Sequence[Any]) -> List[NormalizedCandle]:
    """Normalize Binance-style list candles or dict candles into a strict schema."""
    normalized: List[NormalizedCandle] = []

    for idx, candle in enumerate(candles):
        if isinstance(candle, (list, tuple)):
            if len(candle) < 6:
                raise ValueError(f"Candle at index {idx} has fewer than 6 fields")
            open_time = _to_naive_utc(candle[0])
            close_time = _to_naive_utc(candle[6]) if len(candle) > 6 and candle[6] is not None else open_time
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = float(candle[5])
        elif isinstance(candle, dict):
            open_time = _to_naive_utc(candle.get("open_time", candle.get("timestamp")))
            close_time_raw = candle.get("close_time", candle.get("timestamp"))
            close_time = _to_naive_utc(close_time_raw if close_time_raw is not None else open_time)
            open_price = float(candle["open"])
            high_price = float(candle["high"])
            low_price = float(candle["low"])
            close_price = float(candle["close"])
            volume = float(candle.get("volume", 0.0))
        else:
            raise ValueError(f"Unsupported candle type at index {idx}: {type(candle)}")

        normalized.append(
            NormalizedCandle(
                open_time=open_time,
                close_time=close_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                source_index=idx,
            )
        )

    return normalized


def validate_timestamp_integrity(
    candles: Sequence[NormalizedCandle],
    *,
    require_monotonic_timestamps: bool = True,
) -> Tuple[bool, List[str]]:
    """Validate timestamp ordering and candle span integrity."""
    issues: List[str] = []

    if not candles:
        return False, ["No candles supplied for timestamp validation."]

    seen_open_times = set()
    prev_open: Optional[datetime] = None
    prev_close: Optional[datetime] = None

    for idx, candle in enumerate(candles):
        if candle.close_time < candle.open_time:
            issues.append(
                f"candle[{idx}] close_time {candle.close_time.isoformat()} "
                f"is earlier than open_time {candle.open_time.isoformat()}"
            )

        if candle.open_time in seen_open_times:
            issues.append(f"duplicate open_time at candle[{idx}] -> {candle.open_time.isoformat()}")
        seen_open_times.add(candle.open_time)

        if require_monotonic_timestamps and prev_open is not None:
            if candle.open_time <= prev_open:
                issues.append(
                    f"non-monotonic open_time at candle[{idx}] -> {candle.open_time.isoformat()} "
                    f"(previous {prev_open.isoformat()})"
                )
            if prev_close is not None and candle.open_time < prev_close:
                issues.append(
                    f"overlapping candles around candle[{idx}] -> open_time {candle.open_time.isoformat()} "
                    f"before previous close_time {prev_close.isoformat()}"
                )

        prev_open = candle.open_time
        prev_close = candle.close_time

    return (len(issues) == 0), issues


def _safe_return(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return (current / previous) - 1.0


def _log_returns(closes: Sequence[float]) -> List[float]:
    result: List[float] = []
    if len(closes) < 2:
        return result
    for prev_close, next_close in zip(closes[:-1], closes[1:]):
        if prev_close <= 0 or next_close <= 0:
            continue
        result.append(math.log(next_close / prev_close))
    return result


def _build_basic_features(eligible_candles: Sequence[NormalizedCandle]) -> Dict[str, Dict[str, Any]]:
    closes = [c.close for c in eligible_candles]
    last_candle = eligible_candles[-1]

    ret_1 = _safe_return(closes[-1], closes[-2]) if len(closes) >= 2 else 0.0
    ret_5 = _safe_return(closes[-1], closes[-6]) if len(closes) >= 6 else 0.0

    recent_for_vol = closes[-11:] if len(closes) >= 11 else closes
    lrs = _log_returns(recent_for_vol)
    if not lrs:
        realized_vol = 0.0
    else:
        mean_lr = sum(lrs) / len(lrs)
        variance = sum((x - mean_lr) ** 2 for x in lrs) / len(lrs)
        realized_vol = math.sqrt(max(variance, 0.0))

    return {
        "close_last": {
            "value": float(closes[-1]),
            "feature_timestamp": last_candle.close_time,
            "window_candles": 1,
        },
        "return_1": {
            "value": float(ret_1),
            "feature_timestamp": last_candle.close_time,
            "window_candles": 2 if len(closes) >= 2 else 1,
        },
        "return_5": {
            "value": float(ret_5),
            "feature_timestamp": last_candle.close_time,
            "window_candles": 6 if len(closes) >= 6 else len(closes),
        },
        "realized_volatility_10": {
            "value": float(realized_vol),
            "feature_timestamp": last_candle.close_time,
            "window_candles": min(len(closes), 11),
        },
    }


def build_asof_feature_rows(
    *,
    symbol: str,
    timeframe: str,
    candles: Sequence[Any],
    decision_timestamp: Any,
    source: str = "exchange_ohlcv",
    require_monotonic_timestamps: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build deterministic point-in-time features where each feature is as-of decision timestamp.
    """
    decision_dt = _to_naive_utc(decision_timestamp)
    normalized = normalize_candles(candles)
    ok, issues = validate_timestamp_integrity(
        normalized,
        require_monotonic_timestamps=require_monotonic_timestamps,
    )
    if not ok:
        raise ValueError(f"Timestamp integrity validation failed: {'; '.join(issues)}")

    eligible = [candle for candle in normalized if candle.close_time <= decision_dt]
    if not eligible:
        raise ValueError(
            "No candles available at or before decision timestamp "
            f"{decision_dt.isoformat()} for PIT snapshot."
        )

    features = _build_basic_features(eligible)
    generated_at = datetime.utcnow().isoformat()

    rows: List[Dict[str, Any]] = []
    for feature_name, payload in features.items():
        feature_timestamp = payload["feature_timestamp"]
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "decision_timestamp": decision_dt.isoformat(),
                "feature_name": feature_name,
                "feature_value": float(payload["value"]),
                "feature_timestamp": feature_timestamp.isoformat(),
                "available_timestamp": feature_timestamp.isoformat(),
                "provenance": {
                    "source": source,
                    "generator": "point_in_time.basic_v1",
                    "window_candles": int(payload["window_candles"]),
                    "eligible_candle_count": len(eligible),
                    "generated_at": generated_at,
                },
            }
        )

    return rows


def find_lookahead_violations(feature_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return feature rows that violate point-in-time constraints."""
    violations: List[Dict[str, Any]] = []
    for row in feature_rows:
        decision_ts = _to_naive_utc(row["decision_timestamp"])
        feature_ts = _to_naive_utc(row["feature_timestamp"])
        available_ts = _to_naive_utc(row.get("available_timestamp", row["feature_timestamp"]))
        if feature_ts > decision_ts or available_ts > decision_ts:
            violations.append(row)
    return violations


class PointInTimeDataPipeline:
    """Pipeline that validates market timestamps and emits no-lookahead feature snapshots."""

    def __init__(
        self,
        *,
        require_monotonic_timestamps: bool = True,
        source: str = "exchange_ohlcv",
    ):
        self.require_monotonic_timestamps = require_monotonic_timestamps
        self.source = source

    def build_snapshot(
        self,
        *,
        symbol: str,
        timeframe: str,
        candles: Sequence[Any],
        decision_timestamp: Any,
    ) -> List[Dict[str, Any]]:
        rows = build_asof_feature_rows(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            decision_timestamp=decision_timestamp,
            source=self.source,
            require_monotonic_timestamps=self.require_monotonic_timestamps,
        )
        violations = find_lookahead_violations(rows)
        if violations:
            raise ValueError(
                f"Detected {len(violations)} point-in-time leakage violations in feature snapshot."
            )
        return rows

