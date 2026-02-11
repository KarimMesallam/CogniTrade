import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.data_pipeline import PointInTimeDataPipeline, find_lookahead_violations
from bot.data_pipeline.regime_datasets import (
    RegimeDatasetBuildConfig,
    build_regime_evaluation_datasets,
)
from bot.database import Database
from bot.regime import MarketRegimeDetector


TEST_PIT_DB_PATH = os.path.join(os.path.dirname(__file__), "test_pit_pipeline.db")


@pytest.fixture(scope="function")
def test_db():
    if os.path.exists(TEST_PIT_DB_PATH):
        os.remove(TEST_PIT_DB_PATH)
    db = Database(TEST_PIT_DB_PATH)
    yield db
    if os.path.exists(TEST_PIT_DB_PATH):
        os.remove(TEST_PIT_DB_PATH)


def _build_candles(start_ms: int, interval_ms: int, count: int):
    candles = []
    price = 100.0
    for i in range(count):
        open_ms = start_ms + i * interval_ms
        close_ms = open_ms + interval_ms - 1
        candles.append(
            [
                open_ms,
                f"{price:.2f}",
                f"{price + 1:.2f}",
                f"{price - 1:.2f}",
                f"{price + 0.5:.2f}",
                "10.0",
                close_ms,
                "0",
                0,
                "0",
                "0",
                "0",
            ]
        )
        price += 1.0
    return candles


def _build_regime_candles(start_ms: int, interval_ms: int = 3_600_000):
    phases = [
        ("bull", 90, 0.0020),
        ("sideways", 90, 0.0002),
        ("bear", 90, -0.0020),
        ("high_vol", 90, 0.0),
    ]
    candles = []
    price = 100.0
    idx = 0
    for phase, count, base_change in phases:
        for step in range(count):
            open_ms = start_ms + idx * interval_ms
            close_ms = open_ms + interval_ms - 1
            if phase == "high_vol":
                pct_change = 0.030 if step % 2 == 0 else -0.028
            elif phase == "sideways":
                pct_change = base_change if step % 2 == 0 else -base_change
            else:
                pct_change = base_change

            open_price = price
            close_price = max(1.0, open_price * (1.0 + pct_change))
            high_price = max(open_price, close_price) * (1.0 + abs(pct_change) * 0.35 + 0.001)
            low_price = min(open_price, close_price) * (1.0 - abs(pct_change) * 0.35 - 0.001)

            candles.append(
                [
                    open_ms,
                    f"{open_price:.8f}",
                    f"{high_price:.8f}",
                    f"{low_price:.8f}",
                    f"{close_price:.8f}",
                    f"{(1200 + (idx % 20) * 10):.8f}",
                    close_ms,
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


def test_build_snapshot_has_no_lookahead_violations():
    pipeline = PointInTimeDataPipeline(require_monotonic_timestamps=True)
    start = int(datetime(2026, 2, 1, 0, 0, 0).timestamp() * 1000)
    candles = _build_candles(start, 60_000, 20)
    decision_ts = datetime.utcfromtimestamp((candles[-1][6]) / 1000).isoformat()

    rows = pipeline.build_snapshot(
        symbol="BTCUSDT",
        timeframe="1m",
        candles=candles,
        decision_timestamp=decision_ts,
    )

    assert rows
    assert {row["feature_name"] for row in rows} >= {
        "close_last",
        "return_1",
        "return_5",
        "realized_volatility_10",
    }
    assert find_lookahead_violations(rows) == []


def test_future_candle_is_excluded_from_asof_snapshot():
    pipeline = PointInTimeDataPipeline()
    start = int(datetime(2026, 2, 1, 0, 0, 0).timestamp() * 1000)
    candles = _build_candles(start, 60_000, 5)
    decision_dt = datetime.utcfromtimestamp(candles[-2][6] / 1000)

    rows = pipeline.build_snapshot(
        symbol="BTCUSDT",
        timeframe="1m",
        candles=candles,
        decision_timestamp=decision_dt.isoformat(),
    )

    for row in rows:
        assert row["feature_timestamp"] <= decision_dt.isoformat()
        assert row["available_timestamp"] <= decision_dt.isoformat()


def test_non_monotonic_timestamps_raise_validation_error():
    pipeline = PointInTimeDataPipeline(require_monotonic_timestamps=True)
    start = int(datetime(2026, 2, 1, 0, 0, 0).timestamp() * 1000)
    candles = _build_candles(start, 60_000, 4)
    # Swap to create non-monotonic open times.
    candles[2], candles[3] = candles[3], candles[2]

    with pytest.raises(ValueError, match="Timestamp integrity validation failed"):
        pipeline.build_snapshot(
            symbol="BTCUSDT",
            timeframe="1m",
            candles=candles,
            decision_timestamp=datetime.utcnow().isoformat(),
        )


def test_pit_snapshot_persistence_and_leakage_query(test_db):
    pipeline = PointInTimeDataPipeline()
    start = int(datetime(2026, 2, 1, 0, 0, 0).timestamp() * 1000)
    candles = _build_candles(start, 60_000, 12)
    decision_dt = datetime.utcfromtimestamp(candles[-1][6] / 1000)

    rows = pipeline.build_snapshot(
        symbol="BTCUSDT",
        timeframe="1m",
        candles=candles,
        decision_timestamp=decision_dt.isoformat(),
    )
    inserted = test_db.insert_feature_snapshots(rows)
    assert inserted >= len(rows)

    asof_rows = test_db.get_feature_snapshot_asof(
        "BTCUSDT",
        "1m",
        decision_dt.isoformat(),
    )
    assert len(asof_rows) == len(rows)
    assert test_db.find_feature_leakage_violations("BTCUSDT", "1m") == []

    # Insert an explicit leakage row and verify detection.
    future_dt = (decision_dt + timedelta(minutes=2)).isoformat()
    leak_row = {
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "decision_timestamp": decision_dt.isoformat(),
        "feature_name": "leaky_feature",
        "feature_value": 1.0,
        "feature_timestamp": future_dt,
        "available_timestamp": future_dt,
        "provenance": {"source": "unit-test"},
    }
    test_db.insert_feature_snapshots([leak_row])
    violations = test_db.find_feature_leakage_violations("BTCUSDT", "1m")
    assert any(v["feature_name"] == "leaky_feature" for v in violations)


def test_build_regime_evaluation_dataset_produces_representative_manifest():
    start = int(datetime(2026, 1, 1, 0, 0, 0).timestamp() * 1000)
    candles = _build_regime_candles(start)
    detector = MarketRegimeDetector(lookback_candles=30)
    config = RegimeDatasetBuildConfig(
        symbol="BTCUSDT",
        timeframe="1h",
        min_history_candles=35,
        decision_stride=3,
        train_ratio=0.7,
    )

    first = build_regime_evaluation_datasets(candles=candles, config=config, detector=detector)
    second = build_regime_evaluation_datasets(candles=candles, config=config, detector=detector)

    assert first.manifest["pit"]["lookahead_violation_count"] == 0
    assert first.manifest["pit"]["timestamp_integrity_passed"] is True
    assert first.manifest["missing_representative_regimes"] == []
    assert first.manifest["combined_dataset_fingerprint"]
    assert first.manifest["manifest_fingerprint"] == second.manifest["manifest_fingerprint"]
    assert first.manifest["combined_dataset_fingerprint"] == second.manifest["combined_dataset_fingerprint"]
    assert {"BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY"}.issubset(
        set(first.manifest["representative_regimes_present"])
    )


def test_regime_dataset_train_validation_counts_match_total():
    start = int(datetime(2026, 1, 1, 0, 0, 0).timestamp() * 1000)
    candles = _build_regime_candles(start)
    config = RegimeDatasetBuildConfig(
        symbol="BTCUSDT",
        timeframe="1h",
        min_history_candles=35,
        decision_stride=4,
        train_ratio=0.65,
    )
    result = build_regime_evaluation_datasets(candles=candles, config=config)

    assert not result.combined_frame.empty
    for regime, metadata in result.manifest["regimes"].items():
        total = int(metadata["row_count"])
        train_count = int(metadata["train_row_count"])
        validation_count = int(metadata["validation_row_count"])
        assert train_count + validation_count == total
        if total > 1:
            assert train_count > 0
        if total > 2:
            assert metadata["validation_fingerprint"] != ""
        assert regime in result.per_regime_frames
