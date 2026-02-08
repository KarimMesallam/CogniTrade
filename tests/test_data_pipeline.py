import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.data_pipeline import PointInTimeDataPipeline, find_lookahead_violations
from bot.database import Database


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

