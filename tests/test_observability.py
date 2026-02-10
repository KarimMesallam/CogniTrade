import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.observability.telemetry import ObservabilityManager


def test_trace_lifecycle_emits_events_and_latency_alert():
    manager = ObservabilityManager(
        max_events=200,
        latency_alert_ms=0.1,
        error_rate_alert_threshold=0.8,
        error_rate_min_events=10,
    )

    trace = manager.start_trace(component="unit", operation="trace_test")
    time.sleep(0.002)
    elapsed_ms = manager.end_trace(trace, status="ok")

    assert elapsed_ms >= 0.0

    events = manager.get_recent_events(limit=20)
    event_types = {event.get("event_type") for event in events}
    assert "trace_start" in event_types
    assert "trace_end" in event_types
    assert "latency" in event_types

    alerts = manager.get_recent_alerts(limit=10)
    assert any(alert.get("alert_type") == "latency" for alert in alerts)


def test_error_rate_alert_triggers_on_sustained_failures():
    manager = ObservabilityManager(
        max_events=200,
        latency_alert_ms=999999.0,
        error_rate_alert_threshold=0.5,
        error_rate_min_events=4,
    )

    manager.record_error(component="unit", error_type="e1", message="err1")
    manager.record_error(component="unit", error_type="e2", message="err2")
    manager.record_error(component="unit", error_type="e3", message="err3")
    manager.record_error(component="unit", error_type="e4", message="err4")

    alerts = manager.get_recent_alerts(limit=10)
    assert any(alert.get("alert_type") == "error_rate" for alert in alerts)


def test_dashboard_aggregates_trade_and_pnl_metrics():
    manager = ObservabilityManager(
        max_events=200,
        latency_alert_ms=999999.0,
        error_rate_alert_threshold=1.0,
        error_rate_min_events=100,
    )

    manager.record_pnl_attribution(
        symbol="BTCUSDT",
        strategy_pnl={"simple": 5.0, "technical": -1.0},
        total_pnl=4.0,
        regime="BULL",
    )
    manager.record_trade_decision(
        symbol="BTCUSDT",
        signal_consensus="BUY",
        llm_decision="BUY",
        executed=True,
        trade_mode="SPOT",
        strategies=["simple", "technical"],
    )

    snap = manager.get_dashboard_snapshot(window_minutes=60)
    assert snap["pnl_attribution"]["total_pnl"] == 4.0
    assert snap["pnl_attribution"]["strategy_pnl"]["simple"] == 5.0
    assert snap["trade_decisions"]["count"] == 1.0
    assert snap["trade_decisions"]["executed_count"] == 1.0


def test_persistence_store_retains_events_and_alerts(tmp_path):
    db_url = f"sqlite:///{Path(tmp_path) / 'obs.db'}"
    manager = ObservabilityManager(
        max_events=50,
        latency_alert_ms=0.1,
        error_rate_alert_threshold=1.0,
        error_rate_min_events=100,
        persistence_enabled=True,
        persistence_db_url=db_url,
    )
    manager.reset()

    trace = manager.start_trace(component="persist", operation="op")
    time.sleep(0.002)
    manager.end_trace(trace, status="ok")
    manager.record_error(component="persist", error_type="demo", message="demo error")

    # Force reload from persistent backend by creating a fresh manager
    reloaded = ObservabilityManager(
        max_events=10,
        latency_alert_ms=999999.0,
        error_rate_alert_threshold=1.0,
        error_rate_min_events=100,
        persistence_enabled=True,
        persistence_db_url=db_url,
    )

    events = reloaded.get_recent_events(limit=20)
    assert any(event.get("event_type") == "trace_start" for event in events)
    assert any(event.get("event_type") == "trace_end" for event in events)
    assert any(event.get("event_type") == "error" for event in events)

    alerts = reloaded.get_recent_alerts(limit=20)
    assert any(alert.get("alert_type") == "latency" for alert in alerts)
