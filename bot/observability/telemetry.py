from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Deque, Dict, List, Optional

from sqlalchemy import Column, Integer, MetaData, String, Table, Text, create_engine, delete, insert, select


logger = logging.getLogger("trading_bot")


def _utcnow() -> datetime:
    return datetime.utcnow()


def _iso_now() -> str:
    return _utcnow().isoformat()


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(v) for v in values)
    rank = (len(sorted_values) - 1) * float(q)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


class _ObservabilityPersistence:
    """SQL-backed persistence layer for observability events and alerts."""

    def __init__(self, db_url: str):
        self.db_url = str(db_url)
        self.engine = create_engine(self.db_url, future=True)
        self.metadata = MetaData()
        self.events_table = Table(
            "observability_events",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("timestamp", String(64), nullable=False, index=True),
            Column("event_type", String(64), nullable=False, index=True),
            Column("payload_json", Text, nullable=False),
        )
        self.alerts_table = Table(
            "observability_alerts",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("timestamp", String(64), nullable=False, index=True),
            Column("alert_type", String(64), nullable=False, index=True),
            Column("payload_json", Text, nullable=False),
        )
        self.metadata.create_all(self.engine)

    @staticmethod
    def _dump(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, default=str, sort_keys=False)

    @staticmethod
    def _load(raw: str) -> Dict[str, Any]:
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
        return {}

    def persist_event(self, event: Dict[str, Any]) -> None:
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(self.events_table).values(
                        timestamp=str(event.get("timestamp", _iso_now())),
                        event_type=str(event.get("event_type", "unknown")),
                        payload_json=self._dump(event),
                    )
                )
        except Exception as exc:
            logger.warning("Failed to persist observability event: %s", exc)

    def persist_alert(self, alert: Dict[str, Any]) -> None:
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(self.alerts_table).values(
                        timestamp=str(alert.get("timestamp", _iso_now())),
                        alert_type=str(alert.get("alert_type", "unknown")),
                        payload_json=self._dump(alert),
                    )
                )
        except Exception as exc:
            logger.warning("Failed to persist observability alert: %s", exc)

    def fetch_events(self, *, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            with self.engine.connect() as conn:
                stmt = select(self.events_table.c.payload_json).order_by(self.events_table.c.id.desc()).limit(
                    max(1, int(limit))
                )
                if event_type:
                    stmt = stmt.where(self.events_table.c.event_type == str(event_type))
                rows = conn.execute(stmt).fetchall()
            payloads = [self._load(row[0]) for row in rows if row and row[0]]
            payloads.reverse()
            return payloads
        except Exception as exc:
            logger.warning("Failed to fetch persisted observability events: %s", exc)
            return []

    def fetch_alerts(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            with self.engine.connect() as conn:
                stmt = select(self.alerts_table.c.payload_json).order_by(self.alerts_table.c.id.desc()).limit(
                    max(1, int(limit))
                )
                rows = conn.execute(stmt).fetchall()
            payloads = [self._load(row[0]) for row in rows if row and row[0]]
            payloads.reverse()
            return payloads
        except Exception as exc:
            logger.warning("Failed to fetch persisted observability alerts: %s", exc)
            return []

    def clear(self) -> None:
        try:
            with self.engine.begin() as conn:
                conn.execute(delete(self.events_table))
                conn.execute(delete(self.alerts_table))
        except Exception as exc:
            logger.warning("Failed to clear persisted observability state: %s", exc)


@dataclass(frozen=True)
class TraceContext:
    """Active trace context for operation-level telemetry."""

    trace_id: str
    component: str
    operation: str
    started_at_iso: str
    started_at_perf: float
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ObservabilityManager:
    """
    In-memory observability manager for traces, latency, errors, and PnL attribution.
    """

    def __init__(
        self,
        *,
        max_events: int = 4000,
        latency_alert_ms: float = 2500.0,
        error_rate_alert_threshold: float = 0.25,
        error_rate_min_events: int = 20,
        persistence_enabled: bool = False,
        persistence_db_url: str = "sqlite:///data/observability.db",
    ):
        self.max_events = max(100, int(max_events))
        self.latency_alert_ms = max(0.0, float(latency_alert_ms))
        self.error_rate_alert_threshold = max(0.0, float(error_rate_alert_threshold))
        self.error_rate_min_events = max(1, int(error_rate_min_events))
        self.persistence_enabled = bool(persistence_enabled)
        self.persistence_db_url = str(persistence_db_url)

        self._events: Deque[Dict[str, Any]] = deque(maxlen=self.max_events)
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=self.max_events)
        self._pending_alerts: Deque[Dict[str, Any]] = deque(maxlen=self.max_events)
        self._lock = threading.Lock()
        self._persistence: Optional[_ObservabilityPersistence] = None
        if self.persistence_enabled:
            self._persistence = _ObservabilityPersistence(self.persistence_db_url)

    def reconfigure(
        self,
        *,
        max_events: Optional[int] = None,
        latency_alert_ms: Optional[float] = None,
        error_rate_alert_threshold: Optional[float] = None,
        error_rate_min_events: Optional[int] = None,
        persistence_enabled: Optional[bool] = None,
        persistence_db_url: Optional[str] = None,
    ) -> None:
        """Update thresholds at runtime."""
        with self._lock:
            if max_events is not None and int(max_events) != self.max_events:
                existing_events = list(self._events)
                existing_alerts = list(self._alerts)
                existing_pending = list(self._pending_alerts)
                self.max_events = max(100, int(max_events))
                self._events = deque(existing_events[-self.max_events :], maxlen=self.max_events)
                self._alerts = deque(existing_alerts[-self.max_events :], maxlen=self.max_events)
                self._pending_alerts = deque(existing_pending[-self.max_events :], maxlen=self.max_events)
            if latency_alert_ms is not None:
                self.latency_alert_ms = max(0.0, float(latency_alert_ms))
            if error_rate_alert_threshold is not None:
                self.error_rate_alert_threshold = max(0.0, float(error_rate_alert_threshold))
            if error_rate_min_events is not None:
                self.error_rate_min_events = max(1, int(error_rate_min_events))
            if persistence_enabled is not None:
                self.persistence_enabled = bool(persistence_enabled)
            if persistence_db_url is not None:
                self.persistence_db_url = str(persistence_db_url)

            if self.persistence_enabled:
                if self._persistence is None or self._persistence.db_url != self.persistence_db_url:
                    self._persistence = _ObservabilityPersistence(self.persistence_db_url)
            else:
                self._persistence = None

    def reset(self) -> None:
        """Clear all in-memory telemetry (for tests)."""
        with self._lock:
            self._events.clear()
            self._alerts.clear()
            self._pending_alerts.clear()
            if self._persistence:
                self._persistence.clear()

    def _append_event(self, event: Dict[str, Any]) -> None:
        self._events.append(event)
        if self._persistence:
            self._persistence.persist_event(event)

    def _emit_alert(
        self,
        *,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        alert = {
            "alert_id": uuid.uuid4().hex,
            "timestamp": _iso_now(),
            "alert_type": str(alert_type),
            "severity": str(severity),
            "message": str(message),
            "details": details or {},
        }
        self._alerts.append(alert)
        self._pending_alerts.append(alert)
        if self._persistence:
            self._persistence.persist_alert(alert)
        self._append_event(
            {
                "event_type": "alert",
                "timestamp": alert["timestamp"],
                "alert": alert,
            }
        )
        return alert

    def start_trace(
        self,
        *,
        component: str,
        operation: str,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceContext:
        ctx = TraceContext(
            trace_id=uuid.uuid4().hex,
            component=str(component),
            operation=str(operation),
            started_at_iso=_iso_now(),
            started_at_perf=time.perf_counter(),
            request_id=request_id,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._append_event(
                {
                    "event_type": "trace_start",
                    "timestamp": ctx.started_at_iso,
                    "trace_id": ctx.trace_id,
                    "request_id": ctx.request_id,
                    "component": ctx.component,
                    "operation": ctx.operation,
                    "metadata": ctx.metadata,
                }
            )
        return ctx

    def end_trace(
        self,
        context: TraceContext,
        *,
        status: str = "ok",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        elapsed_ms = max(0.0, (time.perf_counter() - context.started_at_perf) * 1000.0)
        should_record_error = str(status).lower() != "ok" and bool(error)
        with self._lock:
            self._append_event(
                {
                    "event_type": "trace_end",
                    "timestamp": _iso_now(),
                    "trace_id": context.trace_id,
                    "request_id": context.request_id,
                    "component": context.component,
                    "operation": context.operation,
                    "status": str(status),
                    "latency_ms": float(elapsed_ms),
                    "error": str(error) if error else None,
                    "metadata": dict(metadata or {}),
                }
            )
            self._append_event(
                {
                    "event_type": "latency",
                    "timestamp": _iso_now(),
                    "trace_id": context.trace_id,
                    "request_id": context.request_id,
                    "component": context.component,
                    "operation": context.operation,
                    "latency_ms": float(elapsed_ms),
                    "success": str(status).lower() == "ok",
                }
            )
            self._maybe_emit_latency_alert(
                component=context.component,
                operation=context.operation,
                latency_ms=elapsed_ms,
                trace_id=context.trace_id,
                request_id=context.request_id,
            )
        if should_record_error:
            self.record_error(
                component=context.component,
                error_type="trace_error",
                message=str(error),
                severity="high",
                metadata={
                    "operation": context.operation,
                    "trace_id": context.trace_id,
                    "request_id": context.request_id,
                },
            )
        return float(elapsed_ms)

    def record_latency(
        self,
        *,
        component: str,
        operation: str,
        latency_ms: float,
        success: bool = True,
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._append_event(
                {
                    "event_type": "latency",
                    "timestamp": _iso_now(),
                    "trace_id": trace_id,
                    "request_id": request_id,
                    "component": str(component),
                    "operation": str(operation),
                    "latency_ms": float(latency_ms),
                    "success": bool(success),
                    "metadata": dict(metadata or {}),
                }
            )
            self._maybe_emit_latency_alert(
                component=str(component),
                operation=str(operation),
                latency_ms=float(latency_ms),
                trace_id=trace_id,
                request_id=request_id,
            )

    def _maybe_emit_latency_alert(
        self,
        *,
        component: str,
        operation: str,
        latency_ms: float,
        trace_id: Optional[str],
        request_id: Optional[str],
    ) -> None:
        if self.latency_alert_ms <= 0:
            return
        if float(latency_ms) < self.latency_alert_ms:
            return
        self._emit_alert(
            alert_type="latency",
            severity="medium",
            message=(
                f"Latency threshold breached for {component}.{operation}: "
                f"{float(latency_ms):.2f}ms >= {self.latency_alert_ms:.2f}ms"
            ),
            details={
                "component": component,
                "operation": operation,
                "latency_ms": float(latency_ms),
                "threshold_ms": float(self.latency_alert_ms),
                "trace_id": trace_id,
                "request_id": request_id,
            },
        )

    def record_error(
        self,
        *,
        component: str,
        error_type: str,
        message: str,
        severity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._append_event(
                {
                    "event_type": "error",
                    "timestamp": _iso_now(),
                    "component": str(component),
                    "error_type": str(error_type),
                    "message": str(message),
                    "severity": str(severity),
                    "metadata": dict(metadata or {}),
                }
            )
            self._maybe_emit_error_rate_alert()

    def _maybe_emit_error_rate_alert(self) -> None:
        recent = [
            event
            for event in self._events
            if event.get("event_type") in {"error", "trace_end"}
        ]
        recent = recent[-self.error_rate_min_events :]
        if len(recent) < self.error_rate_min_events:
            return

        errors = 0
        for event in recent:
            if event.get("event_type") == "error":
                errors += 1
            elif event.get("event_type") == "trace_end" and str(event.get("status", "")).lower() != "ok":
                errors += 1

        error_rate = float(errors / len(recent))
        if error_rate < self.error_rate_alert_threshold:
            return
        self._emit_alert(
            alert_type="error_rate",
            severity="high",
            message=(
                f"Error-rate threshold breached: {error_rate:.2%} "
                f">= {self.error_rate_alert_threshold:.2%}"
            ),
            details={
                "error_rate": error_rate,
                "sample_size": len(recent),
                "threshold": float(self.error_rate_alert_threshold),
            },
        )

    def record_pnl_attribution(
        self,
        *,
        symbol: str,
        strategy_pnl: Dict[str, float],
        total_pnl: float,
        regime: Optional[str] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._append_event(
                {
                    "event_type": "pnl_attribution",
                    "timestamp": timestamp or _iso_now(),
                    "symbol": str(symbol),
                    "strategy_pnl": {k: float(v) for k, v in (strategy_pnl or {}).items()},
                    "total_pnl": float(total_pnl),
                    "regime": regime,
                    "metadata": dict(metadata or {}),
                }
            )

    def record_trade_decision(
        self,
        *,
        symbol: str,
        signal_consensus: str,
        llm_decision: str,
        executed: bool,
        trade_mode: str,
        strategies: List[str],
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._append_event(
                {
                    "event_type": "trade_decision",
                    "timestamp": _iso_now(),
                    "trace_id": trace_id,
                    "request_id": request_id,
                    "symbol": str(symbol),
                    "signal_consensus": str(signal_consensus),
                    "llm_decision": str(llm_decision),
                    "executed": bool(executed),
                    "trade_mode": str(trade_mode),
                    "strategies": [str(s) for s in strategies],
                    "metadata": dict(metadata or {}),
                }
            )

    def get_recent_events(self, *, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        if self._persistence:
            persisted = self._persistence.fetch_events(event_type=event_type, limit=limit)
            if persisted:
                return persisted
        with self._lock:
            events = list(self._events)
        if event_type:
            events = [event for event in events if event.get("event_type") == event_type]
        return events[-max(1, int(limit)) :]

    def get_recent_alerts(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        if self._persistence:
            persisted = self._persistence.fetch_alerts(limit=limit)
            if persisted:
                return persisted
        with self._lock:
            alerts = list(self._alerts)
        return alerts[-max(1, int(limit)) :]

    def drain_pending_alerts(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        drained: List[Dict[str, Any]] = []
        remaining = max(1, int(limit))
        with self._lock:
            while self._pending_alerts and len(drained) < remaining:
                drained.append(self._pending_alerts.popleft())
        return drained

    def get_dashboard_snapshot(self, *, window_minutes: int = 60) -> Dict[str, Any]:
        window_minutes = max(1, int(window_minutes))
        cutoff = _utcnow() - timedelta(minutes=window_minutes)
        cutoff_iso = cutoff.isoformat()

        with self._lock:
            window_events = [
                event for event in self._events if str(event.get("timestamp", "")) >= cutoff_iso
            ]

        latency_values = [
            float(event.get("latency_ms", 0.0))
            for event in window_events
            if event.get("event_type") == "latency"
        ]
        error_events = [event for event in window_events if event.get("event_type") == "error"]
        trace_end_events = [
            event for event in window_events if event.get("event_type") == "trace_end"
        ]
        failed_trace_events = [
            event for event in trace_end_events if str(event.get("status", "ok")).lower() != "ok"
        ]
        pnl_events = [
            event for event in window_events if event.get("event_type") == "pnl_attribution"
        ]
        trade_decisions = [
            event for event in window_events if event.get("event_type") == "trade_decision"
        ]

        total_ops = max(1, len(trace_end_events))
        error_rate = float((len(error_events) + len(failed_trace_events)) / total_ops)

        strategy_pnl: Dict[str, float] = {}
        total_pnl = 0.0
        for event in pnl_events:
            total_pnl += float(event.get("total_pnl", 0.0))
            for strategy, pnl in (event.get("strategy_pnl") or {}).items():
                strategy_pnl[str(strategy)] = float(strategy_pnl.get(str(strategy), 0.0) + float(pnl))

        executed_decisions = sum(1 for event in trade_decisions if bool(event.get("executed")))
        execution_rate = float(executed_decisions / len(trade_decisions)) if trade_decisions else 0.0

        return {
            "window_minutes": float(window_minutes),
            "event_count": float(len(window_events)),
            "trace_count": float(len(trace_end_events)),
            "latency": {
                "count": float(len(latency_values)),
                "p50_ms": float(median(latency_values)) if latency_values else 0.0,
                "p95_ms": _percentile(latency_values, 0.95),
                "max_ms": float(max(latency_values)) if latency_values else 0.0,
            },
            "errors": {
                "count": float(len(error_events) + len(failed_trace_events)),
                "error_rate": float(error_rate),
            },
            "trade_decisions": {
                "count": float(len(trade_decisions)),
                "executed_count": float(executed_decisions),
                "execution_rate": float(execution_rate),
            },
            "pnl_attribution": {
                "event_count": float(len(pnl_events)),
                "total_pnl": float(total_pnl),
                "strategy_pnl": strategy_pnl,
            },
            "alerts_last_10": self.get_recent_alerts(limit=10),
        }


_GLOBAL_MANAGER: Optional[ObservabilityManager] = None
_GLOBAL_LOCK = threading.Lock()


def get_observability_manager(
    *,
    max_events: int = 4000,
    latency_alert_ms: float = 2500.0,
    error_rate_alert_threshold: float = 0.25,
    error_rate_min_events: int = 20,
    persistence_enabled: bool = False,
    persistence_db_url: str = "sqlite:///data/observability.db",
) -> ObservabilityManager:
    """Get global singleton observability manager."""
    global _GLOBAL_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = ObservabilityManager(
                max_events=max_events,
                latency_alert_ms=latency_alert_ms,
                error_rate_alert_threshold=error_rate_alert_threshold,
                error_rate_min_events=error_rate_min_events,
                persistence_enabled=persistence_enabled,
                persistence_db_url=persistence_db_url,
            )
        else:
            _GLOBAL_MANAGER.reconfigure(
                max_events=max_events,
                latency_alert_ms=latency_alert_ms,
                error_rate_alert_threshold=error_rate_alert_threshold,
                error_rate_min_events=error_rate_min_events,
                persistence_enabled=persistence_enabled,
                persistence_db_url=persistence_db_url,
            )
        return _GLOBAL_MANAGER
