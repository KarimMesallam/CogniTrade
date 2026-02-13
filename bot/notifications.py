"""Telegram notification system for CogniTrade soak monitoring."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("trading_bot")

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class _TokenBucket:
    """Simple token-bucket rate limiter (thread-safe)."""

    def __init__(self, tokens_per_minute: int = 20):
        self._capacity = max(1, int(tokens_per_minute))
        self._tokens = float(self._capacity)
        self._refill_rate = float(self._capacity) / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def wait_and_acquire(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.acquire():
                return True
            time.sleep(0.1)
        return False


class TelegramNotifier:
    """Queue-based, fire-and-forget Telegram notifier running on a daemon thread."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        rate_limit_per_minute: int = 20,
        max_queue_size: int = 200,
    ):
        self._bot_token = str(bot_token)
        self._chat_id = str(chat_id)
        self._enabled = bool(self._bot_token and self._chat_id)
        self._bucket = _TokenBucket(rate_limit_per_minute)
        self._queue: queue.Queue[str] = queue.Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()

        if self._enabled:
            self._thread = threading.Thread(
                target=self._worker, name="telegram-notifier", daemon=True
            )
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(str(text))
        except queue.Full:
            logger.warning("Telegram notification queue full; dropping message")

    def send_alert(self, alert: Dict[str, Any]) -> None:
        severity = str(alert.get("severity", "medium")).upper()
        emoji = {"LOW": "\u2139\ufe0f", "MEDIUM": "\u26a0\ufe0f", "HIGH": "\u2757", "CRITICAL": "\U0001f6a8"}.get(
            severity, "\u26a0\ufe0f"
        )
        message = (
            f"{emoji} *CogniTrade Alert* [{severity}]\n"
            f"{alert.get('message', 'No message')}\n"
            f"Type: {alert.get('alert_type', 'unknown')}\n"
            f"Time: {alert.get('timestamp', datetime.now(timezone.utc).isoformat())}"
        )
        self.send(message)

    def send_lifecycle(self, event: str) -> None:
        emoji = {
            "started": "\u2705",
            "stopped": "\u23f9\ufe0f",
            "crashed": "\U0001f4a5",
        }.get(event.lower(), "\u2139\ufe0f")
        self.send(f"{emoji} *CogniTrade {event.capitalize()}*\nTime: {datetime.now(timezone.utc).isoformat()}")

    def send_trade(self, symbol: str, side: str, order: Dict[str, Any], regime: str = "") -> None:
        emoji = {"BUY": "\U0001f7e2", "SELL": "\U0001f534"}.get(side.upper(), "\U0001f535")
        order_id = order.get("orderId", "N/A")
        qty = order.get("executedQty") or order.get("origQty") or "?"
        price = order.get("price") or "0"
        if price in ("0", "0.00000000", "market"):
            fills = order.get("fills") or []
            if fills:
                price = fills[0].get("price", "market")
            else:
                price = "market"
        cost_str = ""
        try:
            cost = float(qty) * float(price)
            cost_str = f"Cost: {cost:.2f} USDT"
        except (ValueError, TypeError):
            pass
        lines = [
            f"{emoji} *Trade Executed: {side.upper()}*",
            f"Symbol: {symbol}",
            f"Qty: {qty} | Price: {price}",
        ]
        if cost_str:
            lines.append(cost_str)
        lines.append(f"Order ID: {order_id}")
        if regime:
            lines.append(f"Regime: {regime}")
        lines.append(f"Time: {datetime.now(timezone.utc).isoformat()}")
        self.send("\n".join(lines))

    def send_daily_summary(self, summary: Dict[str, Any]) -> None:
        metrics = summary.get("metrics", {})
        checks = summary.get("checks", {})
        overall = "\u2705 PASS" if summary.get("overall_pass") else "\u274c FAIL"
        lines = [
            f"\U0001f4ca *CogniTrade Daily Soak Summary* {overall}",
            f"Rollout: {summary.get('rollout_id', 'N/A')}",
            f"Error rate: {metrics.get('error_rate', 0):.2%}",
            f"Latency P95: {metrics.get('latency_p95_ms', 0):.0f}ms",
            f"Drawdown: {metrics.get('last_drawdown_pct', 0):.2f}%",
            f"Return: {metrics.get('last_total_return_pct', 0):.2f}%",
            f"Events: {int(metrics.get('event_count', 0))} | Traces: {int(metrics.get('trace_count', 0))}",
        ]
        failed = [k for k, v in checks.items() if not v]
        if failed:
            lines.append(f"Failed checks: {', '.join(failed)}")
        self.send("\n".join(lines))

    def shutdown(self, timeout: float = 5.0) -> None:
        self._shutdown.set()
        if self._enabled and hasattr(self, "_thread"):
            self._thread.join(timeout=timeout)

    def _worker(self) -> None:
        url = TELEGRAM_API_URL.format(token=self._bot_token)
        while not self._shutdown.is_set():
            try:
                text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if not self._bucket.wait_and_acquire(timeout=10.0):
                logger.warning("Telegram rate limit exceeded; dropping message")
                continue
            try:
                resp = requests.post(
                    url,
                    json={"chat_id": self._chat_id, "text": text, "parse_mode": "Markdown"},
                    timeout=10,
                )
                if resp.status_code != 200:
                    logger.warning("Telegram API returned %s: %s", resp.status_code, resp.text[:200])
            except Exception as exc:
                logger.warning("Failed to send Telegram message: %s", exc)


class _NoOpNotifier:
    """Silent stand-in when Telegram is not configured."""

    enabled = False

    def send(self, text: str) -> None:
        pass

    def send_alert(self, alert: Dict[str, Any]) -> None:
        pass

    def send_lifecycle(self, event: str) -> None:
        pass

    def send_trade(self, symbol: str, side: str, order: Dict[str, Any], regime: str = "") -> None:
        pass

    def send_daily_summary(self, summary: Dict[str, Any]) -> None:
        pass

    def shutdown(self, timeout: float = 5.0) -> None:
        pass


_GLOBAL_NOTIFIER: Optional[TelegramNotifier] = None
_GLOBAL_LOCK = threading.Lock()


def get_notifier(config: Optional[Dict[str, Any]] = None) -> TelegramNotifier | _NoOpNotifier:
    """Get or create the global Telegram notifier singleton.

    Mirrors the ``get_observability_manager()`` pattern.
    """
    global _GLOBAL_NOTIFIER
    with _GLOBAL_LOCK:
        if _GLOBAL_NOTIFIER is not None:
            return _GLOBAL_NOTIFIER

        cfg = config or {}
        if not cfg.get("enabled", True):
            return _NoOpNotifier()

        bot_token = str(cfg.get("bot_token", "")).strip()
        chat_id = str(cfg.get("chat_id", "")).strip()
        if not bot_token or not chat_id:
            logger.info("Telegram notifications disabled (missing bot_token or chat_id)")
            return _NoOpNotifier()

        notifier = TelegramNotifier(
            bot_token=bot_token,
            chat_id=chat_id,
            rate_limit_per_minute=int(cfg.get("rate_limit_per_minute", 20)),
        )
        _GLOBAL_NOTIFIER = notifier
        return _GLOBAL_NOTIFIER
