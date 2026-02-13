import os
import sys
import time
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.notifications import TelegramNotifier, _NoOpNotifier, _TokenBucket, get_notifier


def test_token_bucket_allows_within_capacity():
    bucket = _TokenBucket(tokens_per_minute=60)
    assert bucket.acquire() is True


def test_token_bucket_denies_when_exhausted():
    bucket = _TokenBucket(tokens_per_minute=1)
    assert bucket.acquire() is True
    assert bucket.acquire() is False


def test_token_bucket_refills_over_time():
    bucket = _TokenBucket(tokens_per_minute=600)
    assert bucket.acquire() is True
    assert bucket.acquire() is True
    time.sleep(0.15)
    assert bucket.acquire() is True


def test_noop_notifier_does_not_raise():
    noop = _NoOpNotifier()
    noop.send("test")
    noop.send_alert({"severity": "high", "message": "test"})
    noop.send_lifecycle("started")
    noop.send_daily_summary({"overall_pass": True, "metrics": {}, "checks": {}})
    noop.shutdown()
    assert noop.enabled is False


@patch("bot.notifications.requests.post")
def test_send_alert_enqueues_and_sends(mock_post):
    mock_post.return_value = MagicMock(status_code=200)
    notifier = TelegramNotifier(
        bot_token="test-token",
        chat_id="12345",
        rate_limit_per_minute=60,
    )
    notifier.send_alert({
        "severity": "critical",
        "message": "Kill switch triggered",
        "alert_type": "error",
        "timestamp": "2026-01-01T00:00:00",
    })
    time.sleep(1.0)
    notifier.shutdown(timeout=3.0)
    assert mock_post.called
    call_kwargs = mock_post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json") or {}
    assert "12345" == body.get("chat_id")
    assert "Kill switch triggered" in body.get("text", "")


@patch("bot.notifications.requests.post")
def test_send_lifecycle_sends_message(mock_post):
    mock_post.return_value = MagicMock(status_code=200)
    notifier = TelegramNotifier(
        bot_token="test-token",
        chat_id="12345",
        rate_limit_per_minute=60,
    )
    notifier.send_lifecycle("started")
    time.sleep(1.0)
    notifier.shutdown(timeout=3.0)
    assert mock_post.called
    body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json") or {}
    assert "Started" in body.get("text", "")


def test_queue_full_does_not_raise():
    notifier = TelegramNotifier(
        bot_token="test-token",
        chat_id="12345",
        rate_limit_per_minute=60,
        max_queue_size=2,
    )
    notifier._shutdown.set()
    for _ in range(10):
        notifier.send("message")
    notifier.shutdown(timeout=1.0)


def test_get_notifier_disabled_returns_noop():
    import bot.notifications as mod
    old = mod._GLOBAL_NOTIFIER
    mod._GLOBAL_NOTIFIER = None
    try:
        noop = get_notifier({"enabled": False})
        assert isinstance(noop, _NoOpNotifier)
    finally:
        mod._GLOBAL_NOTIFIER = old


def test_get_notifier_missing_token_returns_noop():
    import bot.notifications as mod
    old = mod._GLOBAL_NOTIFIER
    mod._GLOBAL_NOTIFIER = None
    try:
        noop = get_notifier({"enabled": True, "bot_token": "", "chat_id": "123"})
        assert isinstance(noop, _NoOpNotifier)
    finally:
        mod._GLOBAL_NOTIFIER = old
