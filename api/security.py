from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Set, Tuple


def _as_set(values) -> Set[str]:
    if not values:
        return set()
    return {str(item).strip() for item in values if str(item).strip()}


@dataclass
class AuthorizationResult:
    allowed: bool
    status_code: int
    reason: str
    role: Optional[str] = None
    api_key: Optional[str] = None


class SlidingWindowRateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, max_requests: int = 120, window_seconds: int = 60):
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1, int(window_seconds))
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def reconfigure(self, *, max_requests: Optional[int] = None, window_seconds: Optional[int] = None) -> None:
        with self._lock:
            if max_requests is not None:
                self.max_requests = max(1, int(max_requests))
            if window_seconds is not None:
                self.window_seconds = max(1, int(window_seconds))

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def allow(self, client_id: str) -> Tuple[bool, int]:
        now = time.time()
        key = str(client_id or "unknown")
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.max_requests:
                retry_after = max(1, int(events[0] + self.window_seconds - now))
                return False, retry_after
            events.append(now)
        return True, 0


class APISecurityController:
    """
    API-key authorization + request rate limiting.

    Roles:
    - read: GET/HEAD/OPTIONS endpoints
    - admin: write endpoints (POST/PUT/PATCH/DELETE)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config: dict = {}
        self.auth_enabled: bool = False
        self.allow_public_health: bool = True
        self.read_api_keys: Set[str] = set()
        self.admin_api_keys: Set[str] = set()
        self.rate_limit_enabled: bool = True
        self.rate_limiter = SlidingWindowRateLimiter()
        self.update_config(config or {})

    def update_config(self, config: dict) -> None:
        cfg = dict(config or {})
        self.config = cfg
        self.auth_enabled = bool(cfg.get("auth_enabled", False))
        self.allow_public_health = bool(cfg.get("allow_public_health", True))
        self.read_api_keys = _as_set(cfg.get("read_api_keys", []))
        self.admin_api_keys = _as_set(cfg.get("admin_api_keys", []))
        self.rate_limit_enabled = bool(cfg.get("rate_limit_enabled", True))
        self.rate_limiter.reconfigure(
            max_requests=int(cfg.get("rate_limit_requests", 120)),
            window_seconds=int(cfg.get("rate_limit_window_seconds", 60)),
        )

    def reset_runtime_state(self) -> None:
        self.rate_limiter.reset()

    @staticmethod
    def _is_read_method(method: str) -> bool:
        return str(method).upper() in {"GET", "HEAD", "OPTIONS"}

    def _is_public_path(self, path: str) -> bool:
        norm = str(path or "").rstrip("/")
        if norm == "":
            norm = "/"
        if self.allow_public_health and norm == "/health":
            return True
        return False

    @staticmethod
    def extract_api_key(request) -> Optional[str]:
        # Preferred header.
        raw = request.headers.get("X-API-Key")
        if raw:
            return str(raw).strip()

        # Optional bearer token fallback.
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
            if token:
                return token
        return None

    @staticmethod
    def client_id(request, api_key: Optional[str]) -> str:
        if api_key:
            return f"key:{api_key}"
        host = "unknown"
        if getattr(request, "client", None) and request.client.host:
            host = str(request.client.host)
        return f"ip:{host}"

    def authorize(self, request) -> AuthorizationResult:
        if not self.auth_enabled:
            return AuthorizationResult(allowed=True, status_code=200, reason="auth_disabled", role="admin")

        path = str(request.url.path)
        method = str(request.method).upper()
        if self._is_public_path(path):
            return AuthorizationResult(allowed=True, status_code=200, reason="public_path", role="public")

        api_key = self.extract_api_key(request)
        if not api_key:
            return AuthorizationResult(
                allowed=False,
                status_code=401,
                reason="missing_api_key",
            )

        required_role = "read" if self._is_read_method(method) else "admin"
        if required_role == "admin":
            if api_key not in self.admin_api_keys:
                return AuthorizationResult(
                    allowed=False,
                    status_code=403,
                    reason="admin_role_required",
                )
            return AuthorizationResult(
                allowed=True,
                status_code=200,
                reason="authorized",
                role="admin",
                api_key=api_key,
            )

        if api_key in self.admin_api_keys:
            return AuthorizationResult(
                allowed=True,
                status_code=200,
                reason="authorized",
                role="admin",
                api_key=api_key,
            )
        if api_key in self.read_api_keys:
            return AuthorizationResult(
                allowed=True,
                status_code=200,
                reason="authorized",
                role="read",
                api_key=api_key,
            )
        return AuthorizationResult(
            allowed=False,
            status_code=403,
            reason="read_role_required",
        )

    def check_rate_limit(self, request, api_key: Optional[str]) -> Tuple[bool, int]:
        if not self.rate_limit_enabled:
            return True, 0
        client_id = self.client_id(request, api_key)
        return self.rate_limiter.allow(client_id)
