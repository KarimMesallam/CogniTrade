import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is open and requests are blocked."""


@dataclass
class CircuitBreaker:
    """Simple circuit breaker for external service calls."""

    name: str
    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    failure_count: int = 0
    opened_at: Optional[float] = None
    state: str = "closed"

    def allow_request(self) -> bool:
        """Return whether the breaker currently allows a call."""
        if self.state != "open":
            return True
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at >= self.cooldown_seconds:
            self.state = "half_open"
            return True
        return False

    def record_success(self) -> None:
        """Reset breaker after a successful call."""
        self.failure_count = 0
        self.opened_at = None
        self.state = "closed"

    def record_failure(self) -> None:
        """Record a failure and open the breaker if threshold is crossed."""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.opened_at = time.time()

    def reset(self) -> None:
        """Force reset breaker state (useful in tests)."""
        self.failure_count = 0
        self.opened_at = None
        self.state = "closed"


def execute_with_resilience(
    operation_name: str,
    operation: Callable[[], T],
    *,
    max_retries: int,
    initial_backoff_seconds: float,
    retry_exceptions: Tuple[Type[BaseException], ...],
    circuit_breaker: Optional[CircuitBreaker] = None,
    logger: Optional[logging.Logger] = None,
) -> T:
    """
    Execute a callable with retries and optional circuit breaker protection.
    """
    if circuit_breaker and not circuit_breaker.allow_request():
        raise CircuitBreakerOpenError(
            f"Circuit breaker '{circuit_breaker.name}' is open for operation '{operation_name}'."
        )

    attempt = 0
    while True:
        try:
            result = operation()
            if circuit_breaker:
                circuit_breaker.record_success()
            return result
        except retry_exceptions as exc:
            if circuit_breaker:
                circuit_breaker.record_failure()

            if attempt >= max_retries:
                raise

            backoff = initial_backoff_seconds * (2 ** attempt)
            if logger:
                logger.warning(
                    "%s failed on attempt %s/%s: %s. Retrying in %.2fs.",
                    operation_name,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    backoff,
                )
            time.sleep(backoff)
            attempt += 1
