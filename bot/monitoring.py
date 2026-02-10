from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Deque, Dict, Optional


@dataclass(frozen=True)
class StrategyEdgeStatus:
    """Current edge health status for a strategy."""

    strategy: str
    sample_count: int
    win_rate: float
    mean_edge_return: float
    state: str
    size_multiplier: float
    disable_trading: bool
    reason: str
    timestamp: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "sample_count": int(self.sample_count),
            "win_rate": float(self.win_rate),
            "mean_edge_return": float(self.mean_edge_return),
            "state": self.state,
            "size_multiplier": float(self.size_multiplier),
            "disable_trading": bool(self.disable_trading),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class EdgeDecayMonitor:
    """
    Monitor strategy edge quality and emit automatic de-risk/disable actions.
    """

    def __init__(
        self,
        *,
        window_size: int = 50,
        min_samples: int = 20,
        derisk_hit_rate_threshold: float = 0.45,
        disable_hit_rate_threshold: float = 0.35,
        derisk_mean_return_threshold: float = -0.0002,
        derisk_size_multiplier: float = 0.5,
        disable_sticky: bool = True,
    ):
        self.window_size = max(5, int(window_size))
        self.min_samples = max(1, int(min_samples))
        self.derisk_hit_rate_threshold = float(derisk_hit_rate_threshold)
        self.disable_hit_rate_threshold = float(disable_hit_rate_threshold)
        self.derisk_mean_return_threshold = float(derisk_mean_return_threshold)
        self.derisk_size_multiplier = min(1.0, max(0.0, float(derisk_size_multiplier)))
        self.disable_sticky = bool(disable_sticky)

        self._edge_histories: Dict[str, Deque[float]] = {}
        self._states: Dict[str, str] = {}

    @staticmethod
    def _now(ts: Optional[str] = None) -> str:
        if ts:
            return str(ts)
        return datetime.utcnow().isoformat()

    @staticmethod
    def _signal_to_direction(signal: str) -> int:
        signal_up = str(signal or "HOLD").upper()
        if signal_up == "BUY":
            return 1
        if signal_up == "SELL":
            return -1
        return 0

    def _history(self, strategy: str) -> Deque[float]:
        if strategy not in self._edge_histories:
            self._edge_histories[strategy] = deque(maxlen=self.window_size)
        return self._edge_histories[strategy]

    def record_outcomes(
        self,
        strategy_signals: Dict[str, str],
        realized_return: float,
        *,
        timestamp: Optional[str] = None,
    ) -> Dict[str, StrategyEdgeStatus]:
        """
        Record one realized return against prior strategy directional calls.

        Edge score is directional return:
        - BUY signal: +return is good, -return is bad
        - SELL signal: -return is good, +return is bad
        """
        ret = float(realized_return)
        for strategy, signal in (strategy_signals or {}).items():
            direction = self._signal_to_direction(signal)
            if direction == 0:
                continue
            score = direction * ret
            self._history(strategy).append(float(score))
        return self.get_all_statuses(timestamp=timestamp)

    def get_status(self, strategy: str, *, timestamp: Optional[str] = None) -> StrategyEdgeStatus:
        scores = list(self._edge_histories.get(strategy, []))
        sample_count = len(scores)
        ts = self._now(timestamp)

        if sample_count == 0:
            status = StrategyEdgeStatus(
                strategy=strategy,
                sample_count=0,
                win_rate=0.0,
                mean_edge_return=0.0,
                state="warmup",
                size_multiplier=1.0,
                disable_trading=False,
                reason="no_samples",
                timestamp=ts,
            )
            self._states[strategy] = status.state
            return status

        wins = sum(1 for score in scores if score > 0)
        win_rate = float(wins / sample_count)
        mean_edge_return = float(mean(scores))
        prev_state = self._states.get(strategy, "healthy")

        if sample_count < self.min_samples:
            state = "warmup"
            multiplier = 1.0
            disable = False
            reason = "insufficient_samples"
        else:
            if prev_state == "disabled" and self.disable_sticky:
                state = "disabled"
                multiplier = 0.0
                disable = True
                reason = "disabled_sticky"
            elif win_rate <= self.disable_hit_rate_threshold and mean_edge_return < 0.0:
                state = "disabled"
                multiplier = 0.0
                disable = True
                reason = "edge_decay_disable"
            elif (
                win_rate <= self.derisk_hit_rate_threshold
                and mean_edge_return <= self.derisk_mean_return_threshold
            ):
                state = "derisked"
                multiplier = self.derisk_size_multiplier
                disable = False
                reason = "edge_decay_derisk"
            else:
                state = "healthy"
                multiplier = 1.0
                disable = False
                reason = "healthy"

        status = StrategyEdgeStatus(
            strategy=strategy,
            sample_count=sample_count,
            win_rate=win_rate,
            mean_edge_return=mean_edge_return,
            state=state,
            size_multiplier=multiplier,
            disable_trading=disable,
            reason=reason,
            timestamp=ts,
        )
        self._states[strategy] = state
        return status

    def get_all_statuses(self, *, timestamp: Optional[str] = None) -> Dict[str, StrategyEdgeStatus]:
        return {
            strategy: self.get_status(strategy, timestamp=timestamp)
            for strategy in sorted(self._edge_histories.keys())
        }

    def clear_strategy(self, strategy: str) -> None:
        self._edge_histories.pop(strategy, None)
        self._states.pop(strategy, None)

    def reset(self) -> None:
        self._edge_histories.clear()
        self._states.clear()
