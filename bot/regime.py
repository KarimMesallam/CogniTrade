"""
Market regime detection utilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence


REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"
REGIME_SIDEWAYS = "SIDEWAYS"
REGIME_HIGH_VOL = "HIGH_VOLATILITY"
REGIME_UNKNOWN = "UNKNOWN"

SUPPORTED_REGIMES = {
    REGIME_BULL,
    REGIME_BEAR,
    REGIME_SIDEWAYS,
    REGIME_HIGH_VOL,
    REGIME_UNKNOWN,
}


@dataclass(frozen=True)
class RegimeState:
    """Current market regime classification."""

    regime: str
    confidence: float
    trend_pct: float
    realized_volatility_pct: float
    lookback_candles: int
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_record(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Convert to DB-ready record payload."""
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "regime": self.regime,
            "confidence": float(self.confidence),
            "trend_pct": float(self.trend_pct),
            "volatility_pct": float(self.realized_volatility_pct),
            "lookback_candles": int(self.lookback_candles),
            "timestamp": self.timestamp,
            "details": dict(self.details),
        }


class MarketRegimeDetector:
    """Deterministic market-regime classifier without lookahead."""

    def __init__(
        self,
        lookback_candles: int = 50,
        trend_threshold_pct: float = 0.02,
        sideways_threshold_pct: float = 0.01,
        high_volatility_threshold_pct: float = 0.015,
    ):
        self.lookback_candles = max(10, int(lookback_candles))
        self.trend_threshold_pct = max(0.0001, float(trend_threshold_pct))
        self.sideways_threshold_pct = max(0.0001, float(sideways_threshold_pct))
        self.high_volatility_threshold_pct = max(0.0001, float(high_volatility_threshold_pct))

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, value))

    @staticmethod
    def _parse_iso_timestamp(timestamp: Optional[str]) -> str:
        if timestamp:
            try:
                return datetime.fromisoformat(str(timestamp)).isoformat()
            except (TypeError, ValueError):
                pass
        return datetime.utcnow().isoformat()

    @staticmethod
    def _extract_closes(candles: Sequence[Any]) -> List[float]:
        closes: List[float] = []
        for candle in candles:
            if isinstance(candle, (list, tuple)) and len(candle) >= 5:
                try:
                    closes.append(float(candle[4]))
                except (TypeError, ValueError):
                    continue
            elif isinstance(candle, dict):
                close = candle.get("close")
                if close is None:
                    continue
                try:
                    closes.append(float(close))
                except (TypeError, ValueError):
                    continue
        return closes

    def _score_trend_confidence(self, trend_abs: float) -> float:
        ratio = trend_abs / self.trend_threshold_pct
        if ratio <= 0:
            return 0.0
        if ratio <= 1:
            return self._clamp(0.3 + (0.25 * ratio), 0.0, 0.6)
        return self._clamp(0.55 + (0.2 * (ratio - 1.0)), 0.55, 0.99)

    def _score_vol_confidence(self, volatility_pct: float) -> float:
        ratio = volatility_pct / self.high_volatility_threshold_pct
        if ratio <= 1:
            return self._clamp(0.3 + (0.2 * ratio), 0.0, 0.6)
        return self._clamp(0.55 + (0.22 * (ratio - 1.0)), 0.55, 0.99)

    def detect_from_closes(self, closes: Iterable[float], timestamp: Optional[str] = None) -> RegimeState:
        """Detect regime from a close-price series."""
        close_values = [float(value) for value in closes if value is not None]
        if len(close_values) < 3:
            return RegimeState(
                regime=REGIME_UNKNOWN,
                confidence=0.0,
                trend_pct=0.0,
                realized_volatility_pct=0.0,
                lookback_candles=len(close_values),
                timestamp=self._parse_iso_timestamp(timestamp),
                details={"reason": "insufficient_close_data"},
            )

        window = close_values[-self.lookback_candles :]
        if len(window) < 3:
            return RegimeState(
                regime=REGIME_UNKNOWN,
                confidence=0.0,
                trend_pct=0.0,
                realized_volatility_pct=0.0,
                lookback_candles=len(window),
                timestamp=self._parse_iso_timestamp(timestamp),
                details={"reason": "insufficient_lookback_window"},
            )

        first_price = window[0]
        if first_price <= 0:
            return RegimeState(
                regime=REGIME_UNKNOWN,
                confidence=0.0,
                trend_pct=0.0,
                realized_volatility_pct=0.0,
                lookback_candles=len(window),
                timestamp=self._parse_iso_timestamp(timestamp),
                details={"reason": "invalid_price_window"},
            )

        trend_pct = (window[-1] / first_price) - 1.0
        returns: List[float] = []
        for idx in range(1, len(window)):
            prev_price = window[idx - 1]
            curr_price = window[idx]
            if prev_price <= 0:
                continue
            returns.append((curr_price / prev_price) - 1.0)
        volatility_pct = float(pstdev(returns)) if len(returns) >= 2 else 0.0

        trend_abs = abs(trend_pct)
        details: Dict[str, Any] = {
            "trend_threshold_pct": self.trend_threshold_pct,
            "sideways_threshold_pct": self.sideways_threshold_pct,
            "high_volatility_threshold_pct": self.high_volatility_threshold_pct,
            "returns_sample_size": len(returns),
        }

        if volatility_pct >= self.high_volatility_threshold_pct:
            regime = REGIME_HIGH_VOL
            confidence = self._score_vol_confidence(volatility_pct)
        elif trend_pct >= self.trend_threshold_pct:
            regime = REGIME_BULL
            confidence = self._score_trend_confidence(trend_abs)
        elif trend_pct <= -self.trend_threshold_pct:
            regime = REGIME_BEAR
            confidence = self._score_trend_confidence(trend_abs)
        elif trend_abs <= self.sideways_threshold_pct:
            regime = REGIME_SIDEWAYS
            centered = 1.0 - (trend_abs / self.sideways_threshold_pct)
            confidence = self._clamp(0.55 + (0.4 * centered), 0.55, 0.99)
        else:
            regime = REGIME_SIDEWAYS
            confidence = 0.55

        return RegimeState(
            regime=regime,
            confidence=float(confidence),
            trend_pct=float(trend_pct),
            realized_volatility_pct=float(volatility_pct),
            lookback_candles=len(window),
            timestamp=self._parse_iso_timestamp(timestamp),
            details=details,
        )

    def detect_from_candles(self, candles: Sequence[Any], timestamp: Optional[str] = None) -> RegimeState:
        """Detect regime from Binance-style candle payloads."""
        closes = self._extract_closes(candles)
        return self.detect_from_closes(closes, timestamp=timestamp)
