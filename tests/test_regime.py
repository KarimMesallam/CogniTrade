import os
import sys

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.regime import (
    MarketRegimeDetector,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_HIGH_VOL,
    REGIME_SIDEWAYS,
    REGIME_UNKNOWN,
)


def _build_candles_from_closes(closes):
    candles = []
    for idx, close in enumerate(closes):
        open_time = 1700000000000 + (idx * 60000)
        candles.append(
            [
                open_time,
                str(close),
                str(close),
                str(close),
                str(close),
                "1.0",
                open_time + 60000,
                "1.0",
                1,
                "1.0",
                "1.0",
                "0",
            ]
        )
    return candles


def test_regime_detector_classifies_bull_market():
    detector = MarketRegimeDetector(
        lookback_candles=50,
        trend_threshold_pct=0.02,
        sideways_threshold_pct=0.01,
        high_volatility_threshold_pct=0.02,
    )
    closes = [100 + (0.2 * i) for i in range(60)]  # ~11.8% trend up over lookback
    state = detector.detect_from_candles(_build_candles_from_closes(closes))

    assert state.regime == REGIME_BULL
    assert state.confidence >= 0.55
    assert state.trend_pct > 0


def test_regime_detector_classifies_bear_market():
    detector = MarketRegimeDetector(
        lookback_candles=50,
        trend_threshold_pct=0.02,
        sideways_threshold_pct=0.01,
        high_volatility_threshold_pct=0.02,
    )
    closes = [120 - (0.3 * i) for i in range(60)]  # clear downtrend
    state = detector.detect_from_candles(_build_candles_from_closes(closes))

    assert state.regime == REGIME_BEAR
    assert state.confidence >= 0.55
    assert state.trend_pct < 0


def test_regime_detector_classifies_high_volatility_market():
    detector = MarketRegimeDetector(
        lookback_candles=40,
        trend_threshold_pct=0.05,
        sideways_threshold_pct=0.02,
        high_volatility_threshold_pct=0.01,
    )
    closes = []
    price = 100.0
    for idx in range(50):
        price *= 1.05 if idx % 2 == 0 else 0.95
        closes.append(price)
    state = detector.detect_from_candles(_build_candles_from_closes(closes))

    assert state.regime == REGIME_HIGH_VOL
    assert state.realized_volatility_pct >= 0.01


def test_regime_detector_classifies_sideways_market():
    detector = MarketRegimeDetector(
        lookback_candles=30,
        trend_threshold_pct=0.03,
        sideways_threshold_pct=0.01,
        high_volatility_threshold_pct=0.02,
    )
    closes = [100.0 + ((-1) ** i) * 0.1 for i in range(45)]
    state = detector.detect_from_candles(_build_candles_from_closes(closes))

    assert state.regime == REGIME_SIDEWAYS
    assert abs(state.trend_pct) <= 0.01


def test_regime_detector_returns_unknown_when_data_is_insufficient():
    detector = MarketRegimeDetector(lookback_candles=20)
    state = detector.detect_from_candles(_build_candles_from_closes([100.0, 101.0]))

    assert state.regime == REGIME_UNKNOWN
    assert state.confidence == 0.0


def test_regime_detector_is_deterministic_for_same_inputs():
    detector = MarketRegimeDetector(lookback_candles=40)
    candles = _build_candles_from_closes([100 + (i * 0.05) for i in range(50)])

    first = detector.detect_from_candles(candles, timestamp="2026-02-08T00:00:00")
    second = detector.detect_from_candles(candles, timestamp="2026-02-08T00:00:00")

    assert first.regime == second.regime
    assert first.confidence == second.confidence
    assert first.trend_pct == second.trend_pct
    assert first.realized_volatility_pct == second.realized_volatility_pct
