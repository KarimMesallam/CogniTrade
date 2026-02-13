#!/usr/bin/env python3
import sys
import os
import json
import asyncio
import logging
import math
import uuid
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add parent directory to path so we can import bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logger = logging.getLogger("trading_bot")

# Import bot modules
from bot import binance_api, strategy, order_manager, database, config, llm_manager
from bot.backtesting import run_backtest, generate_report, optimize_strategy
from bot.backtesting.validation import AdvancedValidationFramework, regime_sliced_evaluation
from bot.observability import get_observability_manager
from bot.deploy_policy import (
    RolloutMetrics,
    ShadowCanaryRolloutGate,
    StrategyQualityThresholds,
    evaluate_strategy_quality,
    thresholds_from_config,
)
from bot.regime import (
    MarketRegimeDetector,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_HIGH_VOL,
    REGIME_SIDEWAYS,
)
from api.security import APISecurityController

# Create a TradingBot class for the API
class TradingBot:
    def __init__(self, symbol='BTCUSDT', interval='1h', trade_amount=100):
        self.symbol = symbol
        self.interval = interval
        self.trade_amount = trade_amount
        self.strategies = []
        self.running = False
    
    def add_strategy(self, name, params):
        self.strategies.append({'name': name, 'params': params})
    
    async def start(self):
        self.running = True
        print(f"Started trading bot with {len(self.strategies)} strategies")
    
    async def stop(self):
        self.running = False
        print("Stopped trading bot")

app = FastAPI(title="Trading Bot API", description="API for the AI Trading Bot")

api_security_config = config.get_api_security_config()

# Add CORS middleware with explicit origin/method/header allowlists.
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_security_config.get("allowed_origins", []),
    allow_credentials=bool(api_security_config.get("allow_credentials", False)),
    allow_methods=api_security_config.get("allow_methods", ["GET", "POST", "OPTIONS"]),
    allow_headers=api_security_config.get("allow_headers", ["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"]),
)

observability_config = config.get_observability_config()
observability = get_observability_manager(
    max_events=observability_config.get("max_events", 4000),
    latency_alert_ms=observability_config.get("latency_alert_ms", 2500.0),
    error_rate_alert_threshold=observability_config.get("error_rate_alert_threshold", 0.25),
    error_rate_min_events=observability_config.get("error_rate_min_events", 20),
    persistence_enabled=observability_config.get("persistence_enabled", True),
    persistence_db_url=observability_config.get("persistence_db_url", "sqlite:///data/observability.db"),
)
api_security = APISecurityController(api_security_config)
rollout_config = config.get_rollout_config()
rollout_gate = ShadowCanaryRolloutGate(thresholds=thresholds_from_config(rollout_config))
rollout_state_store_path = str(rollout_config.get("state_store_path", "data/rollout_gate_state.json"))
if rollout_config.get("enabled", True):
    try:
        rollout_gate.load_from_file(rollout_state_store_path)
    except Exception as exc:
        logger.warning("Failed to load rollout-gate state from %s: %s", rollout_state_store_path, exc)


def _quality_thresholds_from_config() -> StrategyQualityThresholds:
    quality_cfg = config.get_quality_gate_config()
    return StrategyQualityThresholds(
        min_walk_forward_folds=int(quality_cfg.get("min_walk_forward_folds", 3)),
        min_sharpe_ratio=float(quality_cfg.get("min_sharpe_ratio", 0.20)),
        min_calmar_ratio=float(quality_cfg.get("min_calmar_ratio", 0.10)),
        max_drawdown_pct=float(quality_cfg.get("max_drawdown_pct", 25.0)),
        min_regime_samples=int(quality_cfg.get("min_regime_samples", 20)),
        min_regime_win_rate=float(quality_cfg.get("min_regime_win_rate", 0.45)),
        min_regimes_passing=int(quality_cfg.get("min_regimes_passing", 2)),
    )


def _persist_rollout_gate_state() -> None:
    try:
        rollout_gate.save_to_file(rollout_state_store_path)
    except Exception as exc:
        logger.error("Failed to persist rollout-gate state to %s: %s", rollout_state_store_path, exc)


def _auth_error_payload(reason: str) -> Dict[str, str]:
    mapping = {
        "missing_api_key": "Missing API key. Provide X-API-Key or Authorization: Bearer <token>.",
        "admin_role_required": "Admin API key required for write operation.",
        "read_role_required": "Read API key required for this endpoint.",
    }
    return {"detail": mapping.get(reason, "Unauthorized request.")}


@app.middleware("http")
async def observability_request_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    trace = observability.start_trace(
        component="api",
        operation=f"{request.method} {request.url.path}",
        request_id=request_id,
        metadata={"path": request.url.path, "method": request.method},
    )
    request.state.request_id = request_id

    try:
        auth_result = api_security.authorize(request)
        if not auth_result.allowed:
            observability.record_error(
                component="api",
                error_type="auth_rejected",
                message=auth_result.reason,
                severity="high",
                metadata={"path": request.url.path, "method": request.method, "request_id": request_id},
            )
            observability.end_trace(
                trace,
                status="error",
                error=f"auth_rejected:{auth_result.reason}",
                metadata={"status_code": auth_result.status_code},
            )
            response = JSONResponse(
                status_code=auth_result.status_code,
                content=_auth_error_payload(auth_result.reason),
            )
            response.headers["X-Request-ID"] = request_id
            return response

        request.state.api_role = auth_result.role
        rate_allowed, retry_after = api_security.check_rate_limit(request, auth_result.api_key)
        if not rate_allowed:
            observability.record_error(
                component="api",
                error_type="rate_limited",
                message="rate_limit_exceeded",
                severity="medium",
                metadata={
                    "path": request.url.path,
                    "method": request.method,
                    "request_id": request_id,
                    "retry_after_seconds": retry_after,
                },
            )
            observability.end_trace(
                trace,
                status="error",
                error="rate_limit_exceeded",
                metadata={"status_code": 429},
            )
            response = JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Retry later."},
            )
            response.headers["Retry-After"] = str(retry_after)
            response.headers["X-Request-ID"] = request_id
            return response

        response = await call_next(request)
        status = "ok" if response.status_code < 500 else "error"
        observability.end_trace(
            trace,
            status=status,
            metadata={"status_code": response.status_code},
        )
    except Exception as exc:
        observability.record_error(
            component="api",
            error_type="unhandled_exception",
            message=str(exc),
            severity="high",
            metadata={"path": request.url.path, "request_id": request_id},
        )
        observability.end_trace(
            trace,
            status="error",
            error=str(exc),
            metadata={"path": request.url.path},
        )
        raise

    response.headers["X-Request-ID"] = request_id
    return response

# Models for API requests and responses
class ExecutionSimulationConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    spread_bps: Optional[float] = None
    slippage_bps: Optional[float] = None
    latency_bps: Optional[float] = None
    max_volume_participation: Optional[float] = None
    min_partial_fill_ratio: Optional[float] = None
    funding_rate_per_8h: Optional[float] = None

    class Config:
        extra = "forbid"


class BacktestConfig(BaseModel):
    symbol: str
    timeframes: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    commission: float
    strategy_name: str
    strategy_params: Optional[Dict[str, Any]] = None
    trade_mode: Optional[str] = None
    allow_short_positions: Optional[bool] = None
    execution_simulation: Optional[ExecutionSimulationConfigRequest] = None
    run_walk_forward_validation: bool = False
    walk_forward_train_size: int = 240
    walk_forward_test_size: int = 120
    walk_forward_step_size: int = 120
    walk_forward_gap: int = 0
    include_regime_slices: bool = False
    regime_timeframe: Optional[str] = None
    regime_lookback_candles: int = 50

class StrategyConfig(BaseModel):
    name: str
    params: Dict[str, Any]
    active: bool

class TradingConfig(BaseModel):
    symbol: str
    interval: str
    trade_amount: float
    strategies: List[StrategyConfig]

class LLMDecisionRequest(BaseModel):
    symbol: str
    timeframe: str
    market_data: Dict[str, Any]
    context: str
    strategy_signals: Optional[Dict[str, Any]] = None


class RolloutMetricsRequest(BaseModel):
    rollout_id: str
    sample_count: int
    error_rate: float
    drawdown_pct: float = 0.0
    total_return_pct: float = 0.0
    latency_p95_ms: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class StrategyQualityRequest(BaseModel):
    validation_summary: Dict[str, Any]


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    if hasattr(model, "dict"):
        return model.dict(exclude_none=True)
    return dict(model)


def _sanitize_json_value(value: Any) -> Any:
    """Convert non-finite numeric values to JSON-safe values."""
    if isinstance(value, dict):
        return {key: _sanitize_json_value(sub_value) for key, sub_value in value.items()}

    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]

    if isinstance(value, tuple):
        return [_sanitize_json_value(item) for item in value]

    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        return float(value)

    if isinstance(value, np.generic):
        return _sanitize_json_value(value.item())

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    return value


def _records_from_query_result(query_result: Any) -> List[Dict[str, Any]]:
    """Normalize DB query output to a JSON-serializable list of records."""
    if query_result is None:
        return []
    if isinstance(query_result, list):
        return query_result
    if hasattr(query_result, "to_dict"):
        return query_result.to_dict(orient="records")
    raise TypeError(f"Unsupported query result type: {type(query_result)}")


def _normalize_llm_decision_payload(decision: Any) -> Dict[str, Any]:
    """Validate LLM decision payload shape for consistent API responses."""
    if not isinstance(decision, dict):
        raise ValueError("LLM decision must be a dictionary")

    required_keys = ("decision", "confidence", "reasoning")
    missing = [key for key in required_keys if key not in decision]
    if missing:
        raise ValueError(f"LLM decision missing required keys: {', '.join(missing)}")

    return {
        "decision": decision["decision"],
        "confidence": decision["confidence"],
        "reasoning": decision["reasoning"],
    }


def _build_sma_crossover_backtest_strategy(short_period: int, long_period: int, timeframe: str):
    """Build a backtesting-compatible SMA crossover strategy function."""
    def _strategy(data_dict, _symbol):
        frame = data_dict.get(timeframe)
        if frame is None or frame.empty or len(frame) < long_period + 1:
            return "HOLD"

        short_sma = frame["close"].rolling(window=short_period).mean()
        long_sma = frame["close"].rolling(window=long_period).mean()
        prev_short, curr_short = short_sma.iloc[-2], short_sma.iloc[-1]
        prev_long, curr_long = long_sma.iloc[-2], long_sma.iloc[-1]

        if any(value != value for value in [prev_short, curr_short, prev_long, curr_long]):
            return "HOLD"
        if prev_short <= prev_long and curr_short > curr_long:
            return "BUY"
        if prev_short >= prev_long and curr_short < curr_long:
            return "SELL"
        return "HOLD"

    _strategy.__name__ = f"sma_crossover_{short_period}_{long_period}"
    return _strategy


def _build_rsi_backtest_strategy(period: int, overbought: int, oversold: int, timeframe: str):
    """Build a backtesting-compatible RSI strategy function."""
    def _strategy(data_dict, _symbol):
        frame = data_dict.get(timeframe)
        if frame is None or frame.empty or len(frame) < period + 2:
            return "HOLD"

        delta = frame["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        relative_strength = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + relative_strength))

        previous_rsi = rsi.iloc[-2]
        current_rsi = rsi.iloc[-1]
        if previous_rsi != previous_rsi or current_rsi != current_rsi:
            return "HOLD"
        if previous_rsi < oversold and current_rsi > oversold:
            return "BUY"
        if previous_rsi > overbought and current_rsi < overbought:
            return "SELL"
        return "HOLD"

    _strategy.__name__ = f"rsi_{period}_{overbought}_{oversold}"
    return _strategy


def _build_ema_crossover_backtest_strategy(fast_period: int, slow_period: int, timeframe: str):
    """Build an EMA crossover strategy (trend-following; useful in bull phases)."""
    def _strategy(data_dict, _symbol):
        frame = data_dict.get(timeframe)
        if frame is None or frame.empty or len(frame) < slow_period + 2:
            return "HOLD"

        fast_ema = frame["close"].ewm(span=fast_period, adjust=False).mean()
        slow_ema = frame["close"].ewm(span=slow_period, adjust=False).mean()
        prev_fast, curr_fast = fast_ema.iloc[-2], fast_ema.iloc[-1]
        prev_slow, curr_slow = slow_ema.iloc[-2], slow_ema.iloc[-1]
        if any(value != value for value in [prev_fast, curr_fast, prev_slow, curr_slow]):
            return "HOLD"
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return "BUY"
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return "SELL"
        return "HOLD"

    _strategy.__name__ = f"ema_crossover_{fast_period}_{slow_period}"
    return _strategy


def _build_donchian_breakout_backtest_strategy(
    lookback_period: int,
    timeframe: str,
    breakout_buffer_bps: float = 0.0,
):
    """Build a Donchian breakout strategy (can auto-switch long/short on breakouts)."""
    def _strategy(data_dict, _symbol):
        frame = data_dict.get(timeframe)
        if frame is None or frame.empty or len(frame) < lookback_period + 2:
            return "HOLD"

        upper = frame["high"].rolling(window=lookback_period).max().shift(1)
        lower = frame["low"].rolling(window=lookback_period).min().shift(1)
        prev_close = frame["close"].iloc[-2]
        curr_close = frame["close"].iloc[-1]
        curr_upper = upper.iloc[-1]
        curr_lower = lower.iloc[-1]

        if any(value != value for value in [prev_close, curr_close, curr_upper, curr_lower]):
            return "HOLD"

        buffer_multiplier = 1.0 + (max(0.0, float(breakout_buffer_bps)) / 10000.0)
        upper_trigger = curr_upper * buffer_multiplier
        lower_trigger = curr_lower / buffer_multiplier if buffer_multiplier != 0 else curr_lower

        if prev_close <= upper_trigger and curr_close > upper_trigger:
            return "BUY"
        if prev_close >= lower_trigger and curr_close < lower_trigger:
            return "SELL"
        return "HOLD"

    _strategy.__name__ = f"donchian_breakout_{lookback_period}_{int(breakout_buffer_bps)}bps"
    return _strategy


def _build_bear_rally_short_backtest_strategy(
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    rsi_overbought: int,
    rsi_oversold: int,
    timeframe: str,
):
    """Build a bear-market short-bias strategy: short rallies in established downtrends."""
    def _strategy(data_dict, _symbol):
        frame = data_dict.get(timeframe)
        min_candles = max(slow_period, rsi_period) + 2
        if frame is None or frame.empty or len(frame) < min_candles:
            return "HOLD"

        fast_ema = frame["close"].ewm(span=fast_period, adjust=False).mean()
        slow_ema = frame["close"].ewm(span=slow_period, adjust=False).mean()
        delta = frame["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))

        prev_fast, curr_fast = fast_ema.iloc[-2], fast_ema.iloc[-1]
        prev_slow, curr_slow = slow_ema.iloc[-2], slow_ema.iloc[-1]
        prev_rsi, curr_rsi = rsi.iloc[-2], rsi.iloc[-1]

        if any(value != value for value in [prev_fast, curr_fast, prev_slow, curr_slow, prev_rsi, curr_rsi]):
            return "HOLD"

        trend_is_down = curr_fast < curr_slow
        trend_reversal_up = prev_fast <= prev_slow and curr_fast > curr_slow

        if trend_is_down and prev_rsi > rsi_overbought and curr_rsi < rsi_overbought:
            return "SELL"
        if trend_reversal_up:
            return "BUY"
        if prev_rsi < rsi_oversold and curr_rsi > rsi_oversold:
            return "BUY"
        return "HOLD"

    _strategy.__name__ = (
        f"bear_rally_short_{fast_period}_{slow_period}_{rsi_period}_{rsi_overbought}_{rsi_oversold}"
    )
    return _strategy


def _build_regime_switch_adaptive_backtest_strategy(
    timeframe: str,
    regime_lookback_candles: int,
    trend_threshold_pct: float,
    sideways_threshold_pct: float,
    high_volatility_threshold_pct: float,
    min_regime_confidence: float,
    momentum_confirmation_candles: int,
    bull_short_period: int,
    bull_long_period: int,
    bear_fast_period: int,
    bear_slow_period: int,
    bear_rsi_period: int,
    bear_rsi_overbought: int,
    bear_rsi_oversold: int,
    sideways_rsi_period: int,
    sideways_rsi_overbought: int,
    sideways_rsi_oversold: int,
    high_vol_lookback_period: int,
    high_vol_breakout_buffer_bps: float,
):
    """
    Regime-adaptive meta strategy:
    - BULL: SMA trend-following
    - BEAR: bear-rally short strategy
    - HIGH_VOLATILITY: Donchian breakout
    - SIDEWAYS (default fallback): RSI mean-reversion
    """
    bull_strategy = _build_sma_crossover_backtest_strategy(
        short_period=bull_short_period,
        long_period=bull_long_period,
        timeframe=timeframe,
    )
    bear_strategy = _build_bear_rally_short_backtest_strategy(
        fast_period=bear_fast_period,
        slow_period=bear_slow_period,
        rsi_period=bear_rsi_period,
        rsi_overbought=bear_rsi_overbought,
        rsi_oversold=bear_rsi_oversold,
        timeframe=timeframe,
    )
    sideways_strategy = _build_rsi_backtest_strategy(
        period=sideways_rsi_period,
        overbought=sideways_rsi_overbought,
        oversold=sideways_rsi_oversold,
        timeframe=timeframe,
    )
    high_vol_strategy = _build_donchian_breakout_backtest_strategy(
        lookback_period=high_vol_lookback_period,
        timeframe=timeframe,
        breakout_buffer_bps=high_vol_breakout_buffer_bps,
    )
    detector = MarketRegimeDetector(
        lookback_candles=regime_lookback_candles,
        trend_threshold_pct=trend_threshold_pct,
        sideways_threshold_pct=sideways_threshold_pct,
        high_volatility_threshold_pct=high_volatility_threshold_pct,
    )

    def _strategy(data_dict, symbol):
        frame = data_dict.get(timeframe)
        if frame is None or frame.empty or "close" not in frame.columns:
            return "HOLD"

        closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
        min_required = max(3, regime_lookback_candles, momentum_confirmation_candles + 1)
        if len(closes) < min_required:
            return "HOLD"

        timestamp = str(frame["timestamp"].iloc[-1]) if "timestamp" in frame.columns else None
        regime_state = detector.detect_from_closes(closes.tolist(), timestamp=timestamp)
        regime = regime_state.regime
        if float(regime_state.confidence) < float(min_regime_confidence):
            return "HOLD"

        momentum_base = float(closes.iloc[-(momentum_confirmation_candles + 1)])
        if momentum_base == 0:
            return "HOLD"
        momentum_return = float((closes.iloc[-1] / momentum_base) - 1.0)

        if regime == REGIME_BULL:
            if momentum_return <= 0:
                return "HOLD"
            return bull_strategy(data_dict, symbol)
        if regime == REGIME_BEAR:
            if momentum_return >= 0:
                return "HOLD"
            return bear_strategy(data_dict, symbol)
        if regime == REGIME_HIGH_VOL:
            if abs(momentum_return) < float(trend_threshold_pct):
                return "HOLD"
            return high_vol_strategy(data_dict, symbol)
        if regime == REGIME_SIDEWAYS:
            if abs(momentum_return) > float(trend_threshold_pct):
                return "HOLD"
            return sideways_strategy(data_dict, symbol)
        return "HOLD"

    _strategy.__name__ = (
        "regime_switch_adaptive_"
        f"{regime_lookback_candles}_{int(min_regime_confidence * 100)}c_"
        f"{bull_short_period}_{bull_long_period}_"
        f"{bear_fast_period}_{bear_slow_period}_"
        f"{high_vol_lookback_period}_{int(high_vol_breakout_buffer_bps)}bps"
    )
    return _strategy


def _equity_frame_from_result(result) -> pd.DataFrame:
    if not result or not getattr(result, "equity_curve", None):
        return pd.DataFrame()
    frame = pd.DataFrame(
        [
            {
                "timestamp": point.timestamp,
                "equity": float(point.equity),
            }
            for point in result.equity_curve
        ]
    )
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["period_return"] = frame["equity"].pct_change().fillna(0.0)
    return frame


def _evaluate_return_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    test_returns = test_df.get("period_return", pd.Series(dtype=float)).astype(float)
    train_returns = train_df.get("period_return", pd.Series(dtype=float)).astype(float)

    test_mean = float(test_returns.mean()) if len(test_returns) else 0.0
    test_std = float(test_returns.std(ddof=0)) if len(test_returns) > 1 else 0.0

    periods_per_year = 1.0
    if len(test_df) > 1 and "timestamp" in test_df.columns:
        ts = pd.to_datetime(test_df["timestamp"], errors="coerce")
        deltas = ts.diff().dt.total_seconds().dropna()
        if not deltas.empty:
            median_seconds = float(deltas.median())
            if median_seconds > 0:
                periods_per_year = (365.0 * 24.0 * 3600.0) / median_seconds

    sharpe = float((test_mean / test_std) * math.sqrt(periods_per_year)) if test_std > 0 else 0.0

    test_equity = test_df.get("equity", pd.Series(dtype=float)).astype(float)
    if len(test_equity):
        running_peak = test_equity.cummax()
        drawdown_series = ((test_equity - running_peak) / running_peak.replace(0, np.nan)).fillna(0.0) * 100.0
        max_drawdown_pct = float(abs(drawdown_series.min()))
        total_return = float((test_equity.iloc[-1] / test_equity.iloc[0]) - 1.0) if test_equity.iloc[0] != 0 else 0.0
    else:
        max_drawdown_pct = 0.0
        total_return = 0.0

    calmar = float(total_return / (max_drawdown_pct / 100.0)) if max_drawdown_pct > 0 else 0.0
    return {
        "train_mean_return": float(train_returns.mean()) if len(train_returns) else 0.0,
        "test_mean_return": test_mean,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "max_drawdown_pct": max_drawdown_pct,
    }

# Trading bot instance
trading_bot = None
trading_task = None

# Database connection
db = database.Database()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/observability/dashboard")
async def get_observability_dashboard(
    window_minutes: int = Query(default=60, ge=1, le=1440)
):
    dashboard = observability.get_dashboard_snapshot(window_minutes=window_minutes)
    return {"status": "success", "dashboard": dashboard}


@app.get("/observability/events")
async def get_observability_events(
    event_type: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    events = observability.get_recent_events(event_type=event_type, limit=limit)
    return {"status": "success", "events": events}


@app.get("/observability/alerts")
async def get_observability_alerts(limit: int = Query(default=50, ge=1, le=500)):
    alerts = observability.get_recent_alerts(limit=limit)
    return {"status": "success", "alerts": alerts}


@app.post("/rollout/quality/evaluate")
async def evaluate_rollout_quality(request: StrategyQualityRequest):
    thresholds = _quality_thresholds_from_config()
    decision = evaluate_strategy_quality(request.validation_summary, thresholds=thresholds)
    return {
        "status": "success",
        "quality_gate": _sanitize_json_value({
            "passed": decision.passed,
            "reasons": decision.reasons,
            "metrics": decision.metrics,
        }),
    }


@app.post("/rollout/evaluate/shadow")
async def evaluate_shadow_rollout(payload: RolloutMetricsRequest):
    decision = rollout_gate.evaluate_shadow(
        payload.rollout_id,
        RolloutMetrics(
            sample_count=payload.sample_count,
            error_rate=payload.error_rate,
            drawdown_pct=payload.drawdown_pct,
            total_return_pct=payload.total_return_pct,
            latency_p95_ms=payload.latency_p95_ms,
            metadata=dict(payload.metadata or {}),
        ),
    )
    _persist_rollout_gate_state()
    return {"status": "success", "decision": _sanitize_json_value(decision.__dict__)}


@app.post("/rollout/evaluate/canary")
async def evaluate_canary_rollout(payload: RolloutMetricsRequest):
    decision = rollout_gate.evaluate_canary(
        payload.rollout_id,
        RolloutMetrics(
            sample_count=payload.sample_count,
            error_rate=payload.error_rate,
            drawdown_pct=payload.drawdown_pct,
            total_return_pct=payload.total_return_pct,
            latency_p95_ms=payload.latency_p95_ms,
            metadata=dict(payload.metadata or {}),
        ),
    )
    _persist_rollout_gate_state()
    return {"status": "success", "decision": _sanitize_json_value(decision.__dict__)}


@app.post("/rollout/evaluate/production")
async def evaluate_production_rollout(payload: RolloutMetricsRequest):
    metadata = dict(payload.metadata or {})
    if "quality_gate" not in metadata and isinstance(metadata.get("validation_summary"), dict):
        quality_decision = evaluate_strategy_quality(
            metadata["validation_summary"],
            thresholds=_quality_thresholds_from_config(),
        )
        metadata["quality_gate"] = {
            "passed": quality_decision.passed,
            "reasons": quality_decision.reasons,
            "metrics": quality_decision.metrics,
        }

    decision = rollout_gate.evaluate_production(
        payload.rollout_id,
        RolloutMetrics(
            sample_count=payload.sample_count,
            error_rate=payload.error_rate,
            drawdown_pct=payload.drawdown_pct,
            total_return_pct=payload.total_return_pct,
            latency_p95_ms=payload.latency_p95_ms,
            metadata=metadata,
        ),
    )
    _persist_rollout_gate_state()
    return {"status": "success", "decision": _sanitize_json_value(decision.__dict__)}


@app.get("/rollout/status/{rollout_id}")
async def get_rollout_status(rollout_id: str):
    status = rollout_gate.get_rollout_status(rollout_id)
    return {"status": "success", "rollout": _sanitize_json_value(status)}

# Start trading task
@app.post("/trading/start")
async def start_trading(config: TradingConfig, background_tasks: BackgroundTasks):
    global trading_bot, trading_task
    
    if trading_bot is not None:
        raise HTTPException(status_code=400, detail="Trading bot is already running")
    
    # Initialize trading bot with the provided configuration
    trading_bot = TradingBot(
        symbol=config.symbol,
        interval=config.interval,
        trade_amount=config.trade_amount
    )
    
    # Set active strategies
    for strat_config in config.strategies:
        if strat_config.active:
            trading_bot.add_strategy(strat_config.name, strat_config.params)
    
    # Start trading in a background task
    background_tasks.add_task(run_trading_bot, trading_bot)
    
    return {"status": "success", "message": "Trading bot started"}

async def run_trading_bot(bot):
    try:
        # Start the bot
        await bot.start()
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")

# Stop trading
@app.post("/trading/stop")
async def stop_trading():
    global trading_bot
    
    if trading_bot is None:
        raise HTTPException(status_code=400, detail="Trading bot is not running")
    
    await trading_bot.stop()
    trading_bot = None
    
    return {"status": "success", "message": "Trading bot stopped"}

# Get account information
@app.get("/account/info")
async def get_account_info():
    try:
        client = binance_api.get_client()
        account_info = client.get_account()
        
        # Filter and format the response
        balances = [
            {
                "asset": balance["asset"],
                "free": float(balance["free"]),
                "locked": float(balance["locked"])
            }
            for balance in account_info["balances"]
            if float(balance["free"]) > 0 or float(balance["locked"]) > 0
        ]
        
        return {
            "status": "success",
            "account_type": "spot",
            "balances": balances
        }
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch account info from exchange")

# Get available trading pairs
@app.get("/market/symbols")
async def get_symbols():
    try:
        client = binance_api.get_client()
        exchange_info = client.get_exchange_info()
        
        symbols = []
        for symbol_info in exchange_info["symbols"]:
            if symbol_info["status"] == "TRADING":
                symbols.append({
                    "symbol": symbol_info["symbol"],
                    "baseAsset": symbol_info["baseAsset"],
                    "quoteAsset": symbol_info["quoteAsset"]
                })
        
        return {"status": "success", "symbols": symbols}
    except Exception as e:
        logger.error(f"Error getting exchange symbols: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch symbols from exchange")

# Get market data for a specific symbol
@app.get("/market/data/{symbol}/{interval}")
async def get_market_data(symbol: str, interval: str, limit: int = 100):
    try:
        client = binance_api.get_client()
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        # Format the response
        candles = []
        for k in klines:
            candles.append({
                "time": k[0] / 1000,  # Convert to seconds for charting libraries
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            })
        
        return {"status": "success", "candles": candles}
    except Exception as e:
        logger.error(f"Error getting market data for {symbol} ({interval}): {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch market data from exchange")

# Get available strategies
@app.get("/strategies")
async def get_strategies():
    # Return list of available strategies
    strategies = [
        {
            "name": "sma_crossover",
            "display_name": "SMA Crossover",
            "description": "Simple Moving Average Crossover Strategy",
            "parameters": {
                "short_period": {
                    "type": "integer",
                    "min": 5,
                    "max": 50,
                    "default": 10,
                    "description": "Short period for SMA calculation"
                },
                "long_period": {
                    "type": "integer",
                    "min": 20,
                    "max": 200,
                    "default": 50,
                    "description": "Long period for SMA calculation"
                }
            }
        },
        {
            "name": "rsi",
            "display_name": "RSI Strategy",
            "description": "Relative Strength Index Strategy",
            "parameters": {
                "period": {
                    "type": "integer",
                    "min": 7,
                    "max": 30,
                    "default": 14,
                    "description": "Period for RSI calculation"
                },
                "overbought": {
                    "type": "integer",
                    "min": 60,
                    "max": 90,
                    "default": 70,
                    "description": "Overbought threshold"
                },
                "oversold": {
                    "type": "integer",
                    "min": 10,
                    "max": 40,
                    "default": 30,
                    "description": "Oversold threshold"
                }
            }
        },
        {
            "name": "ema_crossover",
            "display_name": "EMA Crossover (Bull Trend)",
            "description": "Trend-following EMA crossover suited for persistent bull trends.",
            "parameters": {
                "fast_period": {
                    "type": "integer",
                    "min": 3,
                    "max": 50,
                    "default": 12,
                    "description": "Fast EMA period"
                },
                "slow_period": {
                    "type": "integer",
                    "min": 10,
                    "max": 200,
                    "default": 50,
                    "description": "Slow EMA period"
                }
            }
        },
        {
            "name": "donchian_breakout",
            "display_name": "Donchian Breakout (Long/Short)",
            "description": "Classic breakout trend strategy that can flip between long and short regimes.",
            "parameters": {
                "lookback_period": {
                    "type": "integer",
                    "min": 10,
                    "max": 200,
                    "default": 20,
                    "description": "Channel breakout lookback period"
                },
                "breakout_buffer_bps": {
                    "type": "number",
                    "min": 0,
                    "max": 100,
                    "default": 0,
                    "description": "Optional breakout confirmation buffer in basis points"
                }
            }
        },
        {
            "name": "bear_rally_short",
            "display_name": "Bear Rally Short",
            "description": "Short-bias bear-market strategy: short overbought rallies during downtrends.",
            "parameters": {
                "fast_period": {
                    "type": "integer",
                    "min": 3,
                    "max": 50,
                    "default": 8,
                    "description": "Fast EMA period"
                },
                "slow_period": {
                    "type": "integer",
                    "min": 10,
                    "max": 200,
                    "default": 30,
                    "description": "Slow EMA period"
                },
                "rsi_period": {
                    "type": "integer",
                    "min": 5,
                    "max": 50,
                    "default": 14,
                    "description": "RSI period"
                },
                "rsi_overbought": {
                    "type": "integer",
                    "min": 55,
                    "max": 95,
                    "default": 65,
                    "description": "Overbought RSI threshold used for short entries"
                },
                "rsi_oversold": {
                    "type": "integer",
                    "min": 5,
                    "max": 45,
                    "default": 35,
                    "description": "Oversold RSI threshold used for short exits"
                }
            }
        },
        {
            "name": "regime_switch_adaptive",
            "display_name": "Regime Switch Adaptive",
            "description": "Adaptive strategy that automatically switches behavior by detected market regime.",
            "parameters": {
                "regime_lookback_candles": {
                    "type": "integer",
                    "min": 20,
                    "max": 400,
                    "default": 50,
                    "description": "Candles used for regime classification"
                },
                "trend_threshold_pct": {
                    "type": "number",
                    "min": 0.001,
                    "max": 0.2,
                    "default": 0.02,
                    "description": "Minimum absolute trend threshold for bull/bear regime"
                },
                "sideways_threshold_pct": {
                    "type": "number",
                    "min": 0.001,
                    "max": 0.1,
                    "default": 0.01,
                    "description": "Maximum absolute trend threshold for sideways regime"
                },
                "high_volatility_threshold_pct": {
                    "type": "number",
                    "min": 0.001,
                    "max": 0.2,
                    "default": 0.015,
                    "description": "Realized volatility threshold for high-volatility regime"
                },
                "min_regime_confidence": {
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.55,
                    "description": "Minimum detector confidence required before any trade signal"
                },
                "momentum_confirmation_candles": {
                    "type": "integer",
                    "min": 1,
                    "max": 20,
                    "default": 4,
                    "description": "Recent candles used for direction confirmation before acting"
                },
                "bull_short_period": {
                    "type": "integer",
                    "min": 2,
                    "max": 50,
                    "default": 8,
                    "description": "Short SMA period used in bull regime"
                },
                "bull_long_period": {
                    "type": "integer",
                    "min": 5,
                    "max": 200,
                    "default": 50,
                    "description": "Long SMA period used in bull regime"
                },
                "bear_fast_period": {
                    "type": "integer",
                    "min": 2,
                    "max": 50,
                    "default": 8,
                    "description": "Fast EMA period used in bear regime"
                },
                "bear_slow_period": {
                    "type": "integer",
                    "min": 5,
                    "max": 200,
                    "default": 30,
                    "description": "Slow EMA period used in bear regime"
                },
                "high_vol_lookback_period": {
                    "type": "integer",
                    "min": 5,
                    "max": 200,
                    "default": 20,
                    "description": "Donchian lookback used in high-volatility regime"
                },
                "high_vol_breakout_buffer_bps": {
                    "type": "number",
                    "min": 0,
                    "max": 100,
                    "default": 5,
                    "description": "Donchian breakout confirmation buffer in basis points"
                }
            }
        },
        {
            "name": "llm_strategy",
            "display_name": "LLM-Enhanced Strategy",
            "description": "Strategy that uses LLM to make trading decisions",
            "parameters": {
                "base_strategy": {
                    "type": "string",
                    "options": [
                        "sma_crossover",
                        "rsi",
                        "ema_crossover",
                        "donchian_breakout",
                        "bear_rally_short",
                        "regime_switch_adaptive"
                    ],
                    "default": "sma_crossover",
                    "description": "Base strategy for LLM to enhance"
                },
                "llm_model": {
                    "type": "string",
                    "options": ["rule_based", "deepseek", "gpt4", "claude"],
                    "default": "rule_based",
                    "description": "LLM model to use"
                }
            }
        }
    ]
    
    return {"status": "success", "strategies": strategies}

# Run backtest
@app.post("/backtest/run")
async def run_backtest_endpoint(config: BacktestConfig):
    try:
        execution_simulation = (
            _model_to_dict(config.execution_simulation)
            if config.execution_simulation is not None
            else None
        )

        # Get the requested strategy function
        strat_func = None
        primary_timeframe = config.timeframes[0]
        if config.strategy_name == "sma_crossover":
            short_period = int(config.strategy_params.get("short_period", 10)) if config.strategy_params else 10
            long_period = int(config.strategy_params.get("long_period", 50)) if config.strategy_params else 50
            if short_period <= 0 or long_period <= 0 or short_period >= long_period:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid SMA parameters: require 0 < short_period < long_period"
                )
            strat_func = _build_sma_crossover_backtest_strategy(short_period, long_period, primary_timeframe)
        elif config.strategy_name == "rsi":
            period = int(config.strategy_params.get("period", 14)) if config.strategy_params else 14
            overbought = int(config.strategy_params.get("overbought", 70)) if config.strategy_params else 70
            oversold = int(config.strategy_params.get("oversold", 30)) if config.strategy_params else 30
            if period <= 1 or not (0 < oversold < overbought < 100):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid RSI parameters: require period > 1 and 0 < oversold < overbought < 100"
                )
            strat_func = _build_rsi_backtest_strategy(period, overbought, oversold, primary_timeframe)
        elif config.strategy_name == "ema_crossover":
            fast_period = int(config.strategy_params.get("fast_period", 12)) if config.strategy_params else 12
            slow_period = int(config.strategy_params.get("slow_period", 50)) if config.strategy_params else 50
            if fast_period <= 0 or slow_period <= 0 or fast_period >= slow_period:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid EMA parameters: require 0 < fast_period < slow_period",
                )
            strat_func = _build_ema_crossover_backtest_strategy(
                fast_period,
                slow_period,
                primary_timeframe,
            )
        elif config.strategy_name == "donchian_breakout":
            lookback_period = (
                int(config.strategy_params.get("lookback_period", 20))
                if config.strategy_params
                else 20
            )
            breakout_buffer_bps = (
                float(config.strategy_params.get("breakout_buffer_bps", 0.0))
                if config.strategy_params
                else 0.0
            )
            if lookback_period < 2 or breakout_buffer_bps < 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid Donchian parameters: require lookback_period >= 2 and breakout_buffer_bps >= 0",
                )
            strat_func = _build_donchian_breakout_backtest_strategy(
                lookback_period,
                primary_timeframe,
                breakout_buffer_bps,
            )
        elif config.strategy_name == "bear_rally_short":
            fast_period = int(config.strategy_params.get("fast_period", 8)) if config.strategy_params else 8
            slow_period = int(config.strategy_params.get("slow_period", 30)) if config.strategy_params else 30
            rsi_period = int(config.strategy_params.get("rsi_period", 14)) if config.strategy_params else 14
            rsi_overbought = (
                int(config.strategy_params.get("rsi_overbought", 65))
                if config.strategy_params
                else 65
            )
            rsi_oversold = (
                int(config.strategy_params.get("rsi_oversold", 35))
                if config.strategy_params
                else 35
            )
            if (
                fast_period <= 0
                or slow_period <= 0
                or fast_period >= slow_period
                or rsi_period <= 1
                or not (0 < rsi_oversold < rsi_overbought < 100)
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid bear strategy parameters: require 0 < fast_period < slow_period, "
                        "rsi_period > 1 and 0 < rsi_oversold < rsi_overbought < 100"
                    ),
                )
            strat_func = _build_bear_rally_short_backtest_strategy(
                fast_period,
                slow_period,
                rsi_period,
                rsi_overbought,
                rsi_oversold,
                primary_timeframe,
            )
        elif config.strategy_name == "regime_switch_adaptive":
            regime_lookback_candles = (
                int(config.strategy_params.get("regime_lookback_candles", 50)) if config.strategy_params else 50
            )
            trend_threshold_pct = (
                float(config.strategy_params.get("trend_threshold_pct", 0.02)) if config.strategy_params else 0.02
            )
            sideways_threshold_pct = (
                float(config.strategy_params.get("sideways_threshold_pct", 0.01)) if config.strategy_params else 0.01
            )
            high_volatility_threshold_pct = (
                float(config.strategy_params.get("high_volatility_threshold_pct", 0.015))
                if config.strategy_params
                else 0.015
            )
            min_regime_confidence = (
                float(config.strategy_params.get("min_regime_confidence", 0.55))
                if config.strategy_params
                else 0.55
            )
            momentum_confirmation_candles = (
                int(config.strategy_params.get("momentum_confirmation_candles", 4))
                if config.strategy_params
                else 4
            )
            bull_short_period = (
                int(config.strategy_params.get("bull_short_period", 8)) if config.strategy_params else 8
            )
            bull_long_period = (
                int(config.strategy_params.get("bull_long_period", 50)) if config.strategy_params else 50
            )
            bear_fast_period = (
                int(config.strategy_params.get("bear_fast_period", 8)) if config.strategy_params else 8
            )
            bear_slow_period = (
                int(config.strategy_params.get("bear_slow_period", 30)) if config.strategy_params else 30
            )
            bear_rsi_period = (
                int(config.strategy_params.get("bear_rsi_period", 14)) if config.strategy_params else 14
            )
            bear_rsi_overbought = (
                int(config.strategy_params.get("bear_rsi_overbought", 60)) if config.strategy_params else 60
            )
            bear_rsi_oversold = (
                int(config.strategy_params.get("bear_rsi_oversold", 30)) if config.strategy_params else 30
            )
            sideways_rsi_period = (
                int(config.strategy_params.get("sideways_rsi_period", 21)) if config.strategy_params else 21
            )
            sideways_rsi_overbought = (
                int(config.strategy_params.get("sideways_rsi_overbought", 65)) if config.strategy_params else 65
            )
            sideways_rsi_oversold = (
                int(config.strategy_params.get("sideways_rsi_oversold", 35)) if config.strategy_params else 35
            )
            high_vol_lookback_period = (
                int(config.strategy_params.get("high_vol_lookback_period", 20)) if config.strategy_params else 20
            )
            high_vol_breakout_buffer_bps = (
                float(config.strategy_params.get("high_vol_breakout_buffer_bps", 5.0))
                if config.strategy_params
                else 5.0
            )

            if (
                regime_lookback_candles < 10
                or not (0 < sideways_threshold_pct < trend_threshold_pct < 1)
                or not (0 < high_volatility_threshold_pct < 1)
                or not (0 <= min_regime_confidence <= 1)
                or momentum_confirmation_candles < 1
                or bull_short_period <= 0
                or bull_long_period <= 0
                or bull_short_period >= bull_long_period
                or bear_fast_period <= 0
                or bear_slow_period <= 0
                or bear_fast_period >= bear_slow_period
                or bear_rsi_period <= 1
                or not (0 < bear_rsi_oversold < bear_rsi_overbought < 100)
                or sideways_rsi_period <= 1
                or not (0 < sideways_rsi_oversold < sideways_rsi_overbought < 100)
                or high_vol_lookback_period < 2
                or high_vol_breakout_buffer_bps < 0
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid regime-switch parameters: require lookback >= 10; "
                        "0 < sideways_threshold_pct < trend_threshold_pct < 1; "
                        "0 < high_volatility_threshold_pct < 1; "
                        "valid SMA/EMA and RSI ranges; and non-negative breakout buffer"
                    ),
                )

            strat_func = _build_regime_switch_adaptive_backtest_strategy(
                timeframe=primary_timeframe,
                regime_lookback_candles=regime_lookback_candles,
                trend_threshold_pct=trend_threshold_pct,
                sideways_threshold_pct=sideways_threshold_pct,
                high_volatility_threshold_pct=high_volatility_threshold_pct,
                min_regime_confidence=min_regime_confidence,
                momentum_confirmation_candles=momentum_confirmation_candles,
                bull_short_period=bull_short_period,
                bull_long_period=bull_long_period,
                bear_fast_period=bear_fast_period,
                bear_slow_period=bear_slow_period,
                bear_rsi_period=bear_rsi_period,
                bear_rsi_overbought=bear_rsi_overbought,
                bear_rsi_oversold=bear_rsi_oversold,
                sideways_rsi_period=sideways_rsi_period,
                sideways_rsi_overbought=sideways_rsi_overbought,
                sideways_rsi_oversold=sideways_rsi_oversold,
                high_vol_lookback_period=high_vol_lookback_period,
                high_vol_breakout_buffer_bps=high_vol_breakout_buffer_bps,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {config.strategy_name}")
        
        resolved_allow_short_positions = config.allow_short_positions
        if resolved_allow_short_positions is None and str(config.trade_mode or "").upper() == "FUTURES":
            resolved_allow_short_positions = True

        # Run the backtest using the new API
        result = run_backtest(
            symbol=config.symbol,
            timeframes=config.timeframes,
            start_date=config.start_date,
            end_date=config.end_date,
            strategy_func=strat_func,
            initial_capital=config.initial_capital,
            commission_rate=config.commission,
            allow_short_positions=resolved_allow_short_positions,
            execution_simulation=execution_simulation,
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Backtest failed to produce results")
        
        # Format the results for the response
        formatted_results = {
            "symbol": config.symbol,
            "timeframes": config.timeframes,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "strategy": config.strategy_name,
            "trade_mode": str(config.trade_mode or "SPOT").upper(),
            "allow_short_positions": bool(resolved_allow_short_positions),
            "initial_capital": config.initial_capital,
            "final_value": result.final_equity,
            "profit_loss": result.final_equity - config.initial_capital,
            "profit_loss_percent": result.metrics.total_return_pct,
            "sharpe_ratio": result.metrics.sharpe_ratio,
            "max_drawdown": result.metrics.max_drawdown_pct,
            "trades": result.total_trades,
            "win_rate": result.metrics.win_rate,
            "avg_profit": result.metrics.avg_win if hasattr(result.metrics, 'avg_win') else 0,
            "avg_loss": result.metrics.avg_loss if hasattr(result.metrics, 'avg_loss') else 0,
            "profit_factor": result.metrics.profit_factor if hasattr(result.metrics, 'profit_factor') else 0
        }

        validation_summary: Dict[str, Any] = {}
        equity_frame = _equity_frame_from_result(result)

        if config.run_walk_forward_validation:
            if len(equity_frame) < max(10, config.walk_forward_train_size + config.walk_forward_test_size):
                validation_summary["walk_forward_error"] = "insufficient_equity_points_for_walk_forward"
                validation_summary["walk_forward_folds"] = 0
                validation_summary["walk_forward_summary"] = {}
            else:
                framework = AdvancedValidationFramework(timestamp_col="timestamp")
                folds = framework.walk_forward_validate(
                    frame=equity_frame,
                    evaluator=_evaluate_return_metrics,
                    train_size=config.walk_forward_train_size,
                    test_size=config.walk_forward_test_size,
                    step_size=config.walk_forward_step_size,
                    gap=config.walk_forward_gap,
                    expanding=True,
                )
                validation_summary["walk_forward_folds"] = len(folds)
                validation_summary["walk_forward_summary"] = framework.summarize_metrics(folds)

        if config.include_regime_slices:
            regime_timeframe = config.regime_timeframe or primary_timeframe
            market_df = db.get_market_data(
                symbol=config.symbol,
                timeframe=regime_timeframe,
                start_time=config.start_date,
                end_time=config.end_date,
                limit=200000,
            )
            if market_df is None or market_df.empty or len(equity_frame) == 0:
                validation_summary["regime_slices_error"] = "missing_market_or_equity_data"
                validation_summary["regime_slices"] = {}
            else:
                market_df = market_df.copy()
                market_df["timestamp"] = pd.to_datetime(market_df["timestamp"])
                market_df = market_df.sort_values("timestamp").reset_index(drop=True)
                closes = market_df["close"].astype(float).tolist()
                detector = MarketRegimeDetector(
                    lookback_candles=int(config.regime_lookback_candles or 50)
                )
                regimes = []
                for idx, _close in enumerate(closes):
                    state = detector.detect_from_closes(closes[: idx + 1], timestamp=str(market_df["timestamp"].iloc[idx]))
                    regimes.append(state.regime)
                regime_frame = pd.DataFrame(
                    {"timestamp": market_df["timestamp"], "regime": regimes}
                ).sort_values("timestamp")
                merged = pd.merge_asof(
                    equity_frame.sort_values("timestamp"),
                    regime_frame,
                    on="timestamp",
                    direction="backward",
                )
                merged["regime"] = merged["regime"].fillna("UNKNOWN")

                # Evaluate regime quality on active-return bars to avoid
                # flat no-position periods diluting per-regime win-rate signals.
                active = merged[np.abs(merged["period_return"].astype(float)) > 1e-12].copy()
                if active.empty:
                    active = merged

                validation_summary["regime_slices"] = regime_sliced_evaluation(
                    active["period_return"].astype(float).tolist(),
                    active["regime"].astype(str).tolist(),
                )

        if validation_summary:
            formatted_results["validation"] = validation_summary
            quality_decision = evaluate_strategy_quality(
                validation_summary,
                thresholds=_quality_thresholds_from_config(),
            )
            formatted_results["quality_gate"] = {
                "passed": quality_decision.passed,
                "reasons": quality_decision.reasons,
                "metrics": quality_decision.metrics,
            }

        return {"status": "success", "results": _sanitize_json_value(formatted_results)}
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")

# Get order history
@app.get("/orders/history")
async def get_order_history(symbol: Optional[str] = None, limit: int = Query(default=100, ge=1, le=1000)):
    try:
        db_connection = database.Database()
        trade_records = db_connection.get_trade_records(symbol=symbol, limit=limit, offset=0)
        orders = []
        for trade in trade_records:
            quantity = float(trade.get("quantity") or 0.0)
            price = float(trade.get("price") or 0.0)
            raw_data = trade.get("raw_data")
            order_type = "market"
            if isinstance(raw_data, dict):
                order_type = str(raw_data.get("type", order_type)).lower()
            orders.append({
                "id": trade.get("order_id") or trade.get("trade_id"),
                "time": trade.get("timestamp"),
                "symbol": trade.get("symbol"),
                "side": str(trade.get("side", "")).lower(),
                "type": order_type,
                "quantity": quantity,
                "price": price,
                "value": round(quantity * price, 8),
                "status": str(trade.get("status", "")).lower()
            })

        return {"status": "success", "orders": orders}
    except Exception as e:
        logger.error(f"Error fetching order history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch order history")

# Get signal history
@app.get("/signals/history")
async def get_signal_history(symbol: Optional[str] = None, limit: int = Query(default=100, ge=1, le=1000)):
    try:
        db_connection = database.Database()
        signal_records = db_connection.get_signal_records(symbol=symbol, limit=limit, offset=0)
        signals = []
        for signal in signal_records:
            signals.append({
                "id": signal.get("signal_id"),
                "time": signal.get("timestamp"),
                "symbol": signal.get("symbol"),
                "timeframe": signal.get("timeframe"),
                "strategy": signal.get("strategy"),
                "signal": str(signal.get("signal", "")).lower(),
                "strength": signal.get("strength", None),
                "price": signal.get("price")
            })

        return {"status": "success", "signals": signals}
    except Exception as e:
        logger.error(f"Error fetching signal history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch signal history")

# LLM Integration endpoints
@app.post("/llm/decision")
async def get_llm_decision(request: LLMDecisionRequest):
    try:
        # Create LLM manager
        manager = llm_manager.LLMManager()
        
        # Get market data in the format expected by the LLM manager
        market_data = request.market_data
        
        # Make decision based on context and market data
        if "llm_model" in request.market_data and request.market_data["llm_model"] != "rule_based":
            # Use the specified LLM model
            decision = manager.make_llm_decision(
                market_data=market_data,
                symbol=request.symbol,
                timeframe=request.timeframe,
                context=request.context,
                strategy_signals=request.strategy_signals
            )
        else:
            # Use rule-based fallback
            decision = manager.make_rule_based_decision(
                market_data=market_data,
                strategy_signals=request.strategy_signals
            )

        normalized_decision = _normalize_llm_decision_payload(decision)
        
        return {
            "status": "success",
            "decision": normalized_decision["decision"],
            "confidence": normalized_decision["confidence"],
            "reasoning": normalized_decision["reasoning"]
        }
    except Exception as e:
        logger.error(f"Error getting LLM decision: {e}")
        raise HTTPException(status_code=500, detail="Failed to get LLM decision")

# Database operations endpoints
@app.get("/database/trades")
async def get_trades(
    symbol: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    try:
        # Get database connection
        db_connection = database.Database()
        
        trade_records = db_connection.get_trade_records(symbol=symbol, limit=limit, offset=offset)

        return {"status": "success", "trades": trade_records}
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch trades")

@app.get("/database/signals")
async def get_signals(
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    try:
        # Get database connection
        db_connection = database.Database()
        
        signal_records = db_connection.get_signal_records(
            symbol=symbol,
            strategy=strategy,
            limit=limit,
            offset=offset
        )

        return {"status": "success", "signals": signal_records}
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch signals")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
