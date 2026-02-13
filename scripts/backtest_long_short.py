#!/usr/bin/env python3
"""
Multi-Window Trend-Following Backtest

Tests a trend-following strategy (SMA crossover + MACD momentum) on three
windows — bullish, mixed/choppy, and bearish — to verify robustness across
all market regimes.

Usage:
    venv/bin/python3 scripts/backtest_long_short.py           # default SMA(10,100)
    venv/bin/python3 scripts/backtest_long_short.py --sweep    # parameter sweep
"""

import argparse
import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance.client import Client as BinanceClient

from bot.backtesting import BacktestEngine
from bot.config import TRADING_CONFIG

# Use mainnet client for historical data downloads.
# The bot's default client uses testnet which only has ~10 days of data.
_data_client = BinanceClient("", "", testnet=False)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/backtest_long_short.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("trading_bot")

# ---------------------------------------------------------------------------
# Test windows — contiguous, non-overlapping, covering ~16 months
# ---------------------------------------------------------------------------
WINDOWS = {
    "bull": {
        "start": "2024-10-01",
        "end": "2025-05-01",
        "label": "BULLISH",
        "desc": "~$60k -> $97k rally",
    },
    "mixed": {
        "start": "2025-05-01",
        "end": "2025-11-01",
        "label": "MIXED",
        "desc": "choppy $97k-$120k range",
    },
    "bear": {
        "start": "2025-11-01",
        "end": "2026-02-13",
        "label": "BEARISH",
        "desc": "~$95k -> $66k decline",
    },
}

SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"
INITIAL_CAPITAL = 10000
COMMISSION_RATE = 0.001

# Trend-following strategies generate fewer, larger trades than mean-reversion.
# Override the promotion_benchmarks threshold (25) with a value appropriate for
# this strategy class.
MIN_TRADES_OVERRIDE = 8


# ---------------------------------------------------------------------------
# ADX computation (Wilder smoothing, matches ATR implementation)
# ---------------------------------------------------------------------------
def _compute_adx(df: pd.DataFrame, period: int = 14) -> dict:
    """
    Compute ADX (Average Directional Index) and +DI/-DI from OHLC data.

    Uses Wilder's smoothing method. Returns a dict of Series aligned to df.index:
        {"adx": Series, "plus_di": Series, "minus_di": Series}
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    n = len(df)
    atr = np.full(n, np.nan)
    smooth_plus = np.full(n, np.nan)
    smooth_minus = np.full(n, np.nan)
    adx = np.full(n, np.nan)

    if n < period + 1:
        nan_series = pd.Series(adx, index=df.index)
        return {"adx": nan_series, "plus_di": nan_series.copy(), "minus_di": nan_series.copy()}

    # Seed with simple averages over the first `period` rows
    atr[period] = tr.iloc[1 : period + 1].mean()
    smooth_plus[period] = plus_dm[1 : period + 1].mean()
    smooth_minus[period] = minus_dm[1 : period + 1].mean()

    # Wilder smoothing
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr.iloc[i]) / period
        smooth_plus[i] = (smooth_plus[i - 1] * (period - 1) + plus_dm[i]) / period
        smooth_minus[i] = (smooth_minus[i - 1] * (period - 1) + minus_dm[i]) / period

    # +DI / -DI
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * smooth_plus / atr
        minus_di = 100.0 * smooth_minus / atr
        dx = 100.0 * np.abs(plus_di - minus_di) / np.where(
            (plus_di + minus_di) == 0, np.nan, plus_di + minus_di
        )

    # ADX = Wilder-smoothed DX
    first_valid = period + period  # need `period` DX values to seed ADX
    if n > first_valid:
        adx[first_valid] = np.nanmean(dx[period + 1 : first_valid + 1])
        for i in range(first_valid + 1, n):
            if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return {
        "adx": pd.Series(adx, index=df.index),
        "plus_di": pd.Series(plus_di, index=df.index),
        "minus_di": pd.Series(minus_di, index=df.index),
    }


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def adx_regime_strategy(data_dict: dict, symbol: str) -> str:
    """
    ADX regime-adaptive strategy with default parameters and DI direction filter.

    DI DIRECTION FILTER (applied to all regimes):
        +DI > -DI  -> only BUY signals allowed (bullish direction)
        -DI > +DI  -> only SELL signals allowed (bearish direction)

    TRENDING (ADX > 25): MACD direction  -> BUY/SELL (filtered by DI)
    CHOPPY   (ADX < 20): RSI mean-revert -> BUY(oversold) / SELL(overbought) (filtered by DI)
    TRANSITION (20-25):  HOLD
    """
    return _adx_regime_logic(data_dict, adx_period=14, trend_thresh=25,
                             chop_thresh=20, rsi_oversold=30, rsi_overbought=70,
                             use_di_filter=True)


def _adx_regime_logic(
    data_dict: dict,
    adx_period: int,
    trend_thresh: float,
    chop_thresh: float,
    rsi_oversold: float,
    rsi_overbought: float,
    use_di_filter: bool = True,
) -> str:
    """Core logic shared by default strategy and factory-built variants."""
    tf = TIMEFRAME
    if tf not in data_dict:
        return "HOLD"

    df = data_dict[tf]
    warmup = adx_period * 2 + 10
    if len(df) < warmup:
        return "HOLD"

    required = ["macd_line", "signal_line", "rsi"]
    if not all(col in df.columns for col in required):
        return "HOLD"

    current = df.iloc[-1]
    macd = current["macd_line"]
    signal_line = current["signal_line"]
    rsi = current["rsi"]

    if pd.isna(macd) or pd.isna(signal_line) or pd.isna(rsi):
        return "HOLD"

    # Compute ADX and DI on-the-fly (not pre-computed by engine)
    adx_data = _compute_adx(df, period=adx_period)
    adx_val = adx_data["adx"].iloc[-1]
    plus_di = adx_data["plus_di"].iloc[-1]
    minus_di = adx_data["minus_di"].iloc[-1]

    if pd.isna(adx_val):
        return "HOLD"

    # DI direction: determines which signals are allowed
    # +DI > -DI = bullish direction → only BUY
    # -DI > +DI = bearish direction → only SELL
    if use_di_filter and not pd.isna(plus_di) and not pd.isna(minus_di):
        bullish_di = plus_di > minus_di
    else:
        bullish_di = None  # no filter

    def _filter_signal(raw_signal: str) -> str:
        """Apply DI direction filter to a raw signal."""
        if bullish_di is None:
            return raw_signal
        if raw_signal == "BUY" and not bullish_di:
            return "HOLD"  # block BUY in bearish direction
        if raw_signal == "SELL" and bullish_di:
            return "HOLD"  # block SELL in bullish direction
        return raw_signal

    # --- TRENDING regime: use MACD for direction ---
    if adx_val >= trend_thresh:
        if macd > signal_line:
            return _filter_signal("BUY")
        if macd < signal_line:
            return _filter_signal("SELL")
        return "HOLD"

    # --- CHOPPY regime: use RSI mean-reversion ---
    if adx_val <= chop_thresh:
        if rsi < rsi_oversold:
            return _filter_signal("BUY")
        if rsi > rsi_overbought:
            return _filter_signal("SELL")
        return "HOLD"

    # --- TRANSITION zone: stay out ---
    return "HOLD"


def make_strategy(
    adx_period: int = 14,
    trend_thresh: float = 25,
    chop_thresh: float = 20,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    use_di_filter: bool = True,
) -> Callable:
    """
    Factory that returns an ADX regime-adaptive strategy with custom parameters.
    use_di_filter: If True, only allow BUY when +DI > -DI, SELL when -DI > +DI.
    """
    def strategy(data_dict: dict, symbol: str) -> str:
        return _adx_regime_logic(
            data_dict, adx_period, trend_thresh, chop_thresh,
            rsi_oversold, rsi_overbought, use_di_filter,
        )

    di_tag = "" if use_di_filter else "_noDI"
    strategy.__name__ = (
        f"ADX({adx_period})_T{trend_thresh}_C{chop_thresh}"
        f"_RSI({rsi_oversold}/{rsi_overbought}){di_tag}"
    )
    return strategy


# ---------------------------------------------------------------------------
# Data download helper
# ---------------------------------------------------------------------------
def download_fresh_data(
    symbol: str, timeframe: str, start_date: str, end_date: str, db_path: str
) -> int:
    """Download klines from Binance and upsert into the DB."""
    import sqlite3

    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

    logger.info(f"Downloading {timeframe} data for {symbol} ({start_date} to {end_date})...")
    klines = _data_client.get_historical_klines(
        symbol=symbol,
        interval=timeframe,
        start_str=str(start_ts),
        end_str=str(end_ts),
    )

    if not klines:
        logger.warning("No klines returned from Binance")
        return 0

    df = pd.DataFrame(
        klines,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col])

    df_store = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df_store["symbol"] = symbol
    df_store["timeframe"] = timeframe
    df_store["timestamp"] = df_store["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(db_path) as conn:
        for _, row in df_store.iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO market_data
                   (symbol, timeframe, timestamp, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["symbol"], row["timeframe"], row["timestamp"],
                    row["open"], row["high"], row["low"], row["close"], row["volume"],
                ),
            )
        conn.commit()

    logger.info(f"Stored {len(df_store)} candles in database")
    return len(df_store)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _dec(v) -> float:
    """Decimal / float -> float."""
    return float(v) if v is not None else 0.0


def _fmt_pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# Backtest result container (lightweight, for comparison tables)
# ---------------------------------------------------------------------------
@dataclass
class WindowResult:
    label: str
    total_return: float
    sharpe: float
    calmar: float
    sortino: float
    max_dd: float
    win_rate: float
    profit_factor: float
    total_trades: int
    n_long: int
    n_short: int
    gates_passed: bool
    raw_result: object  # BacktestResult


def _extract_window_result(label: str, results, gates_passed: bool) -> WindowResult:
    m = results.metrics
    completed = [t for t in results.trades if t.get("profit_loss") is not None]
    n_long = sum(1 for t in completed if t.get("position_side") == "LONG")
    n_short = sum(1 for t in completed if t.get("position_side") == "SHORT")

    return WindowResult(
        label=label,
        total_return=_dec(m.total_return_pct),
        sharpe=_dec(m.sharpe_ratio),
        calmar=_dec(m.calmar_ratio),
        sortino=_dec(m.sortino_ratio),
        max_dd=_dec(m.max_drawdown_pct),
        win_rate=_dec(m.win_rate),
        profit_factor=_dec(m.profit_factor),
        total_trades=results.total_trades,
        n_long=n_long,
        n_short=n_short,
        gates_passed=gates_passed,
        raw_result=results,
    )


# ---------------------------------------------------------------------------
# Single-window backtest runner
# ---------------------------------------------------------------------------
def run_single_backtest(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    strategy: Callable,
    label: str,
) -> WindowResult:
    """
    Downloads data, clears cache, runs the backtest engine, evaluates
    quality gates, prints results, and returns a WindowResult.
    """
    strategy_name = getattr(strategy, "__name__", "unknown")

    print()
    print("=" * 68)
    print(f"  {label}")
    print("=" * 68)
    print(f"  Symbol     : {symbol}")
    print(f"  Timeframe  : {timeframe}")
    print(f"  Period     : {start_date}  ->  {end_date}")
    print(f"  Strategy   : {strategy_name}")
    print(f"  Shorting   : ENABLED")
    print("=" * 68)

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "trading_bot.db",
    )

    # 1. Download fresh data
    print(f"\n  [1/3] Downloading {timeframe} candles from Binance...")
    try:
        _data_client.ping()
        n_rows = download_fresh_data(symbol, timeframe, start_date, end_date, db_path)
        print(f"         Upserted {n_rows} candles into DB")
    except Exception as exc:
        print(f"         Download failed ({exc}), proceeding with cached DB data")

    # 2. Clear pickle cache
    cache_dir = Path("./cache")
    if cache_dir.exists():
        cleared = 0
        for pkl in cache_dir.glob("*.pkl"):
            pkl.unlink()
            cleared += 1
        if cleared:
            print(f"         Cleared {cleared} stale cache files")

    # 3. Build engine and run
    print(f"\n  [2/3] Running backtest...")
    engine = BacktestEngine(
        symbol=symbol,
        timeframes=[timeframe],
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE,
        allow_short_positions=True,
    )
    candle_count = len(engine.market_data.get(timeframe, []))
    print(f"         Loaded {candle_count} candles")

    results = engine.run_backtest(strategy)

    # 4. Extract metrics and print
    m = results.metrics
    total_return = _dec(m.total_return_pct)
    sharpe = _dec(m.sharpe_ratio)
    calmar = _dec(m.calmar_ratio)
    sortino = _dec(m.sortino_ratio)
    max_dd = _dec(m.max_drawdown_pct)
    win_rate = _dec(m.win_rate)
    profit_factor = _dec(m.profit_factor)

    completed = [t for t in results.trades if t.get("profit_loss") is not None]
    n_long = sum(1 for t in completed if t.get("position_side") == "LONG")
    n_short = sum(1 for t in completed if t.get("position_side") == "SHORT")

    print(f"\n  [3/3] Results")
    print("=" * 68)
    print(f"  {'Metric':<30} {'Value':>15}")
    print("-" * 68)
    print(f"  {'Initial Capital':<30} {'${:,.2f}'.format(results.initial_capital):>15}")
    print(f"  {'Final Equity':<30} {'${:,.2f}'.format(results.final_equity):>15}")
    print(f"  {'Total Return':<30} {total_return:>14.2f}%")
    print(f"  {'Sharpe Ratio':<30} {sharpe:>15.4f}")
    print(f"  {'Calmar Ratio':<30} {calmar:>15.4f}")
    print(f"  {'Sortino Ratio':<30} {sortino:>15.4f}")
    print(f"  {'Max Drawdown':<30} {max_dd:>14.2f}%")
    print(f"  {'Win Rate':<30} {win_rate:>14.2f}%")
    print(f"  {'Profit Factor':<30} {profit_factor:>15.4f}")
    print(f"  {'Total Trades':<30} {results.total_trades:>15d}")
    print(f"  {'  |-- LONG exits':<30} {n_long:>15d}")
    print(f"  {'  +-- SHORT exits':<30} {n_short:>15d}")
    print("=" * 68)

    # Trade log (last 30)
    show = completed[-30:]
    count_label = "all" if len(completed) <= 30 else "last 30"
    print(f"\n  Trade Log ({count_label} of {len(completed)} completed trades)")
    print("-" * 92)
    print(f"  {'#':>3}  {'Timestamp':<20} {'Side':<5} {'Pos':<6} {'Entry':>10} {'Exit':>10} {'P/L':>12} {'ROI%':>8}")
    print("-" * 92)
    for idx, t in enumerate(show, start=max(1, len(completed) - 29)):
        ts = str(t.get("timestamp", ""))[:19]
        side = t.get("side", "?")
        pos = t.get("position_side", "?")
        entry = _dec(t.get("entry_price"))
        exit_ = _dec(t.get("exit_price"))
        pl = _dec(t.get("profit_loss"))
        roi = _dec(t.get("roi_pct"))
        print(
            f"  {idx:>3}  {ts:<20} {side:<5} {pos:<6} "
            f"{entry:>10.2f} {exit_:>10.2f} {pl:>+12.2f} {roi:>+7.2f}%"
        )
    print("-" * 92)

    # Long-short switching
    print()
    if n_long > 0 and n_short > 0:
        print(f"  Long-short switching: VERIFIED ({n_long} LONG, {n_short} SHORT)")
    elif n_long > 0:
        print("  Long-short switching: NOT VERIFIED -- only LONG trades fired")
    elif n_short > 0:
        print("  Long-short switching: NOT VERIFIED -- only SHORT trades fired")
    else:
        print("  Long-short switching: NOT VERIFIED -- no completed trades")

    # Quality gates
    gates_passed = evaluate_quality_gates(results, label)

    return _extract_window_result(label, results, gates_passed)


# ---------------------------------------------------------------------------
# Quality gate evaluation
# ---------------------------------------------------------------------------
def evaluate_quality_gates(results, label: str) -> bool:
    """Check promotion benchmarks. Prints gate results. Returns True if all pass."""
    m = results.metrics
    total_return = _dec(m.total_return_pct)
    sharpe = _dec(m.sharpe_ratio)
    calmar = _dec(m.calmar_ratio)
    max_dd = _dec(m.max_drawdown_pct)
    total_trades = results.total_trades

    benchmarks = TRADING_CONFIG["promotion_benchmarks"]
    min_trades = MIN_TRADES_OVERRIDE
    gates = [
        (
            f"Total trades >= {min_trades}",
            total_trades >= min_trades,
        ),
        (
            f"Net return   >= {benchmarks['min_net_return_pct']}%",
            total_return >= benchmarks["min_net_return_pct"],
        ),
        (
            f"Sharpe       >= {benchmarks['min_sharpe_ratio']}",
            sharpe >= benchmarks["min_sharpe_ratio"],
        ),
        (
            f"Calmar       >= {benchmarks['min_calmar_ratio']}",
            calmar >= benchmarks["min_calmar_ratio"],
        ),
        (
            f"Max drawdown <= {benchmarks['max_drawdown_pct']}%",
            abs(max_dd) <= benchmarks["max_drawdown_pct"],
        ),
    ]

    print(f"\n  Quality Gates")
    print("-" * 68)
    all_pass = True
    for gate_label, passed in gates:
        status = _fmt_pass(passed)
        print(f"    [{status:>4}]  {gate_label}")
        if not passed:
            all_pass = False
    print("-" * 68)

    if all_pass:
        print("  OVERALL: ALL GATES PASSED")
    else:
        print("  OVERALL: SOME GATES FAILED -- review metrics above")
    print()

    return all_pass


# ---------------------------------------------------------------------------
# Comparison table (supports 2 or 3 windows)
# ---------------------------------------------------------------------------
def print_comparison(window_results: List[WindowResult]) -> bool:
    """Print side-by-side comparison. Returns True if ALL windows pass."""
    n = len(window_results)
    col_w = 13  # column width for each window

    print()
    print("=" * (30 + n * (col_w + 2) + 12))
    print(f"  {'MULTI-WINDOW COMPARISON' if n > 2 else 'DUAL-WINDOW COMPARISON'}")
    print("=" * (30 + n * (col_w + 2) + 12))

    # Header
    header = f"  {'Metric':<28}"
    for wr in window_results:
        short_label = wr.label.split("(")[0].strip().split("--")[-1].strip()
        header += f" {short_label:>{col_w}}"
    header += f" {'Gate':>{col_w - 3}}"
    print(header)
    print("-" * (30 + n * (col_w + 2) + 12))

    benchmarks = TRADING_CONFIG["promotion_benchmarks"]

    def _all_pass_gate(vals, threshold, op=">="):
        if op == ">=":
            return all(v >= threshold for v in vals)
        return all(abs(v) <= threshold for v in vals)

    metrics = [
        ("Total Return",
         [f"{wr.total_return:+.2f}%" for wr in window_results],
         _all_pass_gate([wr.total_return for wr in window_results],
                        benchmarks["min_net_return_pct"])),
        ("Sharpe Ratio",
         [f"{wr.sharpe:.4f}" for wr in window_results],
         _all_pass_gate([wr.sharpe for wr in window_results],
                        benchmarks["min_sharpe_ratio"])),
        ("Calmar Ratio",
         [f"{wr.calmar:.4f}" for wr in window_results],
         _all_pass_gate([wr.calmar for wr in window_results],
                        benchmarks["min_calmar_ratio"])),
        ("Sortino Ratio",
         [f"{wr.sortino:.4f}" for wr in window_results],
         None),
        ("Max Drawdown",
         [f"{wr.max_dd:.2f}%" for wr in window_results],
         _all_pass_gate([wr.max_dd for wr in window_results],
                        benchmarks["max_drawdown_pct"], op="<=")),
        ("Win Rate",
         [f"{wr.win_rate:.2f}%" for wr in window_results],
         None),
        ("Profit Factor",
         [f"{wr.profit_factor:.4f}" for wr in window_results],
         None),
        ("Total Trades",
         [f"{wr.total_trades}" for wr in window_results],
         _all_pass_gate([wr.total_trades for wr in window_results],
                        MIN_TRADES_OVERRIDE)),
        ("  LONG exits",
         [f"{wr.n_long}" for wr in window_results],
         None),
        ("  SHORT exits",
         [f"{wr.n_short}" for wr in window_results],
         None),
    ]

    for metric_name, vals, gate in metrics:
        row = f"  {metric_name:<28}"
        for v in vals:
            row += f" {v:>{col_w}}"
        gate_str = ""
        if gate is not None:
            gate_str = _fmt_pass(gate)
        row += f" {gate_str:>{col_w - 3}}"
        print(row)

    print("=" * (30 + n * (col_w + 2) + 12))

    all_pass = all(wr.gates_passed for wr in window_results)
    if all_pass:
        print("  VERDICT: ALL WINDOWS PASS -- strategy ready for production promotion")
    else:
        failed = [wr.label.split("(")[0].strip().split("--")[-1].strip()
                  for wr in window_results if not wr.gates_passed]
        print(f"  VERDICT: FAILED -- {', '.join(failed)} window(s) did not pass all gates")

    print()
    return all_pass


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------
def run_parameter_sweep():
    """
    Grid search over ADX regime-adaptive strategy parameters.
    Sweeps: ADX period, trend/chop thresholds, RSI levels.
    Ranks by min Sharpe across all windows.
    """
    # Parameter grid
    adx_periods = [10, 14, 20]
    # (trend_threshold, chop_threshold) pairs — trend > chop always
    threshold_pairs = [
        (20, 15),
        (25, 15),
        (25, 20),
        (30, 20),
        (30, 25),
    ]
    rsi_pairs = [
        (30, 70),   # standard
        (35, 65),   # tighter
        (25, 75),   # wider
    ]

    combos = [
        (ap, tt, ct, ro, rb)
        for ap in adx_periods
        for tt, ct in threshold_pairs
        for ro, rb in rsi_pairs
    ]

    window_keys = list(WINDOWS.keys())

    print()
    print("=" * 100)
    print("  PARAMETER SWEEP: ADX Regime-Adaptive (MACD trend + RSI mean-reversion)")
    print("=" * 100)
    print(f"  Combinations     : {len(combos)}")
    print(f"  ADX periods      : {adx_periods}")
    print(f"  Threshold pairs  : {threshold_pairs}  (trend, chop)")
    print(f"  RSI pairs        : {rsi_pairs}  (oversold, overbought)")
    for wk in window_keys:
        w = WINDOWS[wk]
        print(f"  {w['label']:<12}       : {w['start']} -> {w['end']}  ({w['desc']})")
    print("=" * 100)

    sweep_results: List[dict] = []

    for i, (ap, tt, ct, ro, rb) in enumerate(combos, 1):
        strat = make_strategy(ap, tt, ct, ro, rb)
        combo_label = strat.__name__
        short_label = f"A{ap}_T{tt}C{ct}_R{ro}/{rb}"
        print(f"\n{'='*100}")
        print(f"  [{i}/{len(combos)}] Testing {combo_label}")
        print(f"{'='*100}")

        results_by_window = {}
        for wk in window_keys:
            w = WINDOWS[wk]
            wr = run_single_backtest(
                SYMBOL, TIMEFRAME, w["start"], w["end"],
                strat, f"{short_label} -- {w['label']} Window",
            )
            results_by_window[wk] = wr

        sharpes = [results_by_window[wk].sharpe for wk in window_keys]
        min_sharpe = min(sharpes)

        entry = {
            "combo": short_label,
            "adx_period": ap,
            "trend_thresh": tt,
            "chop_thresh": ct,
            "rsi_os": ro,
            "rsi_ob": rb,
            "min_sharpe": min_sharpe,
            "all_pass": all(results_by_window[wk].gates_passed for wk in window_keys),
        }
        for wk in window_keys:
            wr = results_by_window[wk]
            entry[f"{wk}_return"] = wr.total_return
            entry[f"{wk}_sharpe"] = wr.sharpe
            entry[f"{wk}_dd"] = wr.max_dd
            entry[f"{wk}_trades"] = wr.total_trades
            entry[f"{wk}_pass"] = wr.gates_passed

        sweep_results.append(entry)

    # Rank by min_sharpe descending
    sweep_results.sort(key=lambda r: r["min_sharpe"], reverse=True)

    # Print ranked results table
    print()
    W = 140
    print("=" * W)
    print("  PARAMETER SWEEP RESULTS (ranked by min Sharpe across all windows)")
    print("=" * W)
    print(
        f"  {'Rk':>2}  {'Combo':<22}"
        f"  {'BullRet':>8}  {'MixRet':>8}  {'BearRet':>8}"
        f"  {'BullShp':>8}  {'MixShp':>8}  {'BearShp':>8}  {'MinShp':>7}"
        f"  {'BullDD':>7}  {'MixDD':>7}  {'BearDD':>7}"
        f"  {'BuTr':>4}  {'MxTr':>4}  {'BeTr':>4}"
        f"  {'Pass':>4}"
    )
    print("-" * W)

    for rank, r in enumerate(sweep_results, 1):
        pass_str = "YES" if r["all_pass"] else "no"
        print(
            f"  {rank:>2}  {r['combo']:<22}"
            f"  {r['bull_return']:>+7.1f}%  {r['mixed_return']:>+7.1f}%  {r['bear_return']:>+7.1f}%"
            f"  {r['bull_sharpe']:>8.3f}  {r['mixed_sharpe']:>8.3f}  {r['bear_sharpe']:>8.3f}  {r['min_sharpe']:>7.3f}"
            f"  {r['bull_dd']:>6.1f}%  {r['mixed_dd']:>6.1f}%  {r['bear_dd']:>6.1f}%"
            f"  {r['bull_trades']:>4}  {r['mixed_trades']:>4}  {r['bear_trades']:>4}"
            f"  {pass_str:>4}"
        )

    print("=" * W)

    # Highlight best
    best = sweep_results[0]
    print(f"\n  BEST: {best['combo']}  (min Sharpe = {best['min_sharpe']:.4f})")
    if best["all_pass"]:
        print("  STATUS: All windows PASS all quality gates")
    else:
        print("  STATUS: Does NOT pass all windows -- further tuning needed")

    # Show all passing combos if any
    passing = [r for r in sweep_results if r["all_pass"]]
    if passing:
        print(f"\n  {len(passing)} combo(s) pass ALL windows:")
        for r in passing:
            print(f"    {r['combo']:<22}  min_sharpe={r['min_sharpe']:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main: multi-window runner
# ---------------------------------------------------------------------------
def main():
    """Run the default ADX regime-adaptive strategy on all windows."""
    strategy = adx_regime_strategy
    strategy_name = "ADX(14)_T25_C20_RSI(30/70)"

    print()
    print("#" * 90)
    print(f"  MULTI-WINDOW ADX REGIME-ADAPTIVE BACKTEST")
    print(f"  Strategy: {strategy_name}")
    print("#" * 90)

    window_results = []
    for wk, w in WINDOWS.items():
        wr = run_single_backtest(
            SYMBOL, TIMEFRAME, w["start"], w["end"],
            strategy, f"{w['label']} WINDOW ({w['start']} -> {w['end']})",
        )
        window_results.append(wr)

    print_comparison(window_results)

    return window_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-window trend-following backtest")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    if args.sweep:
        run_parameter_sweep()
    else:
        main()
