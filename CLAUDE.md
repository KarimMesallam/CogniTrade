# CLAUDE.md

## Project Overview

CogniTrade is a production-grade AI-enhanced cryptocurrency trading bot for Binance. Python-based, it combines configurable technical strategies with LLM-assisted decision-making, hard risk controls, staged rollout gates, and full observability. Runs on Binance Testnet by default; explicit opt-in for live trading.

## Commands

```bash
# Tests (use the local runner — it sets safe env overrides)
bash scripts/run_tests_local.sh           # full suite
bash scripts/run_tests_local.sh -x -v     # stop on first failure, verbose
bash scripts/run_tests_local.sh -k "llm"  # keyword filter
venv/bin/python3 -m pytest --cov=bot --cov=api  # coverage (no env safety)

# Run
python run_bot.py                         # start bot (or systemd: cognitrade-bot.service)
cd api && uvicorn main:app --reload       # API server (or systemd: cognitrade-api.service)

# Backtesting
venv/bin/python3 scripts/backtest_long_short.py          # default strategy
venv/bin/python3 scripts/backtest_long_short.py --sweep   # parameter sweep

# Services
sudo systemctl restart cognitrade-bot
sudo systemctl restart cognitrade-api
journalctl -u cognitrade-bot -f           # live logs

# Soak check (runs daily via cron at 06:00 UTC, sends Telegram)
bash scripts/soak_daily_check.sh
```

## Architecture

### Trade Execution Flow
```
Candle Fetch → Regime Detection → Strategy Signals → Consensus →
LLM Validation → Policy Routing → Risk Engine → Order Execution →
DB Recording → Reconciliation → Telegram Alerts
```

### Core Modules (`bot/`)

| Module | Purpose |
|--------|---------|
| `main.py` | Trading loop, initialization, signal pipeline |
| `strategy.py` | Multi-strategy orchestration (simple, technical, trend_following, custom) |
| `config.py` | Env-based configuration with JSON override support |
| `binance_api.py` | Binance client wrapper, time sync, market data |
| `order_manager.py` | SPOT + FUTURES execution, position management, stop-loss/take-profit |
| `llm_manager.py` | Dual-model pipeline (DeepSeek R1 + GPT-5-mini), circuit breakers |
| `database.py` | SQLite schema (trades, signals, market_data, performance) |
| `db_integration.py` | Business logic wrapper for DB |
| `risk_engine.py` | Hard limits: max drawdown, gross exposure, daily loss, kill-switch |
| `regime.py` | Market regime detection (BULL / BEAR / SIDEWAYS / HIGH_VOL) |
| `policy.py` | Regime-based strategy routing with hysteresis and cooldowns |
| `monitoring.py` | Edge-decay detection, de-risk/disable triggers |
| `reconciliation.py` | Exchange position sync on startup + periodic |
| `deploy_policy.py` | Rollout gates: Shadow → Canary → Production |
| `notifications.py` | Telegram alerting with rate limiting |
| `portfolio.py` | Portfolio optimization and allocation |
| `resilience.py` | Circuit breaker + exponential backoff |

### Subsystems

| Directory | Purpose |
|-----------|---------|
| `bot/backtesting/` | Multi-timeframe backtest engine with walk-forward validation, execution simulation, HTML reporting |
| `bot/data_pipeline/` | Point-in-time data validation (no lookahead), regime dataset building |
| `bot/observability/` | Telemetry traces, events, alerts, SQLite persistence, dashboard aggregation |
| `bot/custom_strategies/` | Plugin directory for user-defined strategies |

### API (`api/main.py` — FastAPI on port 8001)

Health, observability dashboard/events/alerts, rollout gate evaluation (shadow/canary/production), trading start/stop, account info, market data, strategy listing, LLM decisions, backtesting, order/signal history, DB queries. Auth via `X-API-Key` header (read/admin roles).

## Strategy System

Strategies return `"BUY"` / `"SELL"` / `"HOLD"` via `generate_signal(symbol, interval)`.

| Strategy | Key | Default TF | Weight | Description |
|----------|-----|-----------|--------|-------------|
| Simple | `ENABLE_SIMPLE_STRATEGY` | 1m | 1.0 | Price momentum (close vs prev close) |
| Technical | `ENABLE_TECHNICAL_STRATEGY` | 1h | 2.0 | RSI + Bollinger Bands + MACD vote |
| **Trend Following** | `ENABLE_TREND_FOLLOWING_STRATEGY` | 4h | 3.0 | **Production strategy**: ADX(10)_T25_C15 regime-adaptive + DI filter |
| Custom | `ENABLE_CUSTOM_STRATEGY` | 4h | 1.0 | Dynamic module loading from `bot/custom_strategies/` |

**Trend Following (ADX regime-adaptive):**
- TRENDING (ADX >= 25): MACD direction → BUY/SELL
- CHOPPY (ADX <= 15): RSI mean-reversion → BUY/SELL
- TRANSITION (15-25): HOLD
- DI filter: +DI > -DI blocks SELL, -DI > +DI blocks BUY

**Consensus:** Weighted majority (configurable). `MIN_STRATEGIES_FOR_DECISION` controls minimum required. LLM agreement optional (`LLM_AGREEMENT_REQUIRED`).

## Configuration (`bot/config.py`)

**Hierarchy:** Environment variables > JSON config file (`CONFIG_FILE`) > defaults.

**Key sections in `TRADING_CONFIG`:**
`strategies`, `decision_making`, `trading`, `regime`, `policy`, `data_pipeline`, `risk_engine`, `reconciliation`, `observability`, `monitoring`, `notifications`, `api_security`, `rollout`, `quality_gate`, `promotion_benchmarks`, `timeframes`, `operation`

**Critical env vars:**
```
TESTNET=True                    # safe default
ENABLE_LIVE_TRADING=False       # explicit opt-in
SYMBOL=BTCUSDT
TRADE_MODE=FUTURES              # SPOT or FUTURES
ENABLE_FUTURES_SHORTS=True
ENABLE_TREND_FOLLOWING_STRATEGY=True
MIN_STRATEGIES_FOR_DECISION=1
LLM_AGREEMENT_REQUIRED=False
```

## Testing

- **33 test files** in `tests/` — unit, integration, API
- **Markers:** `api`, `integration`, `unit` (see `pytest.ini`)
- **Always use `scripts/run_tests_local.sh`** — it backs up `.env`, applies test-safe overrides (disables auth, resets trade mode), and restores on exit
- External services (Binance, LLMs) are mocked in tests

## Infrastructure

- **systemd:** `cognitrade-bot.service` (bot), `cognitrade-api.service` (API on 127.0.0.1:8001)
- **Cron:** `0 6 * * *` runs `scripts/soak_daily_check.sh` (daily health + Telegram summary)
- **Data:** `data/trading_bot.db` (main), `data/observability.db` (telemetry), `order_logs/` (JSON), `data/rollout_gate_state.json`
- **Logs:** `logs/` directory, `cognitrade.log`

## Common Development Tasks

- **Add strategy:** Add function to `bot/strategy.py` returning BUY/SELL/HOLD, add config block to `bot/config.py`, register in `get_all_strategy_signals()`
- **Add API endpoint:** Update `api/main.py`
- **Change DB schema:** Update `bot/database.py`
- **Modify LLM behavior:** Update `bot/llm_manager.py` and `LLM_CONFIG` in config
- **Run backtest sweep:** `venv/bin/python3 scripts/backtest_long_short.py --sweep`

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_tests_local.sh` | Safe test runner with env isolation |
| `scripts/backtest_long_short.py` | Multi-window backtest (bull/mixed/bear) with parameter sweep |
| `scripts/soak_daily_check.sh` | Daily soak health check + Telegram notification |
| `scripts/build_g0_evidence.py` | G0 go/no-go signoff evidence |
| `scripts/build_gap_closure_evidence.py` | Gap closure proof builder |
| `scripts/check_rollout_gate.py` | CLI rollout gate evaluator |
| `scripts/run_gate_tuning_sweep.py` | Gate threshold parameter sweep |
