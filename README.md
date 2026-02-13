# CogniTrade

Production-grade crypto algorithmic trading platform with Binance execution, ADX regime-adaptive strategy, LLM-assisted decisioning, research-grade backtesting, and staged rollout safety gates.

## Status

- Production strategy: **ADX(10)\_T25\_C15 + DI filter** — backtest-validated across bull (+29.5%), mixed (+12.5%), and bear (+5.5%) regimes.
- Bot runs via `systemd` in FUTURES mode with shorts enabled on Binance Testnet.
- Daily soak checks via cron (06:00 UTC) with Telegram notifications.
- Current deployment posture is `GO` for controlled live rollout with staged capital ramp gates.

See:
- `docs/production_execution/GO_NO_GO_CHECKLIST.md`
- `docs/production_execution/GAP_CLOSURE_PLAN.md`
- `docs/production_execution/IMPLEMENTATION_TRACKER.md`
- `docs/production_execution/WORKLOG.md`

## Core Capabilities

### Trading Runtime
- Binance integration for market data and order execution (`SPOT` and guarded `FUTURES` mode).
- Explicit live-trading safety gate:
  - `TESTNET=True` by default.
  - `TESTNET=False` requires `ENABLE_LIVE_TRADING=True`.
- Multi-strategy signal aggregation with configurable consensus (simple, technical, trend\_following, custom).
- Production strategy: ADX regime-adaptive with DI direction filter — MACD trend-following in trending markets, RSI mean-reversion in choppy markets.
- Optional LLM pipeline:
  - Primary model: DeepSeek (`LLM_PRIMARY_MODEL`, default `deepseek-reasoner`).
  - Secondary model: OpenAI (`LLM_SECONDARY_MODEL`, default `gpt-5-mini`).
  - Safe fallback to rule-based behavior.
- Telegram notifications for trades, alerts, lifecycle events, and daily soak summaries.

### Risk and Execution Controls
- Pre-trade notional and exposure limits.
- Short-specific guards: leverage cap, short notional cap, liquidation-buffer checks.
- Hard risk engine with drawdown/daily-loss controls and kill-switch behavior.
- Exchange reconciliation and restart recovery.
- Retry/backoff/circuit-breaker resilience for exchange and LLM transport.

### Regime and Policy System
- Regime detection (`BULL`, `BEAR`, `SIDEWAYS`, `HIGH_VOLATILITY`) with persistence.
- Regime-based policy routing for strategy weights, activation, and sizing.
- Safe switching controls: hysteresis, cooldown, turnover caps, optional shadow switching.
- Edge-decay monitoring with automatic de-risk/disable actions.

### Backtesting and Research
- Modular backtesting engine under `bot/backtesting/`.
- Walk-forward and purged-CV validation support.
- Realistic execution simulation (spread, slippage, latency, partial fills, funding).
- Portfolio optimizer utilities.
- Research registry and reproducibility helpers.
- Cross-engine parity and benchmarking with industry-standard libraries:
  - `vectorbt`
  - `backtrader`

### API and Operations
- FastAPI control plane in `api/main.py`.
- API auth, scoped roles, CORS hardening, and rate limiting.
- Observability: traces, events, alerts, dashboard summaries, persistence support.
- Rollout gates for `shadow -> canary -> production` promotion flow.
- Strategy quality-gate enforcement before production promotion.

## Repository Layout

```text
CogniTrade/
├── bot/                            # Core runtime, risk, policy, data, execution
│   ├── backtesting/                # Modular backtesting engine
│   ├── observability/              # Telemetry, traces, alerts, persistence
│   ├── data_pipeline/              # PIT dataset and regime dataset tooling
│   └── custom_strategies/          # Plugin directory for user-defined strategies
├── api/                            # FastAPI control plane (port 8001)
├── scripts/                        # Test runner, backtests, evidence builders, soak checks
├── tests/                          # 33 pytest test files
├── deploy/systemd/                 # cognitrade-bot.service, cognitrade-api.service
├── docs/production_execution/      # Tracker, worklog, gate checklist, runbooks
├── research/                       # Experiment registry, benchmarks
├── data/                           # SQLite databases (trading_bot.db, observability.db)
├── order_logs/                     # JSON trade execution logs
├── output/                         # Generated evidence and reports
└── run_bot.py                      # Bot launcher
```

## Quick Start

### 1) Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2) Configure `.env`

Minimum required:

```ini
API_KEY=your_binance_key
API_SECRET=your_binance_secret
TESTNET=True
ENABLE_LIVE_TRADING=False
SYMBOL=BTCUSDT
```

Recommended production-safe defaults (already reflected in `.env.example`):

```ini
API_AUTH_ENABLED=True
ENABLE_ROLLOUT_GATES=True
ROLLOUT_ENFORCE_PRODUCTION_GATE=True
ENABLE_RISK_ENGINE=True
ENABLE_OBSERVABILITY=True
```

### 3) Run the Bot

```bash
python run_bot.py
# or via systemd (production)
sudo systemctl start cognitrade-bot
```

### 4) Run the API

```bash
cd api
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
# or via systemd (production)
sudo systemctl start cognitrade-api
```

API docs available at `http://localhost:8001/docs`.

## How to Use

### Start/Stop Trading via API

```bash
curl -X POST http://localhost:8001/trading/start \
  -H "X-API-Key: <admin-key>"

curl -X POST http://localhost:8001/trading/stop \
  -H "X-API-Key: <admin-key>"
```

### Run a Backtest via API

```bash
curl -X POST http://localhost:8001/backtest/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <read-or-admin-key>" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframes": ["1h"],
    "start_date": "2025-01-01",
    "end_date": "2025-02-01",
    "initial_capital": 10000,
    "commission": 0.001,
    "strategy_name": "sma_crossover",
    "strategy_params": {"short_period": 2, "long_period": 3},
    "trade_mode": "FUTURES",
    "allow_short_positions": true,
    "run_walk_forward_validation": true,
    "include_regime_slices": true
  }'
```

### Evaluate Rollout Gates

```bash
# Quality gate
curl -X POST http://localhost:8001/rollout/quality/evaluate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <admin-key>" \
  -d '{"validation_summary": {"walk_forward_folds": 5}}'

# Stage gates
curl -X POST http://localhost:8001/rollout/evaluate/shadow \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <admin-key>" \
  -d '{"rollout_id":"demo","sample_count":120,"error_rate":0.02}'
```

## Production Workflow

Source of truth: `docs/production_execution/README.md`

Recommended cycle:
1. Pick tasks from `IMPLEMENTATION_TRACKER.md`.
2. Implement scoped changes only.
3. Run mapped tests from `TEST_MATRIX.md`.
4. Append evidence to `WORKLOG.md`.
5. Promote only when gates are satisfied.

Useful scripts:

```bash
bash scripts/run_tests_local.sh                          # safe test runner
bash scripts/soak_daily_check.sh                         # daily health + Telegram
venv/bin/python3 scripts/backtest_long_short.py           # multi-window backtest
venv/bin/python3 scripts/backtest_long_short.py --sweep   # parameter sweep
venv/bin/python3 scripts/build_g0_evidence.py
venv/bin/python3 scripts/build_gap_closure_evidence.py
venv/bin/python3 scripts/check_rollout_gate.py --help
```

## Testing

Always use the safe test runner — it backs up `.env`, applies test-safe overrides (disables auth, resets trade mode), and restores on exit:

```bash
bash scripts/run_tests_local.sh              # full suite
bash scripts/run_tests_local.sh -x -v        # stop on first failure, verbose
bash scripts/run_tests_local.sh -k "strategy" # keyword filter
```

Run specific suites:

```bash
bash scripts/run_tests_local.sh tests/test_strategy.py -v
bash scripts/run_tests_local.sh tests/test_main.py -v
bash scripts/run_tests_local.sh tests/test_deploy_policy.py -v
bash scripts/run_tests_local.sh tests/test_gap_closure.py -v
```

Coverage:

```bash
venv/bin/pytest --cov=bot --cov=api
```

## Key Configuration Areas

Use `.env.example` as the complete reference. High-impact sections:

- Strategy selection:
  - `ENABLE_TREND_FOLLOWING_STRATEGY=True` (production strategy)
  - `ENABLE_SIMPLE_STRATEGY`, `ENABLE_TECHNICAL_STRATEGY` (disable for production)
  - `MIN_STRATEGIES_FOR_DECISION=1`
  - `ADX_PERIOD`, `ADX_TREND_THRESH`, `ADX_CHOP_THRESH`, `ADX_USE_DI_FILTER`
- Trading mode and shorting:
  - `TRADE_MODE`
  - `ENABLE_FUTURES_SHORTS`
  - `MAX_SHORT_NOTIONAL_USD`
  - `MAX_SHORT_LEVERAGE`
- Risk engine:
  - `ENABLE_RISK_ENGINE`
  - `RISK_ENGINE_MAX_DRAWDOWN_PCT`
  - `RISK_ENGINE_DAILY_LOSS_LIMIT_USD`
- Notifications:
  - `ENABLE_TELEGRAM_NOTIFICATIONS`
  - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- Rollout gates:
  - `ENABLE_ROLLOUT_GATES`
  - `ROLLOUT_ENFORCE_PRODUCTION_GATE`
  - `ROLLOUT_REQUIRE_QUALITY_GATE`
- Quality and benchmarks:
  - `ENABLE_STRATEGY_QUALITY_GATE`
  - `QUALITY_*`
  - `BENCHMARK_*`
- API hardening:
  - `API_AUTH_ENABLED`
  - `API_READ_KEYS`
  - `API_ADMIN_KEYS`
  - `API_RATE_LIMIT_*`

## Operational Artifacts

Recent gap-closure and go/no-go artifacts are exported under `output/`, including:

- `output/gap_closure_summary_latest.json`
- `output/g2_01_go_no_go_signoff_latest.json`
- `output/g2_02_capital_ramp_plan_latest.json`

## Security Notes

- Never commit real secrets.
- Keep credentials in `.env` only.
- Start on Binance Testnet and enforce staged rollout before any live capital increase.

## Disclaimer

This software is for research and engineering purposes. Trading digital assets carries significant risk. You are responsible for deployment decisions, risk limits, and regulatory compliance.
