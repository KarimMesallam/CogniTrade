# CogniTrade

Production-focused crypto algorithmic trading platform with Binance execution, LLM-assisted decisioning, research-grade backtesting, and staged rollout safety gates.

## Status

- Non-P3 production scope (`P0`/`P1`/`P2` + `R0` + `G0`/`G1`/`G2`) is implemented and test-backed.
- Current deployment posture is `GO` for controlled live rollout with staged capital ramp gates.
- Immediate full-capital deployment is intentionally blocked until ramp stages are completed.

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
- Multi-strategy signal aggregation with configurable consensus.
- Optional LLM pipeline:
  - Primary model: DeepSeek (`LLM_PRIMARY_MODEL`, default `deepseek-reasoner`).
  - Secondary model: OpenAI (`LLM_SECONDARY_MODEL`, default `gpt-5-mini`).
  - Safe fallback to rule-based behavior.

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
trading_bot/
├── bot/                            # Core runtime, risk, policy, data, execution
├── bot/backtesting/                # Modular backtesting engine
├── bot/observability/              # Telemetry + alerting
├── bot/data_pipeline/              # PIT dataset and regime dataset tooling
├── api/main.py                     # FastAPI control plane
├── research/                       # Experiment registry, benchmarks, gap closure
├── scripts/                        # Evidence builders and rollout gate scripts
├── tests/                          # Pytest suites
├── docs/production_execution/      # Tracker, worklog, gate checklist, runbooks
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
# or
python -m bot.main
```

### 4) Run the API

```bash
cd api
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`.

## How to Use

### Start/Stop Trading via API

```bash
curl -X POST http://localhost:8000/trading/start \
  -H "X-API-Key: <admin-key>"

curl -X POST http://localhost:8000/trading/stop \
  -H "X-API-Key: <admin-key>"
```

### Run a Backtest via API

```bash
curl -X POST http://localhost:8000/backtest/run \
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
curl -X POST http://localhost:8000/rollout/quality/evaluate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <admin-key>" \
  -d '{"validation_summary": {"walk_forward_folds": 5}}'

# Stage gates
curl -X POST http://localhost:8000/rollout/evaluate/shadow \
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
venv/bin/python scripts/build_g0_evidence.py
venv/bin/python scripts/build_gap_closure_evidence.py
venv/bin/python scripts/check_rollout_gate.py --help
```

## Testing

Run full suite:

```bash
venv/bin/pytest -q
```

Run key focused suites:

```bash
venv/bin/pytest tests/test_api.py -v
venv/bin/pytest tests/test_main.py -v
venv/bin/pytest tests/test_deploy_policy.py -v
venv/bin/pytest tests/test_gap_closure.py -v
venv/bin/pytest tests/test_validation.py -v
venv/bin/pytest tests/test_execution_simulation.py -v
```

Coverage:

```bash
venv/bin/pytest --cov=bot --cov=api
```

## Key Configuration Areas

Use `.env.example` as the complete reference. High-impact sections:

- Trading mode and shorting:
  - `TRADE_MODE`
  - `ENABLE_FUTURES_SHORTS`
  - `MAX_SHORT_NOTIONAL_USD`
  - `MAX_SHORT_LEVERAGE`
- Risk engine:
  - `ENABLE_RISK_ENGINE`
  - `RISK_ENGINE_MAX_DRAWDOWN_PCT`
  - `RISK_ENGINE_DAILY_LOSS_LIMIT_USD`
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
