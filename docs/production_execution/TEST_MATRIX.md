# Test Matrix

## Command Conventions

- Use repo venv python: `venv/bin/python`
- Use repo venv pytest: `venv/bin/pytest`
- Run targeted tests per task, then run regression gate.

## Task-to-Test Mapping

| Task ID | Required Test Scope | Primary Command(s) | Gate |
|---|---|---|---|
| P0-01 | API error semantics and no fake success fallbacks | `venv/bin/pytest tests/test_api.py -k "error or fallback or database or llm or backtest" -v` | Must pass |
| P0-02 | Backtest endpoint strategy binding contract | `venv/bin/pytest tests/test_api.py -k "backtest" -v` | Must pass |
| P0-03 | LLM endpoint rule-based path + manager method tests | `venv/bin/pytest tests/test_api.py -k "llm" -v` and `venv/bin/pytest tests/test_llm_manager.py -v` | Must pass |
| P0-04 | DB endpoint response structure and filtering | `venv/bin/pytest tests/test_api.py -k "database_trades or database_signals" -v` and `venv/bin/pytest tests/test_database.py -v` | Must pass |
| P0-05 | Safe default mode and live enable gating | `venv/bin/pytest tests/test_main.py -v` | Must pass |
| P0-06 | Real endpoint contract integration checks | `venv/bin/pytest tests/test_api_integration.py -v` | Must pass |
| P1-01 | Risk guardrails at order path | `venv/bin/pytest tests/test_order_manager.py -v` and risk-specific tests | Must pass |
| P1-02 | Trade ID uniqueness and idempotent persistence | `venv/bin/pytest tests/test_order_manager.py -k "trade_id or unique or database" -v` and `venv/bin/pytest tests/test_database.py -k "upsert_trade_idempotent_on_retry" -v` and `venv/bin/pytest tests/test_db_integration.py -k "deterministic_id_from_order" -v` | Must pass |
| P1-03 | Binance filter compliance pre-validation | `venv/bin/pytest tests/test_binance_api.py -v` | Must pass |
| P1-04 | Timeout/retry/circuit-breaker behavior | `venv/bin/pytest tests/test_llm_manager.py -v` and `venv/bin/pytest tests/test_binance_api.py -k "retry and circuit" -v` | Must pass |
| P1-05 | Backtesting short position lifecycle and PnL accuracy | `venv/bin/pytest tests/test_backtesting.py -k "short" -v` and `venv/bin/pytest tests/test_vectorized_backtesting.py -k "short" -v` | Must pass |
| P1-06 | Futures short routing and trade-mode safety gating | `venv/bin/pytest tests/test_main.py -k "futures or short or trade_mode" -v` and `venv/bin/pytest tests/test_order_manager.py -k "futures or short" -v` | Must pass |
| P1-07 | Short-specific risk controls and reject reasons | `venv/bin/pytest tests/test_order_manager.py -k "short and risk" -v` and `venv/bin/pytest tests/test_main.py -k "short and risk" -v` | Must pass |
| P1-08 | Regime detection correctness and state persistence | `venv/bin/pytest tests/test_regime.py -v` and persistence-specific tests | Must pass |
| P1-09 | Regime-based strategy policy routing and sizing | `venv/bin/pytest tests/test_policy.py -v` and `venv/bin/pytest tests/test_main.py -k "regime or policy" -v` | Must pass |
| P1-10 | Switch-stability guards (hysteresis/cooldown/turnover limits) | `venv/bin/pytest tests/test_main.py -k "switch or hysteresis or cooldown" -v` and `venv/bin/pytest tests/test_order_manager.py -k "turnover" -v` | Must pass |
| P2-01 | Point-in-time data integrity and leakage prevention | `venv/bin/pytest tests/test_data_pipeline.py -v` and leakage-specific checks | Must pass |
| P2-02 | Walk-forward + purged CV + regime-slice validation | `venv/bin/pytest tests/test_validation.py -v` | Must pass |
| P2-03 | Execution simulation realism (spread/slippage/latency/funding) | `venv/bin/pytest tests/test_execution_simulation.py -v` | Must pass |
| P2-04 | Portfolio optimizer constraints and allocation correctness | `venv/bin/pytest tests/test_portfolio.py -v` | Must pass |
| P2-05 | Live risk engine invariants and kill-switch behavior | `venv/bin/pytest tests/test_risk_engine.py -v` | Must pass |
| P2-06 | Reconciliation/retry/restart consistency | `venv/bin/pytest tests/test_reconciliation.py -v` and `venv/bin/pytest tests/test_order_manager.py -k "reconcile or retry" -v` | Must pass |
| P2-07 | Edge-decay/drift detection and de-risk automation | `venv/bin/pytest tests/test_monitoring.py -v` | Must pass |
| P2-08 | Observability telemetry and alert-path coverage | `venv/bin/pytest tests/test_observability.py -v` and `venv/bin/pytest tests/test_api.py -k "request_id or telemetry" -v` | Must pass |
| P2-09 | Shadow and canary rollout guardrails | `venv/bin/pytest tests/test_deploy_policy.py -v` | Must pass |
| P2-10 | Research reproducibility and experiment tracking | `venv/bin/pytest tests/test_research.py -v` | Must pass |
| P3-01 | News/sentiment ingestion adapters and normalized persistence | `venv/bin/pytest tests/test_sentiment_pipeline.py -v` | Must pass |
| P3-02 | Sentiment feature engineering and point-in-time joins | `venv/bin/pytest tests/test_sentiment_features.py -v` | Must pass |
| P3-03 | Tool-calling sentiment decision flow and schema/fallback safety | `venv/bin/pytest tests/test_llm_manager.py -k "sentiment or tool" -v` and `venv/bin/pytest tests/test_sentiment_agent.py -v` | Must pass |
| P3-04 | Sentiment-aware policy routing with hard risk bounds | `venv/bin/pytest tests/test_policy.py -k "sentiment" -v` and `venv/bin/pytest tests/test_risk_engine.py -k "sentiment" -v` | Must pass |
| P3-05 | Sentiment ablation validation and rollout promotion gates | `venv/bin/pytest tests/test_validation.py -k "sentiment" -v` and `venv/bin/pytest tests/test_deploy_policy.py -k "sentiment or shadow or canary" -v` | Must pass |

Detailed P3 validation logic and promotion criteria: `docs/production_execution/P3_SENTIMENT_NEWS_PLAN.md`

## Phase Gates

### Phase 0 Gate

1. All `P0-*` tasks marked `done`.
2. Run: `venv/bin/pytest -q`
3. Run (optional but recommended): `venv/bin/pytest --cov=bot --cov=api`

### Phase 1 Gate

1. All `P1-*` tasks marked `done`.
2. Run: `venv/bin/pytest -q`
3. Run targeted resilience/risk suites and record outputs in `WORKLOG.md`.

### Phase 3 Gate

1. All `P3-*` tasks marked `done`.
2. Run: `venv/bin/pytest -q`
3. Run sentiment-specific suites and record ablation/shadow/canary evidence in `WORKLOG.md`.

## Evidence Policy

Each completed task must include:

1. Exact commands run.
2. Pass/fail summary.
3. If failures remain, task cannot move to `done`; use `blocked` with reason.
