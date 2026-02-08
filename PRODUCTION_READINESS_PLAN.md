# Trading Bot Production Readiness Plan

## 1) Executive Summary

This project is **not production-ready** for live trading as of **February 8, 2026**.

The main blockers are:

1. API endpoints that can return mock/default data with `status: success` on real failures.
2. API-to-core contract breaks (missing strategy methods, missing LLM method, database signature mismatches).
3. Incomplete risk controls in live execution (exposure/limits/stop handling not consistently enforced).
4. Backtesting realism issues that can overstate strategy quality.
5. Security and reliability hardening gaps (CORS/auth/rate-limits/timeouts/retries/observability).

This document defines a staged, production-focused plan to remediate those issues quickly while reducing custom surface area by adopting industry-standard tooling.

---

## 2) Audit-Derived Findings To Address First

### Critical

1. Mock/fallback success responses in API endpoints:
   - `api/main.py:167`
   - `api/main.py:196`
   - `api/main.py:229`
   - `api/main.py:411`
   - `api/main.py:450`
   - `api/main.py:517`
   - `api/main.py:540`
   - `api/main.py:572`
2. Backtest endpoint calls strategy functions that do not exist in `bot/strategy.py`:
   - `api/main.py:345`
   - `api/main.py:348`
   - `api/main.py:357`
   - `api/main.py:360`
   - Existing strategy functions are in `bot/strategy.py:11` and `bot/strategy.py:106`.
3. LLM endpoint calls a non-existent public method:
   - `api/main.py:505`
   - Available method is private: `bot/llm_manager.py:606`.
4. DB API contract mismatch and serialization mismatch:
   - `api/main.py:534`
   - `api/main.py:536`
   - `api/main.py:568`
   - DB method signatures are `bot/database.py:298` and `bot/database.py:348`.
5. Unsafe default mode:
   - `bot/config.py:12` defaults `TESTNET` to false-equivalent when env is missing.

### High

1. Risk controls not enforced through execution path:
   - Config exists at `bot/config.py:90`, `bot/config.py:91`, `bot/config.py:93`, `bot/config.py:95`.
   - Execution flow in `bot/main.py:270`, `bot/main.py:273`, `bot/main.py:293`, `bot/main.py:346`.
2. Trade ID collision risk and DB uniqueness failures:
   - `bot/order_manager.py:105`.
3. Incomplete exchange filter handling:
   - `bot/binance_api.py:162`.
4. Backtesting realism and metric distortions:
   - `bot/backtesting/core/engine.py:615`
   - `bot/backtesting/core/engine.py:651`
   - `bot/backtesting/core/engine.py:927`
   - `bot/backtesting/core/engine.py:957`
   - `bot/backtesting/core/engine.py:1150`
   - `bot/backtesting/core/engine.py:1177`
   - `bot/backtesting/models/results.py:276`
5. CORS misconfiguration for credentialed use:
   - `api/main.py:48` to `api/main.py:50`.

### Medium

1. LLM transport resilience and logging safety:
   - No explicit timeout/retry at `bot/llm_manager.py:211`, `bot/llm_manager.py:326`, `bot/llm_manager.py:481`.
   - Logs full model outputs/context at `bot/llm_manager.py:231`, `bot/llm_manager.py:828`.
2. Tests currently mask production breakages through monkeypatch-heavy API tests:
   - `tests/test_api.py:22`
   - `tests/test_api.py:59`
   - `tests/test_api.py:351`
   - `tests/test_api.py:385`
   - `tests/test_api.py:480`
   - `tests/test_api.py:505`
   - `tests/test_api.py:543`

---

## 3) Target Production Architecture

### Recommended near-term direction

1. Keep FastAPI as orchestration and observability surface.
2. Move/standardize live execution and backtesting on a proven crypto framework.
3. Reduce custom exchange logic and custom backtest engine responsibilities.

### Package adoption plan

1. **Execution + backtesting (preferred):** Freqtrade.
2. **Research/rapid prototyping:** vectorbt or backtesting.py.
3. **Exchange abstraction:** CCXT for portability, or Binance official connector for Binance-specific production path.
4. **Optional advanced future path:** NautilusTrader or LEAN if strict backtest/live parity and multi-asset architecture become required.

---

## 4) Reinvented Components To Replace or Minimize

1. Custom backtest engine performance metrics and bias controls.
2. Custom exchange filter compliance logic.
3. Custom live order lifecycle/idempotency and reconciliation.
4. Custom optimization workflow without robust anti-overfit checks.

Keep custom code where it differentiates:

1. Signal logic and strategy IP.
2. Domain-specific risk constraints.
3. Reporting and operational workflows specific to your trading process.

---

## 5) Phased Implementation Roadmap

## Phase 0: Immediate Safety Patch (Week 1)

Goal: Eliminate dangerous behavior and restore truthful API semantics.

Tasks:

1. Remove all API fake/mock success fallbacks in non-test code.
2. Return explicit `4xx/5xx` with structured error payloads and request IDs.
3. Fix API method contract mismatches:
   - Strategy endpoint references.
   - LLM rule-based call path.
   - DB method signatures and response serialization.
4. Change runtime safety defaults:
   - `TESTNET=True` by default.
   - Require explicit `LIVE_TRADING_ENABLED=true` to place live orders.
5. Add kill-switch env var (global emergency stop).

Exit criteria:

1. No production endpoint returns fabricated success data.
2. Backtest and LLM endpoints function with real methods and error correctly when upstream fails.
3. Live mode cannot be entered accidentally.

## Phase 1: Risk & Execution Hardening (Week 2-3)

Goal: Enforce risk invariants before any live pilot.

Tasks:

1. Enforce max order notional and max position exposure.
2. Implement stop-loss/take-profit and daily loss limits at execution layer.
3. Add idempotent trade IDs (UUIDv4/ULID) and deterministic order reconciliation.
4. Implement complete exchange pre-trade validation:
   - quantity step/min
   - price tick
   - min notional/notional checks
5. Add timeout/retry/backoff/circuit-breaker for LLM and exchange HTTP paths.

Exit criteria:

1. Pre-trade risk checks are mandatory and tested.
2. Duplicate insertions/retries cannot create inconsistent trade state.
3. Exchange rejects due preventable filter violations are near-zero.

## Phase 2: Backtesting Standardization (Week 3-5)

Goal: Replace fragile custom assumptions with trusted tooling.

Tasks:

1. Stand up Freqtrade backtesting and dry-run environment.
2. Port top 1-2 strategies into Freqtrade strategy format.
3. Add walk-forward and lookahead-bias checks.
4. Re-run benchmark backtests with realistic fees/slippage assumptions.
5. Keep existing custom engine only for legacy comparison until cutover.

Exit criteria:

1. Strategy results are reproducible on standardized tooling.
2. Bias checks pass.
3. A documented go/no-go threshold exists for each strategy.

## Phase 3: Security, Ops, and SRE Controls (Week 4-6)

Goal: Production-grade operations.

Tasks:

1. Replace permissive CORS with explicit trusted origins.
2. Add API authn/authz and rate limiting.
3. Add structured logging, tracing correlation IDs, metrics dashboards, and alerting.
4. Create runbooks:
   - incident response
   - exchange outage behavior
   - emergency unwind process
5. Establish secrets management and key rotation policy.

Exit criteria:

1. Operational alarms cover order failures, latency spikes, and risk breaches.
2. Security controls are validated in staging.
3. Team has tested runbooks for emergency scenarios.

## Phase 4: Staged Release (Week 6+)

Goal: Controlled production rollout.

Tasks:

1. Paper-trade/dry-run burn-in.
2. Shadow mode with production market data and zero capital risk.
3. Small-capital canary with strict drawdown stop.
4. Progressive scale-up only if SLOs and risk guardrails remain green.

Exit criteria:

1. Canary period completes with no severity-1 incidents.
2. Drawdown and error rates within predefined limits.
3. Leadership sign-off for scaling.

---

## 6) Acceptance Gates (Must Pass)

### Engineering gates

1. No mocked fallback behavior in production handlers.
2. 100% of trade-placement paths pass pre-trade risk and exchange filter validation.
3. Order lifecycle is idempotent across retries and process restarts.
4. API error model is consistent and machine-readable.

### Testing gates

1. Add integration tests for real endpoint contracts (not only monkeypatch mocks).
2. Add end-to-end dry-run scenario tests from signal -> order -> persistence -> API readback.
3. Add failure-path tests for exchange timeout, LLM timeout, DB write failure.
4. Backtest reproducibility tests with fixed seed/data snapshots.

### Operations gates

1. Metrics and alerts active in staging and production.
2. Kill switch validated in live-like environment.
3. On-call runbooks tested with tabletop exercises.

---

## 7) Dependency and Framework Strategy

### Upgrade policy

1. Introduce dependency pinning strategy with periodic upgrade windows.
2. Prioritize upgrades for:
   - FastAPI/Starlette/Pydantic
   - requests/httpx/urllib3
   - numpy/pandas/scipy
   - exchange and LLM SDK dependencies
3. Add compatibility CI matrix before major-version upgrades.

### Framework decision checkpoint

Decision date target: end of Phase 1.

Decision questions:

1. Is Freqtrade sufficient for current strategy family and execution requirements?
2. Is multi-exchange portability needed now (CCXT), or Binance-only for next 2 quarters?
3. Is strict backtest/live engine parity worth moving to Nautilus/LEAN in current timeline?

---

## 8) Roles and Ownership Model

Suggested owners:

1. API contract and safety defaults: backend lead.
2. Risk engine and execution safeguards: trading systems lead.
3. Backtesting migration: quant/dev lead.
4. Security and observability: platform/SRE lead.
5. Release governance and canary sign-off: product + trading owner.

---

## 9) Immediate Work Queue (First 10 Tickets)

1. Remove mock success fallbacks from all API handlers.
2. Fix `/backtest/run` strategy function binding to existing strategy API.
3. Fix `/llm/decision` rule-based call path and method naming consistency.
4. Fix `/database/trades` and `/database/signals` DB method signatures and response serialization.
5. Add `LIVE_TRADING_ENABLED` hard gate and safe defaults.
6. Enforce max order notional and max position exposure checks.
7. Implement UUID/ULID trade IDs and retry-safe persistence.
8. Add complete Binance filter pre-validation.
9. Add HTTP timeout/retry/circuit breaker wrapper for external calls.
10. Add integration tests validating true endpoint behavior without monkeypatch stubs.

---

## 10) Go-Live Definition

Production go-live is permitted only when all are true:

1. Phase 0 and Phase 1 exit criteria are met.
2. Phase 2 standardized backtest and bias checks are completed.
3. Dry-run/shadow/canary sequence is completed without critical incidents.
4. Formal sign-off from engineering, trading, and operations owners.

---

## 11) External References Used For Package Strategy

1. Freqtrade docs: https://docs.freqtrade.io/en/latest/
2. Freqtrade lookahead analysis: https://docs.freqtrade.io/en/latest/lookahead-analysis/
3. vectorbt docs: https://vectorbt.dev/
4. backtesting.py docs: https://kernc.github.io/backtesting.py/
5. NautilusTrader docs: https://nautilustrader.io/docs/latest/getting_started/
6. QuantConnect LEAN: https://github.com/QuantConnect/Lean
7. CCXT manual: https://github.com/ccxt/ccxt/wiki/Manual
8. Binance spot filters: https://developers.binance.com/docs/binance-spot-api-docs/filters
9. Binance Spot testnet FAQ: https://developers.binance.com/docs/binance-spot-api-docs/faqs/testnet
10. FastAPI CORS docs: https://fastapi.tiangolo.com/tutorial/cors/

