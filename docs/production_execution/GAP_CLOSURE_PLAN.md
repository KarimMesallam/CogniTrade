# Production Gap Closure Plan (Non-P3)

Assessment baseline: 2026-02-09  
Current status: `NO-GO` for unrestricted production capital deployment  
Scope: close all non-P3 blockers identified in `docs/production_execution/GO_NO_GO_CHECKLIST.md`

## Goals

1. Promote at least one strategy that passes quality gates on representative market regimes.
2. Prove operational stability on testnet through shadow and canary, including failure-recovery drills.
3. Complete operational readiness (alerts, runbooks, ownership) required for a controlled production cutover.

## Production Exit Criteria (All Required)

1. Quality gate passes for a promotion-candidate strategy:
   - walk-forward folds >= configured minimum
   - Sharpe and Calmar means above configured thresholds
   - regime consistency threshold satisfied
2. Profitability/robustness evidence package exists for candidate:
   - positive net return after modeled costs on evaluation window
   - minimum strategy activity floor met (avoid ultra-low-trade artifacts)
3. Testnet shadow + canary completed without unresolved critical incidents.
4. Reconciliation/restart/kill-switch drills pass with documented timings and outcomes.
5. Operational runbooks and alert routing are validated and owned.

## Recommended Industry-Standard Tooling Additions

1. `vectorbt`: fast parameter sweeps and independent vectorized backtest cross-checks.
2. `backtrader`: event-driven secondary engine validation against internal engine behavior.
3. `quantstats`: standardized performance/tearsheet reporting for promotion packets.

Notes:
- Keep internal engine as the execution source of truth.
- Use external engines to detect model or accounting drift and reduce self-confirmation risk.

## Task Plan

| Task ID | Status | Scope | Deliverables | Evidence / Tests | Depends On | Target Date |
|---|---|---|---|---|---|---|
| G0-01 | todo | Build representative evaluation datasets by regime (`bull`,`bear`,`sideways`,`high_vol`) with PIT guarantees. | Dataset manifest + fingerprints + train/validate splits. | `venv/bin/pytest tests/test_data_pipeline.py -v`; dataset snapshot evidence in `WORKLOG.md`. | P2-01,P2-02 | 2026-02-12 |
| G0-02 | todo | Define benchmark suite and promotion metrics (trade-activity floor, net return floor, quality-gate alignment). | Benchmark spec + acceptance thresholds doc. | `venv/bin/pytest tests/test_research.py -v`; benchmark summary artifact under `output/`. | G0-01,P2-10 | 2026-02-12 |
| G0-03 | todo | Tune/replace candidate strategy for current market conditions and regime coverage. | Candidate strategy config + parameter set + rationale. | `venv/bin/pytest tests/test_validation.py -v`; backtest artifacts for each regime window. | G0-01,G0-02 | 2026-02-14 |
| G0-04 | todo | Add dual-engine validation harness using `vectorbt` + `backtrader` for parity checks. | Cross-engine parity report (returns, drawdown, trade count deltas). | New parity tests (for example `tests/test_backtest_parity.py`) + regression gate. | G0-03 | 2026-02-15 |
| G0-05 | todo | Calibrate execution simulation from real/testnet fill stats (spread/slippage/latency). | Calibration config and error-bound report. | `venv/bin/pytest tests/test_execution_simulation.py -v`; calibration diff report. | G0-03,P2-03 | 2026-02-15 |
| G0-06 | todo | Produce promotion-quality evidence packet and quality-gate pass for selected candidate. | Final validation summary + quality-gate pass JSON + rollout metadata. | `venv/bin/pytest tests/test_api.py -k "rollout_quality or supports_short_execution_and_validation" -v`; successful `/rollout/quality/evaluate`. | G0-04,G0-05,R0-04 | 2026-02-16 |
| G1-01 | todo | Execute testnet shadow run with production-like config and observability capture. | Shadow run logbook + incident log + metrics snapshot. | `/rollout/evaluate/shadow` approved; observability dashboard artifact saved. | G0-06,R0-02,P2-08 | 2026-02-18 |
| G1-02 | todo | Execute testnet canary run with constrained notional and strict risk caps. | Canary run report + risk events + reconciliation evidence. | `/rollout/evaluate/canary` approved; `venv/bin/pytest tests/test_reconciliation.py -v`. | G1-01,P2-06,P2-05 | 2026-02-20 |
| G1-03 | todo | Run resilience drills: restart recovery, API failure bursts, circuit-breaker behavior, kill-switch triggers. | Drill runbook with measured MTTR and pass/fail matrix. | `venv/bin/pytest tests/test_main.py -k "risk_engine or rollout_production_gate" -v` and targeted failure-injection scripts. | G1-02,P1-04,P2-05 | 2026-02-21 |
| G1-04 | todo | Operational readiness package (alerts routing, on-call owner map, incident SOPs). | Runbooks + escalation matrix + alert routing validation. | Alert smoke-test evidence + checklist sign-off in `WORKLOG.md`. | G1-03,P2-08 | 2026-02-22 |
| G2-01 | todo | Final go/no-go review and controlled production approval package. | Signed checklist + frozen config bundle + rollback plan. | Re-run `venv/bin/pytest -q`; checklist in `GO_NO_GO_CHECKLIST.md` fully green. | G1-04 | 2026-02-23 |
| G2-02 | todo | Controlled capital ramp plan (for example 1% -> 5% -> 15% -> 30% -> 50% -> 100%). | Stage-gate ramp schedule with objective rollback triggers per stage. | Stage evidence in `WORKLOG.md`; each stage requires no gate violations for defined duration. | G2-01 | 2026-02-27 |

## Execution Rhythm

1. Run in small batches: `Execute: G0-01,G0-02`, then `G0-03,G0-04`, etc.
2. For each batch:
   - mark tasks `in_progress` in tracker
   - implement scoped changes only
   - run mapped tests + `venv/bin/pytest -q` for shared flow changes
   - update `WORKLOG.md` with command outputs and artifacts
   - mark `done` only if acceptance evidence is complete

## Risks and Mitigations

1. Risk: Overfitting to a single bearish sample.
   - Mitigation: multi-regime, multi-window validation and purged CV.
2. Risk: Backtest/live divergence from fill assumptions.
   - Mitigation: execution calibration and dual-engine parity checks.
3. Risk: Operational failures during cutover.
   - Mitigation: mandatory drills, explicit runbooks, and staged capital ramp.
