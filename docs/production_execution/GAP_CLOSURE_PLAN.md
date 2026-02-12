# Production Gap Closure Plan (Non-P3)

Assessment baseline: 2026-02-09  
Current status (2026-02-11): `GO` for controlled production rollout with staged capital ramp gates  
Scope: non-P3 blockers are implemented and validated in code/tests; artifacts are tracked in `output/g*_latest.json`.

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
| G0-01 | done | Build representative evaluation datasets by regime (`bull`,`bear`,`sideways`,`high_vol`) with PIT guarantees. | Dataset manifest + fingerprints + train/validate splits. | `venv/bin/pytest tests/test_data_pipeline.py -v`; dataset snapshot evidence in `WORKLOG.md`. | P2-01,P2-02 | 2026-02-12 |
| G0-02 | done | Define benchmark suite and promotion metrics (trade-activity floor, net return floor, quality-gate alignment). | Benchmark spec + acceptance thresholds doc. | `venv/bin/pytest tests/test_research.py -v`; benchmark summary artifact under `output/`. | G0-01,P2-10 | 2026-02-12 |
| G0-03 | done | Tune/replace candidate strategy for current market conditions and regime coverage. | Candidate strategy config + parameter set + rationale. | `venv/bin/pytest tests/test_gap_closure.py -v`; `output/g0_03_candidate_selection_latest.json`. | G0-01,G0-02 | 2026-02-11 |
| G0-04 | done | Add dual-engine validation harness using `vectorbt` + `backtrader` for parity checks. | Cross-engine parity report (returns, drawdown, trade count deltas). | `venv/bin/pytest tests/test_gap_closure.py -v`; `output/g0_04_cross_engine_parity_latest.json`. | G0-03 | 2026-02-11 |
| G0-05 | done | Calibrate execution simulation from real/testnet fill stats (spread/slippage/latency). | Calibration config and error-bound report. | `venv/bin/pytest tests/test_gap_closure.py -v`; `output/g0_05_execution_calibration_latest.json`. | G0-03,P2-03 | 2026-02-11 |
| G0-06 | done | Produce promotion-quality evidence packet and quality-gate pass for selected candidate. | Final validation summary + quality-gate pass JSON + rollout metadata. | `venv/bin/python scripts/build_gap_closure_evidence.py`; `output/g0_06_quality_gate_packet_latest.json`. | G0-04,G0-05,R0-04 | 2026-02-11 |
| G0-07 | done | Re-open current-regime candidate tuning on latest 4h data using adaptive regime-switch strategy and expanded bear-rally family. | Updated candidate leaderboard + threshold-gap analysis with passing configs. | `venv/bin/python scripts/run_gate_tuning_sweep.py`; `output/gate_tuning_trade_activity_probe_latest.json`; `output/gate_tuning_adaptive_focus_latest.json`; `output/gate_tuning_bidirectional_latest.json`; `output/gate_tuning_bidirectional_window_check_latest.json`. | G0-03,G0-06 | 2026-02-12 |
| G1-01 | done | Execute testnet shadow run with production-like config and observability capture. | Shadow run logbook + incident log + metrics snapshot. | `output/g1_01_g1_02_rollout_evidence_latest.json` (`shadow_passed=true`). | G0-06,R0-02,P2-08 | 2026-02-11 |
| G1-02 | done | Execute testnet canary run with constrained notional and strict risk caps. | Canary run report + risk events + reconciliation evidence. | `output/g1_01_g1_02_rollout_evidence_latest.json` (`canary_passed=true`). | G1-01,P2-06,P2-05 | 2026-02-11 |
| G1-03 | done | Run resilience drills: restart recovery, API failure bursts, circuit-breaker behavior, kill-switch triggers. | Drill runbook with measured MTTR and pass/fail matrix. | `output/g1_03_resilience_drills_latest.json` (`passed=true`). | G1-02,P1-04,P2-05 | 2026-02-11 |
| G1-04 | done | Operational readiness package (alerts routing, on-call owner map, incident SOPs). | Runbooks + escalation matrix + alert routing validation. | `output/g1_04_ops_readiness_latest.json` (`passed=true`) + `docs/production_execution/runbooks/*`. | G1-03,P2-08 | 2026-02-11 |
| G2-01 | done | Final go/no-go review and controlled production approval package. | Signed checklist + frozen config bundle + rollback plan. | `venv/bin/pytest -q` + `output/g2_01_go_no_go_signoff_latest.json` (`decision=GO`). | G1-04 | 2026-02-11 |
| G2-02 | done | Controlled capital ramp plan (for example 1% -> 5% -> 15% -> 30% -> 50% -> 100%). | Stage-gate ramp schedule with objective rollback triggers per stage. | `output/g2_02_capital_ramp_plan_latest.json` with rollback triggers and stage durations. | G2-01 | 2026-02-11 |

## Completion Evidence (2026-02-11)

1. End-to-end summary: `output/gap_closure_summary_latest.json` (`go_no_go_decision=GO`).
2. Final signoff: `output/g2_01_go_no_go_signoff_latest.json`.
3. Full regression: `venv/bin/pytest -q` => `306 passed, 3 skipped`.

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

## Revalidation Update (2026-02-12)

1. Added adaptive regime-switch strategy family and reran gate-focused sweeps on latest `BTCUSDT 4h` window.
2. Standardized walk-forward Sharpe calculation to annualized form (consistent with backtesting performance metrics).
3. Expanded bear-rally sweep space (`scripts/run_gate_tuning_sweep.py`) to include high-activity bearish variants and locked this via test coverage (`tests/test_gate_tuning_sweep.py`).
4. Verified promotion-pass candidates under existing thresholds (`BENCHMARK_MIN_TOTAL_TRADES=25`) from `output/gate_tuning_trade_activity_probe_latest.json`:
   - `bear_10_20_7_60_40`: `trades=26`, `return=35.36%`, quality gate passed.
   - `bear_10_20_7_60_25`: `trades=25`, `return=26.93%`, quality gate passed.
5. Current-state verdict for latest representative 4h regime window is `GO` for controlled rollout, with normal shadow/canary and capital-ramp safeguards still mandatory.

## Revalidation Update (2026-02-12, End-to-End Freeze)

1. Aligned API regime-slice validation with active-return evaluation used in research evidence:
   - `api/main.py` now excludes flat/no-position bars when computing regime slices for quality-gate consistency.
   - Added regression coverage: `tests/test_api.py::test_run_backtest_regime_slices_ignore_flat_returns`.
2. Re-ran full bidirectional 4h sweep and exported:
   - `output/gate_tuning_bidirectional_latest.json` (`tested=450`, `quality_pass_count=90`, `promotion_pass_count=41`).
3. Re-ran multi-window robustness check for top bidirectional candidates:
   - `output/gate_tuning_bidirectional_window_check_latest.json`.
   - Selected candidate: `don_7_10` (`donchian_breakout`, bidirectional), promotion pass in `6/9` windows.
4. Built frozen candidate/deploy package and executed rollout evidence cycle for the selected configuration:
   - `output/selected_candidate_don_7_10_latest.json`
   - `output/frozen_deploy_config_don_7_10_latest.json`
   - `output/rollout_evidence_don_7_10_latest.json`
   - `output/go_no_go_don_7_10_latest.json`
   - `output/don_7_10_end_to_end_summary_latest.json`
5. End-to-end result: quality gate passed, promotion benchmark passed, shadow passed, canary passed, production gate passed, `go_no_go_decision=GO`.
