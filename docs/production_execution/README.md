# Production Execution Folder

This folder is the source of truth for implementing production-readiness work.

## Files

1. `IMPLEMENTATION_TRACKER.md`: task status, scope, dependencies, acceptance criteria.
2. `TEST_MATRIX.md`: exact test commands and phase gates.
3. `WORKLOG.md`: append-only execution evidence and session outcomes.
4. `P3_SENTIMENT_NEWS_PLAN.md`: research-backed implementation plan for `P3-01` to `P3-05`.
5. `GO_NO_GO_CHECKLIST.md`: current production-readiness verdict and blocking checklist.
6. `GAP_CLOSURE_PLAN.md`: comprehensive non-P3 gap-remediation plan to move from `NO-GO` to controlled `GO`.

## Standard Execution Cycle

1. Pick 1-2 task IDs from `IMPLEMENTATION_TRACKER.md`.
2. Set those IDs to `in_progress`.
3. Implement only scoped changes for those IDs.
4. Run required tests from `TEST_MATRIX.md`.
5. Fix failures or mark task `blocked` with reason.
6. If acceptance criteria are met, set task to `done`.
7. Append command/results evidence to `WORKLOG.md`.

## Prompt Template For Driving Implementation

Use this format for each turn:

```text
Execute: <TASK_ID[,TASK_ID]>
Scope guardrails: only edit files needed for these tasks.
Testing: run required TEST_MATRIX commands + `venv/bin/pytest -q` if task touches shared flow.
Tracking updates: update IMPLEMENTATION_TRACKER.md and WORKLOG.md before finishing.
Output format: summarize changed files, test commands, pass/fail, and remaining risks.
```

## Escalation Rule

If a task cannot be completed without architectural decisions or missing credentials:

1. Mark task `blocked`.
2. Record exact blocker in `WORKLOG.md`.
3. Propose smallest unblocking options.
