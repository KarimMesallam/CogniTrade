# Incident SOP

## Trigger Conditions
1. Production rollout gate failure.
2. Kill-switch activation.
3. Exchange API degradation or reconciliation divergence.

## Immediate Actions
1. Confirm kill-switch state and pause new risk-increasing orders.
2. Capture observability dashboard snapshot and rollout status.
3. Start incident timeline with UTC timestamps.
4. Execute reconciliation and restart recovery runbook steps.

## Exit Criteria
- Root cause identified and mitigated.
- Rollout gate returns to compliant state.
- Postmortem ticket opened with action owners.