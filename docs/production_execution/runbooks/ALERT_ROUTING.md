# Alert Routing

## Primary Channels
- Critical: PagerDuty `CogniTrade-Primary`, Slack `#ops-critical`.
- High: Slack `#ops-trading`, Jira incident project.
- Medium/Low: Daily digest to `#ops-observability`.

## Routing Rules
1. Latency/error-rate alerts route to on-call engineer immediately.
2. Risk-engine kill-switch alerts escalate to trading lead within 5 minutes.
3. Reconciliation mismatches above tolerance trigger incident bridge.