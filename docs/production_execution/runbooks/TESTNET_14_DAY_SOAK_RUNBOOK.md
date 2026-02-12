# 14-Day Testnet Soak Runbook (CogniTrade)

This runbook is the required path before moving to live capital.
It is designed for a remote server running continuously for 14 days on Binance Testnet.

## 1) Objective

Prove production-like stability and risk behavior on testnet using frozen strategy/config, with hard gate checks and evidence capture.

Success condition:
1. 14 continuous days completed on testnet.
2. No unresolved critical incidents.
3. Quality, rollout, risk, and reconciliation constraints remain within limits.
4. Final go/no-go package is green.

## 2) Frozen Candidate for Soak

Use this exact strategy package for the soak window:
1. Candidate artifact: `output/selected_candidate_don_7_10_latest.json`
2. Frozen config artifact: `output/frozen_deploy_config_don_7_10_latest.json`
3. Rollout evidence baseline: `output/rollout_evidence_don_7_10_latest.json`
4. Go/no-go baseline: `output/go_no_go_don_7_10_latest.json`

Freeze rule:
1. Do not change strategy parameters, risk limits, or rollout thresholds mid-soak.
2. If any material config change is required, restart the soak clock from day 0.

## 3) Required Constraints During Soak

### 3.1 Quality Gate Constraints

Must remain compliant with configured quality thresholds:
1. `QUALITY_MIN_WALK_FORWARD_FOLDS >= 3`
2. `QUALITY_MIN_SHARPE_RATIO >= 0.20`
3. `QUALITY_MIN_CALMAR_RATIO >= 0.10`
4. `QUALITY_MAX_DRAWDOWN_PCT <= 25.0`
5. Regime consistency thresholds (`QUALITY_MIN_REGIME_*`) remain satisfied.

### 3.2 Promotion Benchmark Constraints

Must remain compliant with benchmark thresholds:
1. `BENCHMARK_MIN_TOTAL_TRADES >= 25`
2. `BENCHMARK_MIN_NET_RETURN_PCT >= 1.0`
3. `BENCHMARK_MIN_SHARPE_RATIO >= 0.20`
4. `BENCHMARK_MIN_CALMAR_RATIO >= 0.10`
5. `BENCHMARK_MAX_DRAWDOWN_PCT <= 25.0`

### 3.3 Rollout Gate Constraints

Use configured rollout gate constraints:
1. Shadow: `sample_count >= 50`, `error_rate <= 0.20`
2. Canary: `sample_count >= 30`, `error_rate <= 0.15`, `drawdown_pct <= 8.0`, `latency_p95_ms <= 3000`, `total_return_pct >= -1.0`
3. Production gate requires shadow + canary pass and quality gate pass (`ROLLOUT_REQUIRE_QUALITY_GATE=True`).

### 3.4 Operational Constraints

1. No persistent error-rate/latency breaches over 15 minutes.
2. No unresolved reconciliation position mismatches above tolerance.
3. No kill-switch activation caused by strategy instability.
4. Restart recovery works without orphaned risk state.

## 4) Remote Server Deployment (Ubuntu Example)

### 4.1 Server Baseline

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install git python3 python3-venv python3-pip tmux jq curl
sudo useradd --system --create-home --shell /bin/bash cognitrade || true
```

### 4.2 App Install

```bash
sudo -u cognitrade -H bash -lc '
cd ~ &&
if [ ! -d trading_bot ]; then
  git clone <YOUR_REPO_URL> trading_bot
fi &&
cd trading_bot &&
python3 -m venv venv &&
source venv/bin/activate &&
pip install -r requirements.txt
'
```

### 4.3 Environment Setup

```bash
sudo -u cognitrade -H bash -lc '
cd ~/trading_bot &&
cp -n .env.example .env
'
```

Set these minimum soak values in `.env`:
1. `TESTNET=True`
2. `ENABLE_LIVE_TRADING=False`
3. `TRADE_MODE=FUTURES`
4. `ENABLE_FUTURES_SHORTS=True`
5. `API_AUTH_ENABLED=True`
6. `API_ADMIN_KEYS=<strong-admin-key>`
7. `API_READ_KEYS=<strong-read-key>`
8. `ENABLE_ROLLOUT_GATES=True`
9. `ROLLOUT_ENFORCE_PRODUCTION_GATE=True`
10. `ROLLOUT_REQUIRED_ID=don-7-10-soak-YYYYMMDD`
11. `ENABLE_STRATEGY_QUALITY_GATE=True`
12. `BENCHMARK_REQUIRE_QUALITY_GATE=True`

Also configure your Binance testnet credentials and bind API to localhost or private network only.

## 5) Keep It Running Continuously (systemd)

Preferred: install the versioned unit files from this repository:

```bash
sudo cp deploy/systemd/cognitrade-bot.service /etc/systemd/system/
sudo cp deploy/systemd/cognitrade-api.service /etc/systemd/system/
```

If you need to customize paths/users, edit the unit files under `deploy/systemd/` and then copy them to `/etc/systemd/system/`.

Reference template for bot service:

```ini
[Unit]
Description=CogniTrade Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=cognitrade
WorkingDirectory=/home/cognitrade/trading_bot
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=/home/cognitrade/trading_bot/.env
ExecStart=/home/cognitrade/trading_bot/venv/bin/python -m bot.main
Restart=always
RestartSec=10
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=read-only
ReadWritePaths=/home/cognitrade/trading_bot
StandardOutput=append:/home/cognitrade/trading_bot/logs/cognitrade-bot.service.log
StandardError=append:/home/cognitrade/trading_bot/logs/cognitrade-bot.service.log

[Install]
WantedBy=multi-user.target
```

Reference template for API service:

```ini
[Unit]
Description=CogniTrade API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=cognitrade
WorkingDirectory=/home/cognitrade/trading_bot/api
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=/home/cognitrade/trading_bot/.env
ExecStart=/home/cognitrade/trading_bot/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=read-only
ReadWritePaths=/home/cognitrade/trading_bot
StandardOutput=append:/home/cognitrade/trading_bot/logs/cognitrade-api.service.log
StandardError=append:/home/cognitrade/trading_bot/logs/cognitrade-api.service.log

[Install]
WantedBy=multi-user.target
```

Enable/start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now cognitrade-bot.service
sudo systemctl enable --now cognitrade-api.service
sudo systemctl status cognitrade-bot.service --no-pager
sudo systemctl status cognitrade-api.service --no-pager
```

## 6) Day 0 Validation Checklist (Go Live on Testnet)

1. Run local regression from repo root:
```bash
./scripts/run_tests_local.sh -q
```
Expected: green suite.

2. Health check:
```bash
curl -s http://127.0.0.1:8000/health | jq .
```

3. Verify authenticated API access:
```bash
curl -s http://127.0.0.1:8000/observability/dashboard?window_minutes=60 \
  -H "X-API-Key: <read-key>" | jq .
```

4. Verify rollout id is fixed and stage gates are callable:
```bash
python scripts/check_rollout_gate.py --help
```

5. Snapshot baseline artifacts to immutable folder:
```bash
mkdir -p output/soak_baseline
cp output/selected_candidate_don_7_10_latest.json output/soak_baseline/
cp output/frozen_deploy_config_don_7_10_latest.json output/soak_baseline/
cp output/go_no_go_don_7_10_latest.json output/soak_baseline/
```

## 7) Daily Operating Procedure (Days 1-14)

Run once per day (same UTC hour preferred):

1. Run the automated daily collector:
```bash
./scripts/soak_daily_check.sh --strict
```

2. Service uptime and restarts:
```bash
systemctl show cognitrade-bot.service -p ActiveState -p NRestarts
systemctl show cognitrade-api.service -p ActiveState -p NRestarts
```

3. Pull observability snapshot:
```bash
curl -s "http://127.0.0.1:8000/observability/dashboard?window_minutes=1440" \
  -H "X-API-Key: <read-key>" \
  > output/soak_day_$(date -u +%Y%m%d)_dashboard.json
```

4. Pull recent events:
```bash
curl -s "http://127.0.0.1:8000/observability/events?limit=500" \
  -H "X-API-Key: <read-key>" \
  > output/soak_day_$(date -u +%Y%m%d)_events.json
```

5. Confirm rollout status:
```bash
curl -s "http://127.0.0.1:8000/rollout/status/${ROLLOUT_REQUIRED_ID}" \
  -H "X-API-Key: <admin-key>" \
  > output/soak_day_$(date -u +%Y%m%d)_rollout_status.json
```

6. Record reconciliation/risk alerts from DB/logs.

7. Append daily summary to `docs/production_execution/WORKLOG.md`:
   - uptime
   - restarts
   - error_rate
   - p95 latency
   - drawdown
   - any incidents
   - pass/fail against constraints

## 8) Weekly Gate Checks (Day 7 and Day 14)

1. Run strategy quality evaluation endpoint.
2. Re-run representative backtest against frozen candidate window/process.
3. Re-check promotion metrics vs benchmark thresholds.
4. If any quality/benchmark regression appears, stop and investigate.

## 9) Auto-Stop / Rollback Triggers During Soak

Immediately pause forward progression (no production promotion) if any trigger occurs:
1. Shadow/canary gate failure.
2. Quality gate regression.
3. Risk-engine kill-switch activation not attributable to planned drill.
4. Reconciliation position mismatch above tolerance.
5. Persistent latency/error-rate breach over 15 minutes.

Escalation actions:
1. Open incident entry in `docs/production_execution/WORKLOG.md`.
2. Capture last 60 minutes of logs and observability snapshots.
3. Fix root cause.
4. Restart soak from day 0 if behavior/parameters changed materially.

## 10) Promotion to Production (After Day 14)

Only after full soak success:
1. Keep rollout gates enabled.
2. Set production env:
   - `TESTNET=False`
   - `ENABLE_LIVE_TRADING=True`
3. Keep `ROLLOUT_ENFORCE_PRODUCTION_GATE=True`.
4. Execute capital ramp exactly as defined in `output/g2_02_capital_ramp_plan_latest.json`:
   - Stage 1: 1% for 24h
   - Stage 2: 5% for 24h
   - Stage 3: 15% for 48h
   - Stage 4: 30% for 72h
   - Stage 5: 50% for 96h
   - Stage 6: 100% for 120h

Do not skip stages.

## 11) Security and Reliability Recommendations

1. Restrict inbound access with firewall (allow SSH and API only from trusted IPs).
2. Keep API keys out of shell history and scripts.
3. Back up `.env`, `data/`, and `output/` daily.
4. Enable log rotation for `logs/*.log`.
5. Pin deployment to a git commit hash for soak and promotion.

## 12) Quick Command Bundle

```bash
# from project root on server
./scripts/run_tests_local.sh -q
./scripts/soak_daily_check.sh --strict
curl -s http://127.0.0.1:8000/health | jq .
curl -s "http://127.0.0.1:8000/observability/dashboard?window_minutes=60" -H "X-API-Key: <read-key>" | jq .
curl -s "http://127.0.0.1:8000/rollout/status/${ROLLOUT_REQUIRED_ID}" -H "X-API-Key: <admin-key>" | jq .
sudo systemctl status cognitrade-bot.service --no-pager
sudo systemctl status cognitrade-api.service --no-pager
```
