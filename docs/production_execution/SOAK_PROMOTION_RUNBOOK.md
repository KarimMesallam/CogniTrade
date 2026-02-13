# Soak Promotion Runbook — ADX(10)_T25_C15 + DI Filter

**Strategy:** Trend-following, ADX regime-adaptive with DI direction filter
**Soak start:** 2026-02-13
**Earliest evaluation:** 2026-02-27 (14 days)
**Cron:** Daily soak check at 06:00 UTC → Telegram summary

---

## Pre-Evaluation: Collect Evidence

Run these commands and paste output into the sections below.

### 1. Bot health

```bash
sudo systemctl status cognitrade-bot --no-pager
```

### 2. Trade count and summary

```bash
cat /var/www/CogniTrade/order_logs/BTCUSDT_orders.json | python3 -c "
import json, sys
orders = json.load(sys.stdin)
print(f'Total orders: {len(orders)}')
buys = [o for o in orders if o.get('side') == 'BUY']
sells = [o for o in orders if o.get('side') == 'SELL']
print(f'BUY: {len(buys)}, SELL: {len(sells)}')
"
```

### 3. Soak daily summaries (last 7 days)

```bash
for d in $(ls -d /var/www/CogniTrade/output/soak_daily/*/  | tail -7); do
  f="${d}summary.json"
  if [ -s "$f" ]; then
    date=$(basename "$d")
    pass=$(python3 -c "import json; print(json.load(open('$f')).get('overall_pass','?'))")
    echo "$date: overall_pass=$pass"
  fi
done
```

### 4. Latest dashboard metrics

```bash
curl -s http://localhost:8001/observability/dashboard?window_minutes=20160 \
  -H "X-API-Key: $(grep API_READ_KEYS /var/www/CogniTrade/.env | head -1 | cut -d= -f2 | cut -d, -f1)" \
  | python3 -m json.tool
```

### 5. Service restarts

```bash
systemctl show cognitrade-bot.service -p NRestarts
systemctl show cognitrade-api.service -p NRestarts
```

---

## Gate Evaluation

### Pass/Fail Thresholds

| Gate | Threshold | Actual | Pass? |
|------|-----------|--------|-------|
| Total trades | >= 25 | ___ | |
| Net return | >= 1.0% | ___ | |
| Sharpe ratio | >= 0.20 | ___ | |
| Max drawdown | <= 8.0% | ___ | |
| Error rate | <= 15% | ___ | |
| P95 latency | <= 3000ms | ___ | |
| Bot restarts | 0 (or justified) | ___ | |
| Daily soak passes | all or near-all | ___ | |

### Run Shadow Evaluation

```bash
# Get metrics from dashboard first, then run:
curl -s -X POST http://localhost:8001/rollout/evaluate/shadow \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $(grep API_ADMIN_KEYS /var/www/CogniTrade/.env | head -1 | cut -d= -f2 | cut -d, -f1)" \
  -d '{
    "rollout_id": "adx10-t25-c15-di",
    "sample_count": <TOTAL_TRADES>,
    "error_rate": <ERROR_RATE_FROM_DASHBOARD>
  }' | python3 -m json.tool
```

### Run Canary Evaluation (if shadow passes)

```bash
curl -s -X POST http://localhost:8001/rollout/evaluate/canary \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $(grep API_ADMIN_KEYS /var/www/CogniTrade/.env | head -1 | cut -d= -f2 | cut -d, -f1)" \
  -d '{
    "rollout_id": "adx10-t25-c15-di",
    "sample_count": <TOTAL_TRADES>,
    "error_rate": <ERROR_RATE>,
    "total_return_pct": <NET_RETURN>,
    "drawdown_pct": <MAX_DRAWDOWN>,
    "latency_p95_ms": <P95_LATENCY>
  }' | python3 -m json.tool
```

---

## Decision

- [ ] All gates pass
- [ ] Shadow evaluation passes
- [ ] Canary evaluation passes
- [ ] No unexplained restarts or crash loops
- [ ] Daily Telegram summaries received consistently

### If all pass → Promote to live

1. Fund Binance futures wallet with **$500-1,000 USDT** starting capital.

2. Apply live `.env` changes:
   ```bash
   # In .env:
   TESTNET=False
   ENABLE_LIVE_TRADING=True
   DEFAULT_ORDER_AMOUNT_USD=20
   MAX_ORDER_AMOUNT_USD=50
   MAX_ORDER_NOTIONAL_USD=50
   MAX_POSITION_EXPOSURE_USD=250
   MAX_SHORT_NOTIONAL_USD=50
   ```

   **Capital ramp schedule** (increase only after sustained profitability):

   | Phase | Duration | Order size | Max exposure | Capital needed |
   |-------|----------|-----------|--------------|----------------|
   | 1 — Validate | 2 weeks | $20 | $100 | $500 |
   | 2 — Confirm | 2 weeks | $50 | $250 | $1,000 |
   | 3 — Scale | ongoing | $100 | $500 | $2,000+ |

3. Remove the duplicate `DEFAULT_ORDER_AMOUNT_USD` line in `.env` (currently set to both 10 and 100 — keep only one).

4. Restart:
   ```bash
   sudo systemctl restart cognitrade-bot
   ```

5. Verify Telegram "started" lifecycle notification arrives.

6. Continue daily soak monitoring at live scale. Do NOT increase order size until Phase 1 shows positive returns over 2 weeks.

### If gates fail → Diagnose

| Symptom | Action |
|---------|--------|
| 0 trades | Check ADX values in logs — may be stuck in TRANSITION. Consider lowering `ADX_TREND_THRESH` or raising `ADX_CHOP_THRESH`. |
| Negative return | Review trade log for pattern — is DI filter blocking profitable signals? Test with `ADX_USE_DI_FILTER=False`. |
| High drawdown | Check if a single large loss or accumulation. Review stop-loss settings. |
| Latency alerts | Testnet is slow — compare with mainnet. May not be actionable. |
| Bot restarts | Check `journalctl -u cognitrade-bot` for crash stacktraces. |

---

## Evidence Record

_Fill in after evaluation on 2026-02-27 or later._

**Date evaluated:**
**Evaluated by:**
**Shadow result:**
**Canary result:**
**Decision:** GO / NO-GO
**Notes:**
