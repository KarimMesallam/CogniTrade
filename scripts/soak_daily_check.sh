#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
Usage: scripts/soak_daily_check.sh [options]

Options:
  --api-url URL          API base URL (default: http://127.0.0.1:8000)
  --read-key KEY         API read key (fallback: API_READ_KEY or first API_READ_KEYS from .env)
  --admin-key KEY        API admin key (fallback: API_ADMIN_KEY or first API_ADMIN_KEYS from .env)
  --rollout-id ID        Rollout ID (fallback: ROLLOUT_ID or ROLLOUT_REQUIRED_ID from .env)
  --window-minutes N     Dashboard lookback window in minutes (default: 1440)
  --out-dir PATH         Output directory (default: output/soak_daily)
  --strict               Fail script if rollout shadow/canary flags are not both true
  -h, --help             Show this help

Environment fallbacks:
  API_URL, API_READ_KEY, API_ADMIN_KEY, ROLLOUT_ID, WINDOW_MINUTES, SOAK_OUT_DIR
USAGE
}

read_env_var() {
  local key="$1"
  local file=".env"
  if [[ ! -f "$file" ]]; then
    return 1
  fi
  local line
  line=$(grep -E "^${key}=" "$file" | tail -n 1 || true)
  if [[ -z "$line" ]]; then
    return 1
  fi
  line="${line#*=}"
  line="${line%\"}"
  line="${line#\"}"
  echo "$line"
}

first_csv_item() {
  local raw="$1"
  echo "$raw" | cut -d',' -f1 | xargs
}

API_URL="${API_URL:-http://127.0.0.1:8001}"
READ_KEY="${API_READ_KEY:-}"
ADMIN_KEY="${API_ADMIN_KEY:-}"
ROLLOUT_ID="${ROLLOUT_ID:-}"
WINDOW_MINUTES="${WINDOW_MINUTES:-1440}"
OUT_DIR="${SOAK_OUT_DIR:-output/soak_daily}"
STRICT_MODE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-url)
      API_URL="$2"
      shift 2
      ;;
    --read-key)
      READ_KEY="$2"
      shift 2
      ;;
    --admin-key)
      ADMIN_KEY="$2"
      shift 2
      ;;
    --rollout-id)
      ROLLOUT_ID="$2"
      shift 2
      ;;
    --window-minutes)
      WINDOW_MINUTES="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --strict)
      STRICT_MODE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$READ_KEY" ]]; then
  READ_KEY="$(first_csv_item "$(read_env_var API_READ_KEYS || true)")"
fi
if [[ -z "$ADMIN_KEY" ]]; then
  ADMIN_KEY="$(first_csv_item "$(read_env_var API_ADMIN_KEYS || true)")"
fi
if [[ -z "$ROLLOUT_ID" ]]; then
  ROLLOUT_ID="$(read_env_var ROLLOUT_REQUIRED_ID || true)"
fi

if [[ -z "$READ_KEY" ]]; then
  echo "Missing read key. Provide --read-key or set API_READ_KEYS/API_READ_KEY." >&2
  exit 1
fi
if [[ -z "$ADMIN_KEY" ]]; then
  echo "Missing admin key. Provide --admin-key or set API_ADMIN_KEYS/API_ADMIN_KEY." >&2
  exit 1
fi
if [[ -z "$ROLLOUT_ID" ]]; then
  echo "Missing rollout id. Provide --rollout-id or set ROLLOUT_REQUIRED_ID/ROLLOUT_ID." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but not found in PATH." >&2
  exit 1
fi

DATE_UTC="$(date -u +%Y%m%d)"
STAMP_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DAY_DIR="$OUT_DIR/$DATE_UTC"
mkdir -p "$DAY_DIR"

HEALTH_JSON="$DAY_DIR/health.json"
DASHBOARD_JSON="$DAY_DIR/dashboard.json"
EVENTS_JSON="$DAY_DIR/events.json"
ROLLOUT_JSON="$DAY_DIR/rollout_status.json"
SERVICES_TXT="$DAY_DIR/services.txt"
SUMMARY_JSON="$DAY_DIR/summary.json"

curl -fsS "$API_URL/health" > "$HEALTH_JSON"
curl -fsS "$API_URL/observability/dashboard?window_minutes=$WINDOW_MINUTES" \
  -H "X-API-Key: $READ_KEY" > "$DASHBOARD_JSON"
curl -fsS "$API_URL/observability/events?limit=500" \
  -H "X-API-Key: $READ_KEY" > "$EVENTS_JSON"
curl -fsS "$API_URL/rollout/status/$ROLLOUT_ID" \
  -H "X-API-Key: $ADMIN_KEY" > "$ROLLOUT_JSON"

{
  echo "timestamp_utc=$STAMP_UTC"
  if command -v systemctl >/dev/null 2>&1; then
    systemctl show cognitrade-bot.service -p ActiveState -p SubState -p NRestarts || true
    systemctl show cognitrade-api.service -p ActiveState -p SubState -p NRestarts || true
  else
    echo "systemctl not available"
  fi
} > "$SERVICES_TXT"

ERROR_RATE="$(jq -r '.errors.error_rate // 0' "$DASHBOARD_JSON")"
LATENCY_P95_MS="$(jq -r '.latency.p95_ms // 0' "$DASHBOARD_JSON")"
EVENT_COUNT="$(jq -r '.event_count // 0' "$DASHBOARD_JSON")"
TRACE_COUNT="$(jq -r '.trace_count // 0' "$DASHBOARD_JSON")"
ALERT_COUNT="$(jq -r '.alerts_last_10 | length // 0' "$DASHBOARD_JSON")"

SHADOW_PASSED="$(jq -r '.shadow_passed // .rollout_status.shadow_passed // "unknown"' "$ROLLOUT_JSON")"
CANARY_PASSED="$(jq -r '.canary_passed // .rollout_status.canary_passed // "unknown"' "$ROLLOUT_JSON")"
PROD_PASSED="$(jq -r '.production_passed // .rollout_status.production_passed // "unknown"' "$ROLLOUT_JSON")"

LAST_DRAWDOWN="$(jq -r '((.history // .rollout_status.history // []) | (last? // {})) | (.metrics.drawdown_pct // 0)' "$ROLLOUT_JSON")"
LAST_RETURN="$(jq -r '((.history // .rollout_status.history // []) | (last? // {})) | (.metrics.total_return_pct // 0)' "$ROLLOUT_JSON")"

MAX_ERROR_RATE="${SOAK_MAX_ERROR_RATE:-$(read_env_var ROLLOUT_MAX_CANARY_ERROR_RATE || echo 0.15)}"
MAX_LATENCY_MS="${SOAK_MAX_LATENCY_P95_MS:-$(read_env_var ROLLOUT_MAX_CANARY_LATENCY_P95_MS || echo 3000)}"
MAX_DRAWDOWN_PCT="${SOAK_MAX_DRAWDOWN_PCT:-$(read_env_var ROLLOUT_MAX_CANARY_DRAWDOWN_PCT || echo 8.0)}"

python - <<PY > "$SUMMARY_JSON"
import json
from pathlib import Path

def f(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

error_rate = f("$ERROR_RATE")
latency_p95 = f("$LATENCY_P95_MS")
last_drawdown = f("$LAST_DRAWDOWN")
max_error = f("$MAX_ERROR_RATE")
max_latency = f("$MAX_LATENCY_MS")
max_drawdown = f("$MAX_DRAWDOWN_PCT")

checks = {
    "error_rate_within_limit": error_rate <= max_error,
    "latency_p95_within_limit": latency_p95 <= max_latency,
    "drawdown_within_limit": last_drawdown <= max_drawdown,
}
if "$STRICT_MODE" == "true":
    checks["shadow_passed"] = "$SHADOW_PASSED" == "true"
    checks["canary_passed"] = "$CANARY_PASSED" == "true"

summary = {
    "timestamp_utc": "$STAMP_UTC",
    "rollout_id": "$ROLLOUT_ID",
    "api_url": "$API_URL",
    "window_minutes": int("$WINDOW_MINUTES"),
    "metrics": {
        "error_rate": error_rate,
        "latency_p95_ms": latency_p95,
        "event_count": int(float("$EVENT_COUNT")),
        "trace_count": int(float("$TRACE_COUNT")),
        "alert_count_last_10": int(float("$ALERT_COUNT")),
        "last_drawdown_pct": last_drawdown,
        "last_total_return_pct": f("$LAST_RETURN"),
        "shadow_passed": "$SHADOW_PASSED",
        "canary_passed": "$CANARY_PASSED",
        "production_passed": "$PROD_PASSED",
    },
    "thresholds": {
        "max_error_rate": max_error,
        "max_latency_p95_ms": max_latency,
        "max_drawdown_pct": max_drawdown,
    },
    "checks": checks,
    "overall_pass": all(checks.values()),
}
print(json.dumps(summary, indent=2))
PY

echo "Soak daily artifacts written to: $DAY_DIR"
echo "Summary: $SUMMARY_JSON"
cat "$SUMMARY_JSON"

# --- Telegram notification ---
TG_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-$(read_env_var TELEGRAM_BOT_TOKEN || true)}"
TG_CHAT_ID="${TELEGRAM_CHAT_ID:-$(read_env_var TELEGRAM_CHAT_ID || true)}"
TG_ENABLED="${ENABLE_TELEGRAM_NOTIFICATIONS:-$(read_env_var ENABLE_TELEGRAM_NOTIFICATIONS || echo True)}"

if [[ "${TG_ENABLED,,}" =~ ^(true|1|t)$ ]] && [[ -n "$TG_BOT_TOKEN" ]] && [[ -n "$TG_CHAT_ID" ]]; then
  TG_OVERALL=$(jq -r 'if .overall_pass then "PASS ✅" else "FAIL ❌" end' "$SUMMARY_JSON")
  TG_MSG=$(python - <<PYEOF
import json, sys
with open("$SUMMARY_JSON") as f:
    s = json.load(f)
m = s.get("metrics", {})
c = s.get("checks", {})
overall = "✅ PASS" if s.get("overall_pass") else "❌ FAIL"
failed = [k for k, v in c.items() if not v]
lines = [
    f"📊 *CogniTrade Daily Soak Summary* {overall}",
    f"Rollout: {s.get('rollout_id', 'N/A')}",
    f"Error rate: {m.get('error_rate', 0):.2%}",
    f"Latency P95: {m.get('latency_p95_ms', 0):.0f}ms",
    f"Drawdown: {m.get('last_drawdown_pct', 0):.2f}%",
    f"Return: {m.get('last_total_return_pct', 0):.2f}%",
    f"Events: {int(m.get('event_count', 0))} | Traces: {int(m.get('trace_count', 0))}",
]
if failed:
    lines.append(f"Failed checks: {', '.join(failed)}")
print("\n".join(lines))
PYEOF
  )

  curl -sS -X POST "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
    -H "Content-Type: application/json" \
    -d "$(python -c "import json,sys; print(json.dumps({'chat_id': '$TG_CHAT_ID', 'text': sys.stdin.read(), 'parse_mode': 'Markdown'}))" <<< "$TG_MSG")" \
    > /dev/null || true

  echo "Telegram summary sent."
else
  echo "Telegram notifications disabled or not configured; skipping."
fi
