#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "ERROR: .env not found in $ROOT_DIR"
  echo "Create it first (for example: cp .env.example .env)."
  exit 1
fi

if [[ ! -x "venv/bin/pytest" ]]; then
  echo "ERROR: venv/bin/pytest not found or not executable."
  echo "Set up the virtualenv first:"
  echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

backup_file="$(mktemp .env.testbackup.XXXXXX)"
cp .env "$backup_file"

restore_env() {
  if [[ -f "$backup_file" ]]; then
    mv -f "$backup_file" .env
  fi
}
trap restore_env EXIT INT TERM

# Test-safety overrides:
# - Remove hard runtime overrides so tests can validate default behavior.
# - Force LLM decisions enabled for LLM pipeline tests.
# - Remove explicit trading-mode overrides so mode-sensitive tests run under config defaults.
sed -i '/^API_AUTH_ENABLED=/d' .env
sed -i '/^ROLLOUT_ENFORCE_PRODUCTION_GATE=/d' .env
sed -i '/^TRADE_MODE=/d' .env
sed -i '/^ENABLE_FUTURES_SHORTS=/d' .env

if grep -q '^ENABLE_LLM_DECISIONS=' .env; then
  sed -i 's/^ENABLE_LLM_DECISIONS=.*/ENABLE_LLM_DECISIONS=True/' .env
else
  printf '\nENABLE_LLM_DECISIONS=True\n' >> .env
fi

if [[ "$#" -eq 0 ]]; then
  set -- -q
fi

echo "Running tests with temporary local-test env overrides..."
venv/bin/pytest "$@"
