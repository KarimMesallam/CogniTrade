# Repository Guidelines

## Project Structure & Module Organization
Core trading logic lives in `bot/`:
- `bot/main.py` runs initialization and trading loop.
- `bot/strategy.py`, `bot/order_manager.py`, `bot/llm_manager.py`, and `bot/database.py` hold signal, execution, AI, and persistence logic.
- `bot/backtesting/` contains the modular backtesting engine (`core/`, `data/`, `models/`, `reporting/`, `visualization/`).

API endpoints are in `api/main.py` (FastAPI). Tests are under `tests/` and follow `test_*.py` naming. Example scripts are in `examples/`. Runtime artifacts go to `logs/`, `order_logs/`, `output/`, and `data/`.

## Build, Test, and Development Commands
No separate build step; install dependencies and run directly:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run the bot:

```bash
python run_bot.py
# or
python -m bot.main
```

Run the API:

```bash
cd api && uvicorn main:app --reload
```

Run tests and coverage:

```bash
pytest
pytest -v
pytest --cov=bot --cov=api
```

## Coding Style & Naming Conventions
Use Python 3 style with 4-space indentation, `snake_case` for functions/variables/files, and `PascalCase` for classes. Keep modules focused by responsibility (strategy, execution, db, API). Prefer explicit type hints for new or changed public functions, especially in `api/` and backtesting modules.

## Testing Guidelines
Use `pytest` with tests in `tests/test_*.py`. Mirror module scope where possible (e.g., changes in `bot/llm_manager.py` should update `tests/test_llm_manager.py`). For integration-heavy changes, include targeted runs such as:

```bash
pytest tests/test_main_db_integration.py -v
```

Use `tests/run_api_tests.py` only when intentionally running real API-key-based tests.

## Commit & Pull Request Guidelines
Follow the existing imperative commit style seen in history: `Implement ...`, `Enhance ...`, `Update ...`, `Add ...`. Keep commits scoped to one logical change.

PRs should include:
- What changed and why.
- Test evidence (`pytest` command(s) and result summary).
- Config/schema impact (`.env`, DB tables, or API contract changes).
- Sample output paths when relevant (for backtests/reports in `output/`).

## Security & Configuration Tips
Never commit secrets. Keep credentials in `.env` (ignored by git) and start from `.env.example`. Use Binance Testnet for development (`TESTNET=True`) unless a change explicitly requires live-trading behavior.
