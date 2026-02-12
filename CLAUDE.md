# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CogniTrade is an AI-enhanced automated cryptocurrency trading platform that combines traditional trading strategies with modern AI/LLM orchestration. It's a Python-based trading bot for Binance with paper trading support via Binance Testnet.

## Essential Commands

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_strategy.py

# Run with coverage
pytest --cov=bot --cov=api

# Run tests matching pattern
pytest -k "llm"

# Generate HTML coverage report
pytest --cov=bot --cov=api --cov-report=html
```

### Starting the Application
```bash
# Start trading bot
python run_bot.py

# Start API server
cd api && uvicorn main:app --reload
```

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with API keys and configuration
```

## Architecture Overview

### Core Components

1. **Strategy System**: Plugin-based architecture
   - Strategies implement `generate_signal()` returning BUY/SELL/HOLD
   - Custom strategies loaded dynamically from `bot/custom_strategies/`
   - Configuration in `TRADING_CONFIG["strategies"]`

2. **LLM Integration**: Dual-model pipeline
   - Primary model (DeepSeek R1): Deep market analysis
   - Secondary model (GPT-4o): Structured output extraction
   - Fallback to rule-based decisions when unavailable
   - Complete transparency with stored model responses

3. **Database Layer**: Three-tier architecture
   - `database.py`: Core SQLite operations
   - `db_integration.py`: Business logic wrapper
   - Schema includes trades, signals, market_data, performance, alerts

4. **Configuration Hierarchy**
   - Environment variables (highest priority)
   - JSON config file (optional via CONFIG_FILE env var)
   - Default values in config.py

### Key Design Patterns

- **Observer Pattern**: Order status monitoring
- **Factory Pattern**: Strategy creation
- **Plugin Architecture**: Custom strategies
- **Error Handling**: Exponential backoff, custom exceptions
- **Decimal Precision**: All financial calculations use Decimal type

### Trade Execution Flow
```
Signal Generation → Consensus Building → LLM Validation → Risk Management → Order Execution → Database Recording
```

### Important Conventions

1. **Time Sync**: Automatic synchronization with Binance server time
2. **Dual Storage**: JSON order logs + SQLite database
3. **Testing Mode**: Set `TESTING_MODE=True` for development
4. **Logging**: Module-specific loggers to `trading_bot.log`
5. **API Design**: RESTful FastAPI endpoints at `http://localhost:8000`

### Common Development Tasks

- Add new strategy: Create file in `bot/custom_strategies/` implementing required interface
- Modify LLM behavior: Update `bot/llm_manager.py` and `LLM_CONFIG` in config
- Change database schema: Update `bot/database.py` schema definitions
- Add API endpoint: Update `api/main.py` with new route

### Testing Approach

- Unit tests for individual components
- Integration tests for API and database
- Mocked external services (Binance, LLMs)
- Test categories: api, integration, unit (see pytest.ini)

### Critical Files to Understand

- `bot/main.py`: Core trading loop and initialization
- `bot/config.py`: All configuration management
- `bot/strategy.py`: Strategy interface and loading
- `bot/llm_manager.py`: LLM decision orchestration
- `bot/database.py`: Database schema and operations