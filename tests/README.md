# Trading Bot Tests

This directory contains comprehensive test suite for the Trading Bot project. The tests cover all major components of the system including the API, trading strategies, database operations, and LLM integration.

## Test Structure

- **API Tests (`test_api.py`)**: Tests for FastAPI endpoints including trading, market data, strategies, backtesting, and LLM decision making.
- **Binance API Tests (`test_binance_api.py`)**: Tests for the Binance API wrapper with proper mocking of API responses.
- **Strategy Tests (`test_strategy.py`)**: Tests for trading strategies and signal generation.
- **Order Manager Tests (`test_order_manager.py`)**: Tests for order execution and management.
- **Backtesting Tests (`test_backtesting.py`)**: Tests for the backtesting engine.
- **Database Tests (`test_database.py`)**: Tests for database operations.
- **LLM Manager Tests (`test_llm_manager.py`)**: Tests for the LLM integration and decision making.
- **Integration Tests**:
  - `test_db_integration.py`: Tests for database integration layer
  - `test_main_db_integration.py`: Tests for main trading functions with database
  - `test_main.py`: Tests for the main bot loop

## Running Tests

### Running All Tests

```bash
# From the project root directory:
pytest

# With verbose output:
pytest -v

# With test discovery output:
pytest -v --collect-only
```

### Running Specific Test Files

```bash
# Run API tests
pytest tests/test_api.py

# Run database tests
pytest tests/test_database.py

# Run LLM tests
pytest tests/test_llm_manager.py
```

### Running Specific Test Functions

```bash
# Run a specific test function
pytest tests/test_api.py::test_health_check

# Run tests matching a pattern
pytest -k "llm"  # Runs all tests with "llm" in the name
```

## Checking Test Coverage

The test suite uses pytest-cov for coverage reporting:

```bash
# Generate coverage report for all modules
pytest --cov=bot --cov=api

# Generate coverage report for a specific module
pytest --cov=api.main tests/test_api.py

# Generate HTML coverage report
pytest --cov=bot --cov=api --cov-report=html

# The HTML report will be available in htmlcov/index.html
```

## Mocking External Services

The tests use unittest.mock to mock external services:

1. **Binance API**: All API calls are mocked to avoid real network requests
2. **Database**: Database operations are mocked
3. **LLM Services**: Calls to external LLM APIs are mocked
4. **Background Tasks**: FastAPI background tasks are mocked

## Adding New Tests

When adding new functionality to the bot, please add corresponding tests:

1. For new API endpoints, add tests in `test_api.py`
2. For new trading strategies, add tests in `test_strategy.py`
3. For enhanced LLM integration, add tests in `test_llm_manager.py`

## Continuous Integration

These tests are designed to run in a CI environment. You can run them locally to ensure your changes don't break existing functionality before submitting pull requests. 