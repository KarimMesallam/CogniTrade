# AI Trading Bot with Binance & LLM Integration

A scalable Python project for automated trading that combines traditional trading strategies with modern AI/LLM orchestration for decision making. The bot connects to Binance (starting with paper trading via the Testnet) and is designed to easily integrate multiple LLMs to refine trade signals and manage order execution.

## Overview

This project aims to take you from basic trading bot functionality to a robust system that leverages AI:
- **Paper Trading:** Start with simulated trades using Binance Testnet.
- **Scalable Architecture:** Modular design to support multiple strategies and future integration with various LLMs.
- **AI Orchestration:** A dedicated module to incorporate LLMs (e.g., DeepSeek R1, GPT-4) for enhanced decision making.
- **Extensible:** Easily add new exchanges, trading strategies, and real-time data streams.

## Features

- **Binance API Integration:** Uses the official Binance API (via python-binance) for fetching market data, placing orders, and managing accounts.
- **Paper Trading Mode:** Safely test strategies on Binance Testnet before going live.
- **Strategy Module:** Contains logic for trading signals (e.g., based on technical indicators) with an abstract layer for future enhancements.
- **LLM Manager:** A placeholder module to later integrate multiple language models for trade signal orchestration.
- **Robust Project Structure:** Clean separation of concerns with modules for configuration, API calls, strategy logic, order management, and service orchestration.
- **Testing Suite:** Unit tests using pytest to validate functionality as the project scales.

## Project Structure

```
trading_bot/
├── bot/
│   ├── __init__.py
│   ├── config.py           # Configuration loader (API keys, endpoints, etc.)
│   ├── binance_api.py      # Binance API wrapper (using python-binance)
│   ├── strategy.py         # Abstract trading strategy and sample strategy implementation
│   ├── llm_manager.py      # LLM orchestration placeholder (for decision support)
│   ├── order_manager.py    # (Optional) Module for order execution and logging
│   └── main.py             # Main entry point for running the bot
├── tests/                  # Unit tests (pytest)
│   ├── test_binance_api.py
│   ├── test_strategy.py
│   └── test_llm_manager.py
├── requirements.txt        # List of dependencies (python-binance, python-dotenv, requests, pandas, etc.)
├── .env                  # Environment variables (API keys, etc.)
└── README.md               # Project documentation (this file)
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ai-trading-bot.git
   cd ai-trading-bot
   ```

2. **Set Up a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   Create a `.env` file in the root folder with your API credentials:
   ```ini
   API_KEY=your_binance_testnet_api_key
   API_SECRET=your_binance_testnet_api_secret
   TESTNET=True
   SYMBOL=BTCUSDT
   ```

## Usage

### Running the Bot

Start the bot (for paper trading) by running:
```bash
python -m bot.main
```
This script initializes the Binance connection, applies a basic strategy (placeholder logic), and enters a continuous trading loop.

### Running Tests

To run the unit tests:
```bash
pytest
```

## Future Improvements

- **Advanced Trading Strategies:** Implement additional strategies beyond the basic examples.
- **LLM Integration:** Expand `bot/llm_manager.py` to interface with models like DeepSeek R1, GPT-4, etc., for real-time decision support.
- **Live Trading:** After thorough paper trading, switch to live mode (with caution) by toggling configuration parameters.
- **Multi-Exchange Support:** Add wrappers for additional exchanges (e.g., using CCXT) for more diversified trading.
- **Enhanced Logging & Monitoring:** Integrate more robust logging and alerting for operational insights.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests. Please follow the standard GitHub flow and include tests for any new functionality.

## License

This project is licensed under the MIT License.