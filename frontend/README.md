# AI Trading Bot UI

This is the web-based user interface for the AI Trading Bot. It provides a comprehensive dashboard for monitoring, configuring, and controlling your trading bot.

## Features

- **Dashboard:** Overview of trading performance, recent trades, and key metrics
- **Live Trading:** Configure and control your trading bot in real-time
- **Backtesting:** Test and optimize trading strategies using historical data
- **Trade History:** View detailed order and signal history
- **Settings:** Configure API credentials, LLM integration, and other settings

## Technology Stack

- **Frontend:** Next.js 15, Tailwind CSS, Lightweight Charts, Tremor UI
- **Backend:** FastAPI, WebSockets, Python-Binance
- **Communication:** Real-time updates via WebSockets

## Getting Started

### Prerequisites

- Node.js (v18+)
- Python 3.8+
- Binance API key and secret (can use Testnet for paper trading)

### Installation

Clone the repository and set up dependencies:

```bash
# Go to project root directory
cd /path/to/trading_bot

# Install Python dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r api/requirements.txt

# Install Node.js dependencies
cd frontend
npm install
```

### Configuration

1. Configure your API credentials in `.env` file
2. (Optional) Configure LLM settings in `.env` file

### Running the UI

You can run both the API backend and Next.js frontend with a single command:

```bash
# From project root
./run_ui.sh
```

Or run them separately:

```bash
# Start API backend (from project root)
cd api
python main.py

# Start Next.js frontend (from project root)
cd frontend
npm run dev
```

The UI will be available at http://localhost:3000

## Deployment

### Server Requirements

- Ubuntu server 20.04 LTS or newer
- Nginx (for production deployment)
- PM2 (for process management)

### Deployment Steps

1. Clone the repository to your server
2. Configure the `.env` file with your production settings
3. Build the Next.js frontend:

```bash
cd frontend
npm run build
```

4. Set up Nginx as a reverse proxy
5. Use PM2 to manage the API and frontend processes:

```bash
# Install PM2
npm install -g pm2

# Start API server
cd api
pm2 start main.py --name trading-bot-api

# Start Next.js frontend
cd ../frontend
pm2 start npm --name trading-bot-ui -- start
```

## PineScript Integration

The UI supports visualization using PineScript indicators. To add custom indicators:

1. Create your PineScript in TradingView
2. Export as a JavaScript library
3. Add to the frontend project in `/components/indicators/`
4. Register the indicator in the chart configuration

## License

This project is licensed under the MIT License - see the LICENSE file for details.
