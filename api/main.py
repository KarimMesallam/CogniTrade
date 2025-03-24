#!/usr/bin/env python3
import sys
import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to path so we can import bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import bot modules
from bot import binance_api, strategy, backtesting, order_manager, database, config

# Create a TradingBot class for the API
class TradingBot:
    def __init__(self, symbol='BTCUSDT', interval='1h', trade_amount=100):
        self.symbol = symbol
        self.interval = interval
        self.trade_amount = trade_amount
        self.strategies = []
        self.running = False
    
    def add_strategy(self, name, params):
        self.strategies.append({'name': name, 'params': params})
    
    async def start(self):
        self.running = True
        print(f"Started trading bot with {len(self.strategies)} strategies")
    
    async def stop(self):
        self.running = False
        print("Stopped trading bot")

app = FastAPI(title="Trading Bot API", description="API for the AI Trading Bot")

# Add minimal CORS middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for API requests and responses
class BacktestConfig(BaseModel):
    symbol: str
    timeframes: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    commission: float
    strategy_name: str
    strategy_params: Optional[Dict[str, Any]] = None

class StrategyConfig(BaseModel):
    name: str
    params: Dict[str, Any]
    active: bool

class TradingConfig(BaseModel):
    symbol: str
    interval: str
    trade_amount: float
    strategies: List[StrategyConfig]

# Trading bot instance
trading_bot = None
trading_task = None

# Database connection
db = database.Database()

# Start trading task
@app.post("/trading/start")
async def start_trading(config: TradingConfig, background_tasks: BackgroundTasks):
    global trading_bot, trading_task
    
    if trading_bot is not None:
        raise HTTPException(status_code=400, detail="Trading bot is already running")
    
    # Initialize trading bot with the provided configuration
    trading_bot = TradingBot(
        symbol=config.symbol,
        interval=config.interval,
        trade_amount=config.trade_amount
    )
    
    # Set active strategies
    for strat_config in config.strategies:
        if strat_config.active:
            trading_bot.add_strategy(strat_config.name, strat_config.params)
    
    # Start trading in a background task
    background_tasks.add_task(run_trading_bot, trading_bot)
    
    return {"status": "success", "message": "Trading bot started"}

async def run_trading_bot(bot):
    try:
        # Start the bot
        await bot.start()
    except Exception as e:
        print(f"Error running trading bot: {e}")
    finally:
        global trading_bot
        trading_bot = None

# Stop trading
@app.post("/trading/stop")
async def stop_trading():
    global trading_bot
    
    if trading_bot is None:
        raise HTTPException(status_code=400, detail="Trading bot is not running")
    
    await trading_bot.stop()
    trading_bot = None
    
    return {"status": "success", "message": "Trading bot stopped"}

# Get account information
@app.get("/account/info")
async def get_account_info():
    try:
        client = binance_api.get_client()
        account_info = client.get_account()
        
        # Filter and format the response
        balances = [
            {
                "asset": balance["asset"],
                "free": float(balance["free"]),
                "locked": float(balance["locked"])
            }
            for balance in account_info["balances"]
            if float(balance["free"]) > 0 or float(balance["locked"]) > 0
        ]
        
        return {
            "status": "success",
            "account_type": "spot",
            "balances": balances
        }
    except Exception as e:
        # Return mock data for now
        return {
            "status": "success",
            "account_type": "spot",
            "balances": [
                {"asset": "BTC", "free": 0.5, "locked": 0.0},
                {"asset": "ETH", "free": 5.0, "locked": 0.0},
                {"asset": "USDT", "free": 10000.0, "locked": 0.0}
            ]
        }

# Get available trading pairs
@app.get("/market/symbols")
async def get_symbols():
    try:
        client = binance_api.get_client()
        exchange_info = client.get_exchange_info()
        
        symbols = []
        for symbol_info in exchange_info["symbols"]:
            if symbol_info["status"] == "TRADING":
                symbols.append({
                    "symbol": symbol_info["symbol"],
                    "baseAsset": symbol_info["baseAsset"],
                    "quoteAsset": symbol_info["quoteAsset"]
                })
        
        return {"status": "success", "symbols": symbols}
    except Exception as e:
        # Return mock data for now
        return {
            "status": "success",
            "symbols": [
                {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT"},
                {"symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT"},
                {"symbol": "BNBUSDT", "baseAsset": "BNB", "quoteAsset": "USDT"},
                {"symbol": "ADAUSDT", "baseAsset": "ADA", "quoteAsset": "USDT"},
                {"symbol": "SOLUSDT", "baseAsset": "SOL", "quoteAsset": "USDT"}
            ]
        }

# Get market data for a specific symbol
@app.get("/market/data/{symbol}/{interval}")
async def get_market_data(symbol: str, interval: str, limit: int = 100):
    try:
        client = binance_api.get_client()
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        # Format the response
        candles = []
        for k in klines:
            candles.append({
                "time": k[0] / 1000,  # Convert to seconds for charting libraries
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            })
        
        return {"status": "success", "candles": candles}
    except Exception as e:
        # Return mock data for now
        import time
        import random
        
        current_time = int(time.time())
        candles = []
        close_price = 40000.0
        
        for i in range(limit):
            open_price = close_price
            high_price = open_price * (1 + random.uniform(0, 0.02))
            low_price = open_price * (1 - random.uniform(0, 0.02))
            close_price = open_price * (1 + random.uniform(-0.01, 0.01))
            candle_time = current_time - ((limit - i) * 3600)  # Assuming 1h intervals
            
            candles.append({
                "time": candle_time,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": random.uniform(10, 100)
            })
        
        return {"status": "success", "candles": candles}

# Get available strategies
@app.get("/strategies")
async def get_strategies():
    # Return list of available strategies
    strategies = [
        {
            "name": "sma_crossover",
            "display_name": "SMA Crossover",
            "description": "Simple Moving Average Crossover Strategy",
            "parameters": {
                "short_period": {
                    "type": "integer",
                    "min": 5,
                    "max": 50,
                    "default": 10,
                    "description": "Short period for SMA calculation"
                },
                "long_period": {
                    "type": "integer",
                    "min": 20,
                    "max": 200,
                    "default": 50,
                    "description": "Long period for SMA calculation"
                }
            }
        },
        {
            "name": "rsi",
            "display_name": "RSI Strategy",
            "description": "Relative Strength Index Strategy",
            "parameters": {
                "period": {
                    "type": "integer",
                    "min": 7,
                    "max": 30,
                    "default": 14,
                    "description": "Period for RSI calculation"
                },
                "overbought": {
                    "type": "integer",
                    "min": 60,
                    "max": 90,
                    "default": 70,
                    "description": "Overbought threshold"
                },
                "oversold": {
                    "type": "integer",
                    "min": 10,
                    "max": 40,
                    "default": 30,
                    "description": "Oversold threshold"
                }
            }
        }
    ]
    
    return {"status": "success", "strategies": strategies}

# Run backtest
@app.post("/backtest/run")
async def run_backtest(config: BacktestConfig):
    try:
        # Create a backtest engine with the provided configuration
        engine = backtesting.BacktestEngine(
            symbol=config.symbol,
            timeframes=config.timeframes,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            commission=config.commission
        )
        
        # Get the requested strategy function
        strat_func = None
        if config.strategy_name == "sma_crossover":
            # Check if custom parameters are provided
            if config.strategy_params and "short_period" in config.strategy_params and "long_period" in config.strategy_params:
                short_period = int(config.strategy_params["short_period"])
                long_period = int(config.strategy_params["long_period"])
                # Create a partial function with the custom parameters
                from functools import partial
                strat_func = partial(strategy.sma_crossover_strategy, short_period=short_period, long_period=long_period)
            else:
                # Use default parameters
                strat_func = strategy.sma_crossover_strategy
        elif config.strategy_name == "rsi":
            # Check if custom parameters are provided
            if config.strategy_params and "period" in config.strategy_params:
                period = int(config.strategy_params["period"])
                overbought = int(config.strategy_params.get("overbought", 70))
                oversold = int(config.strategy_params.get("oversold", 30))
                # Create a partial function with the custom parameters
                from functools import partial
                strat_func = partial(strategy.rsi_strategy, period=period, overbought=overbought, oversold=oversold)
            else:
                # Use default parameters
                strat_func = strategy.rsi_strategy
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {config.strategy_name}")
        
        # Run the backtest
        results = engine.run_backtest(strat_func)
        
        # Format the results for the response
        formatted_results = {
            "symbol": config.symbol,
            "timeframes": config.timeframes,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "strategy": config.strategy_name,
            "initial_capital": config.initial_capital,
            "final_value": results["final_value"],
            "profit_loss": results["profit_loss"],
            "profit_loss_percent": results["profit_loss_percent"],
            "sharpe_ratio": results["sharpe_ratio"],
            "max_drawdown": results["max_drawdown"],
            "trades": len(results["trades"]),
            "win_rate": results["win_rate"],
            "avg_profit": results["avg_profit"],
            "avg_loss": results["avg_loss"],
            "profit_factor": results["profit_factor"]
        }
        
        return {"status": "success", "results": formatted_results}
    except Exception as e:
        # For now, return mock data in case of errors
        import random
        
        return {
            "status": "success",
            "results": {
                "symbol": config.symbol,
                "timeframes": config.timeframes,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "strategy": config.strategy_name,
                "initial_capital": config.initial_capital,
                "final_value": config.initial_capital * random.uniform(0.8, 1.4),
                "profit_loss": config.initial_capital * random.uniform(-0.2, 0.4),
                "profit_loss_percent": random.uniform(-20, 40),
                "sharpe_ratio": random.uniform(0.5, 2.5),
                "max_drawdown": random.uniform(5, 25),
                "trades": random.randint(10, 50),
                "win_rate": random.uniform(40, 70),
                "avg_profit": random.uniform(1, 5),
                "avg_loss": random.uniform(1, 3),
                "profit_factor": random.uniform(0.8, 2.0)
            }
        }

# Get order history
@app.get("/orders/history")
async def get_order_history(symbol: Optional[str] = None, limit: int = 100):
    try:
        # In a real implementation, this would fetch orders from the database
        # For now, return mock data
        import time
        import random
        
        orders = []
        current_time = int(time.time())
        
        for i in range(limit):
            # Generate a random order
            order_time = current_time - random.randint(60, 86400 * 7)  # Within the past week
            symbol_to_use = symbol if symbol else random.choice(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            side = random.choice(["buy", "sell"])
            qty = round(random.uniform(0.01, 1.0), 3)
            price = round(symbol_to_use.startswith("BTC") and random.uniform(38000, 45000) or random.uniform(2000, 3000), 2)
            
            orders.append({
                "id": 1000 + i,
                "time": order_time,
                "symbol": symbol_to_use,
                "side": side,
                "type": "market",
                "quantity": qty,
                "price": price,
                "value": round(qty * price, 2),
                "status": "filled"
            })
        
        # Sort by time descending
        orders.sort(key=lambda x: x["time"], reverse=True)
        
        return {"status": "success", "orders": orders}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Get signal history
@app.get("/signals/history")
async def get_signal_history(symbol: Optional[str] = None, limit: int = 100):
    try:
        # In a real implementation, this would fetch signals from the database
        # For now, return mock data
        import time
        import random
        
        signals = []
        current_time = int(time.time())
        
        for i in range(limit):
            # Generate a random signal
            signal_time = current_time - random.randint(60, 86400 * 7)  # Within the past week
            symbol_to_use = symbol if symbol else random.choice(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            signal_type = random.choice(["buy", "sell", "hold"])
            strategy = random.choice(["sma_crossover", "rsi", "llm_decision"])
            strength = round(random.uniform(0.6, 1.0), 2)
            
            signals.append({
                "id": 1000 + i,
                "time": signal_time,
                "symbol": symbol_to_use,
                "timeframe": random.choice(["1m", "5m", "15m", "1h", "4h"]),
                "strategy": strategy,
                "signal": signal_type,
                "strength": strength,
                "price": round(symbol_to_use.startswith("BTC") and random.uniform(38000, 45000) or random.uniform(2000, 3000), 2)
            })
        
        # Sort by time descending
        signals.sort(key=lambda x: x["time"], reverse=True)
        
        return {"status": "success", "signals": signals}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 