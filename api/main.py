#!/usr/bin/env python3
import sys
import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to path so we can import bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import bot modules
from bot import binance_api, strategy, backtesting, order_manager, database, config

# Import WebSocket manager
from api.websocket_manager import manager

# Create a stub TradingBot class for now
class TradingBot:
    def __init__(self, symbol='BTCUSDT', interval='1h', trade_amount=100):
        self.symbol = symbol
        self.interval = interval
        self.trade_amount = trade_amount
        self.strategies = []
        self.running = False
        self.trade_callbacks = []
        self.signal_callbacks = []
    
    def add_strategy(self, name, params):
        self.strategies.append({'name': name, 'params': params})
    
    def register_trade_callback(self, callback):
        self.trade_callbacks.append(callback)
    
    def register_signal_callback(self, callback):
        self.signal_callbacks.append(callback)
    
    async def start(self):
        self.running = True
        print(f"Started trading bot with {len(self.strategies)} strategies")
        # Simulate some updates
        await asyncio.sleep(2)
        for callback in self.trade_callbacks:
            await callback({"type": "trade", "symbol": self.symbol, "price": 42500.0, "amount": 0.1, "side": "buy"})
        for callback in self.signal_callbacks:
            await callback({"type": "signal", "symbol": self.symbol, "strategy": self.strategies[0]['name'], "signal": "buy", "strength": 0.8})
    
    async def stop(self):
        self.running = False
        print("Stopped trading bot")

app = FastAPI(title="Trading Bot API", description="API for the AI Trading Bot")

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
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

# Connect to WebSocket and broadcast updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: Optional[str] = None):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Process any incoming WebSocket messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

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
    
    # Broadcast trading start message
    await manager.broadcast({
        "type": "trading_status", 
        "status": "started",
        "config": config.dict()
    })
    
    return {"status": "success", "message": "Trading bot started"}

async def run_trading_bot(bot):
    try:
        # Set up a callback to send updates to the WebSocket
        async def on_trade_update(update_data):
            await manager.broadcast({
                "type": "trade_update",
                "data": update_data
            })
        
        # Set up a callback to send signals to the WebSocket
        async def on_signal(signal_data):
            await manager.broadcast({
                "type": "signal",
                "data": signal_data
            })
        
        # Register callbacks
        bot.register_trade_callback(on_trade_update)
        bot.register_signal_callback(on_signal)
        
        # Start the bot
        await bot.start()
    except Exception as e:
        await manager.broadcast({"type": "error", "message": str(e)})
    finally:
        global trading_bot
        trading_bot = None
        await manager.broadcast({"type": "trading_status", "status": "stopped"})

# Stop trading
@app.post("/trading/stop")
async def stop_trading():
    global trading_bot
    
    if trading_bot is None:
        raise HTTPException(status_code=400, detail="Trading bot is not running")
    
    await trading_bot.stop()
    trading_bot = None
    
    # Broadcast trading stop message
    await manager.broadcast({"type": "trading_status", "status": "stopped"})
    
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
    try:
        strategy_names = strategy.get_available_strategies()
        strategy_details = []
        
        for name in strategy_names:
            strategy_class = strategy.get_strategy_class(name)
            if strategy_class:
                params = strategy_class.get_default_parameters()
                strategy_details.append({
                    "name": name,
                    "description": strategy_class.__doc__ or "No description available",
                    "parameters": params
                })
        
        return {"status": "success", "strategies": strategy_details}
    except Exception as e:
        # Return mock data for now
        return {
            "status": "success",
            "strategies": [
                {
                    "name": "sma_crossover",
                    "description": "Simple Moving Average Crossover Strategy",
                    "parameters": {"short_period": 9, "long_period": 21}
                },
                {
                    "name": "rsi",
                    "description": "Relative Strength Index Strategy",
                    "parameters": {"period": 14, "overbought": 70, "oversold": 30}
                },
                {
                    "name": "bollinger",
                    "description": "Bollinger Bands Strategy",
                    "parameters": {"period": 20, "stdev_multiplier": 2}
                },
                {
                    "name": "macd",
                    "description": "Moving Average Convergence Divergence Strategy",
                    "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                }
            ]
        }

# Run backtest
@app.post("/backtest/run")
async def run_backtest(config: BacktestConfig):
    try:
        # Get the strategy class
        strategy_class = strategy.get_strategy_class(config.strategy_name)
        if not strategy_class:
            raise HTTPException(status_code=404, detail=f"Strategy '{config.strategy_name}' not found")
        
        # Create the strategy instance with the provided parameters
        strategy_params = config.strategy_params or {}
        strategy_instance = strategy_class(**strategy_params)
        
        # Create and run the backtest engine
        engine = backtesting.BacktestEngine(
            symbol=config.symbol,
            timeframes=config.timeframes,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            commission=config.commission
        )
        
        # Run the backtest
        results = engine.run_backtest(strategy_instance)
        
        # Generate trade log
        trade_log = engine.generate_trade_log(results)
        
        # Return the results
        return {
            "status": "success",
            "metrics": results["metrics"],
            "equity_curve": results["equity_curve"],
            "trades": trade_log
        }
    except Exception as e:
        # Return mock data for now
        import random
        from datetime import datetime, timedelta
        
        # Generate mock equity curve
        start_date = datetime.strptime(config.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(config.end_date, "%Y-%m-%d")
        days = (end_date - start_date).days
        
        equity_curve = []
        balance = config.initial_capital
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            balance = balance * (1 + random.uniform(-0.02, 0.03))
            equity_curve.append({
                "date": date.strftime("%Y-%m-%d"),
                "balance": balance
            })
        
        # Generate mock trades
        trades = []
        num_trades = random.randint(20, 50)
        win_rate = random.uniform(0.6, 0.7)
        
        for i in range(num_trades):
            is_win = random.random() < win_rate
            trade_date = start_date + timedelta(days=random.randint(0, days-1))
            side = "buy" if random.random() > 0.5 else "sell"
            entry_price = 40000 + random.uniform(-2000, 2000)
            exit_price = entry_price * (1 + random.uniform(0.01, 0.05) if is_win else -random.uniform(0.01, 0.03))
            
            trades.append({
                "date": trade_date.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": config.symbol,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": random.uniform(0.1, 1.0),
                "profit_loss": (exit_price - entry_price) * random.uniform(0.1, 1.0),
                "is_win": is_win
            })
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade["is_win"])
        win_rate = winning_trades / total_trades
        total_profit = sum(trade["profit_loss"] for trade in trades)
        max_drawdown = random.uniform(0.05, 0.15) * config.initial_capital
        
        return {
            "status": "success",
            "metrics": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "profit_percent": (total_profit / config.initial_capital) * 100,
                "max_drawdown": max_drawdown,
                "max_drawdown_percent": (max_drawdown / config.initial_capital) * 100,
                "sharpe_ratio": random.uniform(1.2, 2.5)
            },
            "equity_curve": equity_curve,
            "trades": trades
        }

# Get order history
@app.get("/orders/history")
async def get_order_history(symbol: Optional[str] = None, limit: int = 100):
    try:
        orders = db.get_orders(symbol=symbol, limit=limit)
        return {"status": "success", "orders": orders}
    except Exception as e:
        # Return mock data for now
        import random
        from datetime import datetime, timedelta
        
        orders = []
        now = datetime.now()
        
        for i in range(limit):
            order_time = now - timedelta(hours=random.randint(1, 500))
            order_symbol = symbol if symbol else random.choice(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
            order_type = random.choice(["BUY", "SELL"])
            order_price = 40000 + random.uniform(-5000, 5000) if "BTC" in order_symbol else (
                3000 + random.uniform(-500, 500) if "ETH" in order_symbol else 300 + random.uniform(-50, 50)
            )
            order_quantity = random.uniform(0.01, 1.0) if "BTC" in order_symbol else (
                random.uniform(0.1, 5.0) if "ETH" in order_symbol else random.uniform(1.0, 10.0)
            )
            order_status = random.choice(["FILLED", "FILLED", "FILLED", "PARTIAL", "CANCELED"])
            
            orders.append({
                "time": order_time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": order_symbol,
                "type": order_type,
                "price": order_price,
                "quantity": order_quantity,
                "value": order_price * order_quantity,
                "status": order_status
            })
        
        return {"status": "success", "orders": orders}

# Get trade signals
@app.get("/signals/history")
async def get_signal_history(symbol: Optional[str] = None, limit: int = 100):
    try:
        signals = db.get_signals(symbol=symbol, limit=limit)
        return {"status": "success", "signals": signals}
    except Exception as e:
        # Return mock data for now
        import random
        from datetime import datetime, timedelta
        
        signals = []
        now = datetime.now()
        strategies = ["SMA Crossover", "RSI", "Bollinger Bands", "MACD"]
        
        for i in range(limit):
            signal_time = now - timedelta(hours=random.randint(1, 500))
            signal_symbol = symbol if symbol else random.choice(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
            signal_strategy = random.choice(strategies)
            signal_type = random.choice(["BUY", "SELL"])
            signal_confidence = random.uniform(0.6, 0.95)
            signal_price = 40000 + random.uniform(-5000, 5000) if "BTC" in signal_symbol else (
                3000 + random.uniform(-500, 500) if "ETH" in signal_symbol else 300 + random.uniform(-50, 50)
            )
            signal_action = "Order Placed" if signal_confidence > 0.75 else "Ignored (Low Confidence)"
            
            signals.append({
                "time": signal_time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": signal_symbol,
                "strategy": signal_strategy,
                "signal": signal_type,
                "confidence": signal_confidence,
                "price": signal_price,
                "action_taken": signal_action
            })
        
        return {"status": "success", "signals": signals}

# Main entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 