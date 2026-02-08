#!/usr/bin/env python3
import sys
import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to path so we can import bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logger = logging.getLogger("trading_bot")

# Import bot modules
from bot import binance_api, strategy, order_manager, database, config, llm_manager
from bot.backtesting import run_backtest, generate_report, optimize_strategy

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

class LLMDecisionRequest(BaseModel):
    symbol: str
    timeframe: str
    market_data: Dict[str, Any]
    context: str
    strategy_signals: Optional[Dict[str, Any]] = None


def _records_from_query_result(query_result: Any) -> List[Dict[str, Any]]:
    """Normalize DB query output to a JSON-serializable list of records."""
    if query_result is None:
        return []
    if isinstance(query_result, list):
        return query_result
    if hasattr(query_result, "to_dict"):
        return query_result.to_dict(orient="records")
    raise TypeError(f"Unsupported query result type: {type(query_result)}")


def _normalize_llm_decision_payload(decision: Any) -> Dict[str, Any]:
    """Validate LLM decision payload shape for consistent API responses."""
    if not isinstance(decision, dict):
        raise ValueError("LLM decision must be a dictionary")

    required_keys = ("decision", "confidence", "reasoning")
    missing = [key for key in required_keys if key not in decision]
    if missing:
        raise ValueError(f"LLM decision missing required keys: {', '.join(missing)}")

    return {
        "decision": decision["decision"],
        "confidence": decision["confidence"],
        "reasoning": decision["reasoning"],
    }


def _build_sma_crossover_backtest_strategy(short_period: int, long_period: int, timeframe: str):
    """Build a backtesting-compatible SMA crossover strategy function."""
    def _strategy(data_dict, _symbol):
        frame = data_dict.get(timeframe)
        if frame is None or frame.empty or len(frame) < long_period + 1:
            return "HOLD"

        short_sma = frame["close"].rolling(window=short_period).mean()
        long_sma = frame["close"].rolling(window=long_period).mean()
        prev_short, curr_short = short_sma.iloc[-2], short_sma.iloc[-1]
        prev_long, curr_long = long_sma.iloc[-2], long_sma.iloc[-1]

        if any(value != value for value in [prev_short, curr_short, prev_long, curr_long]):
            return "HOLD"
        if prev_short <= prev_long and curr_short > curr_long:
            return "BUY"
        if prev_short >= prev_long and curr_short < curr_long:
            return "SELL"
        return "HOLD"

    _strategy.__name__ = f"sma_crossover_{short_period}_{long_period}"
    return _strategy


def _build_rsi_backtest_strategy(period: int, overbought: int, oversold: int, timeframe: str):
    """Build a backtesting-compatible RSI strategy function."""
    def _strategy(data_dict, _symbol):
        frame = data_dict.get(timeframe)
        if frame is None or frame.empty or len(frame) < period + 2:
            return "HOLD"

        delta = frame["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        relative_strength = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + relative_strength))

        previous_rsi = rsi.iloc[-2]
        current_rsi = rsi.iloc[-1]
        if previous_rsi != previous_rsi or current_rsi != current_rsi:
            return "HOLD"
        if previous_rsi < oversold and current_rsi > oversold:
            return "BUY"
        if previous_rsi > overbought and current_rsi < overbought:
            return "SELL"
        return "HOLD"

    _strategy.__name__ = f"rsi_{period}_{overbought}_{oversold}"
    return _strategy

# Trading bot instance
trading_bot = None
trading_task = None

# Database connection
db = database.Database()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

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
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch account info from exchange")

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
        logger.error(f"Error getting exchange symbols: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch symbols from exchange")

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
        logger.error(f"Error getting market data for {symbol} ({interval}): {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch market data from exchange")

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
        },
        {
            "name": "llm_strategy",
            "display_name": "LLM-Enhanced Strategy",
            "description": "Strategy that uses LLM to make trading decisions",
            "parameters": {
                "base_strategy": {
                    "type": "string",
                    "options": ["sma_crossover", "rsi"],
                    "default": "sma_crossover",
                    "description": "Base strategy for LLM to enhance"
                },
                "llm_model": {
                    "type": "string",
                    "options": ["rule_based", "deepseek", "gpt4", "claude"],
                    "default": "rule_based",
                    "description": "LLM model to use"
                }
            }
        }
    ]
    
    return {"status": "success", "strategies": strategies}

# Run backtest
@app.post("/backtest/run")
async def run_backtest_endpoint(config: BacktestConfig):
    try:
        # Get the requested strategy function
        strat_func = None
        primary_timeframe = config.timeframes[0]
        if config.strategy_name == "sma_crossover":
            short_period = int(config.strategy_params.get("short_period", 10)) if config.strategy_params else 10
            long_period = int(config.strategy_params.get("long_period", 50)) if config.strategy_params else 50
            if short_period <= 0 or long_period <= 0 or short_period >= long_period:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid SMA parameters: require 0 < short_period < long_period"
                )
            strat_func = _build_sma_crossover_backtest_strategy(short_period, long_period, primary_timeframe)
        elif config.strategy_name == "rsi":
            period = int(config.strategy_params.get("period", 14)) if config.strategy_params else 14
            overbought = int(config.strategy_params.get("overbought", 70)) if config.strategy_params else 70
            oversold = int(config.strategy_params.get("oversold", 30)) if config.strategy_params else 30
            if period <= 1 or not (0 < oversold < overbought < 100):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid RSI parameters: require period > 1 and 0 < oversold < overbought < 100"
                )
            strat_func = _build_rsi_backtest_strategy(period, overbought, oversold, primary_timeframe)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {config.strategy_name}")
        
        # Run the backtest using the new API
        result = run_backtest(
            symbol=config.symbol,
            timeframes=config.timeframes,
            start_date=config.start_date,
            end_date=config.end_date,
            strategy_func=strat_func,
            initial_capital=config.initial_capital,
            commission_rate=config.commission
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Backtest failed to produce results")
        
        # Format the results for the response
        formatted_results = {
            "symbol": config.symbol,
            "timeframes": config.timeframes,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "strategy": config.strategy_name,
            "initial_capital": config.initial_capital,
            "final_value": result.final_equity,
            "profit_loss": result.final_equity - config.initial_capital,
            "profit_loss_percent": result.metrics.total_return_pct,
            "sharpe_ratio": result.metrics.sharpe_ratio,
            "max_drawdown": result.metrics.max_drawdown_pct,
            "trades": result.total_trades,
            "win_rate": result.metrics.win_rate,
            "avg_profit": result.metrics.avg_win if hasattr(result.metrics, 'avg_win') else 0,
            "avg_loss": result.metrics.avg_loss if hasattr(result.metrics, 'avg_loss') else 0,
            "profit_factor": result.metrics.profit_factor if hasattr(result.metrics, 'profit_factor') else 0
        }
        
        return {"status": "success", "results": formatted_results}
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")

# Get order history
@app.get("/orders/history")
async def get_order_history(symbol: Optional[str] = None, limit: int = Query(default=100, ge=1, le=1000)):
    try:
        db_connection = database.Database()
        trade_records = db_connection.get_trade_records(symbol=symbol, limit=limit, offset=0)
        orders = []
        for trade in trade_records:
            quantity = float(trade.get("quantity") or 0.0)
            price = float(trade.get("price") or 0.0)
            raw_data = trade.get("raw_data")
            order_type = "market"
            if isinstance(raw_data, dict):
                order_type = str(raw_data.get("type", order_type)).lower()
            orders.append({
                "id": trade.get("order_id") or trade.get("trade_id"),
                "time": trade.get("timestamp"),
                "symbol": trade.get("symbol"),
                "side": str(trade.get("side", "")).lower(),
                "type": order_type,
                "quantity": quantity,
                "price": price,
                "value": round(quantity * price, 8),
                "status": str(trade.get("status", "")).lower()
            })

        return {"status": "success", "orders": orders}
    except Exception as e:
        logger.error(f"Error fetching order history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch order history")

# Get signal history
@app.get("/signals/history")
async def get_signal_history(symbol: Optional[str] = None, limit: int = Query(default=100, ge=1, le=1000)):
    try:
        db_connection = database.Database()
        signal_records = db_connection.get_signal_records(symbol=symbol, limit=limit, offset=0)
        signals = []
        for signal in signal_records:
            signals.append({
                "id": signal.get("signal_id"),
                "time": signal.get("timestamp"),
                "symbol": signal.get("symbol"),
                "timeframe": signal.get("timeframe"),
                "strategy": signal.get("strategy"),
                "signal": str(signal.get("signal", "")).lower(),
                "strength": signal.get("strength", None),
                "price": signal.get("price")
            })

        return {"status": "success", "signals": signals}
    except Exception as e:
        logger.error(f"Error fetching signal history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch signal history")

# LLM Integration endpoints
@app.post("/llm/decision")
async def get_llm_decision(request: LLMDecisionRequest):
    try:
        # Create LLM manager
        manager = llm_manager.LLMManager()
        
        # Get market data in the format expected by the LLM manager
        market_data = request.market_data
        
        # Make decision based on context and market data
        if "llm_model" in request.market_data and request.market_data["llm_model"] != "rule_based":
            # Use the specified LLM model
            decision = manager.make_llm_decision(
                market_data=market_data,
                symbol=request.symbol,
                timeframe=request.timeframe,
                context=request.context,
                strategy_signals=request.strategy_signals
            )
        else:
            # Use rule-based fallback
            decision = manager.make_rule_based_decision(
                market_data=market_data,
                strategy_signals=request.strategy_signals
            )

        normalized_decision = _normalize_llm_decision_payload(decision)
        
        return {
            "status": "success",
            "decision": normalized_decision["decision"],
            "confidence": normalized_decision["confidence"],
            "reasoning": normalized_decision["reasoning"]
        }
    except Exception as e:
        logger.error(f"Error getting LLM decision: {e}")
        raise HTTPException(status_code=500, detail="Failed to get LLM decision")

# Database operations endpoints
@app.get("/database/trades")
async def get_trades(
    symbol: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    try:
        # Get database connection
        db_connection = database.Database()
        
        trade_records = db_connection.get_trade_records(symbol=symbol, limit=limit, offset=offset)

        return {"status": "success", "trades": trade_records}
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch trades")

@app.get("/database/signals")
async def get_signals(
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    try:
        # Get database connection
        db_connection = database.Database()
        
        signal_records = db_connection.get_signal_records(
            symbol=symbol,
            strategy=strategy,
            limit=limit,
            offset=offset
        )

        return {"status": "success", "signals": signal_records}
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch signals")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
