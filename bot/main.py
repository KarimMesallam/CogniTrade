import time
import logging
from datetime import datetime
from binance.exceptions import BinanceAPIException, BinanceRequestException
from bot.config import (
    SYMBOL, TESTNET, TRADING_CONFIG, 
    get_trading_parameter, get_loop_interval, 
    get_consensus_method, is_llm_agreement_required
)
from bot.strategy import get_all_strategy_signals, simple_signal, technical_analysis_signal
from bot.binance_api import place_market_buy, place_market_sell, client, get_recent_closes, synchronize_time, get_account_balance
from bot.llm_manager import get_decision_from_llm, log_decision_with_context, LLMManager
from bot.order_manager import OrderManager
from bot.db_integration import DatabaseIntegration
import json
import uuid
import asyncio
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cognitrade.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

# Set this to True when running tests to bypass sleeps
TESTING_MODE = os.environ.get('TESTING_MODE', '0') == '1'

def handle_testnet_balance(symbol='BTC'):
    """
    Check if we're using testnet and if balances are too low.
    
    For Binance Testnet, if balance is zero, we inform the user about how to get more test funds.
    Testnet accounts are automatically funded upon creation but may be reset periodically.
    
    Args:
        symbol: Asset symbol to check (default: BTC)
        
    Returns:
        bool: True if balance is sufficient or handled successfully
    """
    if not TESTNET:
        return True
    
    try:
        # Check balance for the specific asset
        balance = get_account_balance(symbol)
        if balance and balance.get('free', 0) > 0:
            logger.info(f"Testnet {symbol} balance: {balance['free']}")
            return True
        
        # If balance is zero or not found, provide instructions to the user
        logger.warning(f"Your Testnet {symbol} balance is zero or insufficient.")
        logger.info("===== BINANCE TESTNET FUND INFORMATION =====")
        logger.info("The Binance Testnet resets periodically (typically once a month).")
        logger.info("After a reset, your test funds should be automatically replenished.")
        logger.info("Options to get more test funds:")
        logger.info("1. Visit https://testnet.binance.vision/ and login with GitHub")
        logger.info("2. Wait for the next scheduled reset of the testnet")
        logger.info("3. Create a new API key which may trigger a balance refresh")
        logger.info("Note: There is no direct API method to request more test funds.")
        logger.info("==============================================")
        
        # We return True since this is just informational and shouldn't stop the bot
        return True
    except Exception as e:
        logger.error(f"Error checking testnet balance: {e}")
        return False

def initialize_bot():
    """Initialize the trading bot and verify connectivity."""
    try:
        # Synchronize time with Binance server
        synchronize_time()
        
        # Test API connection
        server_time = client.get_server_time()
        logger.info(f"Connected to Binance {'Testnet' if TESTNET else 'Live'} successfully")
        logger.info(f"Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
        
        # Get account information
        balances = get_account_balance()
        if balances:
            logger.info(f"Account balances: {balances}")
        else:
            logger.warning("Could not retrieve account balances")
        
        # Check if trading the configured symbol is possible
        symbol_info = client.get_symbol_info(SYMBOL)
        if not symbol_info:
            logger.error(f"Symbol {SYMBOL} not found or not available for trading")
            return False
        
        # Check and handle testnet balance if necessary
        base_asset = SYMBOL.replace('USDT', '')  # Extract BTC from BTCUSDT
        if TESTNET:
            handle_testnet_balance(base_asset)
        
        # Initialize database
        try:
            db = DatabaseIntegration()
            db.add_system_alert(
                message=f"Bot initialized for trading {SYMBOL} on Binance {'Testnet' if TESTNET else 'Live'}",
                alert_type="info",
                severity="low"
            )
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            logger.warning("Continuing without database support")
        
        logger.info(f"Bot initialized successfully for trading {SYMBOL}")
        return True
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Failed to initialize bot: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        return False

def get_market_data(symbol, interval='1m'):
    """Fetch current market data for analysis."""
    try:
        # Get recent candles
        candles = client.get_klines(symbol=symbol, interval=interval, limit=20)
        
        # Get order book
        order_book = client.get_order_book(symbol=symbol, limit=5)
        
        # Get recent trades
        recent_trades = client.get_recent_trades(symbol=symbol, limit=5)
        
        market_data = {
            'symbol': symbol,
            'candles': candles,
            'order_book': order_book,
            'recent_trades': recent_trades,
            'timestamp': datetime.now().isoformat()
        }
        
        return market_data
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error fetching market data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching market data: {e}")
        return None

def get_signal_consensus(signals):
    """
    Determine a consensus signal from multiple strategies.
    
    Args:
        signals: Dictionary of strategy name -> signal value
        
    Returns:
        String: 'BUY', 'SELL', or 'HOLD'
    """
    # Get consensus method from config
    consensus_method = get_consensus_method()
    
    if consensus_method == "simple_majority":
        # Simple majority vote
        buy_count = sum(1 for signal in signals.values() if signal == "BUY")
        sell_count = sum(1 for signal in signals.values() if signal == "SELL")
        
        # Simple majority rule with bias toward HOLD if tied
        if buy_count > sell_count:
            return "BUY"
        elif sell_count > buy_count:
            return "SELL"
        else:
            return "HOLD"
            
    elif consensus_method == "weighted_majority":
        # Weighted voting based on strategy weights
        buy_weight = 0
        sell_weight = 0
        hold_weight = 0
        
        for strategy_name, signal in signals.items():
            # Get the weight for this strategy from config
            strategy_config = TRADING_CONFIG["strategies"].get(strategy_name, {})
            weight = strategy_config.get("weight", 1.0)
            
            if signal == "BUY":
                buy_weight += weight
            elif signal == "SELL":
                sell_weight += weight
            else:
                hold_weight += weight
        
        # Determine the consensus based on weighted votes
        if buy_weight > sell_weight and buy_weight > hold_weight:
            return "BUY"
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            return "SELL"
        else:
            return "HOLD"
            
    elif consensus_method == "unanimous":
        # All signals must agree for action
        if all(signal == "BUY" for signal in signals.values()):
            return "BUY"
        elif all(signal == "SELL" for signal in signals.values()):
            return "SELL"
        else:
            return "HOLD"
    
    else:
        # Default to simple majority if method is not recognized
        logger.warning(f"Unrecognized consensus method '{consensus_method}', using simple majority")
        buy_count = sum(1 for signal in signals.values() if signal == "BUY")
        sell_count = sum(1 for signal in signals.values() if signal == "SELL")
        
        if buy_count > sell_count:
            return "BUY"
        elif sell_count > buy_count:
            return "SELL"
        else:
            return "HOLD"

def execute_trade(signals, llm_decision, symbol, market_data, order_manager, db_integration=None):
    """Execute a trade based on signals and decisions."""
    try:
        # Get consensus from signals
        signal_consensus = get_signal_consensus(signals)
        logger.info(f"Signal consensus: {signal_consensus}")
        
        # Log the decision with full context for later analysis
        log_decision_with_context(llm_decision, signals, market_data)
        
        # Save signals to database if available
        signal_ids = {}
        if db_integration:
            for strategy_name, signal_value in signals.items():
                # Get timeframe from the strategy configuration
                strategy_config = TRADING_CONFIG["strategies"].get(strategy_name, {})
                timeframe = strategy_config.get("timeframe", "1m")
                
                signal_ids[strategy_name] = db_integration.save_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy=strategy_name,
                    signal=signal_value,
                    llm_decision=llm_decision
                )
        
        # Check if LLM agreement is required from config
        llm_required = is_llm_agreement_required()
        
        # Determine if we should execute a trade
        execute_buy = signal_consensus == "BUY" and (not llm_required or llm_decision == "BUY")
        execute_sell = signal_consensus == "SELL" and (not llm_required or llm_decision == "SELL")
        
        # Get the default order amount from config
        default_order_amount = get_trading_parameter("default_order_amount_usd", 10.0)
        
        if execute_buy:
            logger.info(f"Executing BUY for {symbol}")
            # Use the order manager to execute the buy with the configured USDT amount
            order = order_manager.execute_market_buy(quote_amount=default_order_amount)
            if order:
                logger.info(f"Buy order executed successfully: {order['orderId']}")
                
                # Link signal to trade in database if available
                if db_integration and 'trade_id' in order and signal_ids:
                    for signal_id in signal_ids.values():
                        if signal_id > 0:
                            db_integration.link_signal_to_trade(signal_id, order['trade_id'])
                
                return order
            else:
                logger.warning("Buy order execution failed")
                
        elif execute_sell:
            # For sell, we should check our current position and sell an appropriate amount
            logger.info(f"Executing SELL for {symbol}")
            balance = get_account_balance(symbol.replace('USDT', ''))  # Get BTC balance for BTCUSDT
            
            if balance and balance.get('free', 0) > 0:
                quantity = balance['free']
                order = order_manager.execute_market_sell(quantity)
                if order:
                    logger.info(f"Sell order executed successfully: {order['orderId']}")
                    
                    # Link signal to trade in database if available
                    if db_integration and 'trade_id' in order and signal_ids:
                        for signal_id in signal_ids.values():
                            if signal_id > 0:
                                db_integration.link_signal_to_trade(signal_id, order['trade_id'])
                    
                    return order
                else:
                    logger.warning("Sell order execution failed")
            else:
                logger.warning(f"No balance available to sell for {symbol.replace('USDT', '')}")
                
        else:
            logger.info(f"No trade executed. Signal consensus: {signal_consensus}, LLM decision: {llm_decision}")
        
        return None
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error executing trade: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during trade execution: {e}")
        return None

def trading_loop():
    """Main trading loop."""
    logger.info("Starting trading loop...")
    
    # Initialize database integration
    db_integration = None
    try:
        db_integration = DatabaseIntegration()
        logger.info("Database integration initialized for trading loop")
        
        # Log start of trading session
        db_integration.add_system_alert(
            message="Trading session started",
            alert_type="info",
            severity="low",
            data={"symbol": SYMBOL, "testnet": TESTNET}
        )
    except Exception as e:
        logger.error(f"Failed to initialize database integration: {e}")
        logger.warning("Continuing without database support")
    
    # Initialize order manager with risk percentage from config
    risk_percentage = get_trading_parameter("risk_percentage", 1.0)
    order_manager = OrderManager(SYMBOL, risk_percentage=risk_percentage, use_database=db_integration is not None)
    logger.info(f"Order manager initialized for {SYMBOL} with risk percentage {risk_percentage}%")
    
    # Initialize LLM manager
    llm_manager = LLMManager()
    
    # Track consecutive errors to implement exponential backoff
    consecutive_errors = 0
    max_consecutive_errors = TRADING_CONFIG["operation"].get("max_consecutive_errors", 5)
    max_backoff_seconds = TRADING_CONFIG["operation"].get("max_backoff_seconds", 3600)
    
    # Get loop interval from config
    loop_interval = get_loop_interval()
    
    while True:
        try:
            # Update statuses of existing orders
            updated_orders = order_manager.update_order_statuses()
            if updated_orders:
                logger.info(f"Updated {len(updated_orders)} order statuses")
            
            # Get current market data
            primary_timeframe = TRADING_CONFIG["timeframes"].get("primary", "1m")
            market_data = get_market_data(SYMBOL, primary_timeframe)
            if not market_data:
                raise Exception("Failed to get market data")
            
            # Save market data to database if available
            if db_integration:
                db_integration.save_market_data(market_data, SYMBOL, primary_timeframe)
            
            # Get signals from all enabled strategies
            signals = get_all_strategy_signals(SYMBOL)
            logger.info(f"Strategy signals: {signals}")
            
            # Create a detailed prompt for the LLM with market context
            prompt = (
                f"Current signals for {SYMBOL}: {signals}.\n"
            )
            
            # Add each strategy's signal and timeframe
            for strategy_name, signal in signals.items():
                strategy_config = TRADING_CONFIG["strategies"].get(strategy_name, {})
                timeframe = strategy_config.get("timeframe", "1m")
                prompt += f"{strategy_name.capitalize()} strategy signal ({timeframe} timeframe): {signal}\n"
            
            prompt += f"Given these signals and the latest market data, should we BUY, SELL, or HOLD?"
            
            # Use LLM for decision support (with detailed market data and context)
            llm_result = llm_manager.make_llm_decision(
                market_data=market_data,
                symbol=SYMBOL,
                timeframe=primary_timeframe,
                context=f"Trading {SYMBOL} with {len(signals)} active strategies",
                strategy_signals=signals
            )
            
            llm_decision = llm_result.get("decision", "HOLD")
            llm_confidence = llm_result.get("confidence", 0.5)
            llm_reasoning = llm_result.get("reasoning", "No reasoning provided")
            
            logger.info(f"LLM decision: {llm_decision} (confidence: {llm_confidence:.2f})")
            logger.info(f"LLM reasoning: {llm_reasoning}")
            
            # Execute trade if appropriate
            order = execute_trade(signals, llm_decision, SYMBOL, market_data, order_manager, db_integration)
            
            # Reset error counter on success
            consecutive_errors = 0
            
            # Wait before next iteration
            logger.debug(f"Waiting {loop_interval} seconds for next iteration...")
            if not TESTING_MODE:
                time.sleep(loop_interval)
            
        except (BinanceAPIException, BinanceRequestException) as e:
            consecutive_errors += 1
            backoff_time = min(loop_interval * 2 ** min(consecutive_errors, max_consecutive_errors), max_backoff_seconds)
            logger.error(f"Binance API error: {e}")
            logger.info(f"Retrying in {backoff_time} seconds...")
            
            # Log error to database if available
            if db_integration:
                db_integration.add_system_alert(
                    message=f"Binance API error: {str(e)}",
                    alert_type="error",
                    severity="medium",
                    data={"error_type": "api_error", "retry_in": backoff_time}
                )
            
            if not TESTING_MODE:
                time.sleep(backoff_time)
            
        except Exception as e:
            consecutive_errors += 1
            backoff_time = min(loop_interval * 2 ** min(consecutive_errors, max_consecutive_errors), max_backoff_seconds)
            logger.error(f"Unexpected error in trading loop: {e}")
            logger.info(f"Retrying in {backoff_time} seconds...")
            
            # Log error to database if available
            if db_integration:
                db_integration.add_system_alert(
                    message=f"Unexpected error in trading loop: {str(e)}",
                    alert_type="error",
                    severity="high",
                    data={"error_type": "system_error", "retry_in": backoff_time}
                )
            
            if not TESTING_MODE:
                time.sleep(backoff_time)

if __name__ == '__main__':
    logger.info("=== Starting trading bot ===")
    logger.info(f"Trading symbol: {SYMBOL}")
    logger.info(f"Mode: {'TESTNET' if TESTNET else 'LIVE'}")
    
    if initialize_bot():
        try:
            trading_loop()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            
            # Log shutdown to database
            try:
                db = DatabaseIntegration()
                db.add_system_alert(
                    message="Bot shutdown initiated by user",
                    alert_type="info",
                    severity="low"
                )
            except:
                pass
        except Exception as e:
            logger.critical(f"Critical error: {e}")
            
            # Log critical error to database
            try:
                db = DatabaseIntegration()
                db.add_system_alert(
                    message=f"Critical error: {str(e)}",
                    alert_type="error",
                    severity="critical",
                    data={"error_type": "critical_system_error"}
                )
            except:
                pass
    else:
        logger.critical("Failed to initialize bot. Exiting.")
