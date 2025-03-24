import time
import logging
from datetime import datetime
from binance.exceptions import BinanceAPIException, BinanceRequestException
from bot.config import SYMBOL, TESTNET
from bot.strategy import simple_signal, technical_analysis_signal
from bot.binance_api import place_market_buy, place_market_sell, client, get_recent_closes, synchronize_time, get_account_balance
from bot.llm_manager import get_decision_from_llm, log_decision_with_context
from bot.order_manager import OrderManager
from bot.db_integration import DatabaseIntegration

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
                timeframe = '1m' if strategy_name == 'simple' else '1h'
                signal_ids[strategy_name] = db_integration.save_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy=strategy_name,
                    signal=signal_value,
                    llm_decision=llm_decision
                )
        
        # Only execute if both signals agree
        if signal_consensus == "BUY" and llm_decision == "BUY":
            logger.info(f"Executing BUY for {symbol}")
            # Use the order manager to execute the buy with a fixed USDT amount (adjust as needed)
            order = order_manager.execute_market_buy(quote_amount=10.0)  # 10 USDT for example
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
                
        elif signal_consensus == "SELL" and llm_decision == "SELL":
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

def get_signal_consensus(signals):
    """
    Determine a consensus signal from multiple strategies.
    
    Args:
        signals: Dictionary of strategy name -> signal value
        
    Returns:
        String: 'BUY', 'SELL', or 'HOLD'
    """
    buy_count = sum(1 for signal in signals.values() if signal == "BUY")
    sell_count = sum(1 for signal in signals.values() if signal == "SELL")
    
    # Simple majority rule with bias toward HOLD if tied
    if buy_count > sell_count:
        return "BUY"
    elif sell_count > buy_count:
        return "SELL"
    else:
        return "HOLD"

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
    
    # Initialize order manager
    order_manager = OrderManager(SYMBOL, risk_percentage=1.0, use_database=db_integration is not None)
    logger.info(f"Order manager initialized for {SYMBOL}")
    
    # Track consecutive errors to implement exponential backoff
    consecutive_errors = 0
    
    while True:
        try:
            # Update statuses of existing orders
            updated_orders = order_manager.update_order_statuses()
            if updated_orders:
                logger.info(f"Updated {len(updated_orders)} order statuses")
            
            # Get current market data
            market_data = get_market_data(SYMBOL)
            if not market_data:
                raise Exception("Failed to get market data")
            
            # Save market data to database if available
            if db_integration:
                db_integration.save_market_data(market_data, SYMBOL, '1m')
            
            # Generate trading signals from different strategies
            signals = {
                "simple": simple_signal(SYMBOL, '1m'),
                "technical": technical_analysis_signal(SYMBOL, '1h')
            }
            
            logger.info(f"Signals: {signals}")
            
            # Create a detailed prompt for the LLM with more market context
            prompt = (
                f"Current signals for {SYMBOL}: {signals}.\n"
                f"Simple signal (1m timeframe): {signals['simple']}\n"
                f"Technical analysis signal (1h timeframe): {signals['technical']}\n"
                f"Given these signals and the latest market data, should we BUY, SELL, or HOLD?"
            )
            
            # Use LLM for decision support
            llm_decision = get_decision_from_llm(prompt)
            logger.info(f"LLM decision: {llm_decision}")
            
            # Execute trade if appropriate
            order = execute_trade(signals, llm_decision, SYMBOL, market_data, order_manager, db_integration)
            
            # Reset error counter on success
            consecutive_errors = 0
            
            # Wait before next iteration
            logger.debug("Waiting for next iteration...")
            time.sleep(60)  # 1 minute between iterations
            
        except (BinanceAPIException, BinanceRequestException) as e:
            consecutive_errors += 1
            backoff_time = min(60 * 2 ** consecutive_errors, 3600)  # Exponential backoff, max 1 hour
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
            
            time.sleep(backoff_time)
            
        except Exception as e:
            consecutive_errors += 1
            backoff_time = min(60 * 2 ** consecutive_errors, 3600)
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
