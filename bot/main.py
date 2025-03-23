import time
import logging
from datetime import datetime
from binance.exceptions import BinanceAPIException, BinanceRequestException
from bot.config import SYMBOL, TESTNET
from bot.strategy import simple_signal, technical_analysis_signal
from bot.binance_api import place_market_buy, place_market_sell, client, get_recent_closes, synchronize_time, get_account_balance
from bot.llm_manager import get_decision_from_llm, log_decision_with_context
from bot.order_manager import OrderManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

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

def execute_trade(signals, llm_decision, symbol, market_data, order_manager):
    """Execute a trade based on signals and decisions."""
    try:
        # Get consensus from signals
        signal_consensus = get_signal_consensus(signals)
        logger.info(f"Signal consensus: {signal_consensus}")
        
        # Log the decision with full context for later analysis
        log_decision_with_context(llm_decision, signals, market_data)
        
        # Only execute if both signals agree
        if signal_consensus == "BUY" and llm_decision == "BUY":
            logger.info(f"Executing BUY for {symbol}")
            # Use the order manager to execute the buy with a fixed USDT amount (adjust as needed)
            order = order_manager.execute_market_buy(quote_amount=10.0)  # 10 USDT for example
            if order:
                logger.info(f"Buy order executed successfully: {order['orderId']}")
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
    
    # Initialize order manager
    order_manager = OrderManager(SYMBOL, risk_percentage=1.0)
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
            order = execute_trade(signals, llm_decision, SYMBOL, market_data, order_manager)
            
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
            time.sleep(backoff_time)
            
        except Exception as e:
            consecutive_errors += 1
            backoff_time = min(60 * 2 ** consecutive_errors, 3600)
            logger.error(f"Unexpected error in trading loop: {e}")
            logger.info(f"Retrying in {backoff_time} seconds...")
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
        except Exception as e:
            logger.critical(f"Critical error: {e}")
    else:
        logger.critical("Failed to initialize bot. Exiting.")
