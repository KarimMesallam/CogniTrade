import logging
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from bot.config import API_KEY, API_SECRET, TESTNET

logger = logging.getLogger("trading_bot")

# Initialize Binance client with longer request timeout
# Set recvWindow to 60000 milliseconds when making API calls to allow for time differences
client = Client(API_KEY, API_SECRET, testnet=TESTNET)

# Function to handle time synchronization
def synchronize_time():
    """
    Synchronize local time with Binance server time.
    This helps prevent timestamp errors during API calls.
    
    Returns:
        Integer: Time offset in milliseconds
    """
    try:
        server_time = client.get_server_time()
        server_timestamp = server_time['serverTime']
        local_timestamp = int(time.time() * 1000)
        time_offset = server_timestamp - local_timestamp
        
        logger.info(f"Time offset with Binance server: {time_offset}ms")
        
        # Update client timestamps
        client.timestamp_offset = time_offset
        
        return time_offset
    except Exception as e:
        logger.error(f"Error synchronizing time: {e}")
        return 0

# Synchronize time on module load
time_offset = synchronize_time()

def get_recent_closes(symbol, interval, limit=2):
    """
    Get recent closing prices for a symbol.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Timeframe interval (e.g., '1m', '1h', '1d')
        limit: Number of candles to retrieve
        
    Returns:
        List of closing prices
    """
    try:
        # Don't use recvWindow for get_klines as it's not supported in this context
        candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        # The 5th element in each candle is the close price
        return [float(candle[4]) for candle in candles]
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error getting recent closes: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in get_recent_closes: {e}")
        return []

def get_account_balance(asset=None):
    """
    Get account balance for a specific asset or all assets.
    
    Args:
        asset: Optional asset symbol (e.g., 'BTC', 'USDT')
        
    Returns:
        Dictionary of asset balances or single balance if asset is specified
    """
    try:
        # Use correct timestamp by adding the time offset
        timestamp = int(time.time() * 1000) + time_offset
        
        # Pass timestamp explicitly for accurate time
        account_info = client.get_account(timestamp=timestamp)
        
        if asset:
            # Find the specific asset
            for balance in account_info['balances']:
                if balance['asset'] == asset:
                    return {
                        'free': float(balance['free']),
                        'locked': float(balance['locked']),
                        'total': float(balance['free']) + float(balance['locked'])
                    }
            return None
        else:
            # Return all non-zero balances
            return {
                balance['asset']: {
                    'free': float(balance['free']),
                    'locked': float(balance['locked']),
                    'total': float(balance['free']) + float(balance['locked'])
                }
                for balance in account_info['balances']
                if float(balance['free']) > 0 or float(balance['locked']) > 0
            }
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error getting account balance: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_account_balance: {e}")
        return None

def get_symbol_info(symbol):
    """
    Get detailed information about a trading pair.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        
    Returns:
        Dictionary with symbol information
    """
    try:
        # Symbol info doesn't need recvWindow as it's a public endpoint
        info = client.get_symbol_info(symbol)
        return info
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error getting symbol info: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_symbol_info: {e}")
        return None

def calculate_order_quantity(symbol, quote_amount):
    """
    Calculate the quantity of base asset that can be bought with a given amount of quote asset.
    Respects the lot size filter.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        quote_amount: Amount of quote asset to spend
        
    Returns:
        Quantity of base asset that can be bought
    """
    try:
        # Get ticker price
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        
        # Get symbol info for lot size
        symbol_info = client.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol info not found for {symbol}")
            return None
        
        # Find the LOT_SIZE filter
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter:
            logger.error(f"LOT_SIZE filter not found for {symbol}")
            return None
        
        min_qty = float(lot_size_filter['minQty'])
        step_size = float(lot_size_filter['stepSize'])
        
        # Calculate raw quantity
        raw_qty = quote_amount / price
        
        # Adjust to step size
        decimal_places = len(str(step_size).split('.')[-1].rstrip('0'))
        adjusted_qty = int(raw_qty / step_size) * step_size
        adjusted_qty = round(adjusted_qty, decimal_places)
        
        # Ensure minimum quantity
        if adjusted_qty < min_qty:
            logger.warning(f"Calculated quantity {adjusted_qty} is below minimum {min_qty}")
            return None
        
        return adjusted_qty
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error calculating order quantity: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in calculate_order_quantity: {e}")
        return None

def place_market_buy(symbol, quantity):
    """
    Place a market buy order.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        quantity: Quantity of the base asset to buy
        
    Returns:
        Order response or None on error
    """
    try:
        logger.info(f"Placing market buy order: {symbol}, quantity: {quantity}")
        timestamp = int(time.time() * 1000) + time_offset
        order = client.order_market_buy(symbol=symbol, quantity=quantity, timestamp=timestamp)
        logger.info(f"Market buy order placed successfully: {order}")
        return order
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error placing buy order: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing buy order: {e}")
        return None

def place_market_sell(symbol, quantity):
    """
    Place a market sell order.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        quantity: Quantity of the base asset to sell
        
    Returns:
        Order response or None on error
    """
    try:
        logger.info(f"Placing market sell order: {symbol}, quantity: {quantity}")
        timestamp = int(time.time() * 1000) + time_offset
        order = client.order_market_sell(symbol=symbol, quantity=quantity, timestamp=timestamp)
        logger.info(f"Market sell order placed successfully: {order}")
        return order
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error placing sell order: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing sell order: {e}")
        return None

def place_limit_buy(symbol, quantity, price):
    """
    Place a limit buy order.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        quantity: Quantity of the base asset to buy
        price: Limit price
        
    Returns:
        Order response or None on error
    """
    try:
        logger.info(f"Placing limit buy order: {symbol}, quantity: {quantity}, price: {price}")
        timestamp = int(time.time() * 1000) + time_offset
        order = client.order_limit_buy(symbol=symbol, quantity=quantity, price=price, timestamp=timestamp)
        logger.info(f"Limit buy order placed successfully: {order}")
        return order
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error placing limit buy order: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing limit buy order: {e}")
        return None

def place_limit_sell(symbol, quantity, price):
    """
    Place a limit sell order.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        quantity: Quantity of the base asset to sell
        price: Limit price
        
    Returns:
        Order response or None on error
    """
    try:
        logger.info(f"Placing limit sell order: {symbol}, quantity: {quantity}, price: {price}")
        timestamp = int(time.time() * 1000) + time_offset
        order = client.order_limit_sell(symbol=symbol, quantity=quantity, price=price, timestamp=timestamp)
        logger.info(f"Limit sell order placed successfully: {order}")
        return order
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error placing limit sell order: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing limit sell order: {e}")
        return None

def get_open_orders(symbol=None):
    """
    Get all open orders for a symbol or all symbols.
    
    Args:
        symbol: Optional trading pair symbol (e.g., 'BTCUSDT')
        
    Returns:
        List of open orders
    """
    try:
        timestamp = int(time.time() * 1000) + time_offset
        if symbol:
            orders = client.get_open_orders(symbol=symbol, timestamp=timestamp)
        else:
            orders = client.get_open_orders(timestamp=timestamp)
        return orders
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error getting open orders: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting open orders: {e}")
        return []

def cancel_order(symbol, order_id):
    """
    Cancel an open order.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        order_id: ID of the order to cancel
        
    Returns:
        Cancel response or None on error
    """
    try:
        logger.info(f"Cancelling order: {order_id} for {symbol}")
        timestamp = int(time.time() * 1000) + time_offset
        result = client.cancel_order(symbol=symbol, orderId=order_id, timestamp=timestamp)
        logger.info(f"Order cancelled successfully: {result}")
        return result
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error cancelling order: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error cancelling order: {e}")
        return None

def get_order_status(symbol, order_id):
    """
    Get the status of an order.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        order_id: ID of the order to check
        
    Returns:
        Order status or None on error
    """
    try:
        timestamp = int(time.time() * 1000) + time_offset
        order = client.get_order(symbol=symbol, orderId=order_id, timestamp=timestamp)
        return order
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"Error getting order status: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting order status: {e}")
        return None
