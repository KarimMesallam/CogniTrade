import logging
import time
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, Optional, Tuple

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from bot.config import API_KEY, API_SECRET, TESTNET, get_operation_parameter
from bot.resilience import CircuitBreaker, CircuitBreakerOpenError, execute_with_resilience

logger = logging.getLogger("trading_bot")

KNOWN_QUOTE_ASSETS = (
    "USDT", "USDC", "FDUSD", "BUSD", "TUSD",
    "BTC", "ETH", "BNB", "EUR", "TRY", "USD",
)
ALGO_ORDER_TYPES = {"STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT", "TRAILING_STOP_MARKET"}
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}

EXCHANGE_TIMEOUT_SECONDS = float(get_operation_parameter("exchange_timeout_seconds", 10.0))
EXCHANGE_MAX_RETRIES = int(get_operation_parameter("exchange_max_retries", 2))
EXCHANGE_RETRY_BACKOFF_SECONDS = float(get_operation_parameter("exchange_retry_backoff_seconds", 0.5))
EXCHANGE_CIRCUIT_BREAKER_THRESHOLD = int(get_operation_parameter("exchange_circuit_breaker_threshold", 5))
EXCHANGE_CIRCUIT_BREAKER_COOLDOWN_SECONDS = float(get_operation_parameter("exchange_circuit_breaker_cooldown_seconds", 30.0))

EXCHANGE_CIRCUIT_BREAKER = CircuitBreaker(
    name="binance_exchange",
    failure_threshold=EXCHANGE_CIRCUIT_BREAKER_THRESHOLD,
    cooldown_seconds=EXCHANGE_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
)


def _create_binance_client() -> Client:
    """Initialize Binance client with request timeout when supported."""
    try:
        c = Client(
            API_KEY,
            API_SECRET,
            testnet=TESTNET,
            requests_params={"timeout": EXCHANGE_TIMEOUT_SECONDS},
        )
    except TypeError:
        logger.warning("python-binance Client does not support requests_params timeout in this version.")
        c = Client(API_KEY, API_SECRET, testnet=TESTNET)

    if TESTNET:
        c.FUTURES_URL = c.FUTURES_TESTNET_URL
        c.FUTURES_DATA_URL = c.FUTURES_DATA_TESTNET_URL
        c.FUTURES_COIN_URL = c.FUTURES_COIN_TESTNET_URL
        c.FUTURES_COIN_DATA_URL = c.FUTURES_COIN_DATA_TESTNET_URL
        logger.info("Testnet mode: futures URLs switched to %s", c.FUTURES_URL)

    return c


# Initialize Binance client with request timeout and retry/circuit protection.
client = _create_binance_client()


def get_client():
    """Returns the Binance client instance."""
    return client


def reset_exchange_circuit_breaker():
    """Reset exchange circuit breaker state (useful for tests)."""
    EXCHANGE_CIRCUIT_BREAKER.reset()


def _execute_exchange_call(operation_name: str, operation):
    """Execute Binance API calls with retry/backoff/circuit breaker."""
    return execute_with_resilience(
        operation_name=operation_name,
        operation=operation,
        max_retries=EXCHANGE_MAX_RETRIES,
        initial_backoff_seconds=EXCHANGE_RETRY_BACKOFF_SECONDS,
        retry_exceptions=(BinanceAPIException, BinanceRequestException, CircuitBreakerOpenError, TimeoutError, ConnectionError),
        circuit_breaker=EXCHANGE_CIRCUIT_BREAKER,
        logger=logger,
    )


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Best-effort conversion to Decimal for filter arithmetic."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _is_multiple(value: Decimal, step: Decimal) -> bool:
    """Check whether value is a step multiple."""
    if step <= Decimal("0"):
        return True
    ratio = value / step
    return ratio == ratio.to_integral_value()


def _infer_base_asset(symbol: str) -> str:
    """Infer base asset from a symbol string."""
    for quote_asset in KNOWN_QUOTE_ASSETS:
        if symbol.endswith(quote_asset) and len(symbol) > len(quote_asset):
            return symbol[:-len(quote_asset)]
    if len(symbol) > 4:
        return symbol[:-4]
    if len(symbol) > 3:
        return symbol[:-3]
    return symbol


def _get_weighted_avg_price(symbol: str) -> Optional[Decimal]:
    """Fetch weighted average price used by percent-price filters."""
    try:
        ticker = _execute_exchange_call(
            f"get_ticker:{symbol}",
            lambda: client.get_ticker(symbol=symbol)
        )
        weighted_avg = _to_decimal(ticker.get("weightedAvgPrice")) if ticker else None
        if weighted_avg and weighted_avg > 0:
            return weighted_avg
        last_price = _to_decimal(ticker.get("lastPrice")) if ticker else None
        return last_price
    except Exception as e:
        logger.error("Failed to fetch weighted average price for %s: %s", symbol, e)
        return None


def validate_order_filters(
    symbol: str,
    side: str,
    order_type: str,
    quantity: float,
    price: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate an order against Binance symbol filters before placement.
    """
    try:
        side = str(side).upper()
        order_type = str(order_type).upper()

        qty = _to_decimal(quantity)
        if qty is None or qty <= 0:
            return False, f"Invalid quantity {quantity}; quantity must be > 0."

        symbol_info = get_symbol_info(symbol)
        if not symbol_info or "filters" not in symbol_info:
            return False, f"Symbol info unavailable for {symbol}."

        filters = {f.get("filterType"): f for f in symbol_info.get("filters", [])}
        lot_filter = None
        if order_type == "MARKET":
            lot_filter = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE")
        else:
            lot_filter = filters.get("LOT_SIZE")

        if not lot_filter:
            return False, f"LOT_SIZE filter unavailable for {symbol}."

        min_qty = _to_decimal(lot_filter.get("minQty", "0")) or Decimal("0")
        max_qty = _to_decimal(lot_filter.get("maxQty", "0")) or Decimal("0")
        step_size = _to_decimal(lot_filter.get("stepSize", "0")) or Decimal("0")

        if qty < min_qty:
            return False, f"Quantity {qty} below minQty {min_qty}."
        if max_qty > 0 and qty > max_qty:
            return False, f"Quantity {qty} above maxQty {max_qty}."
        if step_size > 0 and not _is_multiple(qty, step_size):
            return False, f"Quantity {qty} is not a multiple of stepSize {step_size}."

        price_dec = None
        if order_type == "LIMIT":
            if price is None:
                return False, "Limit order requires a price."
            price_dec = _to_decimal(price)
            if price_dec is None or price_dec <= 0:
                return False, f"Invalid limit price {price}."

            price_filter = filters.get("PRICE_FILTER")
            if price_filter:
                min_price = _to_decimal(price_filter.get("minPrice", "0")) or Decimal("0")
                max_price = _to_decimal(price_filter.get("maxPrice", "0")) or Decimal("0")
                tick_size = _to_decimal(price_filter.get("tickSize", "0")) or Decimal("0")

                if min_price > 0 and price_dec < min_price:
                    return False, f"Price {price_dec} below minPrice {min_price}."
                if max_price > 0 and price_dec > max_price:
                    return False, f"Price {price_dec} above maxPrice {max_price}."
                if tick_size > 0 and not _is_multiple(price_dec, tick_size):
                    return False, f"Price {price_dec} is not a multiple of tickSize {tick_size}."

            percent_filter = filters.get("PERCENT_PRICE")
            weighted_avg = _get_weighted_avg_price(symbol) if percent_filter else None
            if percent_filter and weighted_avg and weighted_avg > 0:
                multiplier_up = _to_decimal(percent_filter.get("multiplierUp", "0")) or Decimal("0")
                multiplier_down = _to_decimal(percent_filter.get("multiplierDown", "0")) or Decimal("0")
                max_allowed = weighted_avg * multiplier_up
                min_allowed = weighted_avg * multiplier_down
                if multiplier_up > 0 and price_dec > max_allowed:
                    return False, f"Price {price_dec} above PERCENT_PRICE max {max_allowed}."
                if multiplier_down > 0 and price_dec < min_allowed:
                    return False, f"Price {price_dec} below PERCENT_PRICE min {min_allowed}."

            percent_by_side_filter = filters.get("PERCENT_PRICE_BY_SIDE")
            weighted_avg = _get_weighted_avg_price(symbol) if percent_by_side_filter else weighted_avg
            if percent_by_side_filter and weighted_avg and weighted_avg > 0:
                if side == "BUY":
                    multiplier_up = _to_decimal(percent_by_side_filter.get("bidMultiplierUp", "0")) or Decimal("0")
                    multiplier_down = _to_decimal(percent_by_side_filter.get("bidMultiplierDown", "0")) or Decimal("0")
                else:
                    multiplier_up = _to_decimal(percent_by_side_filter.get("askMultiplierUp", "0")) or Decimal("0")
                    multiplier_down = _to_decimal(percent_by_side_filter.get("askMultiplierDown", "0")) or Decimal("0")
                max_allowed = weighted_avg * multiplier_up
                min_allowed = weighted_avg * multiplier_down
                if multiplier_up > 0 and price_dec > max_allowed:
                    return False, f"Price {price_dec} above side-specific max {max_allowed}."
                if multiplier_down > 0 and price_dec < min_allowed:
                    return False, f"Price {price_dec} below side-specific min {min_allowed}."

        # Notional checks
        reference_price = price_dec
        if reference_price is None:
            ticker = _execute_exchange_call(
                f"get_symbol_ticker:{symbol}",
                lambda: client.get_symbol_ticker(symbol=symbol)
            )
            reference_price = _to_decimal(ticker.get("price")) if ticker else None

        if reference_price is None or reference_price <= 0:
            return False, f"Unable to determine reference price for {symbol} notional validation."

        notional = qty * reference_price

        min_notional_filter = filters.get("MIN_NOTIONAL")
        if min_notional_filter:
            min_notional = _to_decimal(min_notional_filter.get("minNotional", "0")) or Decimal("0")
            apply_to_market = str(min_notional_filter.get("applyToMarket", "true")).lower() == "true"
            should_apply = order_type != "MARKET" or apply_to_market
            if should_apply and min_notional > 0 and notional < min_notional:
                return False, f"Notional {notional} below MIN_NOTIONAL {min_notional}."

        notional_filter = filters.get("NOTIONAL")
        if notional_filter:
            min_notional = _to_decimal(notional_filter.get("minNotional", "0")) or Decimal("0")
            max_notional = _to_decimal(notional_filter.get("maxNotional", "0")) or Decimal("0")
            apply_min_to_market = str(notional_filter.get("applyMinToMarket", "true")).lower() == "true"
            apply_max_to_market = str(notional_filter.get("applyMaxToMarket", "true")).lower() == "true"
            if (order_type != "MARKET" or apply_min_to_market) and min_notional > 0 and notional < min_notional:
                return False, f"Notional {notional} below NOTIONAL min {min_notional}."
            if (order_type != "MARKET" or apply_max_to_market) and max_notional > 0 and notional > max_notional:
                return False, f"Notional {notional} above NOTIONAL max {max_notional}."

        max_orders_filter = filters.get("MAX_NUM_ORDERS")
        if max_orders_filter:
            max_orders = int(max_orders_filter.get("maxNumOrders", 0))
            if max_orders > 0:
                open_orders = get_open_orders(symbol)
                if len(open_orders) >= max_orders:
                    return False, f"Open orders {len(open_orders)} reached MAX_NUM_ORDERS {max_orders}."

        max_algo_orders_filter = filters.get("MAX_NUM_ALGO_ORDERS")
        if max_algo_orders_filter and order_type in ALGO_ORDER_TYPES:
            max_algo_orders = int(max_algo_orders_filter.get("maxNumAlgoOrders", 0))
            if max_algo_orders > 0:
                open_orders = get_open_orders(symbol)
                if len(open_orders) >= max_algo_orders:
                    return False, f"Open algo orders reached MAX_NUM_ALGO_ORDERS {max_algo_orders}."

        max_position_filter = filters.get("MAX_POSITION")
        if max_position_filter and side == "BUY":
            max_position = _to_decimal(max_position_filter.get("maxPosition", "0")) or Decimal("0")
            if max_position > 0:
                base_asset = symbol_info.get("baseAsset") or _infer_base_asset(symbol)
                balance = get_account_balance(base_asset)
                current_position = Decimal(str(balance.get("total", 0))) if balance else Decimal("0")
                projected_position = current_position + qty
                if projected_position > max_position:
                    return False, f"Projected position {projected_position} exceeds MAX_POSITION {max_position}."

        return True, None
    except (BinanceAPIException, BinanceRequestException, CircuitBreakerOpenError) as e:
        return False, f"Validation failed due to exchange call error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def synchronize_time():
    """
    Synchronize local time with Binance server time.
    """
    try:
        server_time = _execute_exchange_call("get_server_time", lambda: client.get_server_time())
        server_timestamp = server_time['serverTime']
        local_timestamp = int(time.time() * 1000)
        computed_offset = server_timestamp - local_timestamp

        logger.info(f"Time offset with Binance server: {computed_offset}ms")
        client.timestamp_offset = computed_offset
        return computed_offset
    except Exception as e:
        logger.error(f"Error synchronizing time: {e}")
        return 0


# Synchronize time on module load.
time_offset = synchronize_time()


def get_recent_closes(symbol, interval, limit=2):
    """Get recent closing prices for a symbol."""
    try:
        candles = _execute_exchange_call(
            f"get_klines:{symbol}:{interval}",
            lambda: client.get_klines(symbol=symbol, interval=interval, limit=limit)
        )
        return [float(candle[4]) for candle in candles]
    except Exception as e:
        logger.error(f"Error getting recent closes: {e}")
        return []


def get_account_balance(asset=None):
    """Get account balance for a specific asset or all assets."""
    try:
        timestamp = int(time.time() * 1000) + time_offset
        account_info = _execute_exchange_call(
            "get_account",
            lambda: client.get_account(timestamp=timestamp)
        )

        balances = account_info['balances']
        if asset:
            for balance in balances:
                if balance['asset'] == asset:
                    return {
                        'free': float(balance['free']),
                        'locked': float(balance['locked']),
                        'total': float(balance['free']) + float(balance['locked'])
                    }
            return None

        return {
            balance['asset']: {
                'free': float(balance['free']),
                'locked': float(balance['locked']),
                'total': float(balance['free']) + float(balance['locked'])
            }
            for balance in balances
            if float(balance['free']) > 0 or float(balance['locked']) > 0
        }
    except Exception as e:
        logger.error(f"Error getting account balance: {e}")
        return None


def get_symbol_info(symbol):
    """Get detailed information about a trading pair."""
    try:
        return _execute_exchange_call(
            f"get_symbol_info:{symbol}",
            lambda: client.get_symbol_info(symbol)
        )
    except Exception as e:
        logger.error(f"Error getting symbol info: {e}")
        return None


def _get_futures_symbol_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Get futures symbol metadata from exchange info payload."""
    try:
        exchange_info = _execute_exchange_call(
            "futures_exchange_info",
            lambda: client.futures_exchange_info()
        )
        for symbol_info in exchange_info.get("symbols", []):
            if symbol_info.get("symbol") == symbol:
                return symbol_info
        return None
    except Exception as e:
        logger.error("Error getting futures symbol info for %s: %s", symbol, e)
        return None


def get_futures_mark_price(symbol: str) -> Optional[float]:
    """Get futures mark price for a symbol."""
    try:
        payload = _execute_exchange_call(
            f"futures_mark_price:{symbol}",
            lambda: client.futures_mark_price(symbol=symbol)
        )
        price = payload.get("markPrice") if payload else None
        if price is None:
            return None
        return float(price)
    except Exception as e:
        logger.error("Error getting futures mark price for %s: %s", symbol, e)
        return None


def get_futures_position_qty(symbol: str) -> Optional[float]:
    """Get signed futures position quantity for a symbol (negative indicates short)."""
    try:
        positions = _execute_exchange_call(
            f"futures_position_information:{symbol}",
            lambda: client.futures_position_information(symbol=symbol)
        )
        if not positions:
            return 0.0

        position = positions[0]
        position_amt = position.get("positionAmt", "0")
        return float(position_amt)
    except Exception as e:
        logger.error("Error getting futures position quantity for %s: %s", symbol, e)
        return None


def calculate_futures_order_quantity(symbol: str, quote_amount: float, leverage: float = 1.0) -> Optional[float]:
    """
    Calculate futures quantity from margin quote amount and leverage while respecting LOT_SIZE.
    """
    try:
        mark_price = get_futures_mark_price(symbol)
        if mark_price is None or mark_price <= 0:
            logger.error("Could not determine futures mark price for %s", symbol)
            return None

        leverage_dec = Decimal(str(leverage))
        if leverage_dec <= 0:
            logger.error("Invalid leverage %s for %s", leverage, symbol)
            return None

        notional = Decimal(str(quote_amount)) * leverage_dec
        raw_qty = notional / Decimal(str(mark_price))

        symbol_info = _get_futures_symbol_info(symbol)
        if not symbol_info:
            logger.error("Futures symbol info not found for %s", symbol)
            return None

        lot_size_filter = next(
            (f for f in symbol_info.get("filters", []) if f.get("filterType") == "LOT_SIZE"),
            None
        )
        if not lot_size_filter:
            logger.error("Futures LOT_SIZE filter not found for %s", symbol)
            return None

        min_qty = Decimal(str(lot_size_filter.get("minQty", "0")))
        step_size = Decimal(str(lot_size_filter.get("stepSize", "0")))
        if step_size <= 0:
            logger.error("Invalid futures step size for %s", symbol)
            return None

        adjusted_steps = (raw_qty / step_size).to_integral_value(rounding=ROUND_DOWN)
        adjusted_qty = adjusted_steps * step_size

        if adjusted_qty < min_qty:
            logger.warning(
                "Calculated futures quantity %s is below minimum %s for %s",
                adjusted_qty,
                min_qty,
                symbol,
            )
            return None

        return float(adjusted_qty)
    except Exception as e:
        logger.error("Error calculating futures order quantity for %s: %s", symbol, e)
        return None


def place_futures_market_order(
    symbol: str,
    side: str,
    quantity: float,
    reduce_only: bool = False,
    leverage: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Place a futures market order with optional leverage update and reduce-only mode."""
    try:
        side = str(side).upper()
        if side not in {"BUY", "SELL"}:
            logger.error("Invalid futures side %s for %s", side, symbol)
            return None
        if quantity <= 0:
            logger.error("Invalid futures quantity %s for %s", quantity, symbol)
            return None

        if leverage is not None:
            leverage_int = int(float(leverage))
            if leverage_int <= 0:
                logger.error("Invalid leverage %s for %s", leverage, symbol)
                return None
            _execute_exchange_call(
                f"futures_change_leverage:{symbol}:{leverage_int}",
                lambda: client.futures_change_leverage(symbol=symbol, leverage=leverage_int)
            )

        timestamp = int(time.time() * 1000) + time_offset
        order = _execute_exchange_call(
            f"futures_create_order:{symbol}:{side}",
            lambda: client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                reduceOnly=reduce_only,
                timestamp=timestamp,
            )
        )
        return order
    except Exception as e:
        logger.error(
            "Error placing futures market order for %s (side=%s, reduce_only=%s): %s",
            symbol,
            side,
            reduce_only,
            e,
        )
        return None


def calculate_order_quantity(symbol, quote_amount):
    """
    Calculate order quantity from quote amount while respecting lot-size and filters.
    """
    try:
        ticker = _execute_exchange_call(
            f"get_symbol_ticker:{symbol}",
            lambda: client.get_symbol_ticker(symbol=symbol)
        )
        price = float(ticker['price'])

        symbol_info = get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol info not found for {symbol}")
            return None

        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter:
            logger.error(f"LOT_SIZE filter not found for {symbol}")
            return None

        min_qty = float(lot_size_filter['minQty'])
        step_size = float(lot_size_filter['stepSize'])

        raw_qty = quote_amount / price
        decimal_places = len(str(step_size).split('.')[-1].rstrip('0'))
        adjusted_qty = int(raw_qty / step_size) * step_size
        adjusted_qty = round(adjusted_qty, decimal_places)

        if adjusted_qty < min_qty:
            logger.warning(f"Calculated quantity {adjusted_qty} is below minimum {min_qty}")
            return None

        is_valid, reason = validate_order_filters(
            symbol=symbol,
            side="BUY",
            order_type="MARKET",
            quantity=adjusted_qty,
        )
        if not is_valid:
            logger.warning("Calculated quantity failed Binance filter validation: %s", reason)
            return None

        return adjusted_qty
    except Exception as e:
        logger.error(f"Error calculating order quantity: {e}")
        return None


def place_market_buy(symbol, quantity):
    """Place a market buy order."""
    try:
        is_valid, reason = validate_order_filters(symbol=symbol, side="BUY", order_type="MARKET", quantity=quantity)
        if not is_valid:
            logger.warning("Market buy blocked by pre-trade validation for %s: %s", symbol, reason)
            return None

        logger.info(f"Placing market buy order: {symbol}, quantity: {quantity}")
        timestamp = int(time.time() * 1000) + time_offset
        order = _execute_exchange_call(
            f"order_market_buy:{symbol}",
            lambda: client.order_market_buy(symbol=symbol, quantity=quantity, timestamp=timestamp)
        )
        logger.info(f"Market buy order placed successfully: {order}")
        return order
    except Exception as e:
        logger.error(f"Error placing buy order: {e}")
        return None


def place_market_sell(symbol, quantity):
    """Place a market sell order."""
    try:
        is_valid, reason = validate_order_filters(symbol=symbol, side="SELL", order_type="MARKET", quantity=quantity)
        if not is_valid:
            logger.warning("Market sell blocked by pre-trade validation for %s: %s", symbol, reason)
            return None

        logger.info(f"Placing market sell order: {symbol}, quantity: {quantity}")
        timestamp = int(time.time() * 1000) + time_offset
        order = _execute_exchange_call(
            f"order_market_sell:{symbol}",
            lambda: client.order_market_sell(symbol=symbol, quantity=quantity, timestamp=timestamp)
        )
        logger.info(f"Market sell order placed successfully: {order}")
        return order
    except Exception as e:
        logger.error(f"Error placing sell order: {e}")
        return None


def place_limit_buy(symbol, quantity, price):
    """Place a limit buy order."""
    try:
        is_valid, reason = validate_order_filters(
            symbol=symbol,
            side="BUY",
            order_type="LIMIT",
            quantity=quantity,
            price=price
        )
        if not is_valid:
            logger.warning("Limit buy blocked by pre-trade validation for %s: %s", symbol, reason)
            return None

        logger.info(f"Placing limit buy order: {symbol}, quantity: {quantity}, price: {price}")
        timestamp = int(time.time() * 1000) + time_offset
        order = _execute_exchange_call(
            f"order_limit_buy:{symbol}",
            lambda: client.order_limit_buy(symbol=symbol, quantity=quantity, price=price, timestamp=timestamp)
        )
        logger.info(f"Limit buy order placed successfully: {order}")
        return order
    except Exception as e:
        logger.error(f"Error placing limit buy order: {e}")
        return None


def place_limit_sell(symbol, quantity, price):
    """Place a limit sell order."""
    try:
        is_valid, reason = validate_order_filters(
            symbol=symbol,
            side="SELL",
            order_type="LIMIT",
            quantity=quantity,
            price=price
        )
        if not is_valid:
            logger.warning("Limit sell blocked by pre-trade validation for %s: %s", symbol, reason)
            return None

        logger.info(f"Placing limit sell order: {symbol}, quantity: {quantity}, price: {price}")
        timestamp = int(time.time() * 1000) + time_offset
        order = _execute_exchange_call(
            f"order_limit_sell:{symbol}",
            lambda: client.order_limit_sell(symbol=symbol, quantity=quantity, price=price, timestamp=timestamp)
        )
        logger.info(f"Limit sell order placed successfully: {order}")
        return order
    except Exception as e:
        logger.error(f"Error placing limit sell order: {e}")
        return None


def get_open_orders(symbol=None, trade_mode: str = "SPOT"):
    """Get all open orders for a symbol or all symbols in SPOT or FUTURES mode."""
    try:
        trade_mode = str(trade_mode or "SPOT").upper()
        timestamp = int(time.time() * 1000) + time_offset
        if trade_mode == "FUTURES":
            if symbol:
                return _execute_exchange_call(
                    f"futures_get_open_orders:{symbol}",
                    lambda: client.futures_get_open_orders(symbol=symbol, timestamp=timestamp)
                )
            return _execute_exchange_call(
                "futures_get_open_orders:all",
                lambda: client.futures_get_open_orders(timestamp=timestamp)
            )
        if symbol:
            return _execute_exchange_call(
                f"get_open_orders:{symbol}",
                lambda: client.get_open_orders(symbol=symbol, timestamp=timestamp)
            )
        return _execute_exchange_call(
            "get_open_orders:all",
            lambda: client.get_open_orders(timestamp=timestamp)
        )
    except Exception as e:
        logger.error(f"Error getting open orders: {e}")
        return []


def cancel_order(symbol, order_id, trade_mode: str = "SPOT"):
    """Cancel an open order in SPOT or FUTURES mode."""
    try:
        trade_mode = str(trade_mode or "SPOT").upper()
        logger.info(f"Cancelling order: {order_id} for {symbol}")
        timestamp = int(time.time() * 1000) + time_offset
        if trade_mode == "FUTURES":
            result = _execute_exchange_call(
                f"futures_cancel_order:{symbol}:{order_id}",
                lambda: client.futures_cancel_order(symbol=symbol, orderId=order_id, timestamp=timestamp)
            )
            logger.info(f"Futures order cancelled successfully: {result}")
            return result
        result = _execute_exchange_call(
            f"cancel_order:{symbol}:{order_id}",
            lambda: client.cancel_order(symbol=symbol, orderId=order_id, timestamp=timestamp)
        )
        logger.info(f"Order cancelled successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        return None


def get_order_status(symbol, order_id, trade_mode: str = "SPOT"):
    """Get the status of an order in SPOT or FUTURES mode."""
    try:
        trade_mode = str(trade_mode or "SPOT").upper()
        timestamp = int(time.time() * 1000) + time_offset
        if trade_mode == "FUTURES":
            return _execute_exchange_call(
                f"futures_get_order:{symbol}:{order_id}",
                lambda: client.futures_get_order(symbol=symbol, orderId=order_id, timestamp=timestamp)
            )
        return _execute_exchange_call(
            f"get_order:{symbol}:{order_id}",
            lambda: client.get_order(symbol=symbol, orderId=order_id, timestamp=timestamp)
        )
    except Exception as e:
        logger.error(f"Error getting order status: {e}")
        return None
