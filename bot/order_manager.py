import logging
import json
import os
import uuid
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Optional, Tuple
from bot.binance_api import (
    client, get_account_balance, get_order_status, get_open_orders, cancel_order,
    place_market_buy, place_market_sell,
    place_limit_buy, place_limit_sell,
    calculate_order_quantity, validate_order_filters
)
from bot.config import get_trading_parameter
from bot.db_integration import DatabaseIntegration

logger = logging.getLogger("trading_bot")

# Directory for storing order history
ORDER_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "order_logs")
os.makedirs(ORDER_LOG_DIR, exist_ok=True)
KNOWN_QUOTE_ASSETS = (
    "USDT", "USDC", "FDUSD", "BUSD", "TUSD",
    "BTC", "ETH", "BNB", "EUR", "TRY", "USD",
)

class OrderManager:
    """
    Handles order execution, tracking, and management.
    """
    def __init__(
        self,
        symbol,
        risk_percentage=1.0,
        use_database=True,
        max_order_notional_usd: Optional[float] = None,
        max_position_exposure_usd: Optional[float] = None
    ):
        """
        Initialize the order manager.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            risk_percentage: Percentage of available funds to risk per trade (1.0 = 1%)
            use_database: Whether to use database for storage (in addition to JSON files)
        """
        self.symbol = symbol
        self.risk_percentage = Decimal(str(risk_percentage))
        self.active_orders = {}
        self.order_history = []
        self.use_database = use_database
        self.order_trade_ids = {}
        self.last_reject_reason = None
        self.base_asset, self.quote_asset = self._infer_assets(symbol)
        resolved_max_order_notional = (
            max_order_notional_usd
            if max_order_notional_usd is not None
            else get_trading_parameter("max_order_notional_usd", get_trading_parameter("max_order_amount_usd", 100.0))
        )
        resolved_max_position_exposure = (
            max_position_exposure_usd
            if max_position_exposure_usd is not None
            else get_trading_parameter("max_position_exposure_usd", 250.0)
        )
        self.max_order_notional_usd = Decimal(str(resolved_max_order_notional))
        self.max_position_exposure_usd = Decimal(str(resolved_max_position_exposure))
        
        # Set Decimal precision
        getcontext().prec = 28
        
        # Initialize database integration if enabled
        if self.use_database:
            try:
                self.db = DatabaseIntegration()
                logger.info("Database integration enabled for order management")
            except Exception as e:
                logger.error(f"Failed to initialize database integration: {e}")
                self.use_database = False
        
        # Load previous orders if log exists
        self._load_order_history()

    @staticmethod
    def _infer_assets(symbol: str) -> Tuple[str, str]:
        """Infer base and quote assets from a symbol (e.g., BTCUSDT -> BTC, USDT)."""
        for quote_asset in KNOWN_QUOTE_ASSETS:
            if symbol.endswith(quote_asset) and len(symbol) > len(quote_asset):
                return symbol[:-len(quote_asset)], quote_asset
        if len(symbol) > 4:
            return symbol[:-4], symbol[-4:]
        if len(symbol) > 3:
            return symbol[:-3], symbol[-3:]
        return symbol, ""

    def _set_reject_reason(self, reason: str) -> None:
        """Store and log the most recent explicit order reject reason."""
        self.last_reject_reason = reason
        logger.warning("Order rejected for %s: %s", self.symbol, reason)

    def _clear_reject_reason(self) -> None:
        """Clear any previous reject reason before processing a new order."""
        self.last_reject_reason = None

    def _build_trade_id(self, order) -> str:
        """Create deterministic UUID trade IDs for exchange orders."""
        order_id = order.get("orderId")
        if order_id is not None:
            if order_id in self.order_trade_ids:
                return self.order_trade_ids[order_id]
            trade_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"binance:{self.symbol}:{order_id}"))
            self.order_trade_ids[order_id] = trade_id
            return trade_id
        return str(uuid.uuid4())

    def _get_reference_price(self) -> Optional[Decimal]:
        """Get best-effort current symbol price for pre-trade checks."""
        try:
            ticker = client.get_symbol_ticker(symbol=self.symbol)
            if not ticker or "price" not in ticker:
                return None
            return Decimal(str(ticker["price"]))
        except Exception as e:
            logger.error("Failed to fetch reference price for %s: %s", self.symbol, e)
            return None

    def _get_current_position_qty(self) -> Optional[Decimal]:
        """Fetch current base-asset position size for exposure checks."""
        try:
            balance = get_account_balance(self.base_asset)
            if not balance:
                return None
            total = balance.get("total")
            if total is None:
                total = float(balance.get("free", 0.0)) + float(balance.get("locked", 0.0))
            return Decimal(str(total))
        except Exception as e:
            logger.error("Failed to fetch account balance for %s: %s", self.base_asset, e)
            return None

    def _check_pre_trade_limits(
        self,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        known_notional: Optional[Decimal] = None
    ) -> bool:
        """Enforce configured pre-trade risk limits before any order placement."""
        if quantity <= Decimal("0"):
            self._set_reject_reason(f"Invalid order quantity {quantity}; quantity must be > 0.")
            return False

        reference_price = price
        requires_reference_price = (
            (known_notional is None and self.max_order_notional_usd > Decimal("0"))
            or (side == "BUY" and self.max_position_exposure_usd > Decimal("0"))
        )
        if reference_price is None and requires_reference_price:
            reference_price = self._get_reference_price()
            if reference_price is None:
                self._set_reject_reason("Unable to fetch reference price for risk validation.")
                return False

        if known_notional is not None:
            order_notional = Decimal(str(known_notional))
        elif self.max_order_notional_usd > Decimal("0"):
            order_notional = quantity * reference_price
        else:
            order_notional = Decimal("0")

        if self.max_order_notional_usd > Decimal("0") and order_notional > self.max_order_notional_usd:
            self._set_reject_reason(
                "Order notional "
                f"{order_notional:.8f} {self.quote_asset or 'quote'} exceeds configured max "
                f"{self.max_order_notional_usd:.8f}."
            )
            return False

        # Exposure limits apply to buy-side accumulation in spot mode.
        if side == "BUY" and self.max_position_exposure_usd > Decimal("0"):
            if reference_price is None:
                reference_price = self._get_reference_price()
                if reference_price is None:
                    self._set_reject_reason("Unable to fetch reference price for exposure validation.")
                    return False

            current_position_qty = self._get_current_position_qty()
            if current_position_qty is None:
                self._set_reject_reason(
                    f"Unable to determine current {self.base_asset} position for exposure validation."
                )
                return False

            projected_exposure = (current_position_qty + quantity) * reference_price
            if projected_exposure > self.max_position_exposure_usd:
                self._set_reject_reason(
                    "Projected position exposure "
                    f"{projected_exposure:.8f} {self.quote_asset or 'quote'} exceeds configured max "
                    f"{self.max_position_exposure_usd:.8f}."
                )
                return False

        return True
    
    def _load_order_history(self):
        """Load order history from disk if available."""
        log_file = os.path.join(ORDER_LOG_DIR, f"{self.symbol}_orders.json")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    self.order_history = json.load(f)
                for order_record in self.order_history:
                    order_id = order_record.get("order_id")
                    trade_id = order_record.get("trade_id")
                    if order_id is not None and trade_id:
                        self.order_trade_ids[order_id] = trade_id
                logger.info(f"Loaded {len(self.order_history)} historical orders")
            except Exception as e:
                logger.error(f"Error loading order history: {e}")
    
    def _save_order_history(self):
        """Save order history to disk."""
        log_file = os.path.join(ORDER_LOG_DIR, f"{self.symbol}_orders.json")
        try:
            with open(log_file, 'w') as f:
                json.dump(self.order_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving order history: {e}")
    
    def _log_order(self, order, action, status):
        """
        Log an order to the history.
        
        Args:
            order: Order data from Binance API
            action: String describing the action (e.g., 'BUY', 'SELL', 'CANCEL')
            status: String status of the order
        """
        if not order:
            return
            
        timestamp = datetime.now().isoformat()
        
        order_log = {
            "timestamp": timestamp,
            "symbol": self.symbol,
            "order_id": order.get("orderId"),
            "client_order_id": order.get("clientOrderId"),
            "action": action,
            "type": order.get("type"),
            "side": order.get("side"),
            "quantity": order.get("origQty"),
            "price": order.get("price"),
            "status": status,
            "fills": order.get("fills", []),
            "raw_response": order
        }
        
        # Generate deterministic UUID trade IDs by exchange order ID.
        trade_id = self._build_trade_id(order)
        order_log["trade_id"] = trade_id
        order["trade_id"] = trade_id
        
        # Add to local history
        self.order_history.append(order_log)
        self._save_order_history()
        
        # If the order is active, add it to active orders
        if status in ["NEW", "PARTIALLY_FILLED"]:
            self.active_orders[order["orderId"]] = order_log
        # If the order is complete, remove it from active orders
        elif status in ["FILLED", "CANCELED", "REJECTED", "EXPIRED"]:
            if order["orderId"] in self.active_orders:
                del self.active_orders[order["orderId"]]
        
        # Save to database if enabled
        if self.use_database:
            try:
                # Create database trade record
                db_trade_data = {
                    "trade_id": trade_id,
                    "symbol": self.symbol,
                    "side": order.get("side"),
                    "quantity": float(order.get("origQty", 0)),  # Keep as float for database
                    "timestamp": timestamp,
                    "order_id": str(order.get("orderId")) if order.get("orderId") is not None else None,
                    "status": status,
                    "strategy": "manual" if not action else action.lower(),
                    "raw_data": order
                }
                
                # Extract price from order
                # For market orders, calculate the average price from fills
                price = order.get("price")
                if (not price or price == "0.00000000") and order.get("fills"):
                    # Use Decimal for precision calculations
                    total_cost = sum(Decimal(str(fill["price"])) * Decimal(str(fill["qty"])) for fill in order["fills"])
                    total_qty = sum(Decimal(str(fill["qty"])) for fill in order["fills"])
                    if total_qty > Decimal('0'):
                        avg_price = total_cost / total_qty
                        db_trade_data["price"] = float(avg_price)  # Convert back to float for database
                else:
                    db_trade_data["price"] = float(price) if price else 0.0
                
                # Calculate profit/loss if possible
                if order.get("fills"):
                    # Use Decimal for precision calculations
                    total_cost = sum(Decimal(str(fill["price"])) * Decimal(str(fill["qty"])) for fill in order["fills"])
                    total_qty = sum(Decimal(str(fill["qty"])) for fill in order["fills"])
                    if total_qty > Decimal('0'):
                        # Add execution time and fees
                        db_trade_data["execution_time"] = order.get("transactTime", 0)
                        # Use Decimal for commission calculations
                        db_trade_data["fees"] = float(sum(Decimal(str(fill.get("commission", 0))) for fill in order["fills"]))
                
                # Save or update in database
                self.db.save_trade(db_trade_data)
                
            except Exception as e:
                logger.error(f"Error saving order to database: {e}")
    
    def execute_market_buy(self, quantity=None, quote_amount=None):
        """
        Execute a market buy order.
        
        Args:
            quantity: Optional quantity to buy (if None, calculated from quote_amount)
            quote_amount: Optional quote asset amount to spend (e.g., USDT)
            
        Returns:
            Order data or None on failure
        """
        try:
            self._clear_reject_reason()
            # If quantity is not provided, calculate it from quote_amount
            quote_amount_decimal = None
            if quantity is None and quote_amount is not None:
                # Convert to Decimal if provided
                if quote_amount is not None:
                    quote_amount_decimal = Decimal(str(quote_amount))
                quantity = calculate_order_quantity(self.symbol, float(quote_amount_decimal))  # API expects float
                if quantity is None:
                    logger.error(f"Failed to calculate quantity for {self.symbol} with {quote_amount_decimal}")
                    return None
            elif quote_amount is not None:
                quote_amount_decimal = Decimal(str(quote_amount))
            
            if quantity is None:
                logger.error("Either quantity or quote_amount must be provided")
                return None

            quantity_decimal = Decimal(str(quantity))
            if not self._check_pre_trade_limits("BUY", quantity_decimal, known_notional=quote_amount_decimal):
                return None

            is_valid, reason = validate_order_filters(
                symbol=self.symbol,
                side="BUY",
                order_type="MARKET",
                quantity=float(quantity_decimal)
            )
            if not is_valid:
                self._set_reject_reason(f"Binance filter validation failed: {reason}")
                return None
                
            logger.info(f"Executing market buy: {self.symbol}, quantity: {quantity}")
            order = place_market_buy(self.symbol, float(quantity_decimal))
            
            if order:
                self._log_order(order, "BUY", order.get("status", "UNKNOWN"))
                logger.info(f"Market buy executed: {order.get('orderId')}")
                return order
            else:
                logger.error("Market buy failed")
                return None
        except Exception as e:
            logger.error(f"Error executing market buy: {e}")
            return None
    
    def execute_market_sell(self, quantity):
        """
        Execute a market sell order.
        
        Args:
            quantity: Quantity to sell
            
        Returns:
            Order data or None on failure
        """
        try:
            self._clear_reject_reason()
            quantity_decimal = Decimal(str(quantity))
            if not self._check_pre_trade_limits("SELL", quantity_decimal):
                return None

            is_valid, reason = validate_order_filters(
                symbol=self.symbol,
                side="SELL",
                order_type="MARKET",
                quantity=float(quantity_decimal)
            )
            if not is_valid:
                self._set_reject_reason(f"Binance filter validation failed: {reason}")
                return None

            logger.info(f"Executing market sell: {self.symbol}, quantity: {quantity}")
            order = place_market_sell(self.symbol, float(quantity_decimal))
            
            if order:
                self._log_order(order, "SELL", order.get("status", "UNKNOWN"))
                logger.info(f"Market sell executed: {order.get('orderId')}")
                return order
            else:
                logger.error("Market sell failed")
                return None
        except Exception as e:
            logger.error(f"Error executing market sell: {e}")
            return None
    
    def execute_limit_buy(self, quantity, price):
        """
        Execute a limit buy order.
        
        Args:
            quantity: Quantity to buy
            price: Limit price
            
        Returns:
            Order data or None on failure
        """
        try:
            self._clear_reject_reason()
            quantity_decimal = Decimal(str(quantity))
            price_decimal = Decimal(str(price))
            if not self._check_pre_trade_limits("BUY", quantity_decimal, price=price_decimal):
                return None

            is_valid, reason = validate_order_filters(
                symbol=self.symbol,
                side="BUY",
                order_type="LIMIT",
                quantity=float(quantity_decimal),
                price=float(price_decimal)
            )
            if not is_valid:
                self._set_reject_reason(f"Binance filter validation failed: {reason}")
                return None

            logger.info(f"Executing limit buy: {self.symbol}, quantity: {quantity}, price: {price}")
            order = place_limit_buy(self.symbol, float(quantity_decimal), float(price_decimal))
            
            if order:
                self._log_order(order, "BUY", order.get("status", "UNKNOWN"))
                logger.info(f"Limit buy executed: {order.get('orderId')}")
                return order
            else:
                logger.error("Limit buy failed")
                return None
        except Exception as e:
            logger.error(f"Error executing limit buy: {e}")
            return None
    
    def execute_limit_sell(self, quantity, price):
        """
        Execute a limit sell order.
        
        Args:
            quantity: Quantity to sell
            price: Limit price
            
        Returns:
            Order data or None on failure
        """
        try:
            self._clear_reject_reason()
            quantity_decimal = Decimal(str(quantity))
            price_decimal = Decimal(str(price))
            if not self._check_pre_trade_limits("SELL", quantity_decimal, price=price_decimal):
                return None

            is_valid, reason = validate_order_filters(
                symbol=self.symbol,
                side="SELL",
                order_type="LIMIT",
                quantity=float(quantity_decimal),
                price=float(price_decimal)
            )
            if not is_valid:
                self._set_reject_reason(f"Binance filter validation failed: {reason}")
                return None

            logger.info(f"Executing limit sell: {self.symbol}, quantity: {quantity}, price: {price}")
            order = place_limit_sell(self.symbol, float(quantity_decimal), float(price_decimal))
            
            if order:
                self._log_order(order, "SELL", order.get("status", "UNKNOWN"))
                logger.info(f"Limit sell executed: {order.get('orderId')}")
                return order
            else:
                logger.error("Limit sell failed")
                return None
        except Exception as e:
            logger.error(f"Error executing limit sell: {e}")
            return None
    
    def cancel_all_orders(self):
        """
        Cancel all open orders for the symbol.
        
        Returns:
            List of canceled orders
        """
        try:
            open_orders = get_open_orders(self.symbol)
            canceled_orders = []
            
            for order in open_orders:
                order_id = order["orderId"]
                result = cancel_order(self.symbol, order_id)
                
                if result:
                    self._log_order(result, "CANCEL", "CANCELED")
                    canceled_orders.append(result)
                    logger.info(f"Canceled order: {order_id}")
                else:
                    logger.error(f"Failed to cancel order: {order_id}")
            
            return canceled_orders
        except Exception as e:
            logger.error(f"Error canceling all orders: {e}")
            return []
    
    def update_order_statuses(self):
        """
        Update the status of all active orders.
        
        Returns:
            Dictionary of updated orders
        """
        updated_orders = {}
        
        for order_id in list(self.active_orders.keys()):
            order_status = get_order_status(self.symbol, order_id)
            
            if order_status:
                current_status = order_status.get("status")
                previous_status = self.active_orders[order_id].get("status")
                
                if current_status != previous_status:
                    logger.info(f"Order {order_id} status changed: {previous_status} -> {current_status}")
                    self._log_order(order_status, self.active_orders[order_id]["action"], current_status)
                    updated_orders[order_id] = order_status
                    
                    # Update in database if enabled
                    if self.use_database and "trade_id" in self.active_orders[order_id]:
                        try:
                            trade_id = self.active_orders[order_id]["trade_id"]
                            update_data = {
                                "status": current_status,
                                "raw_data": order_status
                            }
                            self.db.update_trade(trade_id, update_data)
                        except Exception as e:
                            logger.error(f"Error updating order in database: {e}")
            else:
                logger.warning(f"Failed to get status for order: {order_id}")
        
        return updated_orders
    
    def get_active_orders(self):
        """
        Get all active orders.
        
        Returns:
            Dictionary of active orders
        """
        return self.active_orders
    
    def get_order_history(self, limit=None):
        """
        Get order history.
        
        Args:
            limit: Optional limit on number of orders to return
            
        Returns:
            List of historical orders
        """
        if limit:
            return self.order_history[-limit:]
        return self.order_history 
