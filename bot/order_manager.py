import logging
import json
import os
import uuid
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Optional, Tuple, Dict, Any
from bot.binance_api import (
    client, get_account_balance, get_order_status, get_open_orders, cancel_order,
    place_market_buy, place_market_sell,
    place_limit_buy, place_limit_sell,
    calculate_order_quantity, validate_order_filters,
    calculate_futures_order_quantity, get_futures_mark_price,
    get_futures_position_qty, place_futures_market_order
)
from bot.config import get_trading_parameter, get_trade_mode, is_futures_short_enabled
from bot.db_integration import DatabaseIntegration
from bot.reconciliation import ExchangeReconciler

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
        max_position_exposure_usd: Optional[float] = None,
        trade_mode: Optional[str] = None,
        enable_futures_shorts: Optional[bool] = None,
        max_short_notional_usd: Optional[float] = None,
        default_futures_leverage: Optional[float] = None,
        max_short_leverage: Optional[float] = None,
        min_short_liquidation_buffer_pct: Optional[float] = None,
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
        resolved_trade_mode = (trade_mode or get_trade_mode()).upper()
        self.trade_mode = resolved_trade_mode if resolved_trade_mode in {"SPOT", "FUTURES"} else "SPOT"
        self.enable_futures_shorts = (
            bool(enable_futures_shorts)
            if enable_futures_shorts is not None
            else bool(is_futures_short_enabled())
        )
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
        resolved_max_short_notional = (
            max_short_notional_usd
            if max_short_notional_usd is not None
            else get_trading_parameter("max_short_notional_usd", resolved_max_order_notional)
        )
        resolved_default_futures_leverage = (
            default_futures_leverage
            if default_futures_leverage is not None
            else get_trading_parameter("default_futures_leverage", 2.0)
        )
        resolved_max_short_leverage = (
            max_short_leverage
            if max_short_leverage is not None
            else get_trading_parameter("max_short_leverage", 3.0)
        )
        resolved_min_short_liq_buffer = (
            min_short_liquidation_buffer_pct
            if min_short_liquidation_buffer_pct is not None
            else get_trading_parameter("min_short_liquidation_buffer_pct", 20.0)
        )
        self.max_short_notional_usd = Decimal(str(resolved_max_short_notional))
        self.default_futures_leverage = Decimal(str(resolved_default_futures_leverage))
        self.max_short_leverage = Decimal(str(resolved_max_short_leverage))
        self.min_short_liquidation_buffer_pct = Decimal(str(resolved_min_short_liq_buffer))
        self.reconciler = ExchangeReconciler(symbol=self.symbol, trade_mode=self.trade_mode)
        self.last_reconciliation_report: Optional[Dict[str, Any]] = None
        
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
            if self.trade_mode == "FUTURES":
                mark_price = get_futures_mark_price(self.symbol)
                if mark_price is None:
                    return None
                return Decimal(str(mark_price))

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
            if self.trade_mode == "FUTURES":
                position_qty = get_futures_position_qty(self.symbol)
                if position_qty is None:
                    return None
                return Decimal(str(position_qty))

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

    def get_position_quantity(self) -> Optional[Decimal]:
        """Public accessor for current signed position quantity."""
        return self._get_current_position_qty()

    def _estimate_short_liquidation_buffer_pct(self, leverage: Decimal) -> Decimal:
        """Estimate liquidation buffer percentage from leverage in one-way futures mode."""
        if leverage <= Decimal("0"):
            return Decimal("0")
        return Decimal("100") / leverage

    def _check_pre_trade_limits(
        self,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        known_notional: Optional[Decimal] = None,
        leverage: Optional[Decimal] = None,
        is_short_entry: bool = False,
        enforce_exposure_limit: bool = True,
    ) -> bool:
        """Enforce configured pre-trade risk limits before any order placement."""
        if quantity <= Decimal("0"):
            self._set_reject_reason(f"Invalid order quantity {quantity}; quantity must be > 0.")
            return False

        side = str(side).upper()
        reference_price = price
        requires_reference_price = (
            (known_notional is None and self.max_order_notional_usd > Decimal("0"))
            or (
                enforce_exposure_limit
                and side == "BUY"
                and self.max_position_exposure_usd > Decimal("0")
            )
            or (is_short_entry and (self.max_short_notional_usd > Decimal("0") or self.max_position_exposure_usd > Decimal("0")))
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

        if is_short_entry:
            if self.trade_mode != "FUTURES":
                self._set_reject_reason("Short entry requires TRADE_MODE=FUTURES.")
                return False
            if not self.enable_futures_shorts:
                self._set_reject_reason("Futures shorting is disabled by configuration.")
                return False

            leverage_dec = leverage if leverage is not None else self.default_futures_leverage
            leverage_dec = Decimal(str(leverage_dec))
            if leverage_dec <= Decimal("0"):
                self._set_reject_reason(f"Invalid short leverage {leverage_dec}; leverage must be > 0.")
                return False
            if self.max_short_leverage > Decimal("0") and leverage_dec > self.max_short_leverage:
                self._set_reject_reason(
                    f"Requested short leverage {leverage_dec:.2f} exceeds max {self.max_short_leverage:.2f}."
                )
                return False

            estimated_buffer_pct = self._estimate_short_liquidation_buffer_pct(leverage_dec)
            if (
                self.min_short_liquidation_buffer_pct > Decimal("0")
                and estimated_buffer_pct < self.min_short_liquidation_buffer_pct
            ):
                self._set_reject_reason(
                    "Estimated short liquidation buffer "
                    f"{estimated_buffer_pct:.2f}% is below configured minimum "
                    f"{self.min_short_liquidation_buffer_pct:.2f}%."
                )
                return False

            if self.max_short_notional_usd > Decimal("0") and order_notional > self.max_short_notional_usd:
                self._set_reject_reason(
                    "Short order notional "
                    f"{order_notional:.8f} {self.quote_asset or 'quote'} exceeds configured max "
                    f"{self.max_short_notional_usd:.8f}."
                )
                return False

            if reference_price is None:
                reference_price = self._get_reference_price()
                if reference_price is None:
                    self._set_reject_reason("Unable to fetch reference price for short exposure validation.")
                    return False

            current_position_qty = self._get_current_position_qty()
            if current_position_qty is None:
                self._set_reject_reason("Unable to determine current futures position for short exposure validation.")
                return False
            if current_position_qty > Decimal("0"):
                self._set_reject_reason("Cannot open a short while a long futures position is open.")
                return False

            current_short_qty = abs(min(current_position_qty, Decimal("0")))
            projected_short_qty = current_short_qty + quantity
            projected_short_exposure = projected_short_qty * reference_price
            if (
                self.max_position_exposure_usd > Decimal("0")
                and projected_short_exposure > self.max_position_exposure_usd
            ):
                self._set_reject_reason(
                    "Projected short exposure "
                    f"{projected_short_exposure:.8f} {self.quote_asset or 'quote'} exceeds configured max "
                    f"{self.max_position_exposure_usd:.8f}."
                )
                return False

            return True

        # Exposure limits apply to buy-side accumulation (spot and futures long).
        if (
            enforce_exposure_limit
            and side == "BUY"
            and self.max_position_exposure_usd > Decimal("0")
        ):
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
            if self.trade_mode == "FUTURES":
                self._set_reject_reason(
                    "Use execute_market_cover for futures short exits; spot market buys are disabled in FUTURES mode."
                )
                return None
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
            if self.trade_mode == "FUTURES":
                self._set_reject_reason(
                    "Use execute_market_short for futures shorts; spot market sells are disabled in FUTURES mode."
                )
                return None
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

    def execute_market_short(self, quantity=None, quote_amount=None, leverage=None):
        """
        Execute a futures market short (SELL) order.

        Args:
            quantity: Base-asset quantity to short.
            quote_amount: Optional quote-asset margin amount used with leverage to derive quantity.
            leverage: Optional leverage override.
        """
        try:
            self._clear_reject_reason()
            if self.trade_mode != "FUTURES":
                self._set_reject_reason("Short execution requires TRADE_MODE=FUTURES.")
                return None
            if not self.enable_futures_shorts:
                self._set_reject_reason("Futures shorting is disabled by configuration.")
                return None

            leverage_dec = Decimal(str(leverage)) if leverage is not None else self.default_futures_leverage
            quote_amount_decimal = None
            if quantity is None and quote_amount is not None:
                quote_amount_decimal = Decimal(str(quote_amount))
                quantity = calculate_futures_order_quantity(
                    self.symbol,
                    float(quote_amount_decimal),
                    leverage=float(leverage_dec),
                )
                if quantity is None:
                    self._set_reject_reason("Failed to calculate futures short quantity.")
                    return None
            elif quote_amount is not None:
                quote_amount_decimal = Decimal(str(quote_amount))

            if quantity is None:
                self._set_reject_reason("Either quantity or quote_amount must be provided for futures short.")
                return None

            quantity_decimal = Decimal(str(quantity))
            known_notional = None
            if quote_amount_decimal is not None:
                known_notional = quote_amount_decimal * leverage_dec

            if not self._check_pre_trade_limits(
                side="SELL",
                quantity=quantity_decimal,
                known_notional=known_notional,
                leverage=leverage_dec,
                is_short_entry=True,
                enforce_exposure_limit=False,
            ):
                return None

            logger.info(
                "Executing futures market short: %s, quantity: %s, leverage: %s",
                self.symbol,
                quantity_decimal,
                leverage_dec,
            )
            order = place_futures_market_order(
                symbol=self.symbol,
                side="SELL",
                quantity=float(quantity_decimal),
                reduce_only=False,
                leverage=float(leverage_dec),
            )
            if order:
                self._log_order(order, "SHORT", order.get("status", "UNKNOWN"))
                logger.info("Futures short executed: %s", order.get("orderId"))
                return order

            logger.error("Futures short execution failed")
            return None
        except Exception as e:
            logger.error("Error executing futures market short: %s", e)
            return None

    def execute_market_long(self, quantity=None, quote_amount=None, leverage=None):
        """
        Execute a futures market long (BUY) order.

        Args:
            quantity: Base-asset quantity to go long.
            quote_amount: Optional quote-asset margin amount used with leverage to derive quantity.
            leverage: Optional leverage override.
        """
        try:
            self._clear_reject_reason()
            if self.trade_mode != "FUTURES":
                self._set_reject_reason("Long execution requires TRADE_MODE=FUTURES.")
                return None

            leverage_dec = Decimal(str(leverage)) if leverage is not None else self.default_futures_leverage
            quote_amount_decimal = None
            if quantity is None and quote_amount is not None:
                quote_amount_decimal = Decimal(str(quote_amount))
                quantity = calculate_futures_order_quantity(
                    self.symbol,
                    float(quote_amount_decimal),
                    leverage=float(leverage_dec),
                )
                if quantity is None:
                    self._set_reject_reason("Failed to calculate futures long quantity.")
                    return None
            elif quote_amount is not None:
                quote_amount_decimal = Decimal(str(quote_amount))

            if quantity is None:
                self._set_reject_reason("Either quantity or quote_amount must be provided for futures long.")
                return None

            quantity_decimal = Decimal(str(quantity))

            current_position_qty = self._get_current_position_qty()
            if current_position_qty is not None and current_position_qty < Decimal("0"):
                self._set_reject_reason("Cannot open long while short position exists.")
                return None

            if not self._check_pre_trade_limits(
                side="BUY",
                quantity=quantity_decimal,
                enforce_exposure_limit=True,
            ):
                return None

            logger.info(
                "Executing futures market long: %s, quantity: %s, leverage: %s",
                self.symbol,
                quantity_decimal,
                leverage_dec,
            )
            order = place_futures_market_order(
                symbol=self.symbol,
                side="BUY",
                quantity=float(quantity_decimal),
                reduce_only=False,
                leverage=float(leverage_dec),
            )
            if order:
                self._log_order(order, "LONG", order.get("status", "UNKNOWN"))
                logger.info("Futures long executed: %s", order.get("orderId"))
                return order

            logger.error("Futures long execution failed")
            return None
        except Exception as e:
            logger.error("Error executing futures market long: %s", e)
            return None

    def execute_market_close_long(self, quantity):
        """
        Close an existing futures long using a reduce-only market SELL.
        """
        try:
            self._clear_reject_reason()
            if self.trade_mode != "FUTURES":
                self._set_reject_reason("Close long requires TRADE_MODE=FUTURES.")
                return None

            quantity_decimal = Decimal(str(quantity))
            if quantity_decimal <= Decimal("0"):
                self._set_reject_reason(f"Invalid close long quantity {quantity_decimal}; quantity must be > 0.")
                return None

            current_position_qty = self._get_current_position_qty()
            if current_position_qty is None:
                self._set_reject_reason("Unable to determine current futures position for close long.")
                return None
            if current_position_qty <= Decimal("0"):
                self._set_reject_reason("No open long position to close.")
                return None

            close_quantity = min(current_position_qty, quantity_decimal)
            if close_quantity <= Decimal("0"):
                self._set_reject_reason("Computed close long quantity is zero.")
                return None

            if not self._check_pre_trade_limits(
                side="SELL",
                quantity=close_quantity,
                known_notional=Decimal("0"),
                enforce_exposure_limit=False,
            ):
                return None

            logger.info("Executing futures close long: %s, quantity: %s", self.symbol, close_quantity)
            order = place_futures_market_order(
                symbol=self.symbol,
                side="SELL",
                quantity=float(close_quantity),
                reduce_only=True,
            )
            if order:
                self._log_order(order, "CLOSE_LONG", order.get("status", "UNKNOWN"))
                logger.info("Futures close long executed: %s", order.get("orderId"))
                return order

            logger.error("Futures close long execution failed")
            return None
        except Exception as e:
            logger.error("Error executing futures close long: %s", e)
            return None

    def execute_market_cover(self, quantity):
        """
        Close an existing futures short using a reduce-only market BUY.
        """
        try:
            self._clear_reject_reason()
            if self.trade_mode != "FUTURES":
                self._set_reject_reason("Short cover requires TRADE_MODE=FUTURES.")
                return None

            quantity_decimal = Decimal(str(quantity))
            if quantity_decimal <= Decimal("0"):
                self._set_reject_reason(f"Invalid cover quantity {quantity_decimal}; quantity must be > 0.")
                return None

            current_position_qty = self._get_current_position_qty()
            if current_position_qty is None:
                self._set_reject_reason("Unable to determine current futures position for short cover.")
                return None
            if current_position_qty >= Decimal("0"):
                self._set_reject_reason("No open short position to cover.")
                return None

            cover_quantity = min(abs(current_position_qty), quantity_decimal)
            if cover_quantity <= Decimal("0"):
                self._set_reject_reason("Computed cover quantity is zero.")
                return None

            if not self._check_pre_trade_limits(
                side="BUY",
                quantity=cover_quantity,
                known_notional=Decimal("0"),
                enforce_exposure_limit=False,
            ):
                return None

            logger.info("Executing futures short cover: %s, quantity: %s", self.symbol, cover_quantity)
            order = place_futures_market_order(
                symbol=self.symbol,
                side="BUY",
                quantity=float(cover_quantity),
                reduce_only=True,
            )
            if order:
                self._log_order(order, "COVER", order.get("status", "UNKNOWN"))
                logger.info("Futures short cover executed: %s", order.get("orderId"))
                return order

            logger.error("Futures short cover execution failed")
            return None
        except Exception as e:
            logger.error("Error executing futures short cover: %s", e)
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
            if self.trade_mode == "FUTURES":
                self._set_reject_reason("Spot limit buys are disabled in FUTURES mode.")
                return None
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
            if self.trade_mode == "FUTURES":
                self._set_reject_reason("Spot limit sells are disabled in FUTURES mode.")
                return None
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
            open_orders = get_open_orders(self.symbol, trade_mode=self.trade_mode)
            canceled_orders = []
            
            for order in open_orders:
                order_id = order["orderId"]
                result = cancel_order(self.symbol, order_id, trade_mode=self.trade_mode)
                
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
            order_status = get_order_status(self.symbol, order_id, trade_mode=self.trade_mode)
            
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

    def _build_active_order_entry(self, order_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert exchange order payload into local active-order schema."""
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "order_id": order_payload.get("orderId"),
            "client_order_id": order_payload.get("clientOrderId"),
            "action": str(order_payload.get("side", "UNKNOWN")).upper(),
            "type": order_payload.get("type"),
            "side": order_payload.get("side"),
            "quantity": order_payload.get("origQty", order_payload.get("origQty", "0")),
            "price": order_payload.get("price", order_payload.get("avgPrice", "0")),
            "status": order_payload.get("status", "UNKNOWN"),
            "fills": order_payload.get("fills", []),
            "raw_response": order_payload,
        }

    def _estimate_local_position_qty_from_history(self) -> Optional[Decimal]:
        """
        Estimate signed position from local filled order history.

        This is a best-effort estimate used only for reconciliation diagnostics.
        """
        if not self.order_history:
            return Decimal("0")

        signed_qty = Decimal("0")
        for entry in self.order_history:
            try:
                if str(entry.get("status", "")).upper() != "FILLED":
                    continue
                qty = Decimal(str(entry.get("quantity", "0")))
                if qty <= Decimal("0"):
                    continue

                action = str(entry.get("action", entry.get("side", ""))).upper()
                side = str(entry.get("side", "")).upper()
                if action in {"BUY", "COVER", "LONG"} or side == "BUY":
                    signed_qty += qty
                elif action in {"SELL", "SHORT", "CLOSE_LONG"} or side == "SELL":
                    signed_qty -= qty
            except Exception:
                continue
        return signed_qty

    def recover_state_from_exchange(self) -> Dict[str, Any]:
        """
        Recover local active-order state from exchange open orders.
        """
        report = {
            "symbol": self.symbol,
            "trade_mode": self.trade_mode,
            "recovered": 0,
            "exchange_open_count": 0,
            "reconciled_at": datetime.utcnow().isoformat(),
        }
        try:
            exchange_open_orders = get_open_orders(self.symbol, trade_mode=self.trade_mode)
            report["exchange_open_count"] = len(exchange_open_orders)
            recovered = 0
            for order in exchange_open_orders:
                order_id = order.get("orderId")
                if order_id is None:
                    continue
                if order_id in self.active_orders:
                    continue
                self.active_orders[order_id] = self._build_active_order_entry(order)
                recovered += 1
            report["recovered"] = recovered
            self.last_reconciliation_report = report
            logger.info(
                "Recovered %s/%s active orders from exchange for %s (%s)",
                recovered,
                len(exchange_open_orders),
                self.symbol,
                self.trade_mode,
            )
            return report
        except Exception as e:
            logger.error("Failed to recover state from exchange for %s: %s", self.symbol, e)
            report["error"] = str(e)
            self.last_reconciliation_report = report
            return report

    def reconcile_with_exchange(self, position_tolerance: float = 1e-8) -> Dict[str, Any]:
        """
        Reconcile local active orders against exchange state.
        """
        try:
            exchange_open_orders = get_open_orders(self.symbol, trade_mode=self.trade_mode)
            recon_report = self.reconciler.reconcile_orders(
                local_active_orders=self.active_orders,
                exchange_open_orders=exchange_open_orders,
            )

            exchange_by_id: Dict[str, Dict[str, Any]] = {
                str(order.get("orderId")): order
                for order in exchange_open_orders
                if order.get("orderId") is not None
            }

            for stale_order_id in recon_report.stale_local_order_ids:
                stale_key = int(stale_order_id) if stale_order_id.isdigit() else stale_order_id
                status_payload = get_order_status(
                    self.symbol,
                    stale_key,
                    trade_mode=self.trade_mode,
                )
                if status_payload:
                    previous = self.active_orders.get(stale_key) or self.active_orders.get(stale_order_id) or {}
                    action = previous.get("action", str(status_payload.get("side", "UNKNOWN")).upper())
                    self._log_order(status_payload, action, status_payload.get("status", "UNKNOWN"))
                else:
                    self.active_orders.pop(stale_key, None)
                    self.active_orders.pop(stale_order_id, None)

            for missing_order_id in recon_report.missing_local_order_ids:
                order_payload = exchange_by_id.get(missing_order_id)
                if not order_payload:
                    continue
                order_key = order_payload.get("orderId")
                self.active_orders[order_key] = self._build_active_order_entry(order_payload)

            local_position = self._estimate_local_position_qty_from_history()
            exchange_position = self._get_current_position_qty()
            mismatch, delta = self.reconciler.reconcile_position(
                local_position_qty=float(local_position) if local_position is not None else None,
                exchange_position_qty=float(exchange_position) if exchange_position is not None else None,
                tolerance=position_tolerance,
            )
            recon_report.position_mismatch = mismatch
            recon_report.position_delta = delta

            report_dict = recon_report.to_dict()
            self.last_reconciliation_report = report_dict
            if mismatch:
                logger.warning(
                    "Position mismatch detected for %s (%s): delta=%s",
                    self.symbol,
                    self.trade_mode,
                    delta,
                )
            return report_dict
        except Exception as e:
            logger.error("Error reconciling %s against exchange: %s", self.symbol, e)
            report = {
                "symbol": self.symbol,
                "trade_mode": self.trade_mode,
                "reconciled_at": datetime.utcnow().isoformat(),
                "error": str(e),
            }
            self.last_reconciliation_report = report
            return report

    def get_recent_turnover_notional(self, window_seconds: int = 3600) -> Decimal:
        """
        Estimate filled-order turnover notional over a recent time window.
        """
        if window_seconds <= 0:
            return Decimal("0")

        cutoff_ts = datetime.now().timestamp() - float(window_seconds)
        turnover = Decimal("0")

        for entry in self.order_history:
            try:
                status = str(entry.get("status", "")).upper()
                if status != "FILLED":
                    continue

                timestamp_raw = entry.get("timestamp")
                if not timestamp_raw:
                    continue
                order_ts = datetime.fromisoformat(str(timestamp_raw)).timestamp()
                if order_ts < cutoff_ts:
                    continue

                quantity = Decimal(str(entry.get("quantity", "0")))
                if quantity <= Decimal("0"):
                    continue

                price_raw = entry.get("price")
                if price_raw in (None, "", "0", "0.0", "0.00000000"):
                    fills = entry.get("fills") or []
                    total_cost = Decimal("0")
                    total_qty = Decimal("0")
                    for fill in fills:
                        fill_qty = Decimal(str(fill.get("qty", "0")))
                        fill_price = Decimal(str(fill.get("price", "0")))
                        total_cost += fill_qty * fill_price
                        total_qty += fill_qty
                    if total_qty <= Decimal("0"):
                        continue
                    price = total_cost / total_qty
                else:
                    price = Decimal(str(price_raw))

                if price <= Decimal("0"):
                    continue
                turnover += quantity * price
            except Exception:
                continue

        return turnover

    def exceeds_turnover_limit(
        self,
        projected_notional: float,
        capital_base_usd: float,
        max_turnover_ratio: float,
        window_seconds: int = 3600,
    ) -> bool:
        """
        Check if recent turnover plus projected notional breaches configured cap.
        """
        if max_turnover_ratio <= 0 or capital_base_usd <= 0:
            return False

        projected = Decimal(str(projected_notional))
        if projected <= Decimal("0"):
            return False

        recent_turnover = self.get_recent_turnover_notional(window_seconds=window_seconds)
        turnover_cap = Decimal(str(capital_base_usd)) * Decimal(str(max_turnover_ratio))
        if recent_turnover + projected > turnover_cap:
            self._set_reject_reason(
                "Turnover cap exceeded: "
                f"recent {recent_turnover:.8f} + projected {projected:.8f} > cap {turnover_cap:.8f}"
            )
            return True
        return False
