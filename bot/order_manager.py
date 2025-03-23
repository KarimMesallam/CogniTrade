import logging
import json
import os
import uuid
from datetime import datetime
from bot.binance_api import (
    get_order_status, get_open_orders, cancel_order, 
    place_market_buy, place_market_sell, 
    place_limit_buy, place_limit_sell,
    calculate_order_quantity
)
from bot.db_integration import DatabaseIntegration

logger = logging.getLogger("trading_bot")

# Directory for storing order history
ORDER_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "order_logs")
os.makedirs(ORDER_LOG_DIR, exist_ok=True)

class OrderManager:
    """
    Handles order execution, tracking, and management.
    """
    def __init__(self, symbol, risk_percentage=1.0, use_database=True):
        """
        Initialize the order manager.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            risk_percentage: Percentage of available funds to risk per trade (1.0 = 1%)
            use_database: Whether to use database for storage (in addition to JSON files)
        """
        self.symbol = symbol
        self.risk_percentage = risk_percentage
        self.active_orders = {}
        self.order_history = []
        self.use_database = use_database
        
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
    
    def _load_order_history(self):
        """Load order history from disk if available."""
        log_file = os.path.join(ORDER_LOG_DIR, f"{self.symbol}_orders.json")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    self.order_history = json.load(f)
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
        
        # Generate a unique trade ID if not present
        trade_id = f"{self.symbol}_{action}_{order.get('orderId')}_{int(datetime.now().timestamp())}"
        order_log["trade_id"] = trade_id
        
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
                    "quantity": float(order.get("origQty", 0)),
                    "price": float(order.get("price", 0)) if order.get("price") and order.get("price") != "0.00000000" else None,
                    "timestamp": timestamp,
                    "order_id": str(order.get("orderId")),
                    "status": status,
                    "strategy": "manual" if not action else action.lower(),
                    "raw_data": order
                }
                
                # Calculate profit/loss if possible
                if order.get("fills"):
                    total_cost = sum(float(fill["price"]) * float(fill["qty"]) for fill in order["fills"])
                    total_qty = sum(float(fill["qty"]) for fill in order["fills"])
                    if total_qty > 0:
                        # Add execution time and fees
                        db_trade_data["execution_time"] = order.get("transactTime", 0)
                        db_trade_data["fees"] = sum(float(fill.get("commission", 0)) for fill in order["fills"])
                
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
            # If quantity is not provided, calculate it from quote_amount
            if quantity is None and quote_amount is not None:
                quantity = calculate_order_quantity(self.symbol, quote_amount)
                if quantity is None:
                    logger.error(f"Failed to calculate quantity for {self.symbol} with {quote_amount}")
                    return None
            
            if quantity is None:
                logger.error("Either quantity or quote_amount must be provided")
                return None
                
            logger.info(f"Executing market buy: {self.symbol}, quantity: {quantity}")
            order = place_market_buy(self.symbol, quantity)
            
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
            logger.info(f"Executing market sell: {self.symbol}, quantity: {quantity}")
            order = place_market_sell(self.symbol, quantity)
            
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
            logger.info(f"Executing limit buy: {self.symbol}, quantity: {quantity}, price: {price}")
            order = place_limit_buy(self.symbol, quantity, price)
            
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
            logger.info(f"Executing limit sell: {self.symbol}, quantity: {quantity}, price: {price}")
            order = place_limit_sell(self.symbol, quantity, price)
            
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