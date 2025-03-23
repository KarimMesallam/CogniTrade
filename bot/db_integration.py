import logging
import uuid
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

from bot.database import Database

logger = logging.getLogger("trading_bot")

class DatabaseIntegration:
    """
    Integration layer between the trading system and the database.
    This class provides methods to save trading data to the database.
    """
    
    def __init__(self):
        """Initialize the database integration."""
        try:
            self.db = Database()
            logger.info("Database integration initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database integration: {e}")
            raise
    
    def save_signal(self, symbol: str, timeframe: str, strategy: str, 
                    signal: str, indicators: Dict[str, Any] = None,
                    llm_decision: str = None) -> int:
        """
        Save a trading signal to the database.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe of the signal (e.g., '1h')
            strategy: Strategy that generated the signal
            signal: Signal type ('BUY', 'SELL', 'HOLD')
            indicators: Optional dictionary of indicator values
            llm_decision: Optional LLM decision
            
        Returns:
            Signal ID if successful, -1 otherwise
        """
        try:
            signal_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': strategy,
                'signal': signal,
                'timestamp': datetime.now().isoformat(),
                'indicators': indicators,
                'llm_decision': llm_decision,
                'executed': False
            }
            
            signal_id = self.db.insert_signal(signal_data)
            if signal_id > 0:
                logger.info(f"Saved {signal} signal for {symbol} ({strategy}) to database with ID {signal_id}")
            else:
                logger.warning(f"Failed to save {signal} signal for {symbol} ({strategy}) to database")
            
            return signal_id
        except Exception as e:
            logger.error(f"Error saving signal to database: {e}")
            return -1
    
    def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Save a trade to the database.
        
        Args:
            trade_data: Dictionary containing trade information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure trade has a unique ID
            if 'trade_id' not in trade_data:
                trade_data['trade_id'] = str(uuid.uuid4())
            
            # Ensure timestamp exists
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now().isoformat()
            
            success = self.db.insert_trade(trade_data)
            if success:
                logger.info(f"Saved trade {trade_data['trade_id']} to database")
            else:
                logger.warning(f"Failed to save trade {trade_data.get('trade_id', 'unknown')} to database")
            
            return success
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
            return False
    
    def update_trade(self, trade_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a trade in the database.
        
        Args:
            trade_id: ID of the trade to update
            update_data: Dictionary containing fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.db.update_trade(trade_id, update_data)
            if success:
                logger.info(f"Updated trade {trade_id} in database")
            else:
                logger.warning(f"Failed to update trade {trade_id} in database")
            
            return success
        except Exception as e:
            logger.error(f"Error updating trade in database: {e}")
            return False
    
    def link_signal_to_trade(self, signal_id: int, trade_id: str) -> bool:
        """
        Link a signal to a trade in the database.
        
        Args:
            signal_id: ID of the signal
            trade_id: ID of the trade
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update the signal to mark it as executed and link to trade
            update_data = {
                'executed': True,
                'trade_id': trade_id
            }
            
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE signals SET executed = ?, trade_id = ? WHERE signal_id = ?",
                    (1, trade_id, signal_id)
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Linked signal {signal_id} to trade {trade_id}")
                    return True
                else:
                    logger.warning(f"Failed to link signal {signal_id} to trade {trade_id}")
                    return False
        except Exception as e:
            logger.error(f"Error linking signal to trade in database: {e}")
            return False
    
    def save_market_data(self, market_data: Dict[str, Any], symbol: str, timeframe: str) -> bool:
        """
        Save market data to the database.
        
        Args:
            market_data: Dictionary of market data
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert market data to DataFrame format expected by the database
            if not isinstance(market_data, pd.DataFrame):
                # Extract OHLCV data
                candles = market_data.get('candles', [])
                if candles:
                    df = pd.DataFrame(candles, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 
                        'volume', 'close_time', 'quote_asset_volume',
                        'number_of_trades', 'taker_buy_base_asset_volume',
                        'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Select only necessary columns and convert types
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Convert timestamp from milliseconds to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    logger.warning(f"No candle data found in market_data")
                    return False
            else:
                df = market_data
            
            # Store in database
            success = self.db.store_market_data(df, symbol, timeframe)
            if success:
                logger.info(f"Saved market data for {symbol} {timeframe} to database")
            else:
                logger.warning(f"Failed to save market data for {symbol} {timeframe} to database")
            
            return success
        except Exception as e:
            logger.error(f"Error saving market data to database: {e}")
            return False
    
    def add_system_alert(self, message: str, alert_type: str = "info", 
                        severity: str = "low", data: Dict[str, Any] = None) -> int:
        """
        Add a system alert to the database.
        
        Args:
            message: Alert message
            alert_type: Type of alert ('info', 'warning', 'error', etc.)
            severity: Severity level ('low', 'medium', 'high')
            data: Optional related data
            
        Returns:
            Alert ID if successful, -1 otherwise
        """
        try:
            alert_id = self.db.add_alert(alert_type, severity, message, data)
            if alert_id > 0:
                logger.info(f"Added {severity} {alert_type} alert to database: {message}")
            else:
                logger.warning(f"Failed to add alert to database: {message}")
            
            return alert_id
        except Exception as e:
            logger.error(f"Error adding alert to database: {e}")
            return -1

    def _get_connection(self):
        """Proxy method to get a database connection for custom queries."""
        return self.db._get_connection() 