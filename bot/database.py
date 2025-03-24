import os
import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import time

logger = logging.getLogger("trading_bot")

# Define the database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trading_bot.db")

# Ensure the data directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

class Database:
    """Database manager class for storing trading data and analytics"""
    
    def __init__(self, db_path: str = DB_PATH):
        """Initialize the database connection"""
        self.db_path = db_path
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Create tables if they don't exist
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create trades table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    strategy TEXT,
                    profit_loss REAL,
                    roi_pct REAL,
                    commission REAL,
                    notes TEXT,
                    order_id TEXT,
                    timeframe TEXT,
                    execution_time REAL,
                    fees REAL,
                    raw_data TEXT
                )
                ''')
                
                # Create signals table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    indicators TEXT,
                    llm_decision TEXT,
                    executed INTEGER DEFAULT 0,
                    trade_id TEXT,
                    price REAL
                )
                ''')
                
                # Create market_data table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
                ''')
                
                # Create performance_metrics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    max_drawdown_pct REAL,
                    total_return_pct REAL,
                    profit_loss REAL,
                    win_count INTEGER,
                    loss_count INTEGER,
                    volatility REAL,
                    metrics_data TEXT,
                    timestamp TEXT NOT NULL,
                    max_drawdown REAL
                )
                ''')
                
                # Create alerts table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    related_data TEXT
                )
                ''')
                
                # Create trade_signal_link table for many-to-many relationship
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_signal_link (
                    trade_id TEXT NOT NULL,
                    signal_id INTEGER NOT NULL,
                    PRIMARY KEY (trade_id, signal_id),
                    FOREIGN KEY (trade_id) REFERENCES trades (trade_id),
                    FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
                )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                return True
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False
    
    def insert_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Insert a new trade record into the database
        
        Args:
            trade_data: Dictionary containing trade information
        
        Returns:
            True if successful, False otherwise
        """
        try:
            required_fields = ['trade_id', 'symbol', 'side', 'quantity', 'price', 'timestamp', 'status']
            for field in required_fields:
                if field not in trade_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Convert any dict/object data to JSON string
            if 'raw_data' in trade_data and isinstance(trade_data['raw_data'], dict):
                trade_data['raw_data'] = json.dumps(trade_data['raw_data'])
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build the SQL query dynamically
                fields = ', '.join(trade_data.keys())
                placeholders = ', '.join(['?' for _ in trade_data])
                values = list(trade_data.values())
                
                query = f"INSERT INTO trades ({fields}) VALUES ({placeholders})"
                cursor.execute(query, values)
                conn.commit()
                
                logger.info(f"Trade {trade_data['trade_id']} inserted into database")
                return True
        except Exception as e:
            logger.error(f"Error inserting trade: {e}")
            return False
    
    def update_trade(self, trade_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing trade record in the database
        
        Args:
            trade_id: ID of the trade to update
            update_data: Dictionary containing fields to update
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if 'raw_data' in update_data and isinstance(update_data['raw_data'], dict):
                update_data['raw_data'] = json.dumps(update_data['raw_data'])
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build the SQL update query dynamically
                set_clause = ', '.join([f"{key} = ?" for key in update_data])
                values = list(update_data.values()) + [trade_id]
                
                query = f"UPDATE trades SET {set_clause} WHERE trade_id = ?"
                cursor.execute(query, values)
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Trade {trade_id} updated successfully")
                    return True
                else:
                    logger.warning(f"Trade {trade_id} not found or no changes made")
                    return False
        except Exception as e:
            logger.error(f"Error updating trade: {e}")
            return False
    
    def insert_signal(self, signal_data: Dict[str, Any]) -> int:
        """
        Insert a new trading signal into the database
        
        Args:
            signal_data: Dictionary containing signal information
        
        Returns:
            Signal ID if successful, -1 otherwise
        """
        try:
            required_fields = ['symbol', 'timeframe', 'strategy', 'signal', 'timestamp']
            for field in required_fields:
                if field not in signal_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Convert indicators dict to JSON if present
            if 'indicators' in signal_data and isinstance(signal_data['indicators'], dict):
                signal_data['indicators'] = json.dumps(signal_data['indicators'])
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build the SQL query dynamically
                fields = ', '.join(signal_data.keys())
                placeholders = ', '.join(['?' for _ in signal_data])
                values = list(signal_data.values())
                
                query = f"INSERT INTO signals ({fields}) VALUES ({placeholders})"
                cursor.execute(query, values)
                conn.commit()
                
                # Get the last inserted row id
                signal_id = cursor.lastrowid
                logger.info(f"Signal inserted with ID: {signal_id}")
                return signal_id
        except Exception as e:
            logger.error(f"Error inserting signal: {e}")
            return -1
    
    def get_signals(self, symbol: str = None, timeframe: str = None, 
                    strategy: str = None, limit: int = 100) -> pd.DataFrame:
        """
        Retrieve signals from the database with optional filtering
        
        Args:
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            strategy: Optional strategy filter
            limit: Maximum number of signals to return
        
        Returns:
            DataFrame of signals
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM signals"
                params = []
                
                # Add filters if provided
                conditions = []
                if symbol:
                    conditions.append("symbol = ?")
                    params.append(symbol)
                if timeframe:
                    conditions.append("timeframe = ?")
                    params.append(timeframe)
                if strategy:
                    conditions.append("strategy = ?")
                    params.append(strategy)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += f" ORDER BY timestamp DESC LIMIT {limit}"
                
                # Execute query and convert to DataFrame
                df = pd.read_sql_query(query, conn, params=params)
                
                # Parse indicators json
                if 'indicators' in df.columns:
                    df['indicators'] = df['indicators'].apply(
                        lambda x: json.loads(x) if x else None
                    )
                
                return df
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            return pd.DataFrame()
    
    def get_trades(self, symbol: str = None, strategy: str = None, 
                   status: str = None, limit: int = 100) -> pd.DataFrame:
        """
        Retrieve trades from the database with optional filtering
        
        Args:
            symbol: Optional symbol filter
            strategy: Optional strategy filter
            status: Optional status filter
            limit: Maximum number of trades to return
        
        Returns:
            DataFrame of trades
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM trades"
                params = []
                
                # Add filters if provided
                conditions = []
                if symbol:
                    conditions.append("symbol = ?")
                    params.append(symbol)
                if strategy:
                    conditions.append("strategy = ?")
                    params.append(strategy)
                if status:
                    conditions.append("status = ?")
                    params.append(status)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += f" ORDER BY timestamp DESC LIMIT {limit}"
                
                # Execute query and convert to DataFrame
                df = pd.read_sql_query(query, conn, params=params)
                
                # Parse raw_data json
                if 'raw_data' in df.columns:
                    df['raw_data'] = df['raw_data'].apply(
                        lambda x: json.loads(x) if x else None
                    )
                
                return df
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return pd.DataFrame()
    
    def store_market_data(self, market_data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Store market data in the database.
        
        Args:
            market_data: DataFrame containing OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe of the data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure market_data has required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in market_data.columns:
                    raise ValueError(f"Market data missing required column: {col}")
            
            # Add symbol and timeframe columns
            market_data['symbol'] = symbol
            market_data['timeframe'] = timeframe
            
            # Make sure volume is present (added with 0 as default if missing)
            if 'volume' not in market_data.columns:
                market_data['volume'] = 0.0
            
            # Ensure timestamp is in the correct format
            if not isinstance(market_data['timestamp'].iloc[0], str):
                market_data['timestamp'] = market_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            with sqlite3.connect(self.db_path) as conn:
                # First, create a temporary table for the new data
                temp_table = f"temp_market_data_{int(time.time())}"
                market_data.to_sql(temp_table, conn, if_exists='replace', index=False)
                
                # Insert or replace data in the main table using column names explicitly
                conn.execute(f"""
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    SELECT symbol, timeframe, timestamp, open, high, low, close, volume 
                    FROM {temp_table}
                """)
                
                # Drop the temporary table
                conn.execute(f"DROP TABLE {temp_table}")
                conn.commit()
                
                logger.info(f"Stored {len(market_data)} market data points for {symbol} {timeframe}")
                return True
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            return False
    
    def get_market_data(self, symbol: str, timeframe: str, 
                        start_time: Optional[str] = None, 
                        end_time: Optional[str] = None, 
                        limit: int = 1000) -> pd.DataFrame:
        """
        Retrieve market data for backtesting
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe of the data
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of candles to return
        
        Returns:
            DataFrame of market data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM market_data WHERE symbol = ? AND timeframe = ?"
                params = [symbol, timeframe]
                
                # Add time filters if provided
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += f" ORDER BY timestamp ASC LIMIT {limit}"
                
                # Execute query and convert to DataFrame
                df = pd.read_sql_query(query, conn, params=params)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return pd.DataFrame()
    
    def store_performance_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Store strategy performance metrics
        
        Args:
            metrics: Dictionary containing performance metrics
        
        Returns:
            True if successful, False otherwise
        """
        try:
            required_fields = ['symbol', 'strategy', 'timeframe', 'start_date', 'end_date']
            for field in required_fields:
                if field not in metrics:
                    raise ValueError(f"Missing required field: {field}")
            
            # Convert metrics_data dict to JSON if present
            if 'metrics_data' in metrics and isinstance(metrics['metrics_data'], dict):
                metrics['metrics_data'] = json.dumps(metrics['metrics_data'])
            
            # Add a timestamp if not provided
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now().isoformat()
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build the SQL query dynamically
                fields = ', '.join(metrics.keys())
                placeholders = ', '.join(['?' for _ in metrics])
                values = list(metrics.values())
                
                query = f"INSERT INTO performance ({fields}) VALUES ({placeholders})"
                cursor.execute(query, values)
                conn.commit()
                
                logger.info(f"Performance metrics stored for {metrics['strategy']} on {metrics['symbol']}")
                return True
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
            return False
    
    def add_alert(self, alert_type: str, severity: str, message: str, 
                  related_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a new alert to the database
        
        Args:
            alert_type: Type of alert (e.g., 'error', 'warning', 'info')
            severity: Severity level (e.g., 'high', 'medium', 'low')
            message: Alert message
            related_data: Optional related data
        
        Returns:
            Alert ID if successful, -1 otherwise
        """
        try:
            # Convert related_data to JSON if present
            related_data_json = None
            if related_data:
                related_data_json = json.dumps(related_data)
            
            # Create alert record
            alert = {
                'alert_type': alert_type,
                'severity': severity,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'acknowledged': 0,
                'related_data': related_data_json
            }
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build the SQL query dynamically
                fields = ', '.join(alert.keys())
                placeholders = ', '.join(['?' for _ in alert])
                values = list(alert.values())
                
                query = f"INSERT INTO alerts ({fields}) VALUES ({placeholders})"
                cursor.execute(query, values)
                conn.commit()
                
                # Get the last inserted row id
                alert_id = cursor.lastrowid
                logger.info(f"Alert created with ID: {alert_id}")
                return alert_id
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return -1
    
    def get_alerts(self, acknowledged: bool = False, limit: int = 100) -> pd.DataFrame:
        """
        Retrieve alerts from the database
        
        Args:
            acknowledged: If True, retrieve acknowledged alerts; otherwise, unacknowledged
            limit: Maximum number of alerts to return
        
        Returns:
            DataFrame of alerts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = f"SELECT * FROM alerts WHERE acknowledged = ? ORDER BY timestamp DESC LIMIT {limit}"
                
                # Execute query and convert to DataFrame
                df = pd.read_sql_query(query, conn, params=[int(acknowledged)])
                
                # Parse related_data json
                if 'related_data' in df.columns:
                    df['related_data'] = df['related_data'].apply(
                        lambda x: json.loads(x) if x else None
                    )
                
                return df
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            return pd.DataFrame()
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """
        Mark an alert as acknowledged
        
        Args:
            alert_id: ID of the alert to acknowledge
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = "UPDATE alerts SET acknowledged = 1 WHERE alert_id = ?"
                cursor.execute(query, (alert_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Alert {alert_id} acknowledged")
                    return True
                else:
                    logger.warning(f"Alert {alert_id} not found")
                    return False
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
            
    def _get_connection(self):
        """
        Get a connection to the database.
        For advanced operations requiring direct database access.
        
        Returns:
            sqlite3.Connection object
        """
        return sqlite3.connect(self.db_path) 