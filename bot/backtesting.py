import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import uuid
import json
import os
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from bot.database import Database
from bot.strategy import calculate_rsi, calculate_bollinger_bands, calculate_macd

logger = logging.getLogger("trading_bot")

class BacktestEngine:
    """Engine for backtesting trading strategies on historical data"""
    
    def __init__(self, symbol: str, timeframes: List[str], start_date: str, 
                 end_date: str, initial_capital: float = 10000.0, 
                 commission: float = 0.001, db_path: str = None):
        """
        Initialize the backtesting engine
        
        Args:
            symbol: Trading symbol to backtest
            timeframes: List of timeframes to include (e.g., ['1m', '5m', '1h'])
            start_date: Start date for backtest (ISO format: 'YYYY-MM-DD')
            end_date: End date for backtest (ISO format: 'YYYY-MM-DD')
            initial_capital: Initial capital to start with
            commission: Commission rate as a decimal (e.g., 0.001 for 0.1%)
            db_path: Optional custom path for the database
        """
        self.symbol = symbol
        self.timeframes = timeframes
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission = commission
        
        # Initialize database connection
        self.db = Database(db_path) if db_path else Database()
        
        # Data structures for backtesting
        self.market_data = {}  # Dict of timeframe -> DataFrame
        self.current_capital = initial_capital
        self.position_size = 0.0
        self.trades = []
        self.equity_curve = []
        
        # Load historical data for backtesting
        self._load_market_data()
    
    def _load_market_data(self):
        """Load historical market data for all requested timeframes"""
        for timeframe in self.timeframes:
            data = self.db.get_market_data(
                symbol=self.symbol,
                timeframe=timeframe,
                start_time=self.start_date,
                end_time=self.end_date
            )
            
            if data.empty:
                logger.warning(f"No historical data for {self.symbol} at {timeframe} timeframe")
                continue
            
            # Ensure data is sorted by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            self.market_data[timeframe] = data
            logger.info(f"Loaded {len(data)} candles for {self.symbol} at {timeframe} timeframe")
    
    def download_data(self, client, timeframe: str, limit: int = 1000):
        """
        Download historical data from Binance
        
        Args:
            client: Binance client object
            timeframe: Timeframe to download (e.g., '1h')
            limit: Maximum number of candles to download
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading market data for {self.symbol} at {timeframe} timeframe")
            
            # Convert dates to millisecond timestamps
            start_ts = int(pd.Timestamp(self.start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(self.end_date).timestamp() * 1000)
            
            # Get klines data from Binance
            klines = client.get_historical_klines(
                symbol=self.symbol,
                interval=timeframe,
                start_str=str(start_ts),
                end_str=str(end_ts),
                limit=limit
            )
            
            if not klines:
                logger.warning(f"No data returned for {self.symbol} at {timeframe} timeframe")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamps to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            
            # Store in database
            success = self.db.store_market_data(df, self.symbol, timeframe)
            
            if success:
                # Update local market data
                self.market_data[timeframe] = df
                logger.info(f"Successfully downloaded and stored {len(df)} candles")
                return True
            else:
                logger.error("Failed to store downloaded market data")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading market data: {e}")
            return False
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the market data
        
        Args:
            df: DataFrame containing market data
        
        Returns:
            DataFrame with added indicators
        """
        # Calculate RSI
        df['rsi'] = calculate_rsi(df)
        
        # Calculate Bollinger Bands
        bb_df = calculate_bollinger_bands(df)
        df['upper_band'] = bb_df['upper_band']
        df['middle_band'] = bb_df['sma']
        df['lower_band'] = bb_df['lower_band']
        
        # Calculate MACD
        macd_df = calculate_macd(df)
        df['macd_line'] = macd_df['macd_line']
        df['signal_line'] = macd_df['signal_line']
        df['macd_histogram'] = macd_df['macd_histogram']
        
        return df
    
    def prepare_data(self):
        """Prepare market data by adding necessary indicators"""
        for timeframe in self.timeframes:
            if timeframe in self.market_data:
                self.market_data[timeframe] = self.add_indicators(self.market_data[timeframe])
                logger.info(f"Added indicators to {timeframe} data")
    
    def run_backtest(self, strategy_func: Callable):
        """
        Run a backtest using the specified strategy function
        
        Args:
            strategy_func: A function that takes market data and returns a signal ('BUY', 'SELL', 'HOLD')
        
        Returns:
            Dictionary containing backtest results
        """
        if not self.market_data:
            logger.error("No market data available for backtesting")
            return None
        
        # Prepare data with indicators
        self.prepare_data()
        
        # Get the primary timeframe data (first in the list)
        primary_tf = self.timeframes[0]
        if primary_tf not in self.market_data:
            logger.error(f"Primary timeframe {primary_tf} data not available")
            return None
        
        primary_data = self.market_data[primary_tf]
        
        # Initialize results
        self.current_capital = self.initial_capital
        self.position_size = 0.0
        self.trades = []
        self.equity_curve = [{
            'timestamp': primary_data['timestamp'].iloc[0],
            'equity': self.current_capital,
            'position_size': 0.0
        }]
        
        # Loop through each candle (skipping the first few to ensure indicators are valid)
        min_candles_required = 30  # Adjust as needed based on indicators
        
        for i in range(min_candles_required, len(primary_data)):
            current_time = primary_data['timestamp'].iloc[i]
            current_price = primary_data['close'].iloc[i]
            
            # Prepare data for strategy (up to current candle only)
            data_for_strategy = {}
            for tf in self.timeframes:
                if tf in self.market_data:
                    tf_data = self.market_data[tf]
                    # Filter data up to current timestamp
                    mask = tf_data['timestamp'] <= current_time
                    data_for_strategy[tf] = tf_data[mask].copy()
            
            # Get trading signal from strategy
            signal = strategy_func(data_for_strategy, self.symbol)
            
            # Process signal
            self._process_signal(signal, current_time, current_price)
            
            # Update equity curve
            position_value = self.position_size * current_price
            equity = self.current_capital + position_value
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': equity,
                'position_size': self.position_size
            })
        
        # Close any open position at the end of the backtest
        if self.position_size > 0:
            self._execute_trade('SELL', primary_data['timestamp'].iloc[-1], primary_data['close'].iloc[-1], self.position_size)
        
        # Calculate and return backtest results
        return self._calculate_results()
    
    def _process_signal(self, signal: str, timestamp: datetime, price: float):
        """
        Process a trading signal
        
        Args:
            signal: Trading signal ('BUY', 'SELL', 'HOLD')
            timestamp: Current timestamp
            price: Current price
        """
        if signal == 'BUY' and self.position_size == 0:
            # Calculate position size (use 95% of capital to leave room for commission)
            available_capital = self.current_capital * 0.95
            position_size = available_capital / price
            
            # Execute the trade
            self._execute_trade('BUY', timestamp, price, position_size)
            
        elif signal == 'SELL' and self.position_size > 0:
            # Sell entire position
            self._execute_trade('SELL', timestamp, price, self.position_size)
    
    def _execute_trade(self, side: str, timestamp: datetime, price: float, quantity: float):
        """
        Execute a simulated trade
        
        Args:
            side: Trade side ('BUY' or 'SELL')
            timestamp: Trade timestamp
            price: Trade price
            quantity: Trade quantity
        """
        trade_value = price * quantity
        commission_amount = trade_value * self.commission
        
        trade_id = str(uuid.uuid4())
        
        if side == 'BUY':
            # Update position and capital
            self.position_size = quantity
            self.current_capital -= (trade_value + commission_amount)
            
            # Record the trade
            trade = {
                'trade_id': trade_id,
                'symbol': self.symbol,
                'side': side,
                'timestamp': timestamp,
                'price': price,
                'quantity': quantity,
                'value': trade_value,
                'commission': commission_amount,
                'entry_point': True
            }
            self.trades.append(trade)
            
            logger.debug(f"BUY: {quantity} {self.symbol} at {price} (Value: {trade_value:.2f}, Commission: {commission_amount:.2f})")
            
        elif side == 'SELL':
            # Calculate profit/loss
            matching_buy_trades = [t for t in self.trades if t['side'] == 'BUY' and t['entry_point']]
            if matching_buy_trades:
                # Use the most recent entry
                entry_trade = matching_buy_trades[-1]
                entry_price = entry_trade['price']
                profit_loss = (price - entry_price) * quantity - commission_amount
                roi_pct = (profit_loss / entry_trade['value']) * 100
                
                # Mark the entry trade as closed
                entry_trade['entry_point'] = False
                
            else:
                # No matching entry found
                entry_price = 0
                profit_loss = -commission_amount
                roi_pct = 0
            
            # Update position and capital
            self.current_capital += (trade_value - commission_amount)
            self.position_size = 0
            
            # Record the trade
            trade = {
                'trade_id': trade_id,
                'symbol': self.symbol,
                'side': side,
                'timestamp': timestamp,
                'price': price,
                'quantity': quantity,
                'value': trade_value,
                'commission': commission_amount,
                'entry_price': entry_price,
                'profit_loss': profit_loss,
                'roi_pct': roi_pct,
                'entry_point': False
            }
            self.trades.append(trade)
            
            logger.debug(f"SELL: {quantity} {self.symbol} at {price} (P/L: {profit_loss:.2f}, ROI: {roi_pct:.2f}%)")
    
    def _calculate_results(self) -> Dict[str, Any]:
        """
        Calculate backtest performance metrics
        
        Returns:
            Dictionary containing performance metrics
        """
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Filter to completed trades (SELL trades with profit_loss)
        completed_trades = [t for t in self.trades if t['side'] == 'SELL' and 'profit_loss' in t]
        
        # Calculate metrics
        total_trades = len(completed_trades)
        
        if total_trades == 0:
            logger.warning("No completed trades in backtest")
            return {
                'symbol': self.symbol,
                'timeframes': self.timeframes,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': self.initial_capital,
                'final_equity': self.current_capital,
                'total_return_pct': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'trades': [],
                'equity_curve': equity_df.to_dict(orient='records')
            }
        
        # Calculate win rate
        winning_trades = [t for t in completed_trades if t['profit_loss'] > 0]
        win_count = len(winning_trades)
        win_rate = win_count / total_trades * 100
        
        # Calculate returns
        total_profit = sum(t['profit_loss'] for t in completed_trades)
        total_return_pct = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        # Calculate max drawdown
        equity_df['equity_peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['equity_peak']) / equity_df['equity_peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Calculate Sharpe ratio (using daily returns)
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['daily_return'].dropna()
        
        if len(daily_returns) > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252 ** 0.5)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Prepare results
        results = {
            'symbol': self.symbol,
            'timeframes': self.timeframes,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_equity': self.equity_curve[-1]['equity'],
            'total_profit': total_profit,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': total_trades - win_count,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': completed_trades,
            'equity_curve': equity_df.to_dict(orient='records')
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], strategy_name: str) -> bool:
        """
        Save backtest results to database
        
        Args:
            results: Backtest results dictionary
            strategy_name: Name of the strategy used
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract metrics for performance table
            metrics = {
                'symbol': results['symbol'],
                'strategy': strategy_name,
                'timeframe': ','.join(results['timeframes']),
                'start_date': results['start_date'],
                'end_date': results['end_date'],
                'total_trades': results['total_trades'],
                'win_count': results['win_count'],
                'loss_count': results['loss_count'],
                'win_rate': results['win_rate'],
                'profit_loss': results['total_profit'],
                'max_drawdown': results['max_drawdown'],
                'sharpe_ratio': results['sharpe_ratio'],
                'volatility': results.get('volatility', 0),
                'metrics_data': {
                    'initial_capital': results['initial_capital'],
                    'final_equity': results['final_equity'],
                    'total_return_pct': results['total_return_pct']
                }
            }
            
            # Save to database
            success = self.db.store_performance_metrics(metrics)
            
            # Save trades if requested
            for trade in results['trades']:
                # Create a copy to avoid modifying the original
                trade_copy = trade.copy()
                
                # Handle timestamp serialization
                if 'timestamp' in trade_copy and isinstance(trade_copy['timestamp'], (pd.Timestamp, datetime)):
                    trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
                
                # Convert any nested datetime objects in raw_data for JSON serialization
                raw_data = {}
                for key, value in trade_copy.items():
                    if isinstance(value, (pd.Timestamp, datetime)):
                        raw_data[key] = value.isoformat()
                    else:
                        raw_data[key] = value
                
                # Format trade data for database
                trade_data = {
                    'trade_id': trade_copy.get('trade_id', str(uuid.uuid4())),
                    'symbol': self.symbol,
                    'side': trade_copy['side'],
                    'quantity': trade_copy['quantity'],
                    'price': trade_copy['price'],
                    'timestamp': trade_copy['timestamp'],
                    'status': 'FILLED',
                    'strategy': strategy_name,
                    'timeframe': ','.join(self.timeframes),
                    'profit_loss': trade_copy.get('profit_loss', 0),
                    'fees': trade_copy.get('commission', 0),
                    'notes': f"Backtest trade for {strategy_name}",
                    'raw_data': json.dumps(raw_data)
                }
                
                self.db.insert_trade(trade_data)
            
            return success
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            return False
    
    def plot_results(self, results: Dict[str, Any], filename: Optional[str] = None, 
                     show_indicators: bool = True, custom_indicators: List[str] = None):
        """
        Plot backtest results with enhanced visualization
        
        Args:
            results: Backtest results dictionary
            filename: Optional filename to save the plot
            show_indicators: Whether to show technical indicators used in the strategy
            custom_indicators: List of specific indicators to show (e.g., ['rsi', 'macd'])
        """
        try:
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            
            # Get price data for the main timeframe
            primary_tf = self.timeframes[0]
            if primary_tf not in self.market_data:
                logger.warning(f"Primary timeframe {primary_tf} data not available for visualization")
                return
            
            price_df = self.market_data[primary_tf].copy()
            
            # Determine how many subplots we need
            # 1. Equity curve is always shown
            # 2. Price chart with buy/sell markers is always shown
            # 3. Drawdown is always shown
            # 4. Add additional plots for indicators if requested
            num_subplots = 3  # Equity, Price, Drawdown
            
            # Identify available indicators in the data
            indicator_panels = []
            
            if show_indicators:
                # Check for common indicators
                if custom_indicators is None:
                    custom_indicators = []
                
                # RSI gets its own panel
                if 'rsi' in price_df.columns or 'RSI' in price_df.columns:
                    indicator_panels.append('rsi')
                    num_subplots += 1
                
                # MACD gets its own panel
                if all(x in price_df.columns for x in ['macd_line', 'signal_line', 'macd_histogram']):
                    indicator_panels.append('macd')
                    num_subplots += 1
                
                # Bollinger Bands are shown on the price chart
                has_bb = all(x in price_df.columns for x in ['upper_band', 'middle_band', 'lower_band'])
                
                # Add custom indicators to appropriate panels
                for ind in custom_indicators:
                    if ind not in indicator_panels and ind in price_df.columns:
                        indicator_panels.append(ind)
                        num_subplots += 1
            
            # Create figure with subplots
            fig = plt.figure(figsize=(14, num_subplots * 3))
            gs = fig.add_gridspec(num_subplots, 1, height_ratios=[3] + [2] * (num_subplots - 1))
            
            # 1. Plot equity curve
            ax_equity = fig.add_subplot(gs[0])
            ax_equity.plot(equity_df['timestamp'], equity_df['equity'], label='Equity', color='blue', linewidth=2)
            
            # Add initial and final equity annotations
            ax_equity.axhline(y=results['initial_capital'], color='gray', linestyle='--', alpha=0.5)
            ax_equity.text(equity_df['timestamp'].iloc[0], results['initial_capital'], 
                            f"Initial: ${results['initial_capital']:,.2f}", 
                            verticalalignment='bottom')
            
            ax_equity.text(equity_df['timestamp'].iloc[-1], results['final_equity'], 
                            f"Final: ${results['final_equity']:,.2f} ({results['total_return_pct']:+.2f}%)", 
                            verticalalignment='bottom')
            
            ax_equity.set_title(f"Backtest Results: {results['symbol']} ({results['start_date']} to {results['end_date']})")
            ax_equity.set_ylabel('Equity ($)')
            ax_equity.grid(True)
            
            # Add summary box
            summary_text = (
                f"Initial Capital: ${results['initial_capital']:,.2f}\n"
                f"Final Equity: ${results['final_equity']:,.2f}\n"
                f"Total Return: {results['total_return_pct']:+.2f}%\n"
                f"Total Trades: {results['total_trades']}\n"
                f"Win Rate: {results['win_rate']:.2f}%\n"
                f"Max Drawdown: {results['max_drawdown']:.2f}%\n"
                f"Sharpe Ratio: {results['sharpe_ratio']:.2f}"
            )
            
            # Get current axes coordinates to position the text box
            ax_equity.text(0.02, 0.97, summary_text, transform=ax_equity.transAxes,
                           bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top',
                           fontsize=9)
            
            # 2. Plot price chart with trades
            ax_price = fig.add_subplot(gs[1], sharex=ax_equity)
            ax_price.plot(price_df['timestamp'], price_df['close'], label='Price', color='black')
            
            # Add moving averages if present
            for col in price_df.columns:
                if col.lower().startswith('sma') or col.lower().startswith('ema'):
                    ax_price.plot(price_df['timestamp'], price_df[col], label=col.upper(), alpha=0.7)
            
            # Add Bollinger Bands if present
            if has_bb:
                ax_price.plot(price_df['timestamp'], price_df['upper_band'], label='Upper BB', color='red', alpha=0.3)
                ax_price.plot(price_df['timestamp'], price_df['middle_band'], label='Middle BB', color='orange', alpha=0.3)
                ax_price.plot(price_df['timestamp'], price_df['lower_band'], label='Lower BB', color='green', alpha=0.3)
                ax_price.fill_between(price_df['timestamp'], price_df['upper_band'], price_df['lower_band'], 
                                      color='gray', alpha=0.1)
            
            # Mark trades on the price chart
            buy_trades = [t for t in results['trades'] if t['side'] == 'BUY']
            sell_trades = [t for t in results['trades'] if t['side'] == 'SELL']
            
            if buy_trades:
                buy_times = [pd.to_datetime(t['timestamp']) if isinstance(t['timestamp'], str) 
                             else t['timestamp'] for t in buy_trades]
                buy_prices = [t['price'] for t in buy_trades]
                ax_price.scatter(buy_times, buy_prices, marker='^', color='green', s=100, label='Buy', zorder=5)
                
                # Add profit/loss annotations to buy points
                for i, trade in enumerate(buy_trades):
                    if i < len(sell_trades):
                        profit = sell_trades[i].get('profit_loss', 0)
                        if profit > 0:
                            color = 'green'
                        else:
                            color = 'red'
                        ax_price.annotate(f"{profit:+.2f}", 
                                         (buy_times[i], buy_prices[i]), 
                                         textcoords="offset points",
                                         xytext=(0, 10), 
                                         ha='center',
                                         fontsize=8,
                                         color=color)
            
            if sell_trades:
                sell_times = [pd.to_datetime(t['timestamp']) if isinstance(t['timestamp'], str) 
                              else t['timestamp'] for t in sell_trades]
                sell_prices = [t['price'] for t in sell_trades]
                ax_price.scatter(sell_times, sell_prices, marker='v', color='red', s=100, label='Sell', zorder=5)
            
            ax_price.set_ylabel('Price')
            ax_price.grid(True)
            ax_price.legend(loc='upper left')
            
            # 3. Plot drawdown
            ax_dd = fig.add_subplot(gs[2], sharex=ax_equity)
            
            # Calculate drawdown if not already in equity_df
            if 'drawdown' not in equity_df.columns:
                equity_df['equity_peak'] = equity_df['equity'].cummax()
                equity_df['drawdown'] = (equity_df['equity'] - equity_df['equity_peak']) / equity_df['equity_peak'] * 100
            
            ax_dd.fill_between(equity_df['timestamp'], equity_df['drawdown'], 0, 
                              where=(equity_df['drawdown'] < 0), color='red', alpha=0.3)
            ax_dd.plot(equity_df['timestamp'], equity_df['drawdown'], color='red', label='Drawdown')
            
            # Annotate max drawdown
            max_dd_idx = equity_df['drawdown'].idxmin()
            max_dd_time = equity_df['timestamp'].iloc[max_dd_idx]
            max_dd_value = equity_df['drawdown'].iloc[max_dd_idx]
            
            ax_dd.annotate(f"Max DD: {max_dd_value:.2f}%", 
                           (max_dd_time, max_dd_value),
                           textcoords="offset points",
                           xytext=(0, -20), 
                           ha='center',
                           fontsize=9,
                           color='darkred',
                           arrowprops=dict(arrowstyle="->", color='darkred'))
            
            ax_dd.set_ylabel('Drawdown (%)')
            ax_dd.set_ylim(min(equity_df['drawdown'].min() * 1.5, -1), 1)  # Set ylim to show drawdowns clearly
            ax_dd.grid(True)
            
            # Add indicator subplots
            curr_subplot = 3
            
            # 4. RSI Plot (if available)
            if 'rsi' in indicator_panels:
                rsi_col = 'rsi' if 'rsi' in price_df.columns else 'RSI'
                ax_rsi = fig.add_subplot(gs[curr_subplot], sharex=ax_equity)
                ax_rsi.plot(price_df['timestamp'], price_df[rsi_col], color='purple', label='RSI')
                
                # Add overbought/oversold lines
                ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
                ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
                ax_rsi.axhline(y=50, color='gray', linestyle='-', alpha=0.2)
                
                # Fill overbought/oversold regions
                ax_rsi.fill_between(price_df['timestamp'], 70, price_df[rsi_col], 
                                   where=(price_df[rsi_col] > 70), color='red', alpha=0.2)
                ax_rsi.fill_between(price_df['timestamp'], 30, price_df[rsi_col], 
                                   where=(price_df[rsi_col] < 30), color='green', alpha=0.2)
                
                ax_rsi.set_ylabel('RSI')
                ax_rsi.set_ylim(0, 100)
                ax_rsi.grid(True)
                ax_rsi.legend(loc='upper left')
                
                curr_subplot += 1
            
            # 5. MACD Plot (if available)
            if 'macd' in indicator_panels:
                ax_macd = fig.add_subplot(gs[curr_subplot], sharex=ax_equity)
                
                # Plot MACD components
                ax_macd.plot(price_df['timestamp'], price_df['macd_line'], label='MACD', color='blue')
                ax_macd.plot(price_df['timestamp'], price_df['signal_line'], label='Signal', color='red')
                
                # Plot histogram
                positive_hist = price_df[price_df['macd_histogram'] >= 0]['macd_histogram']
                negative_hist = price_df[price_df['macd_histogram'] < 0]['macd_histogram']
                
                ax_macd.bar(price_df[price_df['macd_histogram'] >= 0]['timestamp'], 
                           positive_hist, color='green', alpha=0.5, width=0.7)
                ax_macd.bar(price_df[price_df['macd_histogram'] < 0]['timestamp'], 
                           negative_hist, color='red', alpha=0.5, width=0.7)
                
                ax_macd.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
                ax_macd.set_ylabel('MACD')
                ax_macd.grid(True)
                ax_macd.legend(loc='upper left')
                
                curr_subplot += 1
            
            # 6. Custom indicators
            for ind in custom_indicators:
                if ind in price_df.columns and ind not in ['rsi', 'RSI', 'macd_line', 'signal_line', 'macd_histogram']:
                    ax_ind = fig.add_subplot(gs[curr_subplot], sharex=ax_equity)
                    ax_ind.plot(price_df['timestamp'], price_df[ind], label=ind.upper())
                    ax_ind.set_ylabel(ind.upper())
                    ax_ind.grid(True)
                    ax_ind.legend(loc='upper left')
                    
                    curr_subplot += 1
            
            # Add a common x-label
            fig.text(0.5, 0.01, 'Date', ha='center', va='center')
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.05)
            
            # Format x-axis with dates
            for ax in fig.axes:
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=5))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Save or show the plot
            if filename:
                plt.savefig(filename, bbox_inches='tight', dpi=100)
                logger.info(f"Saved backtest plot to {filename}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting backtest results: {e}", exc_info=True)
    
    def optimize_parameters(self, strategy_factory: Callable, param_grid: Dict[str, List[Any]]):
        """
        Optimize strategy parameters through grid search
        
        Args:
            strategy_factory: A function that takes parameters and returns a strategy function
            param_grid: Dictionary of parameter names to lists of values to try
        
        Returns:
            Dictionary with best parameters and their performance
        """
        # Generate all parameter combinations
        import itertools
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Running parameter optimization with {len(param_combinations)} combinations")
        
        best_result = None
        best_params = None
        best_sharpe = float('-inf')
        
        # Run backtest for each parameter combination
        for i, combo in enumerate(param_combinations):
            params = dict(zip(param_keys, combo))
            
            # Generate strategy function with these parameters
            strategy_func = strategy_factory(params)
            
            # Run backtest
            result = self.run_backtest(strategy_func)
            
            if result and result['sharpe_ratio'] > best_sharpe:
                best_sharpe = result['sharpe_ratio']
                best_result = result
                best_params = params
            
            logger.info(f"Combination {i+1}/{len(param_combinations)}: {params} - Sharpe: {result['sharpe_ratio']:.2f}")
        
        # Return best parameters and their performance
        return {
            'params': best_params,
            'sharpe_ratio': best_sharpe,
            'result': best_result
        }
        
    def multi_timeframe_analysis(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform multi-timeframe analysis on market data
        
        Args:
            data_dict: Dictionary of timeframe -> DataFrame with market data
            
        Returns:
            Dictionary with analysis results for each timeframe and consolidated view
        """
        if not data_dict:
            logger.warning("No data provided for multi-timeframe analysis")
            return {}
            
        analysis_results = {}
        
        # Analyze each timeframe
        for timeframe, df in data_dict.items():
            if df.empty:
                continue
                
            # Calculate key technical indicators
            df_with_indicators = self.add_indicators(df.copy())
            
            # Get the latest values
            latest = df_with_indicators.iloc[-1]
            
            # Extract key metrics
            analysis_results[timeframe] = {
                'timestamp': latest['timestamp'],
                'close': latest['close'],
                'rsi': latest['rsi'],
                'macd_histogram': latest['macd_histogram'],
                'bb_position': (latest['close'] - latest['lower_band']) / (latest['upper_band'] - latest['lower_band']),
                'trend': 'uptrend' if latest['close'] > latest['middle_band'] else 'downtrend',
                'volatility': (latest['upper_band'] - latest['lower_band']) / latest['middle_band']
            }
        
        # Consolidate signals across timeframes
        consolidated = {
            'bullish_timeframes': [tf for tf, data in analysis_results.items() 
                                 if data['rsi'] < 40 or data['bb_position'] < 0.2 or data['macd_histogram'] > 0],
            'bearish_timeframes': [tf for tf, data in analysis_results.items() 
                                  if data['rsi'] > 60 or data['bb_position'] > 0.8 or data['macd_histogram'] < 0],
            'high_volatility_timeframes': [tf for tf, data in analysis_results.items() 
                                         if data['volatility'] > 0.05]  # 5% volatility threshold
        }
        
        # Add consolidated view to results
        analysis_results['consolidated'] = consolidated
        
        return analysis_results
        
    def monitor_and_alert(self, current_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Monitor backtest progress and generate alerts
        
        Args:
            current_results: Current backtest results
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Check for significant drawdown
        if 'max_drawdown' in current_results and current_results['max_drawdown'] < -15:
            alerts.append({
                'type': 'drawdown',
                'severity': 'high',
                'message': f"High drawdown detected: {current_results['max_drawdown']:.2f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check for low win rate
        if ('win_rate' in current_results and 'total_trades' in current_results 
                and current_results['total_trades'] > 10 and current_results['win_rate'] < 40):
            alerts.append({
                'type': 'win_rate',
                'severity': 'medium',
                'message': f"Low win rate: {current_results['win_rate']:.2f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check for poor Sharpe ratio
        if 'sharpe_ratio' in current_results and current_results['sharpe_ratio'] < 0.5:
            alerts.append({
                'type': 'performance',
                'severity': 'medium',
                'message': f"Poor risk-adjusted returns. Sharpe ratio: {current_results['sharpe_ratio']:.2f}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Record alerts in the database
        for alert in alerts:
            self.db.add_alert(
                alert_type=alert['type'],
                severity=alert['severity'],
                message=alert['message'],
                related_data={'backtest_symbol': self.symbol, 'timeframes': self.timeframes}
            )
        
        return alerts
        
    def generate_trade_log(self, results: Dict[str, Any], filename: Optional[str] = None) -> pd.DataFrame:
        """
        Generate comprehensive trade log for post-trade analysis
        
        Args:
            results: Backtest results dictionary
            filename: Optional filename to save the log as CSV
            
        Returns:
            DataFrame containing detailed trade information
        """
        if 'trades' not in results or not results['trades']:
            logger.warning("No trades available for generating trade log")
            return pd.DataFrame()
            
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(results['trades'])
        
        # Add calculated metrics
        if 'entry_price' in trades_df.columns and 'price' in trades_df.columns:
            sell_trades = trades_df[trades_df['side'] == 'SELL'].copy()
            
            if not sell_trades.empty:
                # Calculate holding period for each trade
                if isinstance(sell_trades['timestamp'].iloc[0], str):
                    sell_trades['timestamp'] = pd.to_datetime(sell_trades['timestamp'])
                
                # Find corresponding buy trades
                buy_trades = trades_df[trades_df['side'] == 'BUY'].copy()
                if not buy_trades.empty and isinstance(buy_trades['timestamp'].iloc[0], str):
                    buy_trades['timestamp'] = pd.to_datetime(buy_trades['timestamp'])
                
                # Calculate additional metrics
                sell_trades['profit_loss_pct'] = sell_trades['roi_pct']
                sell_trades['risk_reward_ratio'] = sell_trades['profit_loss'] / sell_trades['price'] * 100
                
                # Calculate market condition indicators
                market_data = self.market_data.get(self.timeframes[0], pd.DataFrame())
                if not market_data.empty:
                    market_indicators = []
                    
                    for _, trade in sell_trades.iterrows():
                        trade_time = trade['timestamp']
                        # Find closest candle before the trade
                        if 'timestamp' in market_data.columns:
                            nearest_candle = market_data[market_data['timestamp'] <= trade_time].iloc[-1]
                            
                            # Get market condition
                            market_indicators.append({
                                'rsi': nearest_candle.get('rsi', None),
                                'trend': 'bullish' if nearest_candle.get('macd_histogram', 0) > 0 else 'bearish',
                                'volatility': (nearest_candle.get('upper_band', 0) - 
                                              nearest_candle.get('lower_band', 0)) / nearest_candle.get('close', 1)
                            })
                        else:
                            market_indicators.append({})
                    
                    # Add market indicators to sell trades
                    for i, indicators in enumerate(market_indicators):
                        for key, value in indicators.items():
                            if i < len(sell_trades):
                                sell_trades.loc[sell_trades.index[i], f'market_{key}'] = value
        
        # Save to file if specified
        if filename and not trades_df.empty:
            trades_df.to_csv(filename, index=False)
            logger.info(f"Trade log saved to {filename}")
            
        return trades_df
            
    def generate_report(self, results: Dict[str, Any], output_dir: str = 'logs', 
                       report_name: Optional[str] = None) -> str:
        """
        Generate a comprehensive backtest report with visualizations and statistics
        
        Args:
            results: Backtest results dictionary
            output_dir: Directory to save the report files
            report_name: Optional custom name for the report files
        
        Returns:
            Path to the generated report files
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate report filename based on symbol, timeframe and date if not specified
            if not report_name:
                timeframe_str = '_'.join(self.timeframes)
                date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_name = f"{self.symbol}_{timeframe_str}_{date_str}"
            
            # Base path for report files
            base_path = os.path.join(output_dir, report_name)
            
            # 1. Generate trade log CSV
            trade_log_path = f"{base_path}_trades.csv"
            self.generate_trade_log(results, filename=trade_log_path)
            
            # 2. Generate performance chart
            chart_path = f"{base_path}_chart.png"
            self.plot_results(results, filename=chart_path, show_indicators=True)
            
            # 3. Generate detailed statistics HTML report
            html_path = f"{base_path}_report.html"
            
            # Calculate additional performance metrics
            equity_df = pd.DataFrame(results['equity_curve'])
            
            # If there are trades, calculate additional metrics
            if results['total_trades'] > 0:
                trades_df = pd.DataFrame(results['trades'])
                
                # Trade metrics
                win_trades = trades_df[trades_df.get('profit_loss', 0) > 0]
                loss_trades = trades_df[trades_df.get('profit_loss', 0) <= 0]
                
                avg_win = win_trades['profit_loss'].mean() if len(win_trades) > 0 else 0
                avg_loss = loss_trades['profit_loss'].mean() if len(loss_trades) > 0 else 0
                
                # Calculate profit factor
                gross_profit = win_trades['profit_loss'].sum() if len(win_trades) > 0 else 0
                gross_loss = abs(loss_trades['profit_loss'].sum()) if len(loss_trades) > 0 else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Calculate expectancy
                win_rate = results['win_rate'] / 100
                expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
            else:
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
                expectancy = 0
            
            # Calculate additional equity curve metrics
            if len(equity_df) > 1:
                equity_df['daily_return'] = equity_df['equity'].pct_change()
                returns = equity_df['daily_return'].dropna()
                
                if len(returns) > 0:
                    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                    sortino_ratio = returns[returns > 0].mean() / returns[returns < 0].std() * (252 ** 0.5) if len(returns[returns < 0]) > 0 else 0
                    calmar_ratio = results['total_return_pct'] / abs(results['max_drawdown']) if results['max_drawdown'] != 0 else 0
                else:
                    volatility = 0
                    sortino_ratio = 0
                    calmar_ratio = 0
            else:
                volatility = 0
                sortino_ratio = 0
                calmar_ratio = 0
            
            # Generate HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Backtest Report: {self.symbol}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; color: #333; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    .chart-container {{ margin: 20px 0; }}
                    .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
                    .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                    .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                    .section {{ margin-bottom: 30px; }}
                </style>
            </head>
            <body>
                <h1>Backtest Report: {self.symbol}</h1>
                <p>
                    <strong>Strategy:</strong> {results.get('strategy_name', 'Custom Strategy')}<br>
                    <strong>Period:</strong> {results['start_date']} to {results['end_date']}<br>
                    <strong>Timeframes:</strong> {', '.join(results['timeframes'])}<br>
                    <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
                
                <div class="section">
                    <h2>Performance Summary</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <h3>Total Return</h3>
                            <div class="metric-value {'positive' if results['total_return_pct'] > 0 else 'negative'}">
                                {results['total_return_pct']:.2f}%
                            </div>
                        </div>
                        <div class="metric-card">
                            <h3>Initial Capital</h3>
                            <div class="metric-value">${results['initial_capital']:,.2f}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Final Equity</h3>
                            <div class="metric-value">${results['final_equity']:,.2f}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Win Rate</h3>
                            <div class="metric-value">{results['win_rate']:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <h3>Profit Factor</h3>
                            <div class="metric-value">{profit_factor:.2f}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Expectancy</h3>
                            <div class="metric-value {'positive' if expectancy > 0 else 'negative'}">${expectancy:.2f}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Total Trades</h3>
                            <div class="metric-value">{results['total_trades']}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Max Drawdown</h3>
                            <div class="metric-value negative">{results['max_drawdown']:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <h3>Sharpe Ratio</h3>
                            <div class="metric-value">{results['sharpe_ratio']:.2f}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Sortino Ratio</h3>
                            <div class="metric-value">{sortino_ratio:.2f}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Calmar Ratio</h3>
                            <div class="metric-value">{calmar_ratio:.2f}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Volatility (Annualized)</h3>
                            <div class="metric-value">{volatility:.2f}%</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Performance Chart</h2>
                    <div class="chart-container">
                        <img src="{os.path.basename(chart_path)}" alt="Performance Chart" style="max-width:100%;">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Trade Analysis</h2>
                    
                    <h3>Trade Statistics</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Trades</td>
                            <td>{results['total_trades']}</td>
                        </tr>
                        <tr>
                            <td>Winning Trades</td>
                            <td>{results['win_count']}</td>
                        </tr>
                        <tr>
                            <td>Losing Trades</td>
                            <td>{results['loss_count']}</td>
                        </tr>
                        <tr>
                            <td>Win Rate</td>
                            <td>{results['win_rate']:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Average Win</td>
                            <td class="positive">${avg_win:.2f}</td>
                        </tr>
                        <tr>
                            <td>Average Loss</td>
                            <td class="negative">${avg_loss:.2f}</td>
                        </tr>
                        <tr>
                            <td>Profit Factor</td>
                            <td>{profit_factor:.2f}</td>
                        </tr>
                        <tr>
                            <td>Expectancy</td>
                            <td class="{'positive' if expectancy > 0 else 'negative'}">${expectancy:.2f}</td>
                        </tr>
                    </table>
                    
                    <h3>Trade Log</h3>
                    <p>Download the complete trade log: <a href="{os.path.basename(trade_log_path)}">Trade Log CSV</a></p>
                </div>
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated comprehensive backtest report: {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Error generating backtest report: {e}", exc_info=True)
            return ""


class BacktestRunner:
    """Utility class for running multiple backtests with different parameters"""
    
    def __init__(self, db_path: str = None):
        """Initialize with database connection"""
        self.db = Database(db_path) if db_path else Database()
        self.results = {}
        
    def run_multiple_backtests(self, symbols: List[str], timeframes: List[str], 
                              strategies: Dict[str, Callable], 
                              start_date: str, end_date: str,
                              initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Run backtests for multiple symbol/timeframe/strategy combinations
        
        Args:
            symbols: List of trading symbols to test
            timeframes: List of timeframes to test
            strategies: Dictionary of strategy_name -> strategy_func
            start_date: Start date for backtests (ISO format)
            end_date: End date for backtests (ISO format)
            initial_capital: Initial capital for each backtest
            
        Returns:
            Dictionary of backtest results
        """
        all_results = {}
        
        # Progress tracking
        total_tests = len(symbols) * len(strategies)
        current_test = 0
        
        for symbol in symbols:
            symbol_results = {}
            
            for strategy_name, strategy_func in strategies.items():
                current_test += 1
                logger.info(f"Running backtest {current_test}/{total_tests}: {symbol} with {strategy_name}")
                
                # Create and run backtest
                backtest = BacktestEngine(
                    symbol=symbol,
                    timeframes=timeframes,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital
                )
                
                # Run the backtest
                result = backtest.run_backtest(strategy_func)
                
                if result:
                    # Monitor and generate alerts
                    alerts = backtest.monitor_and_alert(result)
                    if alerts:
                        logger.warning(f"Backtest generated {len(alerts)} alerts")
                        
                    # Generate trade log
                    trade_log = backtest.generate_trade_log(
                        result, 
                        filename=f"logs/{symbol}_{strategy_name}_{start_date}_{end_date}.csv"
                    )
                    
                    # Save results
                    backtest.save_results(result, strategy_name)
                    
                    # Store in results dictionary
                    symbol_results[strategy_name] = {
                        'result': result,
                        'alerts': alerts,
                        'trade_count': len(result.get('trades', [])),
                        'sharpe': result.get('sharpe_ratio', 0),
                        'win_rate': result.get('win_rate', 0),
                        'return_pct': result.get('total_return_pct', 0)
                    }
                    
                    logger.info(f"Completed {symbol} {strategy_name}: {result.get('total_return_pct', 0):.2f}% return")
                else:
                    logger.error(f"Backtest failed for {symbol} with {strategy_name}")
            
            all_results[symbol] = symbol_results
        
        self.results = all_results
        return all_results
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare the performance of different strategies across symbols
        
        Returns:
            DataFrame with comparative performance metrics
        """
        if not self.results:
            logger.warning("No backtest results available for comparison")
            return pd.DataFrame()
            
        comparison_data = []
        
        for symbol, strategies in self.results.items():
            for strategy_name, data in strategies.items():
                if 'result' in data:
                    comparison_data.append({
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'total_return_pct': data['result'].get('total_return_pct', 0),
                        'win_rate': data['result'].get('win_rate', 0),
                        'sharpe_ratio': data['result'].get('sharpe_ratio', 0),
                        'max_drawdown': data['result'].get('max_drawdown', 0),
                        'total_trades': data['result'].get('total_trades', 0)
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate strategy rankings
        if not comparison_df.empty:
            comparison_df['return_rank'] = comparison_df.groupby('symbol')['total_return_pct'].rank(ascending=False)
            comparison_df['sharpe_rank'] = comparison_df.groupby('symbol')['sharpe_ratio'].rank(ascending=False)
            comparison_df['overall_rank'] = (comparison_df['return_rank'] + comparison_df['sharpe_rank']) / 2
            
            # Sort by overall rank
            comparison_df = comparison_df.sort_values(['symbol', 'overall_rank'])
        
        return comparison_df
        
    def generate_summary_report(self, output_file: str = None) -> str:
        """
        Generate a summary report of all backtest results
        
        Args:
            output_file: Optional file to save the report
            
        Returns:
            Summary report as string
        """
        if not self.results:
            return "No backtest results available."
            
        comparison = self.compare_strategies()
        
        summary = []
        summary.append("# Backtest Summary Report")
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall statistics
        total_tests = sum(len(strategies) for strategies in self.results.values())
        symbols_tested = len(self.results)
        strategies_tested = len(set(strategy for symbol_results in self.results.values() 
                                   for strategy in symbol_results.keys()))
        
        summary.append(f"Total backtests run: {total_tests}")
        summary.append(f"Symbols tested: {symbols_tested}")
        summary.append(f"Strategies tested: {strategies_tested}\n")
        
        # Best performing strategies by return
        if not comparison.empty:
            best_by_return = comparison.sort_values('total_return_pct', ascending=False).head(3)
            summary.append("## Top Strategies by Return")
            for _, row in best_by_return.iterrows():
                summary.append(f"- {row['strategy']} on {row['symbol']}: {row['total_return_pct']:.2f}% return, "
                              f"{row['win_rate']:.2f}% win rate, {row['total_trades']} trades")
            
            summary.append("")
            
            # Best risk-adjusted returns
            best_by_sharpe = comparison.sort_values('sharpe_ratio', ascending=False).head(3)
            summary.append("## Top Strategies by Risk-Adjusted Return (Sharpe)")
            for _, row in best_by_sharpe.iterrows():
                summary.append(f"- {row['strategy']} on {row['symbol']}: Sharpe {row['sharpe_ratio']:.2f}, "
                              f"{row['total_return_pct']:.2f}% return, {row['max_drawdown']:.2f}% max drawdown")
            
            summary.append("")
        
        # Convert to string
        report = "\n".join(summary)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report)
                logger.info(f"Summary report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving summary report: {e}")
        
        return report


# Test Functions for the Backtesting Module
def test_backtest_engine():
    """Test the basic functionality of the BacktestEngine class"""
    # Simple strategy function for testing
    def test_strategy(data_dict, symbol):
        # Use primary timeframe data
        primary_tf = list(data_dict.keys())[0]
        df = data_dict[primary_tf]
        
        # Simple moving average crossover
        df['sma_short'] = df['close'].rolling(window=10).mean()
        df['sma_long'] = df['close'].rolling(window=30).mean()
        
        # Check for crossover
        if len(df) < 30:
            return 'HOLD'
            
        if df['sma_short'].iloc[-2] < df['sma_long'].iloc[-2] and df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1]:
            return 'BUY'
        elif df['sma_short'].iloc[-2] > df['sma_long'].iloc[-2] and df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1]:
            return 'SELL'
        else:
            return 'HOLD'
    
    # Create a backtest engine
    engine = BacktestEngine(
        symbol='BTCUSDT',
        timeframes=['1h', '4h'],
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    # Create test data if needed
    if not engine.market_data:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='H')
        data = {
            'timestamp': dates,
            'open': np.random.normal(20000, 1000, len(dates)),
            'high': np.random.normal(20500, 1000, len(dates)),
            'low': np.random.normal(19500, 1000, len(dates)),
            'close': np.random.normal(20000, 1000, len(dates)),
            'volume': np.random.normal(100, 30, len(dates))
        }
        df = pd.DataFrame(data)
        df['symbol'] = 'BTCUSDT'
        df['timeframe'] = '1h'
        
        # Store test data
        engine.db.store_market_data(df, 'BTCUSDT', '1h')
        
        # Reload market data
        engine._load_market_data()
    
    # Run backtest
    results = engine.run_backtest(test_strategy)
    
    # Test monitoring and alerting
    alerts = engine.monitor_and_alert(results)
    
    # Test trade logging
    trade_log = engine.generate_trade_log(results, filename='test_trade_log.csv')
    
    # Test multi-timeframe analysis
    mtf_analysis = engine.multi_timeframe_analysis(engine.market_data)
    
    return results, alerts, trade_log, mtf_analysis

def test_backtest_runner():
    """Test the BacktestRunner for multiple strategies and symbols"""
    # Define test strategies
    def strategy_sma_cross(data_dict, symbol):
        primary_tf = list(data_dict.keys())[0]
        df = data_dict[primary_tf]
        if len(df) < 30:
            return 'HOLD'
        df['sma_short'] = df['close'].rolling(window=10).mean()
        df['sma_long'] = df['close'].rolling(window=30).mean()
        if df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1]:
            return 'BUY'
        else:
            return 'SELL'
    
    def strategy_rsi(data_dict, symbol):
        primary_tf = list(data_dict.keys())[0]
        df = data_dict[primary_tf]
        if len(df) < 30:
            return 'HOLD'
        df['rsi'] = calculate_rsi(df)
        if df['rsi'].iloc[-1] < 30:
            return 'BUY'
        elif df['rsi'].iloc[-1] > 70:
            return 'SELL'
        else:
            return 'HOLD'
    
    # Create runner
    runner = BacktestRunner()
    
    # Run multiple backtests
    results = runner.run_multiple_backtests(
        symbols=['BTCUSDT', 'ETHUSDT'],
        timeframes=['1h', '4h'],
        strategies={
            'SMA_Cross': strategy_sma_cross,
            'RSI': strategy_rsi
        },
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    # Compare strategies
    comparison = runner.compare_strategies()
    
    # Generate report
    report = runner.generate_summary_report('backtest_summary_report.txt')
    
    return results, comparison, report

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/backtesting.log"),
            logging.StreamHandler()
        ]
    )
    
    # Run tests
    test_results, test_alerts, test_log, test_mtf = test_backtest_engine()
    logger.info(f"Backtest engine test completed with {len(test_results.get('trades', []))} trades")
    
    # Run multi-strategy test
    multi_results, strategy_comparison, summary_report = test_backtest_runner()
    logger.info("Backtest runner test completed")
    print(summary_report) 