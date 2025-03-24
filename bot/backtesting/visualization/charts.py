"""
Visualization module for plotting backtest results with improved error handling.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import os
from datetime import datetime

from bot.backtesting.config.settings import CHART_SETTINGS
from bot.backtesting.models.results import BacktestResult
from bot.backtesting.exceptions.base import VisualizationError

logger = logging.getLogger("trading_bot.visualization")


class ChartGenerator:
    """Class for generating charts from backtest results."""
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize chart generator.
        
        Args:
            output_dir: Directory to save chart files (default: ./output/charts)
        """
        # Use pathlib for file path handling
        if output_dir is None:
            output_dir = Path("./output/charts")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)
            
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get chart settings from config
        self.figsize = CHART_SETTINGS.get('default_figsize', (14, 10))
        self.dpi = CHART_SETTINGS.get('dpi', 100)
        self.colors = CHART_SETTINGS.get('color_scheme', {})
    
    def plot_equity_curve(self, result: BacktestResult, 
                         filename: Optional[Union[str, Path]] = None, 
                         show_plot: bool = False) -> Optional[Path]:
        """
        Plot equity curve from backtest results.
        
        Args:
            result: Backtest result object
            filename: Filename to save the plot (default: auto-generated)
            show_plot: Whether to display the plot interactively
            
        Returns:
            Path: Path to the saved chart file or None if not saved
            
        Raises:
            VisualizationError: If plotting fails
        """
        try:
            if not result.equity_curve:
                logger.warning("No equity curve data available for plotting")
                return None
                
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame([vars(point) for point in result.equity_curve])
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Plot equity curve
            equity_color = self.colors.get('equity_line', 'blue')
            ax.plot(equity_df['timestamp'], equity_df['equity'], 
                   label=f'Equity ({result.symbol})', 
                   color=equity_color, linewidth=2)
            
            # Add initial capital line
            ax.axhline(y=result.initial_capital, color='gray', linestyle='--', alpha=0.5)
            
            # Calculate metrics to show on the chart
            perf_text = (
                f"Initial: ${result.initial_capital:,.2f}\n"
                f"Final: ${result.final_equity:,.2f}\n"
                f"Return: {result.metrics.total_return_pct:+.2f}%\n"
                f"Trades: {result.total_trades}\n"
                f"Win Rate: {result.metrics.win_rate:.2f}%\n"
                f"Sharpe: {result.metrics.sharpe_ratio:.2f}\n"
                f"Max DD: {result.metrics.max_drawdown_pct:.2f}%"
            )
            
            # Add text box with metrics
            ax.text(0.02, 0.97, perf_text, transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.7), 
                   verticalalignment='top', fontsize=10)
            
            # Set labels and title
            ax.set_title(f"{result.strategy_name} on {result.symbol} ({result.start_date} to {result.end_date})")
            ax.set_ylabel('Equity ($)')
            ax.set_xlabel('Date')
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Determine filename
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = self.output_dir / f"{result.symbol}_{result.strategy_name}_{timestamp}_equity.png"
            elif isinstance(filename, str):
                filename = Path(filename)
                
            # Ensure directory exists
            filename.parent.mkdir(exist_ok=True, parents=True)
            
            # Save figure
            plt.savefig(filename, bbox_inches='tight')
            logger.info(f"Saved equity curve chart to {filename}")
            
            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
            return filename
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}", exc_info=True)
            plt.close()  # Ensure figure is closed on error
            raise VisualizationError(f"Failed to plot equity curve: {str(e)}")
    
    def plot_drawdown(self, result: BacktestResult, 
                     filename: Optional[Union[str, Path]] = None,
                     show_plot: bool = False) -> Optional[Path]:
        """
        Plot drawdown chart from backtest results.
        
        Args:
            result: Backtest result object
            filename: Filename to save the plot (default: auto-generated)
            show_plot: Whether to display the plot interactively
            
        Returns:
            Path: Path to the saved chart file or None if not saved
        """
        try:
            if not result.equity_curve:
                logger.warning("No equity curve data available for plotting drawdown")
                return None
                
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame([vars(point) for point in result.equity_curve])
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            
            # Calculate drawdown if not already available
            if 'drawdown_pct' not in equity_df.columns or equity_df['drawdown_pct'].isna().all():
                equity_df['equity_peak'] = equity_df['equity'].cummax()
                equity_df['drawdown_pct'] = (equity_df['equity'] - equity_df['equity_peak']) / equity_df['equity_peak'] * 100
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Plot drawdown
            dd_color = self.colors.get('drawdown_fill', 'red')
            ax.fill_between(equity_df['timestamp'], equity_df['drawdown_pct'], 0, 
                           where=(equity_df['drawdown_pct'] < 0), 
                           color=dd_color, alpha=0.3)
            
            ax.plot(equity_df['timestamp'], equity_df['drawdown_pct'], 
                   color=dd_color, label='Drawdown')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            
            # Mark maximum drawdown
            min_dd_idx = equity_df['drawdown_pct'].idxmin()
            if not pd.isna(min_dd_idx):
                min_dd_date = equity_df['timestamp'].iloc[min_dd_idx]
                min_dd_value = equity_df['drawdown_pct'].iloc[min_dd_idx]
                
                # Add marker and annotation
                ax.scatter(min_dd_date, min_dd_value, color='darkred', s=100, zorder=5)
                ax.annotate(f"Max DD: {min_dd_value:.2f}%", 
                           (min_dd_date, min_dd_value),
                           xytext=(10, -20),
                           textcoords="offset points",
                           arrowprops=dict(arrowstyle="->", color='darkred'),
                           color='darkred')
            
            # Set labels and title
            ax.set_title(f"Drawdown: {result.strategy_name} on {result.symbol}")
            ax.set_ylabel('Drawdown (%)')
            ax.set_xlabel('Date')
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Set reasonable y limits
            min_dd = equity_df['drawdown_pct'].min()
            if not pd.isna(min_dd):
                ax.set_ylim(min(min_dd * 1.1, -1), 1)  # Give 10% margin below min drawdown
            
            # Determine filename
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = self.output_dir / f"{result.symbol}_{result.strategy_name}_{timestamp}_drawdown.png"
            elif isinstance(filename, str):
                filename = Path(filename)
                
            # Ensure directory exists
            filename.parent.mkdir(exist_ok=True, parents=True)
            
            # Save figure
            plt.savefig(filename, bbox_inches='tight')
            logger.info(f"Saved drawdown chart to {filename}")
            
            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
            return filename
            
        except Exception as e:
            logger.error(f"Error plotting drawdown: {e}", exc_info=True)
            plt.close()  # Ensure figure is closed on error
            raise VisualizationError(f"Failed to plot drawdown: {str(e)}")
    
    def plot_trades(self, result: BacktestResult, market_data: pd.DataFrame,
                   filename: Optional[Union[str, Path]] = None,
                   show_plot: bool = False) -> Optional[Path]:
        """
        Plot price chart with trade markers.
        
        Args:
            result: Backtest result object
            market_data: Market data DataFrame
            filename: Filename to save the plot (default: auto-generated)
            show_plot: Whether to display the plot interactively
            
        Returns:
            Path: Path to the saved chart file or None if not saved
        """
        try:
            if not result.trades:
                logger.warning("No trades available for plotting")
                return None
                
            if market_data is None or market_data.empty:
                logger.warning("No market data available for plotting trades")
                return None
            
            # Ensure market_data has timestamp column as datetime
            if 'timestamp' in market_data.columns:
                market_data = market_data.copy()
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
            else:
                logger.warning("Market data missing timestamp column")
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Plot price
            ax.plot(market_data['timestamp'], market_data['close'], 
                   color='black', label='Price', alpha=0.7)
            
            # Extract trades
            buy_trades = [t for t in result.trades if t.get('side') == 'BUY']
            sell_trades = [t for t in result.trades if t.get('side') == 'SELL']
            
            # Plot buy markers
            if buy_trades:
                try:
                    buy_times = [pd.to_datetime(t['timestamp']) for t in buy_trades]
                    buy_prices = [t['price'] for t in buy_trades]
                    
                    buy_color = self.colors.get('buy_marker', 'green')
                    ax.scatter(buy_times, buy_prices, marker='^', s=100, 
                              color=buy_color, label='Buy', zorder=5)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error plotting buy trades: {e}")
            
            # Plot sell markers
            if sell_trades:
                try:
                    sell_times = [pd.to_datetime(t['timestamp']) for t in sell_trades]
                    sell_prices = [t['price'] for t in sell_trades]
                    
                    sell_color = self.colors.get('sell_marker', 'red')
                    ax.scatter(sell_times, sell_prices, marker='v', s=100, 
                              color=sell_color, label='Sell', zorder=5)
                    
                    # Add profit/loss annotations
                    for i, trade in enumerate(sell_trades):
                        if 'profit_loss' in trade:
                            profit = trade['profit_loss']
                            color = 'green' if profit > 0 else 'red'
                            
                            ax.annotate(f"{profit:+.2f}", 
                                       (sell_times[i], sell_prices[i]), 
                                       textcoords="offset points",
                                       xytext=(5, 5), 
                                       ha='left',
                                       fontsize=8,
                                       color=color)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error plotting sell trades: {e}")
            
            # Add indicators if available in market data
            for indicator in ['sma_20', 'sma_50', 'ema_12', 'ema_26']:
                if indicator in market_data.columns:
                    ax.plot(market_data['timestamp'], market_data[indicator], 
                           label=indicator.upper(), alpha=0.7)
            
            # Add Bollinger Bands if available
            if all(band in market_data.columns for band in ['upper_band', 'middle_band', 'lower_band']):
                ax.plot(market_data['timestamp'], market_data['upper_band'], 
                       color='red', linestyle='--', alpha=0.5, label='Upper BB')
                ax.plot(market_data['timestamp'], market_data['middle_band'], 
                       color='blue', linestyle='--', alpha=0.5, label='Middle BB')
                ax.plot(market_data['timestamp'], market_data['lower_band'], 
                       color='green', linestyle='--', alpha=0.5, label='Lower BB')
            
            # Set labels and title
            ax.set_title(f"{result.strategy_name} Trades on {result.symbol}")
            ax.set_ylabel('Price')
            ax.set_xlabel('Date')
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            # Determine filename
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = self.output_dir / f"{result.symbol}_{result.strategy_name}_{timestamp}_trades.png"
            elif isinstance(filename, str):
                filename = Path(filename)
                
            # Ensure directory exists
            filename.parent.mkdir(exist_ok=True, parents=True)
            
            # Save figure
            plt.savefig(filename, bbox_inches='tight')
            logger.info(f"Saved trades chart to {filename}")
            
            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
            return filename
            
        except Exception as e:
            logger.error(f"Error plotting trades: {e}", exc_info=True)
            plt.close()  # Ensure figure is closed on error
            raise VisualizationError(f"Failed to plot trades: {str(e)}")
    
    def plot_complete_analysis(self, result: BacktestResult, market_data: pd.DataFrame,
                              filename: Optional[Union[str, Path]] = None,
                              show_plot: bool = False) -> Optional[Path]:
        """
        Create a comprehensive chart with equity curve, drawdown, and trades.
        
        Args:
            result: Backtest result object
            market_data: Market data DataFrame
            filename: Filename to save the plot (default: auto-generated)
            show_plot: Whether to display the plot interactively
            
        Returns:
            Path: Path to the saved chart file or None if not saved
        """
        try:
            if not result.equity_curve:
                logger.warning("No equity curve data available for plotting")
                return None
                
            if market_data is None or market_data.empty:
                logger.warning("No market data available for plotting trades")
                return None
                
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame([vars(point) for point in result.equity_curve])
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            
            # Calculate drawdown if not already available
            if 'drawdown_pct' not in equity_df.columns or equity_df['drawdown_pct'].isna().all():
                equity_df['equity_peak'] = equity_df['equity'].cummax()
                equity_df['drawdown_pct'] = (equity_df['equity'] - equity_df['equity_peak']) / equity_df['equity_peak'] * 100
            
            # Ensure market_data has timestamp column as datetime
            market_data = market_data.copy()
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
            
            # Create figure with subplots
            fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * 1.5), dpi=self.dpi)
            
            # Define subplot grid: 3 rows with different heights
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
            
            # 1. Price chart with trades
            ax_price = fig.add_subplot(gs[0])
            
            # Plot price
            ax_price.plot(market_data['timestamp'], market_data['close'], 
                         color='black', label='Price', alpha=0.7)
            
            # Extract trades
            buy_trades = [t for t in result.trades if t.get('side') == 'BUY']
            sell_trades = [t for t in result.trades if t.get('side') == 'SELL']
            
            # Plot buy markers
            if buy_trades:
                try:
                    buy_times = [pd.to_datetime(t['timestamp']) for t in buy_trades]
                    buy_prices = [t['price'] for t in buy_trades]
                    
                    buy_color = self.colors.get('buy_marker', 'green')
                    ax_price.scatter(buy_times, buy_prices, marker='^', s=100, 
                                    color=buy_color, label='Buy', zorder=5)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error plotting buy trades: {e}")
            
            # Plot sell markers
            if sell_trades:
                try:
                    sell_times = [pd.to_datetime(t['timestamp']) for t in sell_trades]
                    sell_prices = [t['price'] for t in sell_trades]
                    
                    sell_color = self.colors.get('sell_marker', 'red')
                    ax_price.scatter(sell_times, sell_prices, marker='v', s=100, 
                                    color=sell_color, label='Sell', zorder=5)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error plotting sell trades: {e}")
            
            # Add indicators if available
            for indicator in ['sma_20', 'sma_50', 'ema_12', 'ema_26']:
                if indicator in market_data.columns:
                    ax_price.plot(market_data['timestamp'], market_data[indicator], 
                                 label=indicator.upper(), alpha=0.7)
            
            # Add Bollinger Bands if available
            if all(band in market_data.columns for band in ['upper_band', 'middle_band', 'lower_band']):
                ax_price.plot(market_data['timestamp'], market_data['upper_band'], 
                             color='red', linestyle='--', alpha=0.5, label='Upper BB')
                ax_price.plot(market_data['timestamp'], market_data['middle_band'], 
                             color='blue', linestyle='--', alpha=0.5, label='Middle BB')
                ax_price.plot(market_data['timestamp'], market_data['lower_band'], 
                             color='green', linestyle='--', alpha=0.5, label='Lower BB')
            
            ax_price.set_title(f"{result.strategy_name} on {result.symbol} ({result.start_date} to {result.end_date})")
            ax_price.set_ylabel('Price')
            ax_price.grid(True, alpha=0.3)
            ax_price.legend(loc='best')
            
            # 2. Equity curve
            ax_equity = fig.add_subplot(gs[1], sharex=ax_price)
            
            equity_color = self.colors.get('equity_line', 'blue')
            ax_equity.plot(equity_df['timestamp'], equity_df['equity'], 
                          label='Equity', color=equity_color, linewidth=2)
            
            # Add initial capital line
            ax_equity.axhline(y=result.initial_capital, color='gray', linestyle='--', alpha=0.5)
            
            # Calculate metrics to show on the chart
            perf_text = (
                f"Initial: ${result.initial_capital:,.2f}\n"
                f"Final: ${result.final_equity:,.2f}\n"
                f"Return: {result.metrics.total_return_pct:+.2f}%\n"
                f"Trades: {result.total_trades}\n"
                f"Win Rate: {result.metrics.win_rate:.2f}%\n"
                f"Sharpe: {result.metrics.sharpe_ratio:.2f}"
            )
            
            # Add text box with metrics
            ax_equity.text(0.02, 0.97, perf_text, transform=ax_equity.transAxes,
                          bbox=dict(facecolor='white', alpha=0.7), 
                          verticalalignment='top', fontsize=10)
            
            ax_equity.set_ylabel('Equity ($)')
            ax_equity.grid(True, alpha=0.3)
            
            # 3. Drawdown chart
            ax_dd = fig.add_subplot(gs[2], sharex=ax_price)
            
            dd_color = self.colors.get('drawdown_fill', 'red')
            ax_dd.fill_between(equity_df['timestamp'], equity_df['drawdown_pct'], 0, 
                              where=(equity_df['drawdown_pct'] < 0), 
                              color=dd_color, alpha=0.3)
            
            ax_dd.plot(equity_df['timestamp'], equity_df['drawdown_pct'], 
                      color=dd_color, label='Drawdown')
            
            # Add zero line
            ax_dd.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            
            # Mark maximum drawdown
            min_dd_idx = equity_df['drawdown_pct'].idxmin()
            if not pd.isna(min_dd_idx):
                min_dd_date = equity_df['timestamp'].iloc[min_dd_idx]
                min_dd_value = equity_df['drawdown_pct'].iloc[min_dd_idx]
                
                # Add marker and annotation
                ax_dd.scatter(min_dd_date, min_dd_value, color='darkred', s=50, zorder=5)
                ax_dd.annotate(f"Max DD: {min_dd_value:.2f}%", 
                              (min_dd_date, min_dd_value),
                              xytext=(10, -20),
                              textcoords="offset points",
                              arrowprops=dict(arrowstyle="->", color='darkred'),
                              color='darkred')
            
            ax_dd.set_ylabel('Drawdown (%)')
            ax_dd.set_xlabel('Date')
            ax_dd.grid(True, alpha=0.3)
            
            # Set reasonable y limits for drawdown
            min_dd = equity_df['drawdown_pct'].min()
            if not pd.isna(min_dd):
                ax_dd.set_ylim(min(min_dd * 1.1, -1), 1)  # Give 10% margin below min drawdown
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Adjust layout
            plt.tight_layout()
            
            # Determine filename
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = self.output_dir / f"{result.symbol}_{result.strategy_name}_{timestamp}_analysis.png"
            elif isinstance(filename, str):
                filename = Path(filename)
                
            # Ensure directory exists
            filename.parent.mkdir(exist_ok=True, parents=True)
            
            # Save figure
            plt.savefig(filename, bbox_inches='tight')
            logger.info(f"Saved complete analysis chart to {filename}")
            
            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
            return filename
            
        except Exception as e:
            logger.error(f"Error creating complete analysis chart: {e}", exc_info=True)
            plt.close()  # Ensure figure is closed on error
            raise VisualizationError(f"Failed to create complete analysis chart: {str(e)}")


def plot_optimization_results(opt_result: Dict[str, Any], 
                            x_param: str, y_param: str = 'sharpe_ratio',
                            filename: Optional[Union[str, Path]] = None,
                            show_plot: bool = False) -> Optional[Path]:
    """
    Plot optimization results showing parameter impact on performance.
    
    Args:
        opt_result: Optimization result dictionary
        x_param: Parameter to show on x-axis
        y_param: Metric to show on y-axis
        filename: Filename to save the plot
        show_plot: Whether to display the plot interactively
        
    Returns:
        Path: Path to the saved chart file or None if not saved
    """
    try:
        # Extract data from optimization results
        all_results = opt_result.get('all_results', [])
        if not all_results:
            logger.warning("No optimization results available for plotting")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=CHART_SETTINGS.get('default_figsize', (10, 6)), 
                              dpi=CHART_SETTINGS.get('dpi', 100))
        
        # Extract parameter values and performance metrics
        x_values = []
        y_values = []
        
        for result in all_results:
            params = result.get('parameters', {})
            if x_param in params:
                x_values.append(params[x_param])
                y_values.append(result.get(y_param, 0))
        
        if not x_values or not y_values:
            logger.warning(f"No data points available for parameters {x_param} and {y_param}")
            plt.close(fig)
            return None
        
        # Plot scatter with trend line
        ax.scatter(x_values, y_values, alpha=0.7, s=50)
        
        # Try to fit a trend line if enough points
        if len(x_values) > 2:
            try:
                # Sort points for line plot
                points = sorted(zip(x_values, y_values))
                x_sorted, y_sorted = zip(*points)
                
                # Fit polynomial
                z = np.polyfit(x_values, y_values, 2)
                p = np.poly1d(z)
                
                # Plot trend line
                x_range = np.linspace(min(x_values), max(x_values), 100)
                y_pred = p(x_range)
                ax.plot(x_range, y_pred, '--', color='red', alpha=0.7)
            except Exception as e:
                logger.warning(f"Could not generate trend line: {e}")
        
        # Mark best value
        best_idx = y_values.index(max(y_values)) if y_param != 'max_drawdown_pct' else y_values.index(min(y_values))
        best_x = x_values[best_idx]
        best_y = y_values[best_idx]
        
        ax.scatter([best_x], [best_y], color='green', s=100, edgecolor='black', zorder=5)
        ax.annotate(f"Best: {best_x}", 
                   (best_x, best_y),
                   xytext=(0, 10),
                   textcoords="offset points",
                   ha='center',
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Set labels and title
        symbol = opt_result.get('symbol', '')
        strategy = opt_result.get('strategy_name', '')
        
        ax.set_title(f"Parameter Optimization: {strategy} on {symbol}")
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param.replace('_', ' ').title())
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Determine filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("./output/charts")
            output_dir.mkdir(exist_ok=True, parents=True)
            filename = output_dir / f"{symbol}_{strategy}_{x_param}_{y_param}_{timestamp}.png"
        elif isinstance(filename, str):
            filename = Path(filename)
            
        # Ensure directory exists
        filename.parent.mkdir(exist_ok=True, parents=True)
        
        # Save figure
        plt.savefig(filename, bbox_inches='tight')
        logger.info(f"Saved optimization chart to {filename}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        return filename
        
    except Exception as e:
        logger.error(f"Error plotting optimization results: {e}", exc_info=True)
        plt.close()  # Ensure figure is closed on error
        return None 