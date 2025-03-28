"""
Data models for backtest results and performance metrics.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import re
from pathlib import Path
from decimal import Decimal, getcontext

# Set Decimal precision
getcontext().prec = 28


@dataclass
class EquityPoint:
    """Single point in the equity curve."""
    timestamp: datetime
    equity: float
    position_size: float = 0.0
    drawdown_pct: Optional[float] = None
    daily_return: Optional[float] = None
    
    
@dataclass
class PerformanceMetrics:
    """Performance metrics for a backtest."""
    total_return_pct: Union[float, Decimal] = 0.0
    annualized_return_pct: Union[float, Decimal] = 0.0
    volatility: Union[float, Decimal] = 0.0
    sharpe_ratio: Union[float, Decimal] = 0.0
    sortino_ratio: Union[float, Decimal] = 0.0
    calmar_ratio: Union[float, Decimal] = 0.0
    max_drawdown_pct: Union[float, Decimal] = 0.0
    max_drawdown_duration_days: Union[float, Decimal] = 0.0
    win_rate: Union[float, Decimal] = 0.0
    profit_factor: Union[float, Decimal] = 0.0
    expectancy: Union[float, Decimal] = 0.0
    avg_trade_profit: Union[float, Decimal] = 0.0
    avg_win: Union[float, Decimal] = 0.0
    avg_loss: Union[float, Decimal] = 0.0
    risk_reward_ratio: Union[float, Decimal] = 0.0
    recovery_factor: Union[float, Decimal] = 0.0
    ulcer_index: Union[float, Decimal] = 0.0
    
    def __post_init__(self):
        # Convert numeric values to Decimal if they aren't already
        self._ensure_decimal('total_return_pct')
        self._ensure_decimal('annualized_return_pct')
        self._ensure_decimal('volatility')
        self._ensure_decimal('sharpe_ratio')
        self._ensure_decimal('sortino_ratio')
        self._ensure_decimal('calmar_ratio')
        self._ensure_decimal('max_drawdown_pct')
        self._ensure_decimal('max_drawdown_duration_days')
        self._ensure_decimal('win_rate')
        self._ensure_decimal('profit_factor')
        self._ensure_decimal('expectancy')
        self._ensure_decimal('avg_trade_profit')
        self._ensure_decimal('avg_win')
        self._ensure_decimal('avg_loss')
        self._ensure_decimal('risk_reward_ratio')
        self._ensure_decimal('recovery_factor')
        self._ensure_decimal('ulcer_index')
    
    def _ensure_decimal(self, attr_name):
        """Ensure an attribute is a Decimal, converting if necessary."""
        value = getattr(self, attr_name)
        if value is not None and not isinstance(value, Decimal):
            setattr(self, attr_name, Decimal(str(value)))
    
    def calculate_additional_metrics(self, returns: np.ndarray, drawdowns: np.ndarray, annualization_factor: int = 252) -> None:
        """
        Calculate additional performance metrics from return and drawdown series.
        
        Args:
            returns: Array of daily/period returns
            drawdowns: Array of drawdown percentages
            annualization_factor: Factor to annualize returns based on data frequency
                                  (252 for daily, 52 for weekly, 12 for monthly, etc.)
        """
        # Skip if no valid data
        if len(returns) == 0 or np.isnan(returns).all():
            return
            
        # Convert annualization factor to Decimal
        ann_factor_dec = Decimal(str(annualization_factor))
        
        # Calculate annualized return
        mean_return = Decimal(str(float(np.mean(returns))))
        self.annualized_return_pct = mean_return * ann_factor_dec * Decimal('100')
        
        # Calculate volatility (annualized)
        std_return = Decimal(str(float(np.std(returns))))
        self.volatility = std_return * Decimal(str(float(np.sqrt(annualization_factor)))) * Decimal('100')
        
        # Risk-adjusted performance metrics
        if self.volatility > Decimal('0'):
            self.sharpe_ratio = self.annualized_return_pct / self.volatility
        
        # Sortino ratio (using only negative returns for denominator)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and np.std(negative_returns) > 0:
            downside_deviation = Decimal(str(float(np.std(negative_returns)))) * \
                                Decimal(str(float(np.sqrt(annualization_factor)))) * Decimal('100')
            self.sortino_ratio = self.annualized_return_pct / downside_deviation
        
        # Calmar ratio
        if self.max_drawdown_pct != Decimal('0'):
            self.calmar_ratio = self.annualized_return_pct / abs(self.max_drawdown_pct)
        
        # Recovery factor
        if self.max_drawdown_pct != Decimal('0'):
            self.recovery_factor = self.total_return_pct / abs(self.max_drawdown_pct)
        
        # Ulcer Index (measure of drawdown severity)
        if len(drawdowns) > 0:
            squared_drawdowns = np.square(drawdowns)
            self.ulcer_index = Decimal(str(float(np.sqrt(np.mean(squared_drawdowns)))))


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""
    # Identification
    symbol: str
    strategy_name: str
    timeframes: List[str]
    start_date: str
    end_date: str
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Capital and trades
    initial_capital: float = 10000.0
    final_equity: float = 10000.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Performance metrics
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Detailed data
    equity_curve: List[EquityPoint] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    
    # Report paths
    html_report_path: Optional[Path] = None
    csv_report_path: Optional[Path] = None
    chart_path: Optional[Path] = None
    
    def calculate_metrics(self) -> None:
        """Calculate performance metrics from equity curve and trade data."""
        if not self.equity_curve:
            return
            
        # Calculate equity curve metrics
        equity_df = pd.DataFrame([vars(point) for point in self.equity_curve])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df = equity_df.sort_values('timestamp')
        
        # Calculate daily returns
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        
        # Calculate drawdowns
        equity_df['equity_peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['equity_peak']) / equity_df['equity_peak'] * 100
        
        # Update equity points with calculations
        for i, point in enumerate(self.equity_curve):
            if i < len(equity_df):
                point.drawdown_pct = equity_df['drawdown'].iloc[i]
                point.daily_return = equity_df['daily_return'].iloc[i]
        
        # Basic metrics
        self.metrics.total_return_pct = Decimal(str((self.final_equity - self.initial_capital) / 
                                         self.initial_capital * 100))
        
        self.metrics.max_drawdown_pct = Decimal(str(float(equity_df['drawdown'].min())))
        
        # Calculate drawdown duration
        if self.metrics.max_drawdown_pct < Decimal('0'):
            underwater = equity_df['equity'] < equity_df['equity_peak']
            underwater_periods = underwater.astype(int).groupby(
                (underwater.astype(int).diff() != 0).cumsum()
            ).sum()
            
            if len(underwater_periods) > 0:
                max_period = underwater_periods.max()
                # Convert to days (assuming daily data points)
                self.metrics.max_drawdown_duration_days = Decimal(str(max_period))
        
        # Trade metrics
        if self.total_trades > 0:
            self.metrics.win_rate = Decimal(str((self.winning_trades / self.total_trades) * 100))
            
            # Calculate profit metrics from trades
            if self.trades:
                # Check if we're dealing with Trade objects or dictionaries
                trade_profits = []
                trade_losses = []
                
                for trade in self.trades:
                    # Handle both Trade objects and dictionaries
                    if hasattr(trade, 'profit_loss'):
                        profit_loss = trade.profit_loss
                    elif isinstance(trade, dict) and 'profit_loss' in trade:
                        profit_loss = trade['profit_loss']
                    else:
                        profit_loss = None
                    
                    if profit_loss is not None:
                        # Ensure profit_loss is a Decimal
                        if not isinstance(profit_loss, Decimal):
                            profit_loss = Decimal(str(profit_loss))
                            
                        if profit_loss > Decimal('0'):
                            trade_profits.append(profit_loss)
                        elif profit_loss < Decimal('0'):
                            trade_losses.append(profit_loss)
                
                # Calculate metrics with the collected profits/losses
                if trade_profits or trade_losses:
                    if trade_profits and trade_losses:
                        self.metrics.avg_trade_profit = (sum(trade_profits) + sum(trade_losses)) / Decimal(str(self.total_trades))
                    elif trade_profits:
                        self.metrics.avg_trade_profit = sum(trade_profits) / Decimal(str(self.total_trades))
                    elif trade_losses:
                        self.metrics.avg_trade_profit = sum(trade_losses) / Decimal(str(self.total_trades))
                    
                    if trade_profits:
                        self.metrics.avg_win = sum(trade_profits) / Decimal(str(len(trade_profits)))
                    
                    if trade_losses:
                        self.metrics.avg_loss = sum(trade_losses) / Decimal(str(len(trade_losses)))
                    
                    if trade_losses and self.metrics.avg_loss != Decimal('0'):
                        self.metrics.risk_reward_ratio = abs(self.metrics.avg_win / self.metrics.avg_loss)
                    
                    # Profit factor
                    total_profit = sum(trade_profits) if trade_profits else Decimal('0')
                    total_loss = abs(sum(trade_losses)) if trade_losses else Decimal('0')
                    
                    if total_loss > Decimal('0'):
                        self.metrics.profit_factor = total_profit / total_loss
                    else:
                        self.metrics.profit_factor = Decimal('Infinity')
                    
                    # Expectancy - ensure all values are Decimal
                    win_rate_dec = self.metrics.win_rate / Decimal('100')
                    self.metrics.expectancy = (
                        (win_rate_dec * self.metrics.avg_win) + 
                        ((Decimal('1') - win_rate_dec) * self.metrics.avg_loss)
                    )
        
        # Additional metrics using numpy
        returns = equity_df['daily_return'].dropna().values
        drawdowns = equity_df['drawdown'].values
        
        # Determine annualization factor based on timeframe
        annualization_factor = 252  # Default for daily data
        
        # Get primary timeframe if available
        if self.timeframes and len(self.timeframes) > 0:
            primary_timeframe = self.timeframes[0]
            
            # Extract the numeric part and unit from timeframe (e.g., '1h', '4h', '1d')
            match = re.match(r'(\d+)([mhdwM])', primary_timeframe)
            if match:
                value, unit = match.groups()
                value = int(value)
                
                if unit == 'm':  # Minutes
                    minutes_per_year = 252 * 6.5 * 60  # ~252 trading days * 6.5 trading hours * 60 minutes
                    annualization_factor = minutes_per_year / value
                elif unit == 'h':  # Hours
                    hours_per_year = 252 * 6.5  # ~252 trading days * 6.5 trading hours
                    annualization_factor = hours_per_year / value
                elif unit == 'd':  # Days
                    annualization_factor = 252 / value
                elif unit == 'w':  # Weeks
                    annualization_factor = 52 / value
                elif unit == 'M':  # Months
                    annualization_factor = 12 / value
        
        self.metrics.calculate_additional_metrics(returns, drawdowns, int(annualization_factor))


@dataclass
class OptimizationResult:
    """Results from a parameter optimization run."""
    strategy_name: str
    symbol: str
    timeframes: List[str]
    parameter_grid: Dict[str, List[Any]]
    best_parameters: Dict[str, Any]
    best_backtest: BacktestResult
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    optimization_time_seconds: float = 0.0
    
    def add_result(self, params: Dict[str, Any], result: BacktestResult) -> None:
        """Add a parameter combination result."""
        self.all_results.append({
            'parameters': params,
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'total_return_pct': result.metrics.total_return_pct,
            'max_drawdown_pct': result.metrics.max_drawdown_pct,
            'win_rate': result.metrics.win_rate,
            'total_trades': result.total_trades
        })
    
    def find_best_parameters(self, sort_key: str = 'sharpe_ratio', ascending: bool = False) -> None:
        """
        Find the best parameters based on the specified metric.
        
        Args:
            sort_key: Metric to use for ranking (e.g., 'sharpe_ratio', 'total_return_pct')
            ascending: Whether lower values are better (True) or higher values (False)
        """
        if not self.all_results:
            return
            
        sorted_results = sorted(
            self.all_results, 
            key=lambda x: x.get(sort_key, 0), 
            reverse=not ascending
        )
        
        if sorted_results:
            self.best_parameters = sorted_results[0]['parameters'] 