"""
Data models for backtest results and performance metrics.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path


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
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_trade_profit: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    risk_reward_ratio: float = 0.0
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0
    
    def calculate_additional_metrics(self, returns: np.ndarray, drawdowns: np.ndarray) -> None:
        """
        Calculate additional performance metrics from return and drawdown series.
        
        Args:
            returns: Array of daily/period returns
            drawdowns: Array of drawdown percentages
        """
        # Skip if no valid data
        if len(returns) == 0 or np.isnan(returns).all():
            return
            
        # Calculate annualized return (assuming daily returns)
        self.annualized_return_pct = np.mean(returns) * 252 * 100
        
        # Calculate volatility (annualized)
        self.volatility = np.std(returns) * np.sqrt(252) * 100
        
        # Risk-adjusted performance metrics
        if self.volatility > 0:
            self.sharpe_ratio = self.annualized_return_pct / self.volatility
        
        # Sortino ratio (using only negative returns for denominator)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and np.std(negative_returns) > 0:
            downside_deviation = np.std(negative_returns) * np.sqrt(252) * 100
            self.sortino_ratio = self.annualized_return_pct / downside_deviation
        
        # Calmar ratio
        if self.max_drawdown_pct != 0:
            self.calmar_ratio = self.annualized_return_pct / abs(self.max_drawdown_pct)
        
        # Recovery factor
        if self.max_drawdown_pct != 0:
            self.recovery_factor = self.total_return_pct / abs(self.max_drawdown_pct)
        
        # Ulcer Index (measure of drawdown severity)
        if len(drawdowns) > 0:
            squared_drawdowns = np.square(drawdowns)
            self.ulcer_index = np.sqrt(np.mean(squared_drawdowns))


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
        self.metrics.total_return_pct = ((self.final_equity - self.initial_capital) / 
                                         self.initial_capital * 100)
        
        self.metrics.max_drawdown_pct = equity_df['drawdown'].min()
        
        # Calculate drawdown duration
        if self.metrics.max_drawdown_pct < 0:
            underwater = equity_df['equity'] < equity_df['equity_peak']
            underwater_periods = underwater.astype(int).groupby(
                (underwater.astype(int).diff() != 0).cumsum()
            ).sum()
            
            if len(underwater_periods) > 0:
                max_period = underwater_periods.max()
                # Convert to days (assuming daily data points)
                self.metrics.max_drawdown_duration_days = max_period
        
        # Trade metrics
        if self.total_trades > 0:
            self.metrics.win_rate = (self.winning_trades / self.total_trades) * 100
            
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
                        if profit_loss > 0:
                            trade_profits.append(profit_loss)
                        elif profit_loss < 0:
                            trade_losses.append(profit_loss)
                
                # Calculate metrics with the collected profits/losses
                if trade_profits or trade_losses:
                    self.metrics.avg_trade_profit = sum(trade_profits + trade_losses) / self.total_trades
                    
                    if trade_profits:
                        self.metrics.avg_win = sum(trade_profits) / len(trade_profits)
                    
                    if trade_losses:
                        self.metrics.avg_loss = sum(trade_losses) / len(trade_losses)
                    
                    if trade_losses and self.metrics.avg_loss != 0:
                        self.metrics.risk_reward_ratio = abs(self.metrics.avg_win / self.metrics.avg_loss)
                    
                    # Profit factor
                    total_profit = sum(trade_profits)
                    total_loss = abs(sum(trade_losses))
                    
                    if total_loss > 0:
                        self.metrics.profit_factor = total_profit / total_loss
                    else:
                        self.metrics.profit_factor = float('inf')
                    
                    # Expectancy
                    self.metrics.expectancy = (
                        (self.metrics.win_rate / 100 * self.metrics.avg_win) + 
                        ((1 - self.metrics.win_rate / 100) * self.metrics.avg_loss)
                    )
        
        # Additional metrics using numpy
        returns = equity_df['daily_return'].dropna().values
        drawdowns = equity_df['drawdown'].values
        
        self.metrics.calculate_additional_metrics(returns, drawdowns)


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