"""
Data models for trades and trade-related information.
Uses dataclasses for clean, typed data representations.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid
from decimal import Decimal, getcontext

# Set Decimal precision
getcontext().prec = 28

@dataclass
class Trade:
    """Represents a trade executed during backtesting."""
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: str = ""  # 'BUY' or 'SELL'
    timestamp: datetime = field(default_factory=datetime.now)
    price: Decimal = field(default_factory=lambda: Decimal('0.0'))
    quantity: Decimal = field(default_factory=lambda: Decimal('0.0'))
    commission: Decimal = field(default_factory=lambda: Decimal('0.0'))
    status: str = "FILLED"  # Trade status (e.g., 'FILLED', 'CANCELED')
    
    # Added when a trade is completed
    entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    profit_loss: Optional[Decimal] = None
    roi_pct: Optional[Decimal] = None
    
    # For advanced analysis
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    holding_period_hours: Optional[float] = None
    strategy: Optional[str] = None
    timeframe: Optional[str] = None
    entry_signal: Optional[str] = None
    exit_signal: Optional[str] = None
    entry_point: bool = False
    
    # Market conditions at entry/exit
    market_indicators: Dict[str, Any] = field(default_factory=dict)
    
    # Raw data (e.g., from exchange)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert numeric values to Decimal if they aren't already"""
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if not isinstance(self.commission, Decimal):
            self.commission = Decimal(str(self.commission))
        if self.entry_price is not None and not isinstance(self.entry_price, Decimal):
            self.entry_price = Decimal(str(self.entry_price))
        if self.exit_price is not None and not isinstance(self.exit_price, Decimal):
            self.exit_price = Decimal(str(self.exit_price))
        if self.profit_loss is not None and not isinstance(self.profit_loss, Decimal):
            self.profit_loss = Decimal(str(self.profit_loss))
        if self.roi_pct is not None and not isinstance(self.roi_pct, Decimal):
            self.roi_pct = Decimal(str(self.roi_pct))
    
    @property
    def value(self) -> Decimal:
        """Calculate the total trade value excluding commission."""
        return self.price * self.quantity
    
    @property
    def total_cost(self) -> Decimal:
        """Calculate the total cost including commission."""
        return self.value + self.commission


@dataclass
class TradeAnalytics:
    """Analytics for a set of trades."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal('0.0'))
    average_win: Decimal = field(default_factory=lambda: Decimal('0.0'))
    average_loss: Decimal = field(default_factory=lambda: Decimal('0.0'))
    largest_win: Decimal = field(default_factory=lambda: Decimal('0.0'))
    largest_loss: Decimal = field(default_factory=lambda: Decimal('0.0'))
    profit_factor: Decimal = field(default_factory=lambda: Decimal('0.0'))
    expectancy: Decimal = field(default_factory=lambda: Decimal('0.0'))
    avg_holding_period: float = 0.0
    
    # Trade distribution
    trade_count_by_hour: Dict[int, int] = field(default_factory=dict)
    trade_count_by_day: Dict[str, int] = field(default_factory=dict)
    profit_by_hour: Dict[int, Decimal] = field(default_factory=dict)
    profit_by_day: Dict[str, Decimal] = field(default_factory=dict)
    
    # Market condition statistics
    trades_in_uptrend: int = 0
    trades_in_downtrend: int = 0
    trades_in_high_volatility: int = 0
    trades_in_low_volatility: int = 0
    
    def __post_init__(self):
        """Convert numeric values to Decimal if they aren't already"""
        if not isinstance(self.win_rate, Decimal):
            self.win_rate = Decimal(str(self.win_rate))
        if not isinstance(self.average_win, Decimal):
            self.average_win = Decimal(str(self.average_win))
        if not isinstance(self.average_loss, Decimal):
            self.average_loss = Decimal(str(self.average_loss))
        if not isinstance(self.largest_win, Decimal):
            self.largest_win = Decimal(str(self.largest_win))
        if not isinstance(self.largest_loss, Decimal):
            self.largest_loss = Decimal(str(self.largest_loss))
        if not isinstance(self.profit_factor, Decimal):
            self.profit_factor = Decimal(str(self.profit_factor))
        if not isinstance(self.expectancy, Decimal):
            self.expectancy = Decimal(str(self.expectancy))
        # Convert profit_by_hour and profit_by_day values to Decimal
        for hour, profit in self.profit_by_hour.items():
            if not isinstance(profit, Decimal):
                self.profit_by_hour[hour] = Decimal(str(profit))
        for day, profit in self.profit_by_day.items():
            if not isinstance(profit, Decimal):
                self.profit_by_day[day] = Decimal(str(profit))


@dataclass
class TradeBatch:
    """Batch of trades for analysis and storage."""
    trades: List[Trade] = field(default_factory=list)
    analytics: TradeAnalytics = field(default_factory=TradeAnalytics)
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the batch."""
        self.trades.append(trade)
        
    def calculate_analytics(self) -> None:
        """Calculate analytics for the trades in this batch."""
        if not self.trades:
            return
        
        self.analytics.total_trades = len(self.trades)
        
        # Set symbol and strategy if not already set
        if not self.symbol and self.trades:
            self.symbol = self.trades[0].symbol
        
        if not self.strategy and self.trades:
            self.strategy = self.trades[0].strategy
        
        winning_trades = [t for t in self.trades if t.profit_loss and t.profit_loss > Decimal('0')]
        losing_trades = [t for t in self.trades if t.profit_loss and t.profit_loss < Decimal('0')]
        breakeven_trades = [t for t in self.trades if t.profit_loss == Decimal('0')]
        
        self.analytics.winning_trades = len(winning_trades)
        self.analytics.losing_trades = len(losing_trades)
        self.analytics.breakeven_trades = len(breakeven_trades)
        
        # Calculate win rate
        if self.analytics.total_trades > 0:
            self.analytics.win_rate = (Decimal(str(self.analytics.winning_trades)) / Decimal(str(self.analytics.total_trades))) * Decimal('100')
        
        # Calculate average win/loss
        if winning_trades:
            self.analytics.average_win = sum(t.profit_loss for t in winning_trades) / Decimal(str(len(winning_trades)))
            self.analytics.largest_win = max(t.profit_loss for t in winning_trades)
        
        if losing_trades:
            self.analytics.average_loss = sum(t.profit_loss for t in losing_trades) / Decimal(str(len(losing_trades)))
            self.analytics.largest_loss = min(t.profit_loss for t in losing_trades)
        
        # Calculate profit factor
        total_profit = sum(t.profit_loss for t in winning_trades)
        total_loss = abs(sum(t.profit_loss for t in losing_trades))
        self.analytics.profit_factor = total_profit / total_loss if total_loss > Decimal('0') else Decimal('inf')
        
        # Calculate expectancy
        if self.analytics.total_trades > 0:
            self.analytics.expectancy = (
                (self.analytics.win_rate / Decimal('100') * self.analytics.average_win) + 
                ((Decimal('1') - self.analytics.win_rate / Decimal('100')) * self.analytics.average_loss)
            )
            
        # Calculate holding period
        trades_with_holding_period = [t for t in self.trades if t.holding_period_hours is not None]
        if trades_with_holding_period:
            self.analytics.avg_holding_period = (
                sum(t.holding_period_hours for t in trades_with_holding_period) / 
                len(trades_with_holding_period)
            )
            
        # Calculate trade distribution by time
        for trade in self.trades:
            if not trade.timestamp:
                continue
                
            hour = trade.timestamp.hour
            day = trade.timestamp.strftime('%A')  # Day of week
            
            # Count trades by hour and day
            self.analytics.trade_count_by_hour[hour] = self.analytics.trade_count_by_hour.get(hour, 0) + 1
            self.analytics.trade_count_by_day[day] = self.analytics.trade_count_by_day.get(day, 0) + 1
            
            # Sum profit by hour and day
            if trade.profit_loss:
                self.analytics.profit_by_hour[hour] = (
                    self.analytics.profit_by_hour.get(hour, Decimal('0')) + trade.profit_loss
                )
                self.analytics.profit_by_day[day] = (
                    self.analytics.profit_by_day.get(day, Decimal('0')) + trade.profit_loss
                )
                
        # Calculate market condition statistics
        for trade in self.trades:
            indicators = trade.market_indicators
            
            if indicators.get('trend') == 'uptrend':
                self.analytics.trades_in_uptrend += 1
            elif indicators.get('trend') == 'downtrend':
                self.analytics.trades_in_downtrend += 1
                
            if indicators.get('volatility', 0) > 0.03:  # 3% threshold
                self.analytics.trades_in_high_volatility += 1
            else:
                self.analytics.trades_in_low_volatility += 1 