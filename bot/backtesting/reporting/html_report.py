"""
Reporting module using Jinja2 for HTML templates.
"""
import pandas as pd
import logging
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import jinja2
import os

from bot.backtesting.models.results import BacktestResult
from bot.backtesting.config.settings import REPORT_SETTINGS
from bot.backtesting.exceptions.base import ReportingError

logger = logging.getLogger("trading_bot.reporting")


class HTMLReportGenerator:
    """Generates HTML reports from backtest results using Jinja2 templates."""
    
    def __init__(self, template_dir: Optional[Union[str, Path]] = None, 
                output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the HTML report generator.
        
        Args:
            template_dir: Directory containing templates (default: built-in template)
            output_dir: Directory to save reports (default: ./output/reports)
        """
        # Set up template directory
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        elif isinstance(template_dir, str):
            template_dir = Path(template_dir)
            
        self.template_dir = template_dir
        
        # Create template directory if it doesn't exist
        self.template_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up output directory
        if output_dir is None:
            output_dir = Path("./output/reports")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)
            
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up Jinja2 environment
        try:
            self.env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(self.template_dir)),
                autoescape=jinja2.select_autoescape(['html', 'xml'])
            )
            # Create default template if it doesn't exist
            self._ensure_default_template()
        except Exception as e:
            logger.error(f"Error setting up Jinja2 environment: {e}", exc_info=True)
            raise ReportingError(f"Failed to initialize HTML report generator: {str(e)}")
    
    def _ensure_default_template(self) -> None:
        """
        Ensure that the default template exists.
        Creates it if it doesn't exist.
        """
        template_path = self.template_dir / "backtest_report.html"
        
        if not template_path.exists():
            # Create a basic template
            default_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report: {{ result.symbol }}</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            line-height: 1.6; 
            margin: 20px; 
            color: #333; 
        }
        h1, h2, h3 { color: #2c3e50; }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin-bottom: 20px; 
        }
        th, td { 
            text-align: left; 
            padding: 8px; 
            border-bottom: 1px solid #ddd; 
        }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .positive { color: green; }
        .negative { color: red; }
        .chart-container { 
            margin: 20px 0; 
            text-align: center; 
        }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); 
            gap: 20px; 
        }
        .metric-card { 
            background-color: #f9f9f9; 
            padding: 15px; 
            border-radius: 5px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        }
        .metric-value { 
            font-size: 24px; 
            font-weight: bold; 
            margin: 10px 0; 
        }
        .section { margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>Backtest Report: {{ result.symbol }}</h1>
    <p>
        <strong>Strategy:</strong> {{ result.strategy_name }}<br>
        <strong>Period:</strong> {{ result.start_date }} to {{ result.end_date }}<br>
        <strong>Timeframes:</strong> {{ result.timeframes|join(', ') }}<br>
        <strong>Generated:</strong> {{ generation_time }}
    </p>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Return</h3>
                <div class="metric-value {% if result.metrics.total_return_pct > 0 %}positive{% else %}negative{% endif %}">
                    {{ result.metrics.total_return_pct|round(2) }}%
                </div>
            </div>
            <div class="metric-card">
                <h3>Initial Capital</h3>
                <div class="metric-value">${{ result.initial_capital|round(2) }}</div>
            </div>
            <div class="metric-card">
                <h3>Final Equity</h3>
                <div class="metric-value">${{ result.final_equity|round(2) }}</div>
            </div>
            <div class="metric-card">
                <h3>Win Rate</h3>
                <div class="metric-value">{{ result.metrics.win_rate|round(2) }}%</div>
            </div>
            <div class="metric-card">
                <h3>Profit Factor</h3>
                <div class="metric-value">{{ result.metrics.profit_factor|round(2) }}</div>
            </div>
            <div class="metric-card">
                <h3>Expectancy</h3>
                <div class="metric-value {% if result.metrics.expectancy > 0 %}positive{% else %}negative{% endif %}">
                    ${{ result.metrics.expectancy|round(2) }}
                </div>
            </div>
            <div class="metric-card">
                <h3>Total Trades</h3>
                <div class="metric-value">{{ result.total_trades }}</div>
            </div>
            <div class="metric-card">
                <h3>Max Drawdown</h3>
                <div class="metric-value negative">{{ result.metrics.max_drawdown_pct|round(2) }}%</div>
            </div>
            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <div class="metric-value">{{ result.metrics.sharpe_ratio|round(2) }}</div>
            </div>
            <div class="metric-card">
                <h3>Sortino Ratio</h3>
                <div class="metric-value">{{ result.metrics.sortino_ratio|round(2) }}</div>
            </div>
            <div class="metric-card">
                <h3>Calmar Ratio</h3>
                <div class="metric-value">{{ result.metrics.calmar_ratio|round(2) }}</div>
            </div>
            <div class="metric-card">
                <h3>Recovery Factor</h3>
                <div class="metric-value">{{ result.metrics.recovery_factor|round(2) }}</div>
            </div>
        </div>
    </div>
    
    {% if chart_paths %}
    <div class="section">
        <h2>Performance Charts</h2>
        {% for name, path in chart_paths.items() %}
        <div class="chart-container">
            <h3>{{ name }}</h3>
            <img src="{{ path }}" alt="{{ name }}" style="max-width:100%;">
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    {% if include_trades and result.trades %}
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
                <td>{{ result.total_trades }}</td>
            </tr>
            <tr>
                <td>Winning Trades</td>
                <td>{{ result.winning_trades }}</td>
            </tr>
            <tr>
                <td>Losing Trades</td>
                <td>{{ result.losing_trades }}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{{ result.metrics.win_rate|round(2) }}%</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td class="positive">${{ result.metrics.avg_win|round(2) }}</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td class="negative">${{ result.metrics.avg_loss|round(2) }}</td>
            </tr>
            <tr>
                <td>Risk/Reward Ratio</td>
                <td>{{ result.metrics.risk_reward_ratio|round(2) }}</td>
            </tr>
        </table>
        
        <h3>Trade List</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Side</th>
                <th>Price</th>
                <th>Quantity</th>
                <th>P/L</th>
                <th>ROI %</th>
            </tr>
            {% for trade in result.trades %}
            <tr>
                <td>{{ trade.timestamp }}</td>
                <td>{{ trade.side }}</td>
                <td>{{ trade.price|round(2) }}</td>
                <td>{{ trade.quantity|round(6) }}</td>
                <td class="{% if trade.profit_loss and trade.profit_loss > 0 %}positive{% elif trade.profit_loss and trade.profit_loss < 0 %}negative{% endif %}">
                    {% if trade.profit_loss %}{{ trade.profit_loss|round(2) }}{% endif %}
                </td>
                <td class="{% if trade.roi_pct and trade.roi_pct > 0 %}positive{% elif trade.roi_pct and trade.roi_pct < 0 %}negative{% endif %}">
                    {% if trade.roi_pct %}{{ trade.roi_pct|round(2) }}%{% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
</body>
</html>"""
            
            # Write the template to file
            with open(template_path, 'w') as f:
                f.write(default_template)
                
            logger.info(f"Created default HTML report template at {template_path}")
    
    def generate_report(self, result: BacktestResult, 
                      template_name: str = "backtest_report.html",
                      include_trades: bool = True,
                      chart_paths: Optional[Dict[str, str]] = None) -> Path:
        """
        Generate an HTML report from backtest results.
        
        Args:
            result: BacktestResult object containing performance data
            template_name: Name of the template file to use
            include_trades: Whether to include trade details in the report
            chart_paths: Optional dictionary of chart paths to include
            
        Returns:
            Path: Path to the generated HTML report
        """
        try:
            # Load template
            template = self.env.get_template(template_name)
            
            # Generate filename based on symbol, strategy and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.symbol}_{result.strategy_name}_{timestamp}_report.html"
            output_path = self.output_dir / filename
            
            # Convert chart paths to relative paths if provided
            relative_chart_paths = {}
            if chart_paths:
                for name, path in chart_paths.items():
                    if path:
                        # Convert to relative path for HTML (this assumes charts are in the same output directory)
                        relative_chart_paths[name] = os.path.basename(str(path))
            
            # Get max trades to include in report
            max_trades = REPORT_SETTINGS.get('max_trades_in_report', 100)
            
            # Process and limit trades if needed
            trades_for_report = result.trades
            if max_trades > 0 and len(result.trades) > max_trades:
                trades_for_report = result.trades[-max_trades:]  # Take most recent trades
                logger.info(f"Limiting report to {max_trades} most recent trades")
            
            # Render template
            html_content = template.render(
                result=result,
                include_trades=include_trades,
                chart_paths=relative_chart_paths,
                trades=trades_for_report,
                generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Write HTML to file
            with open(output_path, 'w') as f:
                f.write(html_content)
                
            logger.info(f"Generated HTML report: {output_path}")
            return output_path
            
        except jinja2.exceptions.TemplateError as e:
            logger.error(f"Template error generating report: {e}", exc_info=True)
            raise ReportingError(f"Failed to generate HTML report due to template error: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}", exc_info=True)
            raise ReportingError(f"Failed to generate HTML report: {str(e)}")
    
    def generate_comparison_report(self, results: List[BacktestResult], 
                                 template_name: str = "comparison_report.html") -> Path:
        """
        Generate a comparison report for multiple backtest results.
        
        Args:
            results: List of BacktestResult objects to compare
            template_name: Name of the template file to use
            
        Returns:
            Path: Path to the generated HTML report
        """
        try:
            # Check if comparison template exists, create it if it doesn't
            comparison_template_path = self.template_dir / template_name
            if not comparison_template_path.exists():
                self._create_comparison_template(comparison_template_path)
            
            # Load template
            template = self.env.get_template(template_name)
            
            # Generate filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{timestamp}_report.html"
            output_path = self.output_dir / filename
            
            # Calculate summary statistics for comparison
            summary_data = []
            for result in results:
                summary_data.append({
                    'symbol': result.symbol,
                    'strategy': result.strategy_name,
                    'total_return_pct': result.metrics.total_return_pct,
                    'sharpe_ratio': result.metrics.sharpe_ratio,
                    'win_rate': result.metrics.win_rate,
                    'max_drawdown_pct': result.metrics.max_drawdown_pct,
                    'total_trades': result.total_trades
                })
            
            # Sort by Sharpe ratio
            summary_data = sorted(summary_data, key=lambda x: x['sharpe_ratio'], reverse=True)
            
            # Render template
            html_content = template.render(
                results=results,
                summary=summary_data,
                generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Write HTML to file
            with open(output_path, 'w') as f:
                f.write(html_content)
                
            logger.info(f"Generated comparison report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}", exc_info=True)
            raise ReportingError(f"Failed to generate comparison report: {str(e)}")
    
    def _create_comparison_template(self, template_path: Path) -> None:
        """
        Create a default comparison report template.
        
        Args:
            template_path: Path to save the template
        """
        default_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strategy Comparison Report</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            line-height: 1.6; 
            margin: 20px; 
            color: #333; 
        }
        h1, h2, h3 { color: #2c3e50; }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin-bottom: 20px; 
        }
        th, td { 
            text-align: left; 
            padding: 8px; 
            border-bottom: 1px solid #ddd; 
        }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .positive { color: green; }
        .negative { color: red; }
        .section { margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>Strategy Comparison Report</h1>
    <p><strong>Generated:</strong> {{ generation_time }}</p>
    
    <div class="section">
        <h2>Strategy Rankings (by Sharpe Ratio)</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Symbol</th>
                <th>Strategy</th>
                <th>Return (%)</th>
                <th>Sharpe Ratio</th>
                <th>Win Rate (%)</th>
                <th>Max Drawdown (%)</th>
                <th>Total Trades</th>
            </tr>
            {% for item in summary %}
            <tr>
                <td>{{ loop.index }}</td>
                <td>{{ item.symbol }}</td>
                <td>{{ item.strategy }}</td>
                <td class="{% if item.total_return_pct > 0 %}positive{% else %}negative{% endif %}">
                    {{ item.total_return_pct|round(2) }}%
                </td>
                <td>{{ item.sharpe_ratio|round(2) }}</td>
                <td>{{ item.win_rate|round(2) }}%</td>
                <td class="negative">{{ item.max_drawdown_pct|round(2) }}%</td>
                <td>{{ item.total_trades }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Individual Backtest Details</h2>
        {% for result in results %}
        <div class="strategy-summary">
            <h3>{{ result.strategy_name }} on {{ result.symbol }}</h3>
            <p>
                <strong>Period:</strong> {{ result.start_date }} to {{ result.end_date }}<br>
                <strong>Timeframes:</strong> {{ result.timeframes|join(', ') }}<br>
                <strong>Initial Capital:</strong> ${{ result.initial_capital|round(2) }}<br>
                <strong>Final Equity:</strong> ${{ result.final_equity|round(2) }}<br>
                <strong>Total Return:</strong> 
                <span class="{% if result.metrics.total_return_pct > 0 %}positive{% else %}negative{% endif %}">
                    {{ result.metrics.total_return_pct|round(2) }}%
                </span><br>
                <strong>Sharpe Ratio:</strong> {{ result.metrics.sharpe_ratio|round(2) }}<br>
                <strong>Win Rate:</strong> {{ result.metrics.win_rate|round(2) }}%<br>
                <strong>Max Drawdown:</strong> <span class="negative">{{ result.metrics.max_drawdown_pct|round(2) }}%</span><br>
                <strong>Total Trades:</strong> {{ result.total_trades }}
            </p>
        </div>
        {% endfor %}
    </div>
</body>
</html>"""
        
        # Write the template to file
        with open(template_path, 'w') as f:
            f.write(default_template)
            
        logger.info(f"Created comparison report template at {template_path}")
    
    def generate_csv_report(self, result: BacktestResult) -> Path:
        """
        Generate a CSV report of trades from backtest results.
        
        Args:
            result: BacktestResult object containing performance data
            
        Returns:
            Path: Path to the generated CSV file
        """
        try:
            # Generate filename based on symbol, strategy and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.symbol}_{result.strategy_name}_{timestamp}_trades.csv"
            output_path = self.output_dir / filename
            
            # Convert trades to DataFrame
            trade_dicts = []
            
            for trade in result.trades:
                # Handle both Trade objects and dictionaries
                if hasattr(trade, '__dict__'):
                    # For Trade objects, convert to dict
                    trade_dict = {k: v for k, v in trade.__dict__.items() 
                                 if not k.startswith('_') and k != 'raw_data' and k != 'market_indicators'}
                else:
                    # For dictionaries, use as is
                    trade_dict = trade.copy()
                
                trade_dicts.append(trade_dict)
            
            # Create DataFrame and save to CSV
            if trade_dicts:
                trades_df = pd.DataFrame(trade_dicts)
                trades_df.to_csv(output_path, index=False)
                logger.info(f"Generated CSV report: {output_path}")
            else:
                logger.warning("No trades available for CSV report")
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating CSV report: {e}", exc_info=True)
            raise ReportingError(f"Failed to generate CSV report: {str(e)}")
    
    def generate_json_report(self, result: BacktestResult) -> Path:
        """
        Generate a JSON report of backtest results.
        
        Args:
            result: BacktestResult object containing performance data
            
        Returns:
            Path: Path to the generated JSON file
        """
        try:
            # Generate filename based on symbol, strategy and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.symbol}_{result.strategy_name}_{timestamp}_report.json"
            output_path = self.output_dir / filename
            
            # Create a serializable dict from the result
            result_dict = {
                'symbol': result.symbol,
                'strategy_name': result.strategy_name,
                'timeframes': result.timeframes,
                'start_date': result.start_date,
                'end_date': result.end_date,
                'run_id': result.run_id,
                'initial_capital': result.initial_capital,
                'final_equity': result.final_equity,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'metrics': {
                    'total_return_pct': result.metrics.total_return_pct,
                    'annualized_return_pct': result.metrics.annualized_return_pct,
                    'volatility': result.metrics.volatility,
                    'sharpe_ratio': result.metrics.sharpe_ratio,
                    'sortino_ratio': result.metrics.sortino_ratio,
                    'calmar_ratio': result.metrics.calmar_ratio,
                    'max_drawdown_pct': result.metrics.max_drawdown_pct,
                    'max_drawdown_duration_days': result.metrics.max_drawdown_duration_days,
                    'win_rate': result.metrics.win_rate,
                    'profit_factor': result.metrics.profit_factor,
                    'expectancy': result.metrics.expectancy,
                    'avg_trade_profit': result.metrics.avg_trade_profit,
                    'avg_win': result.metrics.avg_win,
                    'avg_loss': result.metrics.avg_loss,
                    'risk_reward_ratio': result.metrics.risk_reward_ratio,
                    'recovery_factor': result.metrics.recovery_factor,
                    'ulcer_index': result.metrics.ulcer_index
                },
                'trades': []
            }
            
            # Add trades (limit to 1000 to keep file size reasonable)
            max_trades = min(len(result.trades), 1000)
            for i in range(max_trades):
                trade = result.trades[i]
                
                # Convert to dict
                if hasattr(trade, '__dict__'):
                    # For Trade objects
                    trade_dict = {}
                    for k, v in trade.__dict__.items():
                        if not k.startswith('_') and k not in ['raw_data', 'market_indicators']:
                            # Handle datetime objects
                            if isinstance(v, datetime):
                                trade_dict[k] = v.isoformat()
                            else:
                                trade_dict[k] = v
                else:
                    # For dictionaries
                    trade_dict = {}
                    for k, v in trade.items():
                        if k not in ['raw_data', 'market_indicators']:
                            # Handle datetime objects
                            if isinstance(v, datetime):
                                trade_dict[k] = v.isoformat()
                            else:
                                trade_dict[k] = v
                                
                result_dict['trades'].append(trade_dict)
            
            # Write JSON to file
            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
                
            logger.info(f"Generated JSON report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}", exc_info=True)
            raise ReportingError(f"Failed to generate JSON report: {str(e)}")


# Create the templates directory if it doesn't exist
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True, parents=True) 