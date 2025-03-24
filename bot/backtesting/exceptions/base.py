"""
Custom exceptions for the backtesting engine.
This module defines specific exception types for clear error handling.
"""
from typing import Optional, Any


class BacktestError(Exception):
    """Base exception for all backtesting errors."""
    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        super().__init__(message)


class DataError(BacktestError):
    """Exception raised for errors related to market data."""
    pass


class MissingDataError(DataError):
    """Exception raised when required market data is missing."""
    pass


class InvalidDataError(DataError):
    """Exception raised when market data is invalid or corrupted."""
    pass


class DatabaseError(BacktestError):
    """Exception raised for database operation errors."""
    pass


class StrategyError(BacktestError):
    """Exception raised for errors in strategy implementation or execution."""
    pass


class InvalidParameterError(BacktestError):
    """Exception raised when an invalid parameter is provided."""
    pass


class ConfigurationError(BacktestError):
    """Exception raised for configuration errors."""
    pass


class ReportingError(BacktestError):
    """Exception raised for errors in report generation."""
    pass


class TradeExecutionError(BacktestError):
    """Exception raised for errors during trade execution."""
    pass


class OptimizationError(BacktestError):
    """Exception raised for errors during parameter optimization."""
    pass


class VisualizationError(BacktestError):
    """Exception raised for errors in visualization."""
    pass 