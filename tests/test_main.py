import os
import sys
import pytest
import time
from unittest.mock import patch, MagicMock, call
from datetime import datetime

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.main import (
    initialize_bot, get_market_data, execute_trade, get_signal_consensus,
    trading_loop
)
from bot.config import SYMBOL

@pytest.fixture
def mock_client():
    """Mock the Binance client."""
    with patch('bot.binance_api.client') as mock:
        mock.get_server_time.return_value = {"serverTime": int(time.time() * 1000)}
        mock.get_symbol_info.return_value = {"symbol": "BTCUSDT", "status": "TRADING"}
        yield mock

@pytest.fixture
def mock_market_data():
    """Fixture for mock market data."""
    return {
        'symbol': "BTCUSDT",
        'candles': [[
            int(time.time() * 1000),  # Open time
            "50000",  # Open
            "51000",  # High
            "49000",  # Low
            "50500",  # Close
            "10",     # Volume
            int(time.time() * 1000),  # Close time
            "505000", # Quote asset volume
            100,      # Number of trades
            "5",      # Taker buy base asset volume
            "250000", # Taker buy quote asset volume
            "0"       # Ignore
        ]],
        'order_book': {
            'lastUpdateId': 1234567890,
            'bids': [["50000", "1.5"], ["49900", "2.5"]],
            'asks': [["50100", "1.0"], ["50200", "3.0"]]
        },
        'recent_trades': [
            {'id': 1, 'price': '50050', 'qty': '0.1', 'time': int(time.time() * 1000)},
            {'id': 2, 'price': '50075', 'qty': '0.2', 'time': int(time.time() * 1000)}
        ],
        'timestamp': datetime.now().isoformat()
    }

@pytest.fixture
def mock_order_manager():
    """Mock OrderManager fixture."""
    with patch('bot.main.OrderManager') as mock_cls:
        mock_manager = MagicMock()
        mock_cls.return_value = mock_manager
        mock_manager.execute_market_buy.return_value = {"orderId": 123, "status": "FILLED"}
        mock_manager.execute_market_sell.return_value = {"orderId": 456, "status": "FILLED"}
        mock_manager.update_order_statuses.return_value = {}
        yield mock_manager

class TestMain:
    
    @patch('bot.main.get_account_balance')
    @patch('bot.main.synchronize_time')
    @patch('bot.main.client')  # Patch the client import in main.py, not in binance_api
    def test_initialize_bot(self, mock_client, mock_sync_time, mock_balances):
        """Test bot initialization."""
        mock_sync_time.return_value = 100
        mock_balances.return_value = {"BTC": {"free": 1.0, "locked": 0.0}, "USDT": {"free": 5000.0, "locked": 0.0}}
        mock_client.get_server_time.return_value = {"serverTime": int(time.time() * 1000)}
        mock_client.get_symbol_info.return_value = {"symbol": "BTCUSDT", "status": "TRADING"}
        
        result = initialize_bot()
        assert result is True
        mock_sync_time.assert_called_once()
        mock_client.get_server_time.assert_called_once()
        assert mock_balances.called
        mock_client.get_symbol_info.assert_called_once_with(SYMBOL)
    
    @patch('bot.main.client')
    def test_get_market_data(self, mock_client, mock_market_data):
        """Test getting market data."""
        # Setup mocks to return predefined data
        mock_client.get_klines.return_value = mock_market_data['candles']
        mock_client.get_order_book.return_value = mock_market_data['order_book']
        mock_client.get_recent_trades.return_value = mock_market_data['recent_trades']
        
        result = get_market_data(SYMBOL)
        
        assert result is not None
        assert result['symbol'] == SYMBOL
        assert 'candles' in result
        assert 'order_book' in result
        assert 'recent_trades' in result
        mock_client.get_klines.assert_called_once()
        mock_client.get_order_book.assert_called_once()
        mock_client.get_recent_trades.assert_called_once()
    
    def test_get_signal_consensus(self):
        """Test signal consensus calculation."""
        # Test BUY consensus
        signals = {"strategy1": "BUY", "strategy2": "BUY", "strategy3": "HOLD"}
        assert get_signal_consensus(signals) == "BUY"
        
        # Test SELL consensus
        signals = {"strategy1": "SELL", "strategy2": "SELL", "strategy3": "HOLD"}
        assert get_signal_consensus(signals) == "SELL"
        
        # Test HOLD consensus (equal BUY and SELL)
        signals = {"strategy1": "BUY", "strategy2": "SELL", "strategy3": "HOLD"}
        assert get_signal_consensus(signals) == "HOLD"
        
        # Test HOLD consensus (all HOLD)
        signals = {"strategy1": "HOLD", "strategy2": "HOLD", "strategy3": "HOLD"}
        assert get_signal_consensus(signals) == "HOLD"
    
    @patch('bot.main.log_decision_with_context')
    @patch('bot.main.get_account_balance')
    def test_execute_trade_buy(self, mock_balance, mock_log, mock_order_manager, mock_market_data):
        """Test executing a buy trade."""
        signals = {"simple": "BUY", "technical": "BUY"}
        llm_decision = "BUY"
        
        result = execute_trade(signals, llm_decision, SYMBOL, mock_market_data, mock_order_manager)
        
        assert result == {"orderId": 123, "status": "FILLED"}
        mock_log.assert_called_once()
        mock_order_manager.execute_market_buy.assert_called_once_with(quote_amount=10.0)
    
    @patch('bot.main.log_decision_with_context')
    @patch('bot.main.get_account_balance')
    def test_execute_trade_sell(self, mock_balance, mock_log, mock_order_manager, mock_market_data):
        """Test executing a sell trade."""
        signals = {"simple": "SELL", "technical": "SELL"}
        llm_decision = "SELL"
        
        # Mock the balance return for the sell
        mock_balance.return_value = {"free": 0.1, "locked": 0.0}
        
        result = execute_trade(signals, llm_decision, SYMBOL, mock_market_data, mock_order_manager)
        
        assert result == {"orderId": 456, "status": "FILLED"}
        mock_log.assert_called_once()
        mock_balance.assert_called_once()
        mock_order_manager.execute_market_sell.assert_called_once_with(0.1)
    
    @patch('bot.main.log_decision_with_context')
    def test_execute_trade_hold(self, mock_log, mock_order_manager, mock_market_data):
        """Test no trade executed when signals or LLM disagree."""
        # Signals agree but LLM disagrees
        signals = {"simple": "BUY", "technical": "BUY"}
        llm_decision = "HOLD"
        
        result = execute_trade(signals, llm_decision, SYMBOL, mock_market_data, mock_order_manager)
        
        assert result is None
        mock_log.assert_called_once()
        mock_order_manager.execute_market_buy.assert_not_called()
        mock_order_manager.execute_market_sell.assert_not_called()
        
        # Reset mocks
        mock_log.reset_mock()
        
        # LLM agrees but signals disagree
        signals = {"simple": "BUY", "technical": "SELL"}
        llm_decision = "BUY"
        
        result = execute_trade(signals, llm_decision, SYMBOL, mock_market_data, mock_order_manager)
        
        assert result is None
        mock_log.assert_called_once()
        mock_order_manager.execute_market_buy.assert_not_called()
        mock_order_manager.execute_market_sell.assert_not_called()
    
    @patch('bot.main.get_market_data')
    @patch('bot.main.get_all_strategy_signals')
    @patch('bot.main.get_decision_from_llm')
    @patch('bot.main.execute_trade')
    @patch('bot.main.time.sleep', side_effect=KeyboardInterrupt)  # Stop after first iteration
    @patch('bot.main.DatabaseIntegration')  # Add mock for DatabaseIntegration
    @patch('bot.main.OrderManager')  # Add mock for OrderManager
    @patch('bot.main.LLMManager')  # Add mock for LLMManager
    def test_trading_loop(self, mock_llm_manager, mock_order_manager_cls, mock_db_integration, 
                         mock_sleep, mock_execute_trade, mock_llm, mock_get_all_signals, 
                         mock_market_data):
        """Test the main trading loop with mocked dependencies."""
        # Setup mocks
        mock_market_data.return_value = {"symbol": SYMBOL, "data": "test"}
        mock_get_all_signals.return_value = {"simple": "BUY", "technical": "BUY"}
        mock_llm.return_value = {"decision": "BUY", "confidence": 0.9, "reasoning": "Mocked reasoning"}
        mock_execute_trade.return_value = {"orderId": 123}
        
        # Set up mock order manager instance
        mock_order_manager = MagicMock()
        mock_order_manager_cls.return_value = mock_order_manager
        
        # Set up mock db integration
        mock_db = MagicMock()
        mock_db_integration.return_value = mock_db
        
        # Set up mock LLM manager
        mock_llm_instance = MagicMock()
        mock_llm_instance.make_llm_decision.return_value = {"decision": "BUY", "confidence": 0.9, "reasoning": "Mocked reasoning"}
        mock_llm_manager.return_value = mock_llm_instance
        
        # Call trading_loop - it will raise KeyboardInterrupt after one iteration
        with pytest.raises(KeyboardInterrupt):
            trading_loop()
        
        # Verify the correct sequence of calls
        mock_order_manager.update_order_statuses.assert_called_once()
        assert mock_market_data.call_count == 1
        assert SYMBOL in mock_market_data.call_args[0]
        mock_get_all_signals.assert_called_once_with(SYMBOL)
        mock_llm_instance.make_llm_decision.assert_called_once()
        mock_execute_trade.assert_called_once()
        mock_sleep.assert_called_once_with(60)
    
    @patch('bot.main.get_market_data')
    @patch('bot.main.time.sleep')
    def test_trading_loop_error_handling(self, mock_sleep, mock_market_data, mock_order_manager):
        """Test error handling in the trading loop."""
        # Configure mock to raise exception on first call, then work, then raise KeyboardInterrupt
        mock_market_data.side_effect = [
            Exception("Test error"),
            KeyboardInterrupt()
        ]
        
        # Call trading_loop - it will raise KeyboardInterrupt after second iteration
        with pytest.raises(KeyboardInterrupt):
            trading_loop()
        
        # Verify error handling (exponential backoff)
        mock_sleep.assert_called_once_with(120)  # 60 * 2^1 = 120 seconds for first error
        assert mock_market_data.call_count == 2 