import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import json

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.llm_manager import (
    get_decision_from_llm, make_rule_based_decision, call_real_llm_api, log_decision_with_context
)

@pytest.fixture
def sample_data():
    """Sample market data and signals for testing."""
    return {
        'market_data': {
            'symbol': 'BTCUSDT',
            'candles': [
                [1672531200000, "50000", "50100", "49900", "50050", "100", 1672534800000, "5000000", 123, "50", "2500000", "0"]
            ],
            'order_book': {
                'bids': [["50000", "1.0"]],
                'asks': [["50100", "1.0"]]
            },
            'recent_trades': [
                {"id": 1, "price": "50050", "qty": "0.1", "time": 1672534800000}
            ],
            'timestamp': '2023-01-01T00:00:00'
        },
        'signals': {
            'simple': 'BUY',
            'technical': 'SELL'
        }
    }

def test_make_rule_based_decision():
    """Test the rule-based decision making function."""
    # Test BUY signal from both strategies
    prompt = "Simple signal: BUY. Technical analysis signal: BUY. Should we trade?"
    decision = make_rule_based_decision(prompt)
    assert decision == 'BUY'
    
    # Test conflicting signals
    prompt = "Simple signal: BUY. Technical analysis signal: SELL. Should we trade?"
    decision = make_rule_based_decision(prompt)
    assert decision == 'HOLD'
    
    # Test SELL signal from both strategies
    prompt = "Simple signal: SELL. Technical analysis signal: SELL. Should we trade?"
    decision = make_rule_based_decision(prompt)
    assert decision == 'SELL'
    
    # Test with technical signal given priority
    prompt = "Simple signal: HOLD. Technical analysis signal: BUY. Should we trade?"
    decision = make_rule_based_decision(prompt)
    assert decision == 'BUY'

@patch('bot.llm_manager.LLM_API_KEY', 'fake_key')
@patch('bot.llm_manager.LLM_API_ENDPOINT', 'https://api.example.com')
@patch('bot.llm_manager.LLM_MODEL', 'model-test')
@patch('bot.llm_manager.call_real_llm_api')
def test_get_decision_from_llm_with_api(mock_call_real_llm_api):
    """Test the LLM decision function with API configured."""
    mock_call_real_llm_api.return_value = 'BUY'
    
    prompt = "Should we trade BTCUSDT?"
    decision = get_decision_from_llm(prompt)
    
    # Verify the API was called
    mock_call_real_llm_api.assert_called_once_with(prompt)
    assert decision == 'BUY'

@patch('bot.llm_manager.LLM_API_KEY', '')
@patch('bot.llm_manager.make_rule_based_decision')
def test_get_decision_from_llm_without_api(mock_make_rule_based_decision):
    """Test the LLM decision function without API configured."""
    mock_make_rule_based_decision.return_value = 'HOLD'
    
    prompt = "Should we trade BTCUSDT?"
    decision = get_decision_from_llm(prompt)
    
    # Verify the rule-based function was called
    mock_make_rule_based_decision.assert_called_once_with(prompt)
    assert decision == 'HOLD'

def test_call_real_llm_api_success():
    """Test the real LLM API call function success case."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [
            {'message': {'content': 'BUY'}}
        ]
    }
    
    prompt = "Should we trade BTCUSDT?"
    with patch('bot.llm_manager.LLM_API_KEY', 'fake_key'), \
         patch('bot.llm_manager.LLM_API_ENDPOINT', 'https://api.example.com'), \
         patch('bot.llm_manager.LLM_MODEL', 'model-test'), \
         patch('bot.llm_manager.requests.post', return_value=mock_response):
        decision = call_real_llm_api(prompt)
    
    # Verify the correct decision was returned
    assert decision == 'BUY'

def test_call_real_llm_api_error():
    """Test the real LLM API call function error case."""
    # Mock the API response for error
    mock_response = MagicMock()
    mock_response.status_code = 500
    
    prompt = "Should we trade BTCUSDT?"
    with patch('bot.llm_manager.LLM_API_KEY', 'fake_key'), \
         patch('bot.llm_manager.LLM_API_ENDPOINT', 'https://api.example.com'), \
         patch('bot.llm_manager.LLM_MODEL', 'model-test'), \
         patch('bot.llm_manager.requests.post', return_value=mock_response):
        decision = call_real_llm_api(prompt)
    
    # Should return HOLD on error
    assert decision == 'HOLD'

def test_log_decision_with_context(sample_data):
    """Test the decision logging function."""
    decision = 'BUY'
    
    with patch('bot.llm_manager.logger') as mock_logger:
        log_decision_with_context(decision, sample_data['signals'], sample_data['market_data'])
        
        # Verify that the logger was called
        mock_logger.info.assert_called_once()
        
        # The first argument to the call should be a string containing JSON
        call_args = mock_logger.info.call_args[0][0]
        assert call_args.startswith("Decision log: {")
        
        # Try to parse the JSON to verify it's valid
        json_str = call_args.replace("Decision log: ", "")
        log_data = json.loads(json_str)
        
        # Verify the key information was logged
        assert log_data['decision'] == 'BUY'
        assert log_data['signals'] == sample_data['signals']
        assert log_data['market_data_summary']['symbol'] == 'BTCUSDT'
