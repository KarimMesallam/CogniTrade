import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import json
import re

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables first thing
print(f"Current directory: {os.getcwd()}")
env_file = os.path.join(os.getcwd(), '.env')
print(f".env file path: {env_file}")
print(f".env file exists: {os.path.exists(env_file)}")

# Explicitly load the .env file with absolute path
if os.path.exists(env_file):
    load_dotenv(dotenv_path=env_file, verbose=True)
else:
    print("WARNING: .env file not found!")

"""
==== IMPORTANT: HOW TO RUN TESTS WITH REAL APIs ====

For the tests to access real API keys, use one of the following methods:

1. Direct environment variables:
   LLM_API_KEY="your-deepseek-key" OPENAI_API_KEY="your-openai-key" USE_REAL_API=1 pytest tests/test_llm_manager.py

2. Using .env file with bash:
   (source .env && USE_REAL_API=1 pytest tests/test_llm_manager.py)

3. Using helper script:
   python tests/run_api_tests.py

Note: The .env file should contain the following variables:
- LLM_API_KEY=sk-...        # DeepSeek R1 API key
- OPENAI_API_KEY=sk-...     # OpenAI API key
- USE_REAL_API=1            # Enable real API testing
"""

# Print warnings for missing API keys
if 'pytest' in sys.modules:
    if not os.getenv('LLM_API_KEY'):
        print("WARNING: LLM_API_KEY not set. Tests using DeepSeek R1 API will be skipped.")
    if not os.getenv('OPENAI_API_KEY'):
        print("WARNING: OPENAI_API_KEY not set. Tests using GPT-4o API will be skipped.")

# Debug: Print all environment variables related to API keys
print(f"Environment variables after setup:")
print(f"LLM_API_KEY: {'[SET]' if os.getenv('LLM_API_KEY') else '[NOT SET]'}")
print(f"OPENAI_API_KEY: {'[SET]' if os.getenv('OPENAI_API_KEY') else '[NOT SET]'}")
print(f"USE_REAL_API: {os.getenv('USE_REAL_API')}")

# Import after setting environment variables
from bot.llm_manager import (
    get_decision_from_llm, make_rule_based_decision, call_real_llm_api, log_decision_with_context,
    LLMManager, OPENAI_API_KEY, OPENAI_MODEL
)

# Flag to control whether tests use real API calls
# Check if USE_REAL_API is set in environment
USE_REAL_API = bool(int(os.getenv('USE_REAL_API', '0')))

# Check if API keys are available
DEEPSEEK_KEY_AVAILABLE = bool(os.getenv('LLM_API_KEY'))
OPENAI_KEY_AVAILABLE = bool(os.getenv('OPENAI_API_KEY'))

# Print debug info about environment when running tests
print(f"Test environment: USE_REAL_API={USE_REAL_API}, DeepSeek API available: {DEEPSEEK_KEY_AVAILABLE}, OpenAI API available: {OPENAI_KEY_AVAILABLE}")

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

@pytest.fixture
def generate_real_responses(use_real_api, llm_manager):
    """Generate real model responses if real API testing is enabled.
    
    Returns a tuple of (deepseek_response, gpt4o_response) either from real APIs
    or using the sample fixtures.
    
    Args:
        use_real_api: Flag to determine if real API calls should be used
        llm_manager: LLM manager instance for API calls
    """
    if not use_real_api:
        # Return None to use the sample fixtures
        return None, None
        
    # Check for required API keys
    deepseek_key = os.getenv('LLM_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not deepseek_key or not openai_key:
        # Return None to use the sample fixtures
        return None, None
    
    # Simple prompt for real API analysis
    prompt = "Analyze BTC/USDT price action. Current price is 50,000 USDT with RSI at 65 and positive MACD crossover."
    
    try:
        # Generate real DeepSeek R1 response
        deepseek_response = llm_manager._call_deepseek_api_raw(prompt)
        
        # Generate real GPT-4o response by processing the DeepSeek response
        gpt4o_response = llm_manager._process_with_gpt4o(deepseek_response, "BTCUSDT")
        
        # Return the real responses
        return deepseek_response, gpt4o_response
    except Exception as e:
        print(f"Error generating real responses: {e}")
        return None, None

@pytest.fixture
def real_deepseek_response(generate_real_responses, sample_deepseek_response):
    """Get a real DeepSeek R1 response if available, otherwise use the sample."""
    deepseek_response, _ = generate_real_responses
    return deepseek_response if deepseek_response else sample_deepseek_response

@pytest.fixture
def real_gpt4o_response(generate_real_responses, sample_gpt4o_response):
    """Get a real GPT-4o response if available, otherwise use the sample."""
    _, gpt4o_response = generate_real_responses
    return gpt4o_response if gpt4o_response else sample_gpt4o_response

@pytest.fixture
def use_real_api():
    """Flag to determine if real API calls should be used for testing.
    
    Set USE_REAL_API=1 in environment to enable real API testing.
    Returns True only if both USE_REAL_API is set and API keys are available.
    """
    # Return True only if USE_REAL_API is set AND we have at least one API key
    return USE_REAL_API and (DEEPSEEK_KEY_AVAILABLE or OPENAI_KEY_AVAILABLE)

@pytest.fixture
def llm_manager(use_real_api):
    """Create a test instance of LLMManager.
    
    Args:
        use_real_api: Whether to use real API keys for testing
    """
    if use_real_api:
        # Use real API keys from environment for testing
        return LLMManager()
    else:
        # Use mock API keys
        with patch('bot.llm_manager.LLM_API_KEY', 'fake_key'), \
             patch('bot.llm_manager.LLM_API_ENDPOINT', 'https://api.deepseek.com/v1/chat/completions'), \
             patch('bot.llm_manager.LLM_MODEL', 'deepseek-reasoner'), \
             patch('bot.llm_manager.OPENAI_API_KEY', 'fake-openai-key'), \
             patch('bot.llm_manager.OPENAI_MODEL', 'gpt-4o'):
            return LLMManager()

@pytest.fixture
def sample_deepseek_response():
    """Sample response from DeepSeek R1 with reasoning and decision."""
    return """<think>
    Let me analyze the current market conditions for BTCUSDT.
    
    1. Price Analysis:
       - Current price is 50050 USDT
       - Price is above the 200-day moving average of 48500 USDT, which is bullish
       - We've seen a 3% increase in the last 24 hours
    
    2. Technical Indicators:
       - RSI is at 68, approaching overbought (70+) but not quite there yet
       - MACD shows a bullish crossover recently
       - Trading volume has increased by 15% compared to the 7-day average
    
    3. Strategy Signals:
       - Moving average strategy: BUY 
       - RSI strategy: NEUTRAL
       - Volume analysis: BUY
    
    4. Market Context:
       - Overall market sentiment appears positive
       - No significant negative news events recently
       - Bitcoin dominance has been increasing slightly
    
    Given these factors, particularly the price being above key moving averages, the recent MACD bullish crossover, and increasing volume, the signals point towards a BUY recommendation.
    
    However, the RSI approaching 70 suggests caution as the asset may be nearing overbought territory.
    
    Weighing all factors, I believe the positive signals outweigh the caution signs.
    </think>
    
    Based on my analysis of BTCUSDT, I recommend a BUY decision for the following reasons:
    
    1. Price momentum is positive, with the current price (50050 USDT) above the 200-day moving average
    2. MACD shows a recent bullish crossover, indicating increasing positive momentum
    3. Trading volume has increased 15% compared to the 7-day average, confirming the price movement
    4. Multiple strategy signals (moving average and volume analysis) suggest buying
    
    While the RSI at 68 is approaching overbought territory, the other indicators strongly support a buying opportunity at this time.
    
    BUY"""

@pytest.fixture
def sample_gpt4o_response():
    """Sample structured response from GPT-4o."""
    return {
        "decision": "BUY",
        "confidence": 0.85,
        "reasoning": "Strong bullish signals including price above 200-day MA, positive MACD crossover, and increased trading volume, despite RSI approaching overbought territory at 68."
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
@patch('bot.llm_manager.LLM_API_ENDPOINT', 'https://api.deepseek.com/v1/chat/completions')
@patch('bot.llm_manager.LLM_MODEL', 'deepseek-reasoner')
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
         patch('bot.llm_manager.LLM_API_ENDPOINT', 'https://api.deepseek.com/v1/chat/completions'), \
         patch('bot.llm_manager.LLM_MODEL', 'deepseek-reasoner'), \
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
         patch('bot.llm_manager.LLM_API_ENDPOINT', 'https://api.deepseek.com/v1/chat/completions'), \
         patch('bot.llm_manager.LLM_MODEL', 'deepseek-reasoner'), \
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

def test_clean_deepseek_response(llm_manager, sample_deepseek_response):
    """Test the _clean_deepseek_response method."""
    # Test with a response containing <think> tags
    cleaned_response = llm_manager._clean_deepseek_response(sample_deepseek_response)
    
    # Verify that <think> tags were removed
    assert "<think>" not in cleaned_response
    assert "</think>" not in cleaned_response
    
    # Verify that the actual decision content is preserved
    assert "Based on my analysis of BTCUSDT" in cleaned_response
    assert "BUY" in cleaned_response
    
    # Test with a response not containing <think> tags
    regular_response = "I analyzed the market and recommend a BUY."
    cleaned_regular = llm_manager._clean_deepseek_response(regular_response)
    assert cleaned_regular == regular_response  # Should remain unchanged

def test_call_deepseek_api_raw(llm_manager):
    """Test the _call_deepseek_api_raw method."""
    # Mock the DeepSeek API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [
            {'message': {'content': 'Test response from DeepSeek R1'}}
        ]
    }
    
    prompt = "Analyze BTC/USDT trading pair"
    
    with patch('bot.llm_manager.requests.post', return_value=mock_response):
        result = llm_manager._call_deepseek_api_raw(prompt)
    
    # Verify that the raw content is returned
    assert result == 'Test response from DeepSeek R1'

def test_call_deepseek_api_raw_error(llm_manager):
    """Test the _call_deepseek_api_raw method with an API error."""
    # Mock error response
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    
    prompt = "Analyze BTC/USDT trading pair"
    
    with patch('bot.llm_manager.requests.post', return_value=mock_response), \
         patch('bot.llm_manager.logger') as mock_logger:
        result = llm_manager._call_deepseek_api_raw(prompt)
    
    # Verify that an error message is returned
    assert "Error calling DeepSeek API" in result
    mock_logger.error.assert_called_once()

def test_process_with_gpt4o(llm_manager, sample_deepseek_response, sample_gpt4o_response, use_real_api):
    """Test the _process_with_gpt4o method.
    
    This test can use either mock data or real API calls depending on the use_real_api flag.
    """
    if use_real_api:
        # Skip this test if no OpenAI API key is available
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("Skipping real API test: No OpenAI API key available")
            
        # Use real API call to process the sample DeepSeek response
        result = llm_manager._process_with_gpt4o(sample_deepseek_response, "BTCUSDT")
        
        # Verify we got a valid structured response
        assert "decision" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert result["decision"] in ["BUY", "SELL", "HOLD"]
        assert 0.5 <= result["confidence"] <= 1.0
        assert len(result["reasoning"]) > 10
    else:
        # Use mocked response for regular testing
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [
                {'message': {'content': json.dumps(sample_gpt4o_response)}}
            ]
        }
        
        with patch('bot.llm_manager.requests.post', return_value=mock_response), \
             patch.object(llm_manager, '_clean_deepseek_response', return_value=sample_deepseek_response):
            result = llm_manager._process_with_gpt4o(sample_deepseek_response, "BTCUSDT")
        
        # Verify that the structured data is returned
        assert result == sample_gpt4o_response
        assert result["decision"] == "BUY"
        assert result["confidence"] == 0.85
        assert "bullish signals" in result["reasoning"]

def test_process_with_gpt4o_error(llm_manager, sample_deepseek_response):
    """Test the _process_with_gpt4o method with an API error."""
    # Mock error response
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    
    with patch('bot.llm_manager.requests.post', return_value=mock_response), \
         patch.object(llm_manager, '_clean_deepseek_response', return_value=sample_deepseek_response), \
         patch('bot.llm_manager.logger') as mock_logger:
        result = llm_manager._process_with_gpt4o(sample_deepseek_response, "BTCUSDT")
    
    # Verify fallback behavior
    assert result["decision"] == "HOLD"
    assert result["confidence"] == 0.5
    assert "Error processing with GPT-4o" in result["reasoning"]
    mock_logger.error.assert_called_once()

def test_call_deepseek_api_raw_real(use_real_api, llm_manager):
    """Test the _call_deepseek_api_raw method with real API.
    
    This test uses the real DeepSeek R1 API if the USE_REAL_API flag is set.
    """
    if not use_real_api:
        pytest.skip("Skipping real DeepSeek API test - set USE_REAL_API=1 to enable")
        
    # Check if DeepSeek API key is available
    deepseek_key = os.getenv('LLM_API_KEY')
    if not deepseek_key:
        pytest.skip("Skipping real DeepSeek API test: No LLM_API_KEY available")
    
    # Simple prompt for analysis
    prompt = "Analyze BTC/USDT price action. Current price is 50,000 USDT with RSI at 65 and positive MACD crossover."
    
    # Make real API call
    result = llm_manager._call_deepseek_api_raw(prompt)
    
    # Verify we got a valid response
    assert result
    assert len(result) > 100  # Should have substantial content
    assert "BUY" in result or "SELL" in result or "HOLD" in result
    
    # Log the actual API-generated response
    print(f"\nReal DeepSeek R1 Response:\n{result[:500]}...")  # Print first 500 chars

def test_make_llm_decision_with_gpt4o(llm_manager, sample_data, sample_deepseek_response, sample_gpt4o_response):
    """Test the make_llm_decision method with GPT-4o processing."""
    with patch.object(llm_manager, '_call_deepseek_api_raw', return_value=sample_deepseek_response), \
         patch.object(llm_manager, '_process_with_gpt4o', return_value=sample_gpt4o_response):
        result = llm_manager.make_llm_decision(
            sample_data['market_data'], 
            'BTCUSDT', 
            '1h', 
            'Current market analysis', 
            sample_data['signals']
        )
    
    # Verify the result uses GPT-4o structured output
    assert result["decision"] == "buy"  # Should be lowercase
    assert result["confidence"] == 0.85
    assert "bullish signals" in result["reasoning"]

def test_make_llm_decision_without_openai(llm_manager, sample_data):
    """Test the make_llm_decision method without OpenAI API key."""
    # Create a new manager with no OpenAI API key
    with patch.object(llm_manager, 'openai_api_key', ''), \
         patch.object(llm_manager, '_call_deepseek_api', return_value={
             "decision": "SELL",
             "confidence": 0.7,
             "reasoning": "Bearish market indicators"
         }):
        
        result = llm_manager.make_llm_decision(
            sample_data['market_data'], 
            'BTCUSDT', 
            '1h', 
            'Current market analysis', 
            sample_data['signals']
        )
    
    # Should fall back to DeepSeek R1 only
    assert result["decision"] == "sell"
    assert result["confidence"] == 0.7
    assert "Bearish market indicators" in result["reasoning"]

def test_llm_manager_fallback_to_rule_based(llm_manager, sample_data):
    """Test that make_llm_decision falls back to rule-based when API is unavailable."""
    # Set API key to empty to force fallback
    with patch.object(llm_manager, 'api_key', ''), \
         patch('bot.llm_manager.make_rule_based_decision', return_value="SELL"):
        
        result = llm_manager.make_llm_decision(
            sample_data['market_data'], 
            'BTCUSDT', 
            '1h', 
            'Current market analysis', 
            sample_data['signals']
        )
    
    # Check the result uses rule-based values
    assert result["decision"] == "sell"  # Should be lowercase
    assert result["confidence"] == 0.6  # Rule-based confidence
    assert "rule-based" in result["reasoning"].lower()

def test_real_api_integration(use_real_api, sample_data):
    """Integration test using real API calls to both DeepSeek R1 and GPT-4o.
    
    This test is skipped unless USE_REAL_API=1 is set in the environment.
    """
    # Detailed debug information about environment
    print(f"\nTest environment details:")
    print(f"- USE_REAL_API env var: {os.getenv('USE_REAL_API')}")
    print(f"- use_real_api flag: {use_real_api}")
    print(f"- LLM_API_KEY available: {bool(os.getenv('LLM_API_KEY'))}")
    print(f"- OPENAI_API_KEY available: {bool(os.getenv('OPENAI_API_KEY'))}")
    print(f"- LLM_API_ENDPOINT: {os.getenv('LLM_API_ENDPOINT')}")
    print(f"- LLM_MODEL: {os.getenv('LLM_MODEL')}")
    print(f"- OPENAI_MODEL: {os.getenv('OPENAI_MODEL')}")
    
    if not use_real_api:
        pytest.skip("Skipping real API integration test - set USE_REAL_API=1 to enable")
        
    # Check if API keys are available
    deepseek_key = os.getenv('LLM_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not deepseek_key:
        pytest.skip("Skipping real API test: Missing DeepSeek API key (LLM_API_KEY)")
    
    if not openai_key:
        pytest.skip("Skipping real API test: Missing OpenAI API key (OPENAI_API_KEY)")
    
    # Create a real LLM manager instance
    manager = LLMManager()
    
    # Ensure API endpoint and model are set correctly
    if not manager.api_endpoint:
        print(f"Setting DeepSeek API endpoint manually")
        manager.api_endpoint = "https://api.deepseek.com/v1/chat/completions"
        
    if not manager.model:
        print(f"Setting DeepSeek model manually")
        manager.model = "deepseek-reasoner"
    
    # Log the API configuration from manager
    print(f"LLM Manager configuration (after fixes):")
    print(f"- DeepSeek API key available: {bool(manager.api_key)}")
    print(f"- DeepSeek API endpoint: {manager.api_endpoint}")
    print(f"- DeepSeek Model: {manager.model}")
    print(f"- OpenAI API key available: {bool(manager.openai_api_key)}")
    
    # Directly test both API connections before the real test
    print("\nTesting direct DeepSeek API connection...")
    test_prompt = "Analyze BTC price trend with RSI=60."
    deepseek_result = manager._call_deepseek_api_raw(test_prompt)
    print(f"DeepSeek API response length: {len(deepseek_result)}")
    print(f"DeepSeek response excerpt: {deepseek_result[:100]}...")
    
    # Make sure it's not an error message
    assert not deepseek_result.startswith("Error:"), f"DeepSeek API error: {deepseek_result}"
    
    # Make a real decision with both APIs
    result = manager.make_llm_decision(
        sample_data['market_data'],
        'BTCUSDT',
        '1h',
        'Current market analysis for BTCUSDT with indicators',
        sample_data['signals']
    )
    
    # Verify the response structure without assuming specific content
    assert "decision" in result
    assert "confidence" in result
    assert "reasoning" in result
    assert result["decision"] in ["buy", "sell", "hold"]
    assert 0.5 <= result["confidence"] <= 1.0
    assert len(result["reasoning"]) > 10
    
    # Make sure we're not getting the rule-based fallback
    assert "rule-based" not in result["reasoning"].lower(), "Test is using rule-based fallback instead of real APIs"
    
    # Log the actual API-generated response
    print(f"\nReal API Response:\n{json.dumps(result, indent=2)}")

def test_model_generated_responses(use_real_api, real_deepseek_response, real_gpt4o_response):
    """Test using model-generated responses from fixtures.
    
    This test verifies the format of responses from DeepSeek R1 and GPT-4o,
    either from real APIs or from sample fixtures, depending on the use_real_api flag.
    """
    # Check DeepSeek R1 response
    assert real_deepseek_response
    assert len(real_deepseek_response) > 50
    assert isinstance(real_deepseek_response, str)
    
    # Log the response source
    if use_real_api and os.getenv('LLM_API_KEY'):
        print("\nUsing real DeepSeek R1 response")
    else:
        print("\nUsing sample DeepSeek R1 response")
    
    # Check GPT-4o response
    assert real_gpt4o_response
    assert "decision" in real_gpt4o_response
    assert "confidence" in real_gpt4o_response
    assert "reasoning" in real_gpt4o_response
    
    # Log the response source
    if use_real_api and os.getenv('OPENAI_API_KEY'):
        print("Using real GPT-4o response")
    else:
        print("Using sample GPT-4o response")
    
    # Print the actual responses
    print(f"\nDeepSeek R1 Response (excerpt):\n{real_deepseek_response[:200]}...")
    print(f"\nGPT-4o Response:\n{json.dumps(real_gpt4o_response, indent=2)}")
