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
    get_decision_from_llm, log_decision_with_context,
    LLMManager, get_primary_model_config, get_secondary_model_config
)
from bot.config import TRADING_CONFIG

# Flag to control whether tests use real API calls
# Check if USE_REAL_API is set in environment
USE_REAL_API = bool(int(os.getenv('USE_REAL_API', '0')))

# Check if API keys are available
DEEPSEEK_KEY_AVAILABLE = bool(os.getenv('LLM_API_KEY'))
OPENAI_KEY_AVAILABLE = bool(os.getenv('OPENAI_API_KEY'))

# Get model names from config for testing
PRIMARY_MODEL = get_primary_model_config().get("model", "deepseek-reasoner")
SECONDARY_MODEL = get_secondary_model_config().get("model", "gpt-4o")

# Print debug info about environment when running tests
print(f"Test environment: USE_REAL_API={USE_REAL_API}, DeepSeek API available: {DEEPSEEK_KEY_AVAILABLE}, OpenAI API available: {OPENAI_KEY_AVAILABLE}")

@pytest.fixture
def sample_data():
    """Sample market data and signals for testing."""
    return {
        'market_data': {
            'symbol': 'BTCUSDT',
            'candles': [
                [1672531200000, "16500.0", "16550.0", "16480.0", "16525.0", "100.5", 1672534800000, "1660000.0", 500, "60.3", "995000.0", "0"],
                [1672534800000, "16525.0", "16575.0", "16510.0", "16550.0", "120.7", 1672538400000, "1995000.0", 600, "72.4", "1197000.0", "0"],
                [1672538400000, "16550.0", "16600.0", "16540.0", "16580.0", "150.2", 1672542000000, "2490000.0", 700, "90.1", "1494000.0", "0"]
            ],
            'order_book': {
                'bids': [["16570.0", "2.5"], ["16565.0", "5.0"]],
                'asks': [["16585.0", "1.8"], ["16590.0", "4.2"]]
            },
            'timestamp': '2023-01-01T03:00:00.000000'
        },
        'signals': {
            'simple': 'BUY',
            'technical': 'HOLD'
        }
    }

@pytest.fixture
def mock_trading_config():
    """Mock configuration for testing."""
    return {
        "decision_making": {
            "llm": {
                "enabled": True,
                "required_confidence": 0.6,
                "models": {
                    "primary": {
                        "provider": "deepseek",
                        "model": "deepseek-reasoner",
                        "api_key": "fake_deepseek_key",
                        "api_endpoint": "https://api.deepseek.com/v1/chat/completions",
                        "temperature": 0.3
                    },
                    "secondary": {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "fake_openai_key",
                        "api_endpoint": "https://api.openai.com/v1/chat/completions",
                        "temperature": 0.1
                    }
                }
            }
        }
    }

@pytest.fixture
def generate_real_responses(use_real_api, llm_manager):
    """
    Generate real API responses if real API testing is enabled.
    
    Returns:
        Function to get responses
    """
    def _generate_response(api_type="deepseek"):
        if not use_real_api:
            return None
            
        if api_type == "deepseek" and not DEEPSEEK_KEY_AVAILABLE:
            return None
            
        if api_type == "openai" and not OPENAI_KEY_AVAILABLE:
            return None
        
        if api_type == "deepseek":
            prompt = "Bitcoin price has been rising for the past 3 days, RSI is 65, and MACD shows a bullish crossover. Should I buy, sell, or hold?"
            response = llm_manager._call_primary_model(prompt)
            return response
            
        if api_type == "openai":
            analysis = """
            After analyzing the market data for BTCUSDT, I can see the following:
            
            1. Price has been steadily increasing over the last 3 candles, from 16525 to 16580.
            2. The RSI is at 65, which is approaching overbought territory but not yet there.
            3. The MACD shows a bullish crossover, indicating positive momentum.
            4. Trading volume has been increasing, showing stronger buying pressure.
            
            Given these indicators suggesting upward momentum, I believe we should BUY.
            """
            response = llm_manager._process_with_secondary_model(analysis, "BTCUSDT")
            return response
            
        return None
        
    return _generate_response

@pytest.fixture
def real_deepseek_response(generate_real_responses, sample_deepseek_response):
    """Get a real DeepSeek R1 API response or use the sample if real API is not available."""
    return generate_real_responses("deepseek") or sample_deepseek_response

@pytest.fixture
def real_gpt4o_response(generate_real_responses, sample_gpt4o_response):
    """Get a real GPT-4o API response or use the sample if real API is not available."""
    return generate_real_responses("openai") or sample_gpt4o_response

@pytest.fixture
def use_real_api():
    """
    Fixture to control whether tests use real API calls.
    
    Returns:
        bool: True if using real API, False otherwise
    """
    return USE_REAL_API

@pytest.fixture
def llm_manager(use_real_api, mock_trading_config):
    """
    Create an LLMManager instance for testing.
    
    If USE_REAL_API is True, use real API keys from environment.
    Otherwise, use mock keys and return mocked responses.
    """
    # For real API tests, use actual API keys
    if use_real_api:
        primary_config = get_primary_model_config()
        secondary_config = get_secondary_model_config()
        
        # Use environment variables directly to ensure we have the latest values
        primary_api_key = os.getenv('LLM_API_KEY', primary_config.get('api_key', ''))
        secondary_api_key = os.getenv('OPENAI_API_KEY', secondary_config.get('api_key', ''))
        
        with patch('bot.llm_manager.get_primary_model_config', return_value={
                "provider": primary_config.get("provider", "deepseek"),
                "model": primary_config.get("model", "deepseek-reasoner"),
                "api_key": primary_api_key,
                "api_endpoint": primary_config.get("api_endpoint", "https://api.deepseek.com/v1/chat/completions"),
                "temperature": primary_config.get("temperature", 0.3)
            }), \
            patch('bot.llm_manager.get_secondary_model_config', return_value={
                "provider": secondary_config.get("provider", "openai"),
                "model": secondary_config.get("model", "gpt-4o"),
                "api_key": secondary_api_key,
                "api_endpoint": secondary_config.get("api_endpoint", "https://api.openai.com/v1/chat/completions"),
                "temperature": secondary_config.get("temperature", 0.1)
            }):
            return LLMManager()
    
    # For mock tests, use fake API keys
    with patch('bot.llm_manager.get_primary_model_config', return_value=mock_trading_config["decision_making"]["llm"]["models"]["primary"]), \
         patch('bot.llm_manager.get_secondary_model_config', return_value=mock_trading_config["decision_making"]["llm"]["models"]["secondary"]):
        return LLMManager()

@pytest.fixture
def sample_deepseek_response():
    """Sample response from DeepSeek R1 API for testing."""
    return """
    <think>
    Let me analyze the current market data and signals for BTCUSDT:
    
    1. Simple signal: BUY - This is likely based on short-term price action.
    2. Technical signal: HOLD - This suggests that technical indicators are mixed.
    
    The market data shows that the price has been increasing over the last few candles, from 16525 to 16580. The order book shows more buying pressure at bid levels than selling pressure at ask levels.
    
    Given the positive short-term momentum but potential caution from technical indicators, I should consider both perspectives before making a recommendation.
    </think>
    
    Based on the analysis of current signals and market data for BTCUSDT, I can see a positive short-term momentum with the simple signal indicating BUY, while the technical analysis is more cautious with a HOLD signal.
    
    The recent price action shows a steady upward trend with each candle closing higher than the previous one (16525 → 16550 → 16580). The order book also reveals decent buying pressure at the bid levels of 16570 and 16565.
    
    While the technical indicators suggest caution (likely due to potential overbought conditions or resistance levels), the consistent price appreciation and positive order book depth indicate a favorable short-term outlook.
    
    Given the balance of these signals and the apparent strength in buying momentum, I recommend a BUY position, though with appropriate risk management due to the cautionary technical indicators.
    
    BUY
    """

@pytest.fixture
def sample_gpt4o_response():
    """Sample response from GPT-4o API for testing."""
    # GPT-4o with structured output will return a JSON string
    return json.dumps({
        "decision": "BUY",
        "confidence": 0.75,
        "reasoning": "Price is in an uptrend with increasing volume, and order book shows strong bid support."
    })

def test_make_rule_based_decision():
    """Test the rule-based decision making fallback."""
    # Test prompt with BUY signal from simple, HOLD from technical
    prompt = "Simple signal: BUY\nTechnical analysis signal: HOLD"
    
    with patch('bot.llm_manager.make_rule_based_decision', wraps=lambda x: "BUY") as mock_func:
        llm_manager = LLMManager()
        result = llm_manager._make_rule_based_decision({}, {"simple": "BUY", "technical": "HOLD"})
        
        # Check that the result has the expected format and decision
        assert isinstance(result, dict)
        assert "decision" in result
        assert "confidence" in result
        assert "reasoning" in result
        
        # Rule-based decisions should have a lower confidence value
        assert result["confidence"] == 0.6
        
        # Check that reasoning is included
        assert "rule-based" in result["reasoning"].lower()

@patch('bot.llm_manager.is_llm_enabled', return_value=True)
def test_get_decision_from_llm(mock_is_llm_enabled, llm_manager):
    """Test the get_decision_from_llm function with mocked LLMManager."""
    with patch.object(LLMManager, 'make_llm_decision', return_value={
        "decision": "BUY",
        "confidence": 0.8, 
        "reasoning": "Test reasoning"
    }):
        decision = get_decision_from_llm("Test prompt")
        assert decision == "BUY"
        
        # Test with different return value
        with patch.object(LLMManager, 'make_llm_decision', return_value={
            "decision": "SELL",
            "confidence": 0.7,
            "reasoning": "Different reasoning"
        }):
            decision = get_decision_from_llm("Another test prompt")
            assert decision == "SELL"

@patch('bot.llm_manager.is_llm_enabled', return_value=False)
def test_get_decision_from_llm_disabled(mock_is_llm_enabled):
    """Test the get_decision_from_llm function when LLM is disabled."""
    decision = get_decision_from_llm("Test prompt")
    assert decision == "HOLD"  # Should default to HOLD when disabled

def test_log_decision_with_context(sample_data):
    """Test logging of decisions with context for audit trail."""
    with patch('bot.llm_manager.logger') as mock_logger:
        decision = "BUY"
        signals = sample_data['signals']
        market_data = sample_data['market_data']
        
        log_decision_with_context(decision, signals, market_data)
        
        # Verify that the logger was called with appropriate JSON
        mock_logger.info.assert_called_once()
        log_call_args = mock_logger.info.call_args[0][0]
        assert "Decision log:" in log_call_args
        
        # Extract the JSON from the log message
        log_data_str = log_call_args.replace("Decision log: ", "")
        log_data = json.loads(log_data_str)
        
        # Verify the structure of the logged data
        assert "timestamp" in log_data
        assert "decision" in log_data
        assert log_data["decision"] == "BUY"
        assert "signals" in log_data
        assert log_data["signals"]["simple"] == "BUY"
        assert log_data["signals"]["technical"] == "HOLD"
        assert "market_data_summary" in log_data

def test_clean_primary_response(llm_manager):
    """Test cleaning of primary model responses to remove special tags."""
    # Sample response with <think> tags
    response_with_tags = """
    <think>
    Internal reasoning process that shouldn't be shown to the user.
    Calculating various technical indicators.
    </think>
    
    Based on the analysis, I recommend a BUY action for the following reasons:
    1. Upward price momentum
    2. Strong support levels
    3. Positive technical indicators
    
    BUY
    """
    
    cleaned = llm_manager._clean_primary_response(response_with_tags)
    
    # Verify that <think> tags were removed
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    
    # But the main content is preserved
    assert "Based on the analysis" in cleaned
    assert "BUY" in cleaned
    
    # Test with no tags
    response_without_tags = "This is a clean response with no tags. BUY"
    cleaned_no_tags = llm_manager._clean_primary_response(response_without_tags)
    assert cleaned_no_tags == response_without_tags

def test_call_primary_model(llm_manager):
    """Test calling the primary LLM model."""
    with patch('requests.post') as mock_post:
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Based on the analysis, I recommend BUY."}}
            ]
        }
        mock_post.return_value = mock_response
        
        prompt = "Should I buy or sell based on RSI=65 and MACD bullish crossover?"
        response = llm_manager._call_primary_model(prompt)
        
        # Verify that the response content was extracted correctly
        assert "Based on the analysis" in response
        assert "BUY" in response
        
        # Verify that the API was called with correct parameters
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = json.loads(kwargs['data'])
        
        # Verify the request structure
        assert "model" in payload
        assert "messages" in payload
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == prompt

def test_call_primary_model_error(llm_manager):
    """Test error handling when calling the primary LLM model."""
    with patch('requests.post') as mock_post:
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        prompt = "Test prompt"
        response = llm_manager._call_primary_model(prompt)
        
        # Response should contain error information
        assert "Error" in response
        assert "500" in response
        
        # Test connection error
        mock_post.side_effect = Exception("Connection failed")
        response = llm_manager._call_primary_model(prompt)
        assert "Error" in response
        assert "Connection failed" in response

def test_process_with_secondary_model(llm_manager, sample_deepseek_response):
    """Test processing the primary model's response with the secondary model."""
    with patch('requests.post') as mock_post:
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({
                    "decision": "BUY",
                    "confidence": 0.75,
                    "reasoning": "Strong uptrend with increasing volume"
                })}}
            ]
        }
        mock_post.return_value = mock_response
        
        # Set up the LLMManager with secondary API key
        llm_manager.secondary_api_key = "fake_key"
        
        # Process the sample response
        result = llm_manager._process_with_secondary_model(sample_deepseek_response, "BTCUSDT")
        
        # Verify the result structure
        assert "decision" in result
        assert "confidence" in result
        assert "reasoning" in result
        
        # Check the extracted values
        assert result["decision"] == "BUY"
        assert result["confidence"] == 0.75
        assert "uptrend" in result["reasoning"]
        
        # Verify that the API was called with correct parameters
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = json.loads(kwargs['data'])
        
        # Verify the request structure
        assert "model" in payload
        assert "messages" in payload
        assert "response_format" in payload
        assert payload["response_format"]["type"] == "json_object"

def test_process_with_secondary_model_error(llm_manager, sample_deepseek_response):
    """Test error handling when processing with the secondary model."""
    # Test missing API key
    llm_manager.secondary_api_key = ""
    result = llm_manager._process_with_secondary_model(sample_deepseek_response, "BTCUSDT")
    
    # Verify default values on error
    assert result["decision"] == "HOLD"
    assert result["confidence"] == 0.5
    assert "unavailable" in result["reasoning"]
    
    # Test API error
    llm_manager.secondary_api_key = "fake_key"
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        result = llm_manager._process_with_secondary_model(sample_deepseek_response, "BTCUSDT")
        
        # Verify default values on error
        assert result["decision"] == "HOLD"
        assert result["confidence"] == 0.5
        assert "Error" in result["reasoning"]
        assert "500" in result["reasoning"]

@pytest.mark.skipif(not USE_REAL_API or not DEEPSEEK_KEY_AVAILABLE, 
                   reason="Skipping real API test - need USE_REAL_API=1 and LLM_API_KEY set")
def test_call_primary_model_real(use_real_api, llm_manager):
    """Test calling the primary LLM model with real API (if enabled)."""
    prompt = "Bitcoin price has been rising for the past 3 days, RSI is 65, and MACD shows a bullish crossover. Should I buy, sell, or hold?"
    response = llm_manager._call_primary_model(prompt)
    
    # Verify that we received a valid response
    assert response
    assert isinstance(response, str)
    assert len(response) > 50
    
    # Check if the response contains common trading terms
    common_terms = ["buy", "sell", "hold", "market", "price", "trend", "rsi", "macd"]
    contains_terms = any(term in response.lower() for term in common_terms)
    assert contains_terms, "Response doesn't contain expected trading terminology"
    
    # Log the response for debugging
    print(f"\nReal DeepSeek API Response:\n{response}\n")

def test_make_llm_decision(llm_manager, sample_data, sample_deepseek_response):
    """Test the complete LLM decision making pipeline."""
    # Mock both API calls
    with patch.object(llm_manager, '_call_primary_model', return_value=sample_deepseek_response), \
         patch.object(llm_manager, '_process_with_secondary_model', return_value={
             "decision": "BUY",
             "confidence": 0.8,
             "reasoning": "Test reasoning"
         }):
        result = llm_manager.make_llm_decision(
            sample_data['market_data'],
            'BTCUSDT',
            '1h',
            'Testing LLM decision pipeline',
            sample_data['signals']
        )
        
        # Verify the result
        assert result["decision"] == "BUY"
        assert result["confidence"] == 0.8
        assert result["reasoning"] == "Test reasoning"

@patch('bot.llm_manager.is_llm_enabled', return_value=False)
def test_make_llm_decision_disabled(mock_is_llm_enabled, llm_manager, sample_data):
    """Test the LLM decision making when LLM is disabled in config."""
    with patch.object(llm_manager, '_make_rule_based_decision', return_value={
        "decision": "HOLD",
        "confidence": 0.6, 
        "reasoning": "Rule-based decision"
    }):
        result = llm_manager.make_llm_decision(
            sample_data['market_data'],
            'BTCUSDT',
            '1h',
            'Testing disabled LLM',
            sample_data['signals']
        )
        
        # Should use rule-based decision
        assert result["decision"] == "HOLD"
        assert result["confidence"] == 0.6
        assert "Rule-based" in result["reasoning"]

@pytest.mark.skipif(not USE_REAL_API or not DEEPSEEK_KEY_AVAILABLE or not OPENAI_KEY_AVAILABLE,
                   reason="Skipping real API integration test - need both API keys")
def test_real_api_integration(use_real_api, sample_data):
    """Full integration test with real APIs (if enabled)."""
    # This test only runs if USE_REAL_API=1 and both API keys are available
    llm_manager = LLMManager()  # Use a real, non-mocked instance
    
    # Run the complete decision pipeline
    result = llm_manager.make_llm_decision(
        sample_data['market_data'],
        'BTCUSDT',
        '1h',
        'Real API integration test',
        sample_data['signals']
    )
    
    # Verify that we received a valid response
    assert isinstance(result, dict)
    assert "decision" in result
    assert "confidence" in result
    assert "reasoning" in result
    
    # Decision should be one of the valid options
    assert result["decision"] in ["BUY", "SELL", "HOLD"]
    
    # Confidence should be in the valid range
    assert 0.5 <= result["confidence"] <= 1.0
    
    # Reasoning should be non-empty
    assert result["reasoning"]
    assert len(result["reasoning"]) > 10
    
    # Log the result for debugging
    print(f"\nReal API Integration Result:\n{json.dumps(result, indent=2)}\n")

@pytest.mark.skipif(not USE_REAL_API, reason="Skipping real response test - need USE_REAL_API=1")
def test_model_generated_responses(use_real_api, real_deepseek_response, real_gpt4o_response):
    """Test parsing of real responses from the APIs."""
    if real_deepseek_response:
        print(f"\nReal DeepSeek response sample:\n{real_deepseek_response[:200]}...\n")
        assert len(real_deepseek_response) > 50
        contains_terms = any(term in real_deepseek_response.lower() for term in ["buy", "sell", "hold"])
        assert contains_terms, "DeepSeek response doesn't contain expected decision terms"
    
    if real_gpt4o_response:
        print(f"\nReal GPT-4o response sample:\n{real_gpt4o_response}\n")
        if isinstance(real_gpt4o_response, str) and real_gpt4o_response.startswith("{"):
            # Try to parse as JSON
            try:
                data = json.loads(real_gpt4o_response)
                assert "decision" in data
                assert "confidence" in data
                assert "reasoning" in data
            except json.JSONDecodeError:
                # Not JSON, should have decision terms
                contains_terms = any(term in real_gpt4o_response.lower() for term in ["buy", "sell", "hold"])
                assert contains_terms, "GPT-4o response doesn't contain expected decision terms"
        else:
            # Object response
            assert "decision" in real_gpt4o_response
            assert "confidence" in real_gpt4o_response
            assert "reasoning" in real_gpt4o_response
