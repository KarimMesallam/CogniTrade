#!/usr/bin/env python
"""
Test the complete LLM workflow with DeepSeek R1 and GPT-4o.
This script loads API keys from the .env file and uses the LLMManager
to generate a trading decision using both models.
"""

import os
import sys
import json
from pprint import pprint
from dotenv import load_dotenv

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # Load environment variables from .env
    load_dotenv()
    
    # Get API keys and settings
    deepseek_key = os.getenv('LLM_API_KEY')
    deepseek_endpoint = os.getenv('LLM_API_ENDPOINT', 'https://api.deepseek.com/v1/chat/completions')
    deepseek_model = os.getenv('LLM_MODEL', 'deepseek-reasoner')
    
    openai_key = os.getenv('OPENAI_API_KEY')
    openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o')
    
    # Check if keys are available
    if not deepseek_key:
        print("ERROR: LLM_API_KEY not found in environment")
        return
        
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return
    
    # Print configuration
    print(f"API Configuration:")
    print(f"- DeepSeek API Key: {'*' * 5}{deepseek_key[-5:]}")
    print(f"- DeepSeek API Endpoint: {deepseek_endpoint}")
    print(f"- DeepSeek Model: {deepseek_model}")
    print(f"- OpenAI API Key: {'*' * 5}{openai_key[-5:]}")
    print(f"- OpenAI Model: {openai_model}")
    
    # Import the LLMManager after setting environment variables
    from bot.llm_manager import LLMManager
    
    # Create an instance of LLMManager
    print("\nInitializing LLMManager...")
    manager = LLMManager()
    
    # Set endpoint and model explicitly
    if not manager.api_endpoint:
        print("Setting DeepSeek endpoint explicitly")
        manager.api_endpoint = deepseek_endpoint
        
    if not manager.model:
        print("Setting DeepSeek model explicitly")
        manager.model = deepseek_model
    
    # Create sample market data
    sample_data = {
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
    
    # Test DeepSeek R1 API directly
    print("\nTesting DeepSeek R1 API...")
    prompt = "Analyze BTC/USDT trading pair with RSI at 60 and positive MACD."
    deepseek_response = manager._call_deepseek_api_raw(prompt)
    print(f"DeepSeek response length: {len(deepseek_response)}")
    print(f"DeepSeek response excerpt: {deepseek_response[:200]}...")
    
    # Test GPT-4o processing with the DeepSeek response
    print("\nTesting GPT-4o processing...")
    gpt4o_result = manager._process_with_gpt4o(deepseek_response, "BTCUSDT")
    print("GPT-4o structured output:")
    pprint(gpt4o_result)
    
    # Test the complete workflow
    print("\nTesting complete workflow with both models...")
    result = manager.make_llm_decision(
        sample_data['market_data'],
        'BTCUSDT',
        '1h',
        'Current market analysis',
        sample_data['signals']
    )
    
    print("\nFinal trading decision:")
    pprint(result)
    
    return True

if __name__ == "__main__":
    main() 