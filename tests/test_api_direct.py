#!/usr/bin/env python
"""
Test the DeepSeek API connection directly.
This script loads API keys from the .env file and makes a direct
API call to DeepSeek to verify connectivity.
"""

import os
import json
import requests
from dotenv import load_dotenv

def main():
    # Load environment variables from .env
    load_dotenv()
    
    # Get API keys and settings
    api_key = os.getenv('LLM_API_KEY')
    api_endpoint = os.getenv('LLM_API_ENDPOINT', 'https://api.deepseek.com/v1/chat/completions')
    model = os.getenv('LLM_MODEL', 'deepseek-reasoner')
    
    if not api_key:
        print("ERROR: LLM_API_KEY not found in environment")
        return
    
    print(f"DeepSeek API Key: {'*' * 5}{api_key[-5:]}")
    print(f"API Endpoint: {api_endpoint}")
    print(f"Model: {model}")
    
    # Test prompt
    prompt = "Analyze BTC price action with RSI at 60."
    
    # Make the API call
    print(f"\nMaking API call to DeepSeek...")
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a trading assistant that helps make trading decisions."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        response = requests.post(
            api_endpoint,
            headers=headers,
            data=json.dumps(payload)
        )
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            print("\nAPI Response (first 300 chars):")
            print(f"{content[:300]}...")
            
            # Check for BUY/SELL/HOLD decision
            if "BUY" in content or "SELL" in content or "HOLD" in content:
                print("\nDecision found in response:")
                if "BUY" in content:
                    print("Decision: BUY")
                elif "SELL" in content:
                    print("Decision: SELL")
                elif "HOLD" in content:
                    print("Decision: HOLD")
            
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception during API call: {e}")
        return False

if __name__ == "__main__":
    main() 