import logging
import requests
import json
import os
from datetime import datetime

logger = logging.getLogger("trading_bot")

# Example of LLM API configuration (would come from .env in production)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

def get_decision_from_llm(prompt):
    """
    This function interfaces with LLMs for trading decisions.
    For now, it uses a simplified approach but can be expanded to call real LLM APIs.
    
    Args:
        prompt: String with context about signals and market data
        
    Returns:
        String: "BUY", "SELL", or "HOLD" decision
    """
    # Log the prompt for debugging
    logger.info(f"LLM received prompt: {prompt}")
    
    # Check if we have real LLM API credentials configured
    if LLM_API_KEY and LLM_API_ENDPOINT and LLM_MODEL:
        try:
            return call_real_llm_api(prompt)
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            # Fall back to rule-based decision
            return make_rule_based_decision(prompt)
    else:
        # Use rule-based decision logic when no LLM API is configured
        logger.info("No LLM API configured, using rule-based decision logic")
        return make_rule_based_decision(prompt)

def call_real_llm_api(prompt):
    """
    Call an actual LLM API service. This is a template function that can be
    customized for specific LLM providers (OpenAI, DeepSeek, etc).
    
    Returns:
        String: "BUY", "SELL", or "HOLD" decision
    """
    try:
        # Example request structure - modify for your specific LLM API
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": """
                    You are a trading assistant that helps make decisions based on technical indicators and market data.
                    You should respond ONLY with "BUY", "SELL", or "HOLD".
                    Consider technical indicators carefully and be conservative with your recommendations.
                """},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10,
            "temperature": 0.3
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}"
        }
        
        response = requests.post(
            LLM_API_ENDPOINT,
            headers=headers,
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            # Extract the decision from the response format
            # Adjust this according to the actual LLM API response structure
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"].strip().upper()
            
            # Validate and normalize the response
            if "BUY" in content:
                return "BUY"
            elif "SELL" in content:
                return "SELL"
            else:
                return "HOLD"
        else:
            logger.error(f"LLM API error: {response.status_code}, {response.text}")
            return "HOLD"  # Default to conservative action
            
    except Exception as e:
        logger.error(f"Exception calling LLM API: {e}")
        return "HOLD"  # Default to conservative action

def make_rule_based_decision(prompt):
    """
    A fallback method that implements simple rules to parse the prompt and make decisions.
    This simulates what an LLM might decide based on the provided signals.
    
    Returns:
        String: "BUY", "SELL", or "HOLD" decision
    """
    # Parse the prompt to identify strategy signals
    # This is a very basic parser - in production this would be more robust
    simple_signal = "HOLD"
    technical_signal = "HOLD"
    
    if "simple signal" in prompt.lower():
        if "simple signal" in prompt.lower() and "buy" in prompt.lower().split("simple signal")[1].split("\n")[0].lower():
            simple_signal = "BUY"
        elif "simple signal" in prompt.lower() and "sell" in prompt.lower().split("simple signal")[1].split("\n")[0].lower():
            simple_signal = "SELL"
    
    if "technical analysis signal" in prompt.lower():
        if "technical analysis signal" in prompt.lower() and "buy" in prompt.lower().split("technical analysis signal")[1].split("\n")[0].lower():
            technical_signal = "BUY"
        elif "technical analysis signal" in prompt.lower() and "sell" in prompt.lower().split("technical analysis signal")[1].split("\n")[0].lower():
            technical_signal = "SELL"
    
    logger.info(f"Rule-based parsed signals - Simple: {simple_signal}, Technical: {technical_signal}")
    
    # Apply decision rules
    # Give more weight to technical signals as they incorporate more indicators
    if technical_signal == "BUY" and simple_signal != "SELL":
        return "BUY"
    elif technical_signal == "SELL" and simple_signal != "BUY":
        return "SELL"
    elif simple_signal == "BUY" and technical_signal != "SELL":
        return "BUY"
    elif simple_signal == "SELL" and technical_signal != "BUY":
        return "SELL"
    else:
        # When in doubt, HOLD
        return "HOLD"

def log_decision_with_context(decision, signals, market_data):
    """
    Logs the decision along with relevant context for later analysis.
    This creates an audit trail for understanding why decisions were made.
    
    Args:
        decision: The trading decision ("BUY", "SELL", "HOLD")
        signals: Dictionary of strategy signals
        market_data: Dictionary of market data used for the decision
    """
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "decision": decision,
        "signals": signals,
        "market_data_summary": {
            "symbol": market_data.get("symbol"),
            "latest_close": market_data.get("candles", [])[-1][4] if market_data.get("candles") else None,
            "order_book_summary": {
                "top_bid": market_data.get("order_book", {}).get("bids", [])[0] if market_data.get("order_book", {}).get("bids") else None,
                "top_ask": market_data.get("order_book", {}).get("asks", [])[0] if market_data.get("order_book", {}).get("asks") else None,
            }
        }
    }
    
    logger.info(f"Decision log: {json.dumps(log_entry)}")
    
    # In a production system, you might store this in a database for analysis
