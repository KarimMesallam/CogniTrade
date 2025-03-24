import logging
import requests
import json
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional
from bot.config import TRADING_CONFIG, is_llm_enabled, get_required_llm_confidence

logger = logging.getLogger("trading_bot")

# Get LLM configuration from the new config system
def get_llm_config():
    """
    Get the LLM configuration from the trading config.
    """
    llm_config = TRADING_CONFIG.get("decision_making", {}).get("llm", {})
    return llm_config

def get_primary_model_config():
    """
    Get the primary LLM model configuration.
    """
    llm_config = get_llm_config()
    return llm_config.get("models", {}).get("primary", {})

def get_secondary_model_config():
    """
    Get the secondary LLM model configuration.
    """
    llm_config = get_llm_config()
    return llm_config.get("models", {}).get("secondary", {})

class LLMManager:
    """
    A class to manage LLM interactions for trading decisions using configured models.
    """
    def __init__(self):
        # Get primary model config
        primary_config = get_primary_model_config()
        self.api_key = primary_config.get("api_key", "")
        self.api_endpoint = primary_config.get("api_endpoint", "")
        self.model = primary_config.get("model", "")
        self.temperature = primary_config.get("temperature", 0.3)
        
        # Get secondary model config
        secondary_config = get_secondary_model_config()
        self.secondary_api_key = secondary_config.get("api_key", "")
        self.secondary_api_endpoint = secondary_config.get("api_endpoint", "")
        self.secondary_model = secondary_config.get("model", "")
        self.secondary_temperature = secondary_config.get("temperature", 0.1)
        
        # Read required confidence from config
        self.required_confidence = get_required_llm_confidence()
    
    def make_llm_decision(self, market_data, symbol, timeframe, context, strategy_signals=None):
        """
        Make a trading decision using LLMs based on market data and context.
        
        Args:
            market_data: Dictionary with market data
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            context: Additional context for the LLM
            strategy_signals: Optional signals from traditional strategies
            
        Returns:
            Dictionary with decision, confidence, and reasoning
        """
        # Check if LLM decisions are enabled in config
        if not is_llm_enabled():
            logger.info("LLM-based decisions are disabled in config. Using rule-based decision.")
            return self._make_rule_based_decision(market_data, strategy_signals)
        
        # Prepare context string from input data
        prompt = self._prepare_prompt(market_data, symbol, timeframe, context, strategy_signals)
        
        try:
            # Full pipeline (primary + secondary model)
            if self.api_key and self.api_endpoint and self.model and self.secondary_api_key:
                # Call primary LLM API first
                primary_response = self._call_primary_model(prompt)
                
                # Process the response with secondary model for structured output
                decision_result = self._process_with_secondary_model(primary_response, symbol)
                
                decision = decision_result.get("decision", "HOLD")
                confidence = decision_result.get("confidence", 0.5)
                reasoning = decision_result.get("reasoning", "Decision processed with secondary model")
                
                # Apply confidence threshold
                if confidence < self.required_confidence:
                    logger.info(f"LLM decision confidence {confidence} below required threshold {self.required_confidence}. Defaulting to HOLD.")
                    decision = "HOLD"
            
            # Primary model only
            elif self.api_key and self.api_endpoint and self.model:
                # Fall back to primary model only if secondary API key is not available
                logger.warning("Secondary model API key not available, falling back to primary model only")
                decision_result = self._call_primary_model_for_decision(prompt)
                decision = decision_result["decision"]
                confidence = decision_result["confidence"]
                reasoning = decision_result["reasoning"]
                
                # Apply confidence threshold
                if confidence < self.required_confidence:
                    logger.info(f"LLM decision confidence {confidence} below required threshold {self.required_confidence}. Defaulting to HOLD.")
                    decision = "HOLD"
            
            # Rule-based fallback
            else:
                return self._make_rule_based_decision(market_data, strategy_signals)
                
            return {
                "decision": decision.upper(),
                "confidence": confidence,
                "reasoning": reasoning
            }
        except Exception as e:
            logger.error(f"Error making LLM decision: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Error in LLM decision process: {str(e)}"
            }
    
    def _call_primary_model(self, prompt):
        """
        Call the primary LLM model API and return the raw response content.
        
        Args:
            prompt: String with context about signals and market data
            
        Returns:
            String: Raw response content from the primary model
        """
        try:
            # Get provider-specific settings
            provider = get_primary_model_config().get("provider", "").lower()
            
            # Prepare the request payload based on the provider
            if provider == "deepseek":
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": """
                            You are a trading assistant that helps make decisions based on technical indicators and market data.
                            Analyze the provided market data, indicators, and signals to make a trading decision.
                            Consider technical indicators carefully and be conservative with your recommendations.
                            First explain your reasoning and analysis process in detail.
                            Then conclude with ONLY ONE of these terms: "BUY", "SELL", or "HOLD".
                        """},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": self.temperature
                }
            elif provider == "openai":
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": """
                            You are a trading assistant that helps make decisions based on technical indicators and market data.
                            Analyze the provided market data, indicators, and signals to make a trading decision.
                            Consider technical indicators carefully and be conservative with your recommendations.
                            First explain your reasoning and analysis process in detail.
                            Then conclude with ONLY ONE of these terms: "BUY", "SELL", or "HOLD".
                        """},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 1000
                }
            else:
                # Generic payload format as fallback
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a trading assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature
                }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                # Extract just the raw content from the response
                response_data = response.json()
                
                # Handle different response formats from different providers
                if provider == "deepseek" or provider == "openai":
                    content = response_data["choices"][0]["message"]["content"]
                else:
                    # Generic fallback
                    content = response_data.get("output", response_data.get("content", str(response_data)))
                
                # Log the raw response for debugging
                logger.debug(f"Raw {provider} response: {content}")
                
                return content
            else:
                logger.error(f"LLM API error: {response.status_code}, {response.text}")
                return f"Error calling {provider} API: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Exception calling primary LLM API: {e}")
            return f"Error: {str(e)}"
    
    def _process_with_secondary_model(self, primary_response, symbol):
        """
        Process the primary model response with the secondary model to get structured output.
        
        Args:
            primary_response: Raw response from the primary model
            symbol: Trading pair symbol for context
            
        Returns:
            Dictionary with structured decision data
        """
        if not self.secondary_api_key:
            logger.error("Secondary model API key not available for processing")
            return {
                "decision": "HOLD",
                "confidence": 0.5,
                "reasoning": "Secondary model processing unavailable"
            }
        
        try:
            # Remove <think>...</think> tags if they exist in the primary response
            clean_response = self._clean_primary_response(primary_response)
            
            # Structure for secondary model to extract decision data
            prompt = f"""
            Analyze this trading analysis for {symbol} and extract the key information:

            {clean_response}

            Based on the analysis, provide a structured output with:
            1. The final trading decision (BUY, SELL, or HOLD)
            2. A confidence score between 0.5 and 1.0
            3. A summary of the reasoning behind this decision
            """
            
            # Get provider-specific settings
            provider = get_secondary_model_config().get("provider", "").lower()
            
            # Different handling for different providers
            if provider == "openai":
                # Define the response structure we want the model to follow
                response_format = {
                    "type": "json_object"
                }
                
                # Define the system message with schema information
                system_message = """
                You are a financial analysis assistant that extracts key trading insights from detailed analyses.
                
                Extract the following from the user's input:
                - trading decision (BUY, SELL, or HOLD)
                - confidence level (a number between 0.5 and 1.0)
                - reasoning behind the decision (brief summary)
                
                Return your response as a JSON object with the following structure:
                {
                    "decision": "BUY" or "SELL" or "HOLD",
                    "confidence": <number between 0.5 and 1.0>,
                    "reasoning": "<brief summary of the reasoning>"
                }
                """
                
                payload = {
                    "model": self.secondary_model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": response_format,
                    "temperature": self.secondary_temperature,
                    "max_tokens": 500
                }
            else:
                # Generic payload for other providers
                payload = {
                    "model": self.secondary_model,
                    "messages": [
                        {"role": "system", "content": "Extract a trading decision from this analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.secondary_temperature
                }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.secondary_api_key}"
            }
            
            response = requests.post(
                self.secondary_api_endpoint,
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Handle different response formats from different providers
                if provider == "openai":
                    content = response_data["choices"][0]["message"]["content"]
                else:
                    # Generic fallback
                    content = response_data.get("output", response_data.get("content", str(response_data)))
                
                # Parse the response
                if provider == "openai":
                    try:
                        # Try to parse as JSON if it's OpenAI with structured output
                        structured_result = json.loads(content)
                    except json.JSONDecodeError:
                        # Fall back to extraction if it's not valid JSON
                        structured_result = self._extract_decision_from_text(content)
                else:
                    # For other providers, attempt extraction
                    structured_result = self._extract_decision_from_text(content)
                    
                logger.info(f"Secondary model structured output: {structured_result}")
                
                return structured_result
            else:
                logger.error(f"Secondary model API error: {response.status_code}, {response.text}")
                return {
                    "decision": "HOLD",
                    "confidence": 0.5,
                    "reasoning": f"Error processing with secondary model: API error {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Exception processing with secondary model: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Error processing with secondary model: {str(e)}"
            }
    
    def _clean_primary_response(self, response):
        """
        Clean the response from the primary model.
        Removes special tags like <think>...</think> if present.
        
        Args:
            response: The raw response text
            
        Returns:
            Cleaned response text
        """
        # Remove <think>...</think> tags if present
        cleaned = re.sub(r'<think>\s*(.*?)\s*</think>', r'\1', response, flags=re.DOTALL)
        
        # If the response has become too short after cleaning, use the original
        if len(cleaned.strip()) < 50 and len(response.strip()) > len(cleaned.strip()):
            return response
            
        return cleaned
    
    def _call_primary_model_for_decision(self, prompt):
        """
        Call the primary LLM model API to get a trading decision.
        
        Args:
            prompt: String with context about signals and market data
            
        Returns:
            Dictionary with decision, confidence and reasoning
        """
        try:
            # DeepSeek R1 specific request structure
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": """
                        You are a trading assistant that helps make decisions based on technical indicators and market data.
                        Analyze the provided market data, indicators, and signals to make a trading decision.
                        Consider technical indicators carefully and be conservative with your recommendations.
                        First explain your reasoning and analysis process in detail.
                        Then conclude with ONLY ONE of these terms: "BUY", "SELL", or "HOLD".
                    """},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                # Extract the decision from the response
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                
                # Extract decision, reasoning and estimate confidence
                reasoning, decision = self._extract_decision_and_reasoning(content)
                confidence = self._estimate_confidence(content)
                
                return {
                    "decision": decision,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
            else:
                logger.error(f"DeepSeek API error: {response.status_code}, {response.text}")
                return {
                    "decision": "HOLD",
                    "confidence": 0.5,
                    "reasoning": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Exception calling primary LLM API for decision: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0.5,
                "reasoning": f"API exception: {str(e)}"
            }
    
    def _extract_decision_and_reasoning(self, content):
        """
        Extract the final decision and reasoning from the LLM response.
        
        Args:
            content: String response from the LLM
            
        Returns:
            Tuple of (reasoning, decision)
        """
        # Get the last few lines for the decision
        lines = content.strip().split('\n')
        
        # Extract decision from the last non-empty line
        decision = "HOLD"  # Default
        for line in reversed(lines):
            line = line.strip().upper()
            if line:
                if "BUY" in line:
                    decision = "BUY"
                    break
                elif "SELL" in line:
                    decision = "SELL"
                    break
                elif "HOLD" in line:
                    decision = "HOLD"
                    break
        
        # Use everything except the last line as reasoning
        reasoning_lines = lines[:-1] if len(lines) > 1 else []
        reasoning = "\n".join(reasoning_lines).strip()
        
        # If no reasoning was extracted or it's too short, use the content itself excluding the decision
        if len(reasoning) < 10:
            # Extract content up to the decision keyword
            content_lower = content.lower()
            for keyword in ["buy", "sell", "hold"]:
                if keyword in content_lower:
                    index = content_lower.find(keyword)
                    if index > 5:  # Ensure we have some content before the decision
                        reasoning = content[:index].strip()
                        break
        
        # If we still don't have meaningful reasoning, use a generic message
        if len(reasoning) < 10:
            reasoning = "Decision based on analysis of market indicators and signals."
            
        return reasoning, decision
    
    def _estimate_confidence(self, content):
        """
        Estimate the confidence level based on the LLM response.
        
        Args:
            content: String response from the LLM
            
        Returns:
            Float confidence level between 0.5 and 1.0
        """
        # Simple heuristic: higher confidence if the content contains certainty words
        content_lower = content.lower()
        
        # High confidence words
        high_confidence_words = ["certain", "confident", "definitely", "clearly", "strongly", "sure", 
                                 "guarantee", "doubtless", "undoubtedly", "absolutely"]
        
        # Medium confidence words
        medium_confidence_words = ["likely", "probably", "suggests", "indicates", "appears", 
                                  "should", "would", "recommend", "believe", "think"]
        
        # Low confidence words
        low_confidence_words = ["possibly", "might", "could", "uncertain", "unclear", "maybe", 
                               "perhaps", "potential", "risky", "doubt", "chance", "hesitant"]
        
        # Count occurrences of confidence words
        high_count = sum(1 for word in high_confidence_words if word in content_lower)
        medium_count = sum(1 for word in medium_confidence_words if word in content_lower)
        low_count = sum(1 for word in low_confidence_words if word in content_lower)
        
        # Calculate confidence score
        if high_count > 0 and low_count == 0:
            return 0.9  # High confidence
        elif medium_count > 0 and low_count <= 1:
            return 0.8  # Medium confidence
        elif low_count > medium_count:
            return 0.6  # Low confidence
        else:
            return 0.7  # Default moderate confidence
    
    def _make_rule_based_decision(self, market_data, strategy_signals=None):
        """
        Make a rule-based trading decision without using an LLM.
        
        Args:
            market_data: Dictionary with market data
            strategy_signals: Optional signals from traditional strategies
            
        Returns:
            Dictionary with decision, confidence, and reasoning
        """
        # Create a simple prompt from the available data
        prompt = "Market data: "
        if "price" in market_data:
            prompt += f"Price: {market_data['price']} "
        
        if "indicators" in market_data:
            prompt += "Indicators: "
            for indicator, value in market_data["indicators"].items():
                prompt += f"{indicator}: {value} "
        
        if strategy_signals:
            prompt += "Signals: "
            for strategy, signal in strategy_signals.items():
                prompt += f"simple signal: {signal} "
        
        # Get decision using rule-based method
        decision = make_rule_based_decision(prompt)
        
        return {
            "decision": decision.lower(),
            "confidence": 0.6,  # Standard confidence for rule-based decisions
            "reasoning": "Decision based on rule-based analysis of market indicators"
        }
    
    def _prepare_prompt(self, market_data, symbol, timeframe, context, strategy_signals):
        """
        Prepare a prompt string from the input data for the LLM.
        
        Returns:
            String prompt for the LLM
        """
        prompt = f"Trading analysis for {symbol} on {timeframe} timeframe.\n"
        prompt += f"Context: {context}\n"
        
        if "price" in market_data:
            prompt += f"Current price: {market_data['price']}\n"
        
        if "indicators" in market_data:
            prompt += "Technical indicators:\n"
            for indicator, value in market_data["indicators"].items():
                prompt += f"- {indicator}: {value}\n"
        
        if strategy_signals:
            prompt += "Strategy signals:\n"
            for strategy, signal in strategy_signals.items():
                prompt += f"- {strategy}: {signal}\n"
        
        prompt += "\nBased on this information, should I buy, sell, or hold? Provide detailed reasoning for your decision."
        
        return prompt

def get_decision_from_llm(prompt):
    """
    Get a trading decision from the LLM.
    
    This is a wrapper function for backward compatibility with older code.
    New code should directly use the LLMManager class.
    
    Args:
        prompt: A string with market data and context for the LLM
        
    Returns:
        String: "BUY", "SELL", or "HOLD"
    """
    try:
        # Check if LLM is enabled
        if not is_llm_enabled():
            logger.info("LLM decision making is disabled in configuration")
            return "HOLD"
            
        # Create the LLM manager instance
        llm_manager = LLMManager()
        
        # Call the LLM using the manager with placeholder data
        # Since the old API only provides the prompt, we need to create dummy context
        result = llm_manager.make_llm_decision(
            market_data={},  # Empty market data
            symbol="",       # No symbol specified
            timeframe="",    # No timeframe
            context=prompt,  # Use the prompt as context
            strategy_signals=None  # No strategy signals
        )
        
        # Extract the decision from the result
        decision = result.get("decision", "HOLD").upper()
        
        return decision
    except Exception as e:
        logger.error(f"Error getting decision from LLM: {e}")
        return "HOLD"

def call_real_llm_api(prompt):
    """
    Call the DeepSeek R1 API service.
    
    Returns:
        String: "BUY", "SELL", or "HOLD" decision
    """
    try:
        # DeepSeek R1 specific request structure
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
