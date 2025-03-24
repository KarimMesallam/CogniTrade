import logging
import requests
import json
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("trading_bot")

# DeepSeek R1 API configuration (would come from .env in production)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-reasoner")  # deepseek-reasoner is the R1 model

# OpenAI GPT-4o configuration for structured output processing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # Default to GPT-4o

class LLMManager:
    """
    A class to manage LLM interactions for trading decisions using DeepSeek R1
    with GPT-4o structured output processing.
    """
    def __init__(self):
        self.api_key = LLM_API_KEY
        self.api_endpoint = LLM_API_ENDPOINT
        self.model = LLM_MODEL
        self.openai_api_key = OPENAI_API_KEY
    
    def make_llm_decision(self, market_data, symbol, timeframe, context, strategy_signals=None):
        """
        Make a trading decision using DeepSeek R1 based on market data and context,
        then process the response with GPT-4o for structured output.
        
        Args:
            market_data: Dictionary with market data
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            context: Additional context for the LLM
            strategy_signals: Optional signals from traditional strategies
            
        Returns:
            Dictionary with decision, confidence, and reasoning
        """
        # Prepare context string from input data
        prompt = self._prepare_prompt(market_data, symbol, timeframe, context, strategy_signals)
        
        try:
            if self.api_key and self.api_endpoint and self.model:
                # If OpenAI API key is available, use DeepSeek + GPT-4o pipeline
                if self.openai_api_key:
                    # Call DeepSeek R1 API first
                    deepseek_response = self._call_deepseek_api_raw(prompt)
                    
                    # Process the response with GPT-4o for structured output
                    decision_result = self._process_with_gpt4o(deepseek_response, symbol)
                    
                    decision = decision_result.get("decision", "HOLD")
                    confidence = decision_result.get("confidence", 0.5)
                    reasoning = decision_result.get("reasoning", "Decision processed with GPT-4o")
                else:
                    # Fall back to DeepSeek R1 only if OpenAI API key is not available
                    logger.warning("OpenAI API key not available, falling back to DeepSeek R1 only")
                    decision_result = self._call_deepseek_api(prompt)
                    decision = decision_result["decision"]
                    confidence = decision_result["confidence"]
                    reasoning = decision_result["reasoning"]
            else:
                # Fall back to rule-based if no DeepSeek API credentials
                decision = make_rule_based_decision(prompt)
                confidence = 0.6  # Lower confidence for rule-based
                reasoning = "Decision based on rule-based analysis (LLM unavailable)"
                
            return {
                "decision": decision.lower(),
                "confidence": confidence,
                "reasoning": reasoning
            }
        except Exception as e:
            logger.error(f"Error making LLM decision: {e}")
            return {
                "decision": "hold",
                "confidence": 0.5,
                "reasoning": f"Error in LLM decision process: {str(e)}"
            }
    
    def _call_deepseek_api_raw(self, prompt):
        """
        Call the DeepSeek R1 API and return the raw response content.
        This is used as input for GPT-4o processing.
        
        Args:
            prompt: String with context about signals and market data
            
        Returns:
            String: Raw response content from DeepSeek R1
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
                "max_tokens": 1000,
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
                # Extract just the raw content from the response
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                
                # Log the raw response for debugging
                logger.debug(f"Raw DeepSeek response: {content}")
                
                return content
            else:
                logger.error(f"DeepSeek API error: {response.status_code}, {response.text}")
                return "Error calling DeepSeek API"
                
        except Exception as e:
            logger.error(f"Exception calling DeepSeek API raw: {e}")
            return f"Error: {str(e)}"
    
    def _process_with_gpt4o(self, deepseek_response, symbol):
        """
        Process the DeepSeek R1 response with GPT-4o to get structured output.
        
        Args:
            deepseek_response: Raw response from DeepSeek R1
            symbol: Trading pair symbol for context
            
        Returns:
            Dictionary with structured decision data
        """
        if not self.openai_api_key:
            logger.error("OpenAI API key not available for GPT-4o processing")
            return {
                "decision": "HOLD",
                "confidence": 0.5,
                "reasoning": "GPT-4o processing unavailable"
            }
        
        try:
            # Remove <think>...</think> tags if they exist in the DeepSeek response
            # DeepSeek R1 often wraps its reasoning in these tags
            clean_response = self._clean_deepseek_response(deepseek_response)
            
            # Structure for GPT-4o to extract decision data
            prompt = f"""
            Analyze this trading analysis for {symbol} and extract the key information:

            {clean_response}

            Based on the analysis, provide a structured output with:
            1. The final trading decision (BUY, SELL, or HOLD)
            2. A confidence score between 0.5 and 1.0
            3. A summary of the reasoning behind this decision
            """
            
            # Define the response structure we want GPT-4o to follow
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
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "response_format": response_format,
                "temperature": 0.1,  # Low temperature for more deterministic results
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                
                # Parse the JSON response
                structured_result = json.loads(content)
                logger.info(f"GPT-4o structured output: {structured_result}")
                
                return structured_result
            else:
                logger.error(f"OpenAI API error: {response.status_code}, {response.text}")
                return {
                    "decision": "HOLD",
                    "confidence": 0.5,
                    "reasoning": f"Error processing with GPT-4o: API error {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Exception processing with GPT-4o: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Error processing with GPT-4o: {str(e)}"
            }
    
    def _clean_deepseek_response(self, response):
        """
        Clean the DeepSeek R1 response by removing think tags and other formatting.
        
        Args:
            response: Raw response from DeepSeek R1
            
        Returns:
            String: Cleaned response content
        """
        # Remove <think>...</think> tags if present
        # DeepSeek R1 often wraps reasoning in these tags
        cleaned = re.sub(r'<think>\s*(.*?)\s*</think>', r'\1', response, flags=re.DOTALL)
        
        # If the response has become too short after cleaning, use the original
        if len(cleaned.strip()) < 50 and len(response.strip()) > len(cleaned.strip()):
            return response
            
        return cleaned
    
    def _call_deepseek_api(self, prompt):
        """
        Call the DeepSeek R1 API to get a trading decision.
        
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
            logger.error(f"Exception calling DeepSeek API: {e}")
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
    
    def make_rule_based_decision(self, market_data, strategy_signals=None):
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
    This function interfaces with DeepSeek R1 for trading decisions.
    
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
