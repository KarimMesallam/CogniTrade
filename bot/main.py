import time
from bot.config import SYMBOL
from bot.strategy import simple_signal
from bot.binance_api import place_market_buy, place_market_sell
from bot.llm_manager import get_decision_from_llm

def trading_loop():
    while True:
        signal = simple_signal(SYMBOL)
        print("Signal from strategy:", signal)
        
        # Use LLM to get an additional layer of decision making (optional)
        prompt = f"Signal is {signal}. Given current market data, should we BUY, SELL, or HOLD?"
        llm_decision = get_decision_from_llm(prompt)
        print("LLM decision:", llm_decision)
        
        # For this basic demo, we only execute on a BUY signal (paper trading)
        if signal == "BUY" and llm_decision == "BUY":
            print("Placing market buy order...")
            order = place_market_buy(SYMBOL, 0.001)
            print("Order response:", order)
        # In a real bot, add logic for selling, position management, risk controls, etc.
        
        # Wait a minute between iterations
        time.sleep(60)

if __name__ == '__main__':
    print("Starting trading bot...")
    trading_loop()
