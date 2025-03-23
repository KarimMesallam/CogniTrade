from binance.client import Client
from bot.config import API_KEY, API_SECRET, TESTNET

client = Client(API_KEY, API_SECRET, testnet=TESTNET)

def get_recent_closes(symbol, interval, limit=2):
    candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    # The 5th element in each candle is the close price
    return [float(candle[4]) for candle in candles]

def place_market_buy(symbol, quantity):
    try:
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        return order
    except Exception as e:
        print("Error placing buy order:", e)
        return None

def place_market_sell(symbol, quantity):
    try:
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
        return order
    except Exception as e:
        print("Error placing sell order:", e)
        return None
