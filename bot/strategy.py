from bot.binance_api import get_recent_closes

def simple_signal(symbol, interval='1m'):
    closes = get_recent_closes(symbol, interval)
    if len(closes) < 2:
        return None
    # Simple strategy: if the latest close is higher than the previous, signal a buy
    return 'BUY' if closes[-1] > closes[-2] else 'SELL'
