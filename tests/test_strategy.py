import pytest
from bot.strategy import simple_signal

# We'll use monkeypatch to override the get_recent_closes function from bot.binance_api
def fake_get_recent_closes_buy(symbol, interval, limit=2):
    # Simulate that the latest close is higher than the previous close
    return [100, 101]

def fake_get_recent_closes_sell(symbol, interval, limit=2):
    # Simulate that the latest close is lower than the previous close
    return [101, 100]

def test_simple_signal_buy(monkeypatch):
    from bot.strategy import simple_signal
    monkeypatch.setattr('bot.strategy.get_recent_closes', fake_get_recent_closes_buy)
    signal = simple_signal('BTCUSDT', '1m')
    assert signal == 'BUY'

def test_simple_signal_sell(monkeypatch):
    from bot.strategy import simple_signal
    monkeypatch.setattr('bot.strategy.get_recent_closes', fake_get_recent_closes_sell)
    signal = simple_signal('BTCUSDT', '1m')
    assert signal == 'SELL'

