def fake_get_klines(symbol, interval, limit):
    # Create a fake kline response with the close price at index 4
    # Each candle is a list where the 5th element is the close price.
    return [
        ["", "", "", "", "50000", "", "", "", "", ""],
        ["", "", "", "", "50100", "", "", "", "", ""]
    ]

def test_get_recent_closes(monkeypatch):
    from bot.binance_api import get_recent_closes, client
    # Monkey-patch the client's get_klines method
    monkeypatch.setattr(client, 'get_klines', fake_get_klines)
    closes = get_recent_closes('BTCUSDT', '1m')
    # Ensure that the close prices are parsed correctly as floats.
    assert closes == [50000.0, 50100.0]
