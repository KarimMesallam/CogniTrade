import os
from dotenv import load_dotenv

load_dotenv() 

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

TESTNET = os.getenv('TESTNET', 'False').lower() in ('true', '1', 't')

SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')