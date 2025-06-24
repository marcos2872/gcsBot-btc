# src/data_fetcher.py

import ccxt
from src.config import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET

def get_binance_client():
    if BINANCE_TESTNET:
        return ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'urls': {
                'api': 'https://testnet.binance.vision/api',  # Testnet API URL
            }
        })
    else:
        return ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET
        })

def get_last_price(symbol="BTC/USDT"):
    """Busca o último preço do símbolo (ex: BTC/USDT)"""
    client = get_binance_client()
    ticker = client.fetch_ticker(symbol)
    return ticker['last']

def get_balance():
    """Busca o saldo disponível de USDT e BTC na conta"""
    client = get_binance_client()
    balance = client.fetch_balance()
    return balance['total']['USDT'], balance['total']['BTC']
