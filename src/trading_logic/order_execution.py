# src/trading_logic/order_execution.py

import logging
from src.data_fetcher import get_binance_client

def place_order(symbol="BTC/USDT", amount=0.001, order_type="buy"):
    """Coloca uma ordem de compra ou venda"""
    client = get_binance_client()
    if order_type == "buy":
        logging.info(f"Colocando ordem de compra de {amount} BTC a {symbol}")
        order = client.create_market_buy_order(symbol, amount)
    elif order_type == "sell":
        logging.info(f"Colocando ordem de venda de {amount} BTC a {symbol}")
        order = client.create_market_sell_order(symbol, amount)
    return order
