# src/trading_logic/strategies.py

from .indicators import calculate_rsi, calculate_bollinger_bands

def rsi_strategy(prices, rsi_period=14, overbought=70, oversold=30):
    """Estratégia RSI"""
    rsi = calculate_rsi(prices, rsi_period)
    if rsi.iloc[-1] < oversold:
        return "buy"
    elif rsi.iloc[-1] > overbought:
        return "sell"
    return "hold"

def bollinger_bands_strategy(prices, window=20, num_std_dev=2):
    """Estratégia Bandas de Bollinger"""
    upper_band, lower_band = calculate_bollinger_bands(prices, window, num_std_dev)
    if prices[-1] < lower_band.iloc[-1]:
        return "buy"
    elif prices[-1] > upper_band.iloc[-1]:
        return "sell"
    return "hold"
