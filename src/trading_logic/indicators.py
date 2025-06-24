# src/trading_logic/indicators.py

import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    """Calcula o Índice de Força Relativa (RSI)"""
    df = pd.DataFrame(prices, columns=['close'])
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, window=20, num_std_dev=2):
    """Calcula as Bandas de Bollinger"""
    df = pd.DataFrame(prices, columns=['close'])
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    upper_band = rolling_mean + (num_std_dev * rolling_std)
    lower_band = rolling_mean - (num_std_dev * rolling_std)
    return upper_band, lower_band
