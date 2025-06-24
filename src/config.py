# src/config.py

from dotenv import load_dotenv
import os

# Carrega as variáveis do arquivo .env
load_dotenv()

# Configurações da Binance API
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Configuração do ambiente de Testnet ou Real
REAL_WALLET = os.getenv("REAL_WALLET", "False") == "True"
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "False") == "True"
