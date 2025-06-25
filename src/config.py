# src/config.py
from dotenv import load_dotenv
import os

# Carrega as variáveis do arquivo .env
load_dotenv()

# Configurações da Binance API
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY")
TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET")

# Definir se deve usar a Testnet ou a rede real
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() == "true"

# Parâmetros de Trading
SYMBOL = "BTCUSDT"
INITIAL_CAPITAL = 1000
RISK_TOLERANCE = 0.02
PROFIT_TARGET = 0.05

print(f"Configuração carregada - USE_TESTNET: {USE_TESTNET}")