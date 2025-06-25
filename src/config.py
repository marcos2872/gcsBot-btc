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
USE_TESTNET = os.getenv("USE_TESTNET") == 'True'  # Verifica se USE_TESTNET é True no .env

# Parâmetros de Trading
SYMBOL = "BTCUSDT"
INITIAL_CAPITAL = 1000  # Capital inicial em USDT
STARTING_BTC = 0  # Quantidade inicial de BTC (caso não tenha BTC, começa comprando)
CAPITAL_ON_BTC = True  # Se o bot começa comprando BTC ou fica em USDT

# Parâmetros de Machine Learning (os parâmetros serão ajustados dinamicamente)
RISK_TOLERANCE = 0.02  # Percentual de risco aceito para a operação
PROFIT_TARGET = 0.05   # Percentual de ganho desejado antes de vender
LEARNING_RATE = 0.01   # Taxa de aprendizado para ajuste contínuo
