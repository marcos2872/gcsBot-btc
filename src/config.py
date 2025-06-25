from dotenv import load_dotenv
import os

# Carrega as variáveis do arquivo .env
load_dotenv()

# --- Configurações da Binance API ---
# (Esta parte permanece a mesma)
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY")
TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET")

# --- Configurações Gerais de Execução ---
# Define se o bot deve se conectar à Testnet da Binance (para obter preços)
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() == "true"
# Define o par de moedas a ser negociado
SYMBOL = "BTCUSDT"
# Define se o bot deve rodar em modo de simulação (carteira virtual)
SIMULATION_MODE = True

# --- Parâmetros para o Modo de Simulação ---
# (Ignorados se SIMULATION_MODE for False)
SIMULATION_INITIAL_USDT = 100.0  # Capital inicial em USDT para a simulação
SIMULATION_INITIAL_BTC = 0.0     # Capital inicial em BTC para a simulação
SIMULATION_TRADE_RATIO = 0.1     # Fração da carteira a ser usada em cada transação (10%)

print(f"Configuração carregada - USE_TESTNET: {USE_TESTNET}, SIMULATION_MODE: {SIMULATION_MODE}")