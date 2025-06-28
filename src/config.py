# src/config.py

from dotenv import load_dotenv
import os
from src.logger import logger

# Carrega as variáveis do arquivo .env para o ambiente
load_dotenv()

def get_config_var(var_name, default_value=None):
    """
    Função auxiliar para ler uma variável de ambiente e limpá-la.
    Remove espaços em branco e aspas simples/duplas.
    """
    value = os.getenv(var_name, default_value)
    if isinstance(value, str):
        # Remove espaços no início/fim e remove aspas
        return value.strip().strip("'\"")
    return value

# --- MODO DE OPERAÇÃO ---
MODE = get_config_var("MODE", "optimize").lower()

# --- CONFIGURAÇÕES DA BINANCE ---
SYMBOL = get_config_var("SYMBOL", "BTCUSDT").upper() # Garante que o símbolo seja sempre maiúsculo

# Chaves de API Real (Mainnet)
BINANCE_API_KEY = get_config_var("BINANCE_API_KEY")
BINANCE_API_SECRET = get_config_var("BINANCE_API_SECRET")

# Chaves de API de Teste (Testnet)
BINANCE_TESTNET_API_KEY = get_config_var("BINANCE_TESTNET_API_KEY")
BINANCE_TESTNET_API_SECRET = get_config_var("BINANCE_TESTNET_API_SECRET")

# Define se o modo de teste está ativo (apenas se MODE='test')
USE_TESTNET = (MODE == 'test')

# Seleciona as chaves corretas com base no modo
API_KEY = BINANCE_TESTNET_API_KEY if USE_TESTNET else BINANCE_API_KEY
API_SECRET = BINANCE_TESTNET_API_SECRET if USE_TESTNET else BINANCE_API_SECRET

# Validação das chaves para os modos que precisam delas
if MODE in ['test', 'trade']:
    if not API_KEY or not API_SECRET:
        raise ValueError(f"Para MODE='{MODE}', as chaves da API correspondentes devem ser configuradas no .env")

# --- CONFIGURAÇÕES DO MODELO E DADOS ---
DATA_DIR = "data"
LOGS_DIR = "logs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

KAGGLE_BOOTSTRAP_FILE = os.path.join(DATA_DIR, "kaggle_btc_1m_bootstrap.csv")
HISTORICAL_DATA_FILE = os.path.join(DATA_DIR, "full_historical_BTCUSDT_1m.csv")
MODEL_FILE = os.path.join(DATA_DIR, "trading_model.pkl")
SCALER_FILE = os.path.join(DATA_DIR, "scaler.pkl")
TRADES_LOG_FILE = os.path.join(DATA_DIR, "trades_log.csv")
OPTIMIZATION_RESULTS_FILE = os.path.join(DATA_DIR, "optimization_results.json")
BOT_STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")
WFO_STATE_FILE = os.path.join(DATA_DIR, "wfo_optimization_state.json")
STRATEGY_PARAMS_FILE = os.path.join(DATA_DIR, "strategy_params.json")

# --- PARÂMETROS DE TRADING INICIAIS ---
TRADE_AMOUNT_USDT = float(get_config_var("TRADE_AMOUNT_USDT", 100.0))

# --- PARÂMETROS PARA A OTIMIZAÇÃO WALK-FORWARD ---
WFO_TRAIN_MINUTES = int(get_config_var("WFO_TRAIN_MINUTES", 43200)) # ~30 dias
WFO_TEST_MINUTES = int(get_config_var("WFO_TEST_MINUTES", 10080))  # ~7 dias
WFO_STEP_MINUTES = int(get_config_var("WFO_STEP_MINUTES", 10080))  # ~7 dias