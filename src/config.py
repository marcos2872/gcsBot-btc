# src/config.py

from dotenv import load_dotenv
import os
from src.logger import logger

# Carrega as variáveis do arquivo .env para o ambiente
load_dotenv()

# --- MODO DE OPERAÇÃO ---
MODE = os.getenv("MODE", "optimize").lower()
if MODE not in ['train', 'trade', 'optimize']:
    raise ValueError(f"MODE inválido no .env: '{MODE}'. Use 'train', 'trade' ou 'optimize'.")

# --- CONFIGURAÇÕES DA BINANCE ---
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() == "true"
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")

API_KEY = os.getenv("BINANCE_TESTNET_API_KEY") if USE_TESTNET else os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET") if USE_TESTNET else os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    logger.warning("Chaves da API não configuradas no .env. O bot só funcionará para otimização offline.")
    if MODE == 'trade':
        raise ValueError(f"Para MODE='trade', as chaves da API devem ser configuradas.")

# --- CONFIGURAÇÕES DO MODELO E DADOS ---
DATA_DIR = "data"
LOGS_DIR = "logs"

# Caminhos para os datasets
KAGGLE_BOOTSTRAP_FILE = os.path.join(DATA_DIR, "kaggle_btc_1m_bootstrap.csv")
HISTORICAL_DATA_FILE = os.path.join(DATA_DIR, "full_historical_BTCUSDT_1m.csv")

MODEL_FILE = os.path.join(DATA_DIR, "trading_model.pkl")
SCALER_FILE = os.path.join(DATA_DIR, "scaler.pkl")
TRADES_LOG_FILE = os.path.join(DATA_DIR, "trades_log.csv")
OPTIMIZATION_RESULTS_FILE = os.path.join(DATA_DIR, "optimization_results.json")
BOT_STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")

WFO_STATE_FILE = os.path.join(DATA_DIR, "wfo_optimization_state.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- PARÂMETROS DE TRADING INICIAIS (Serão otimizados) ---
TRADE_AMOUNT_USDT = 100.0

# --- PARÂMETROS PARA A OTIMIZAÇÃO WALK-FORWARD BASEADA EM MINUTOS ---
# Quantos minutos (velas de 1m) em cada janela?
WFO_TRAIN_MINUTES = 43200  # Treinar com ~30 dias de dados (30*24*60)
WFO_TEST_MINUTES = 10080   # Validar nos ~7 dias seguintes (7*24*60)
WFO_STEP_MINUTES = 10080   # Deslizar a janela para frente no mesmo tamanho do teste.

logger.info("---------- CONFIGURAÇÃO CARREGADA ----------")
logger.info(f"MODO DE EXECUÇÃO: {MODE.upper()}")
logger.info(f"SÍMBOLO: {SYMBOL}")
logger.info(f"USANDO TESTNET: {USE_TESTNET}")
logger.info(f"WFO Janela de Treino: {WFO_TRAIN_MINUTES} minutos (~{WFO_TRAIN_MINUTES/1440:.1f} dias)")
logger.info(f"WFO Janela de Teste:  {WFO_TEST_MINUTES} minutos (~{WFO_TEST_MINUTES/1440:.1f} dias)")
logger.info(f"WFO Passo da Janela:  {WFO_STEP_MINUTES} minutos (~{WFO_STEP_MINUTES/1440:.1f} dias)")
logger.info("------------------------------------------")