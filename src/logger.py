# src/logger.py
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import os

# Importa o MODO para que o logger saiba em qual ambiente está rodando
# Usamos um try-except para evitar erros de importação circular se este arquivo for importado antes do config
try:
    from src.config import MODE
except ImportError:
    MODE = os.getenv("MODE", "optimize").lower()

# Cria o diretório de logs se ele não existir
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configuração base do logger
logger = logging.getLogger("gcsBot")
logger.setLevel(logging.DEBUG)  # Captura TODOS os níveis de log

# Evita adicionar handlers duplicados
if not logger.handlers:
    # --- Handler para Arquivo (Sempre Ativo) ---
    # Este é o nosso log principal e seguro. Ele grava tudo em um arquivo.
    log_file_path = os.path.join(log_dir, 'gcs_bot.log')
    file_handler = TimedRotatingFileHandler(
        log_file_path,
        when="midnight",
        interval=1,
        backupCount=14, # Aumentei para 14 dias de histórico
        encoding='utf-8'
    )
    # Define um formato detalhado para o arquivo de log
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG) # Grava TUDO no arquivo
    logger.addHandler(file_handler)

    # --- Handler para Console (Inteligente e Leve) ---
    console_handler = logging.StreamHandler(sys.stdout)
    
    # ### A MÁGICA ACONTECE AQUI ###
    # O formato e o nível do log no console dependem do MODO
    if MODE == 'optimize':
        # No modo de otimização, queremos um console LIMPO.
        # Mostraremos apenas logs de nível INFO e acima, e com formato simples.
        console_formatter = logging.Formatter('%(message)s') # Formato super limpo
        console_handler.setLevel(logging.INFO) # Nível mais alto para não poluir
    else: # Nos modos 'test' e 'trade', queremos mais detalhes no console
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setLevel(logging.INFO) # INFO é um bom padrão para monitoramento

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

logger.info(f"Logger configurado para o modo: '{MODE.upper()}'")