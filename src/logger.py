# src/logger.py
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import os

# Cria o diretório de logs se ele não existir
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configuração do logger
logger = logging.getLogger("gcsBot")
logger.setLevel(logging.INFO)

# Evita adicionar handlers duplicados se o módulo for importado várias vezes
if not logger.handlers:
    # Formato da mensagem de log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler para o console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Handler para arquivo com rotação diária e encoding UTF-8
    file_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, 'trading.log'),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # Adiciona os handlers ao logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)