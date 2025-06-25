# src/logger.py
import logging
import os
from datetime import datetime

def setup_logger():
    """Configura o sistema de logging"""
    # Criar diretório de logs
    os.makedirs('logs', exist_ok=True)
    
    # Configurar logger
    logger = logging.getLogger('trading_bot')
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Handler para arquivo
    file_handler = logging.FileHandler(
        f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Limpar handlers existentes
    logger.handlers.clear()
    
    # Adicionar handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()