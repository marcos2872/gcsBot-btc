# src/logger.py

import logging

def setup_logger():
    logging.basicConfig(
        filename="bot_trades.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    logging.info("Logger configurado com sucesso.")
