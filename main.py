# main.py

import sys
from src.config import MODE, SYMBOL
from src.logger import logger
from src.data_manager import DataManager
from src.optimizer import WalkForwardOptimizer

def main():
    if MODE == 'optimize':
        logger.info("--- MODO OTIMIZAÇÃO WALK-FORWARD ---")
        dm = DataManager()
        # O método agora não precisa mais do days_to_load
        df = dm.update_and_load_data(SYMBOL, '1m') 
        df = df[df.index >= '2018-01-01']
        if df.empty:
            logger.error("Sem dados para otimizar. Abortando.")
            sys.exit(1)
        optimizer = WalkForwardOptimizer(df)
        optimizer.run()

    elif MODE == 'trade':
        from src.trading_bot import TradingBot
        logger.info("--- MODO TRADING ---")
        bot = TradingBot()
        bot.run()

    else:
        # A validação agora está no config.py, mas é bom manter um fallback.
        logger.error("Modo inválido. Use 'optimize' ou 'trade'.")
        sys.exit(1)

if __name__ == "__main__":
    main()