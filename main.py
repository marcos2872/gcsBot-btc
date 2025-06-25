# main.py
from src.trader import TradingBot
from src.logger import logger

if __name__ == "__main__":
    try:
        # Criar e executar o bot
        bot = TradingBot()
        
        # Executar 1 ciclo (para teste)
        bot.run(cycles=1)
        
        # Para execução contínua, descomente a linha abaixo:
        # bot.run(cycles=24, interval_minutes=60)  # 24 horas, 1 hora por ciclo
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}")