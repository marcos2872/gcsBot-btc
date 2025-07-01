# main.py

import sys
import os
import pandas as pd
from src.config import MODE, SYMBOL, BACKTEST_START_DATE, BACKTEST_END_DATE
from src.logger import logger

def main():
    """
    Ponto de entrada principal do bot.
    Lê o modo de operação do arquivo de configuração e inicia o processo correspondente,
    seja a otimização da estratégia ou a execução do trading ao vivo.
    """
    
    # Garante que os diretórios de dados e logs existam antes de qualquer operação.
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if MODE == 'optimize':
        logger.info("--- MODO: OTIMIZAÇÃO WALK-FORWARD ---")
        logger.info("Iniciando o processo de otimização para encontrar os melhores parâmetros...")
        
        # Importações específicas para este modo, mantendo o código limpo e eficiente.
        from src.data_manager import DataManager
        from src.optimizer import WalkForwardOptimizer
        
        dm = DataManager()
        # Carrega todo o histórico de dados disponível, combinando fontes e atualizando com o mais recente.
        full_historical_data = dm.update_and_load_data(SYMBOL, '1m')
        
        # Filtra para usar dados a partir de 2018 para maior relevância, se disponível.
        if not full_historical_data.empty and pd.to_datetime("2018-01-01", utc=True) < full_historical_data.index.max():
             full_historical_data = full_historical_data[full_historical_data.index >= '2018-01-01']

        if full_historical_data.empty:
            logger.error("Não há dados disponíveis para otimizar. Verifique a conexão ou os arquivos de dados. Abortando.")
            sys.exit(1)
            
        optimizer = WalkForwardOptimizer(full_historical_data)
        optimizer.run()

    elif MODE == 'backtest':
        logger.info(f"--- MODO: BACKTEST RÁPIDO ---")
        from src.quick_tester import QuickTester
        
        tester = QuickTester()
        tester.run(start_date_str=BACKTEST_START_DATE, end_date_str=BACKTEST_END_DATE)

    elif MODE == 'test' or MODE == 'trade':
        logger.info(f"--- MODO: {MODE.upper()} ---")
        
        if MODE == 'test':
            logger.info("************************************************************")
            logger.info(">>> OPERANDO NA BINANCE TESTNET (CARTEIRA DE TESTE) <<<")
            logger.info("************************************************************")
        else: # modo 'trade'
            logger.warning("************************************************************")
            logger.warning(">>> ATENÇÃO: OPERANDO NA BINANCE REAL (CARTEIRA REAL) <<<")
            logger.warning("************************************************************")

        # Importa o bot de trading somente quando necessário.
        from src.trading_bot import TradingBot
        
        bot = TradingBot()
        bot.run()

    else:
        # Esta verificação é um fallback de segurança, já que o config.py deve barrar modos inválidos.
        logger.error(f"Modo '{MODE}' não reconhecido no .env. Use 'optimize', 'test', ou 'trade'.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExecução interrompida pelo usuário.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Um erro crítico encerrou a aplicação: {e}", exc_info=True)
        sys.exit(1)