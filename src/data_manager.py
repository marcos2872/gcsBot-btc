# src/data_manager.py

import os
import datetime
import time
import pandas as pd
from binance.client import Client
from src.logger import logger
from src.config import API_KEY, API_SECRET, USE_TESTNET, HISTORICAL_DATA_FILE, KAGGLE_BOOTSTRAP_FILE

class DataManager:
    def __init__(self):
        self.client = Client(
            API_KEY, API_SECRET,
            testnet=USE_TESTNET,
            requests_params={"timeout": 60}
        )
        logger.info("Cliente Binance inicializado no DataManager.")

    def _preprocess_kaggle_data(self, df):
        """Prepara o CSV do Kaggle para o formato que usamos."""
        logger.info("Pré-processando dados do Kaggle...")
        # Ajuste os nomes das colunas conforme seu arquivo Kaggle.
        # Exemplo para o dataset 'Bitstamp BTCUSD' que usa timestamp UNIX.
        df = df.rename(columns={
            'Timestamp': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Converte timestamp UNIX para datetime e localiza em UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        df.index = df.index.tz_localize('UTC')
        
        # Seleciona, ordena e limpa os dados
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df.astype(float)
        df.dropna(inplace=True)
        
        logger.info(f"Processamento concluído. {len(df)} registros válidos carregados do Kaggle.")
        return df

    def get_historical_data_by_batch(self, symbol, interval, start_date, end_date):
        all_dfs = []
        cursor = start_date
        while cursor < end_date:
            nxt = min(cursor + datetime.timedelta(days=30), end_date)
            logger.info(f"Baixando lote da Binance: {cursor:%Y-%m-%d} → {nxt:%Y-%m-%d}")
            klines = self.client.get_historical_klines(
                symbol, interval,
                start_str=cursor.strftime("%Y-%m-%d %H:%M:%S UTC"),
                end_str=  nxt.strftime("%Y-%m-%d %H:%M:%S UTC")
            )
            if not klines:
                break
            df = pd.DataFrame(klines, columns=[
                'timestamp','open','high','low','close','volume',
                'close_time','qav','nt','tbbav','tbqav','ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open','high','low','close','volume']].astype(float)
            df.index = df.index.tz_localize('UTC')
            all_dfs.append(df)
            cursor = nxt
            time.sleep(0.5)
        return pd.concat(all_dfs) if all_dfs else pd.DataFrame()

    def update_and_load_data(self, symbol, interval='1m'):
        end_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if os.path.exists(HISTORICAL_DATA_FILE):
            logger.info(f"Arquivo de dados local encontrado em '{HISTORICAL_DATA_FILE}'. Carregando e atualizando...")
            df = pd.read_csv(HISTORICAL_DATA_FILE, index_col=0, parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            last_timestamp = df.index.max()
            if last_timestamp < end_utc:
                logger.info("Buscando novos dados da Binance para atualizar o arquivo local...")
                df_new = self.get_historical_data_by_batch(symbol, interval, last_timestamp, end_utc)
                if not df_new.empty:
                    df = pd.concat([df, df_new])
                    df = df[~df.index.duplicated(keep='last')].sort_index()
                    df.to_csv(HISTORICAL_DATA_FILE)
                    logger.info(f"Arquivo atualizado com {len(df_new)} novas velas.")
            else:
                logger.info("Os dados locais já estão atualizados.")
            return df

        elif os.path.exists(KAGGLE_BOOTSTRAP_FILE):
            logger.info(f"Arquivo mestre não encontrado. Iniciando a partir do arquivo Kaggle: '{KAGGLE_BOOTSTRAP_FILE}'")
            df_kaggle = pd.read_csv(KAGGLE_BOOTSTRAP_FILE, low_memory=False)
            df = self._preprocess_kaggle_data(df_kaggle)
            
            last_timestamp = df.index.max()
            if last_timestamp < end_utc:
                logger.info("Atualizando dados do Kaggle com os dados mais recentes da Binance...")
                df_new = self.get_historical_data_by_batch(symbol, interval, last_timestamp, end_utc)
                if not df_new.empty:
                    df = pd.concat([df, df_new])
                    df = df[~df.index.duplicated(keep='last')].sort_index()

            logger.info(f"Salvando o novo arquivo de dados mestre combinado em '{HISTORICAL_DATA_FILE}'.")
            df.to_csv(HISTORICAL_DATA_FILE)
            return df

        else:
            logger.warning("Nenhum arquivo local encontrado (nem mestre, nem Kaggle). Baixando o último ano da Binance como fallback.")
            start_utc = end_utc - datetime.timedelta(days=365)
            df = self.get_historical_data_by_batch(symbol, interval, start_utc, end_utc)
            if not df.empty:
                logger.info(f"Salvando dados baixados em '{HISTORICAL_DATA_FILE}'.")
                df.to_csv(HISTORICAL_DATA_FILE)
            return df