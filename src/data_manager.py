# src/data_manager.py

import os
import datetime
import time
import pandas as pd
from binance.client import Client
import yfinance as yf
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

    def _preprocess_kaggle_data(self, df_kaggle: pd.DataFrame) -> pd.DataFrame:
        logger.info("Pré-processando dados do Kaggle...")
        column_mapping = {
            'Timestamp': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'
        }
        possible_volume_names = ['Volume_(BTC)', 'Volume', 'Volume (BTC)', 'Volume (Currency)', 'Volume USD']
        found_volume_col = next((name for name in possible_volume_names if name in df_kaggle.columns), None)
        
        if not found_volume_col:
            raise ValueError(f"Não foi possível encontrar uma coluna de volume no arquivo Kaggle. Nomes tentados: {possible_volume_names}")
        
        logger.info(f"Coluna de volume encontrada no arquivo Kaggle: '{found_volume_col}'")
        column_mapping[found_volume_col] = 'volume'
        df_kaggle.rename(columns=column_mapping, inplace=True)
        
        df_kaggle['timestamp'] = pd.to_datetime(df_kaggle['timestamp'], unit='s')
        df_kaggle.set_index('timestamp', inplace=True)
        df_kaggle.index = df_kaggle.index.tz_localize('UTC')
        
        final_columns = ['open', 'high', 'low', 'close', 'volume']
        df = df_kaggle[final_columns].copy()
        df.dropna(inplace=True)
        df = df.astype(float)
        
        logger.info(f"Processamento do Kaggle concluído. {len(df)} registros válidos carregados.")
        return df

    def get_historical_data_by_batch(self, symbol, interval, start_date_dt, end_date_dt):
        all_dfs = []
        cursor = start_date_dt
        while cursor < end_date_dt:
            next_cursor = min(cursor + datetime.timedelta(days=30), end_date_dt)
            logger.info(f"Baixando lote da Binance: {cursor:%Y-%m-%d} -> {next_cursor:%Y-%m-%d}")
            start_str, end_str = cursor.strftime("%Y-%m-%d %H:%M:%S"), next_cursor.strftime("%Y-%m-%d %H:%M:%S")
            klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)
            if not klines: break
            df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','qav','nt','tbbav','tbqav','ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.index = df.index.tz_localize('UTC')
            df = df[['open','high','low','close','volume']].astype(float)
            all_dfs.append(df)
            cursor = next_cursor
            time.sleep(0.5)
        return pd.concat(all_dfs) if all_dfs else pd.DataFrame()

    def _fetch_and_manage_btc_data(self, symbol, interval='1m'):
        end_utc = datetime.datetime.now(datetime.timezone.utc)
        if os.path.exists(HISTORICAL_DATA_FILE):
            logger.info(f"Arquivo de dados local encontrado em '{HISTORICAL_DATA_FILE}'. Carregando e atualizando...")
            df = pd.read_csv(HISTORICAL_DATA_FILE, index_col=0, parse_dates=True)
            df.index = df.index.tz_convert('UTC')
            last_timestamp = df.index.max()
            if last_timestamp < end_utc:
                logger.info("Buscando novos dados da Binance para atualizar o arquivo local...")
                df_new = self.get_historical_data_by_batch(symbol, interval, last_timestamp, end_utc)
                if not df_new.empty:
                    # ### CORREÇÃO DEFINITIVA ###
                    # A lógica de concatenação e remoção de duplicatas foi separada para evitar o erro de tamanho.
                    df = pd.concat([df, df_new])
                    df = df.loc[~df.index.duplicated(keep='last')]
                    df.sort_index(inplace=True)
                    df.to_csv(HISTORICAL_DATA_FILE)
                    logger.info(f"Arquivo atualizado com {len(df_new)} novas velas.")
            return df
        if os.path.exists(KAGGLE_BOOTSTRAP_FILE):
            logger.info(f"Arquivo mestre não encontrado. Iniciando a partir do arquivo Kaggle: '{KAGGLE_BOOTSTRAP_FILE}'")
            df_kaggle = pd.read_csv(KAGGLE_BOOTSTRAP_FILE, low_memory=False, on_bad_lines='skip')
            df = self._preprocess_kaggle_data(df_kaggle)
            last_timestamp = df.index.max()
            if last_timestamp < end_utc:
                logger.info("Atualizando dados do Kaggle com os dados mais recentes da Binance...")
                df_new = self.get_historical_data_by_batch(symbol, interval, last_timestamp, end_utc)
                if not df_new.empty:
                    # ### CORREÇÃO DEFINITIVA ###
                    df = pd.concat([df, df_new])
                    df = df.loc[~df.index.duplicated(keep='last')]
                    df.sort_index(inplace=True)
            logger.info(f"Salvando o novo arquivo de dados mestre em '{HISTORICAL_DATA_FILE}'.")
            df.to_csv(HISTORICAL_DATA_FILE)
            return df
        logger.warning("Nenhum arquivo local encontrado. Baixando o último ano da Binance como fallback.")
        start_utc = end_utc - datetime.timedelta(days=365)
        df = self.get_historical_data_by_batch(symbol, interval, start_utc, end_utc)
        if not df.empty:
            df.to_csv(HISTORICAL_DATA_FILE)
        return df

    def _fetch_dxy_data_hybrid(self, start_date, end_date):
        logger.info("Iniciando busca híbrida de dados do DXY...")
        dxy_ticker = yf.Ticker("DX-Y.NYB")
        today = datetime.datetime.now(datetime.timezone.utc)
        cutoff_date = today - datetime.timedelta(days=7)
        all_dxy_dfs = []
        try:
            logger.info(f"Buscando dados históricos diários do DXY de {start_date.year} até {cutoff_date.year}...")
            df_dxy_daily = dxy_ticker.history(start=start_date, end=cutoff_date, interval="1d")
            if not df_dxy_daily.empty:
                all_dxy_dfs.append(df_dxy_daily[['Close']])
        except Exception as e:
            logger.warning(f"Não foi possível buscar dados diários do DXY: {e}")
        try:
            logger.info(f"Buscando dados recentes de hora em hora do DXY dos últimos 7 dias...")
            df_dxy_hourly = dxy_ticker.history(period="7d", interval="1h")
            if not df_dxy_hourly.empty:
                all_dxy_dfs.append(df_dxy_hourly[['Close']])
        except Exception as e:
            logger.warning(f"Não foi possível buscar dados de hora em hora do DXY: {e}")
        if not all_dxy_dfs:
            logger.error("Falha total ao buscar quaisquer dados do DXY.")
            return pd.DataFrame()
        combined_df = pd.concat(all_dxy_dfs)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
        combined_df.index = combined_df.index.tz_convert('UTC')
        return combined_df.rename(columns={'Close': 'dxy_close'})

    def update_and_load_data(self, symbol, interval='1m'):
        df_btc = self._fetch_and_manage_btc_data(symbol, interval)
        if df_btc.empty:
            logger.error("Falha ao obter dados do BTC. Não é possível continuar.")
            return pd.DataFrame()
        start_date, end_date = df_btc.index.min(), df_btc.index.max()
        df_dxy = self._fetch_dxy_data_hybrid(start_date, end_date)
        if not df_dxy.empty:
            logger.info("Combinando e reamostrando dados do BTC e DXY...")
            full_range_index = pd.date_range(start=start_date, end=end_date, freq='1T', tz='UTC')
            df_dxy_resampled = df_dxy.reindex(full_range_index).ffill().bfill()
            df_combined = df_btc.join(df_dxy_resampled, how='left')
            df_combined['dxy_close'].ffill(inplace=True)
            df_combined['dxy_close'].bfill(inplace=True)
            logger.info("Dados do BTC e DXY combinados com sucesso.")
            return df_combined
        else:
            logger.warning("Não foi possível obter dados do DXY. Continuando apenas com dados do BTC.")
            df_btc['dxy_close'] = 0
            return df_btc