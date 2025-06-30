# src/data_manager.py

import os
import datetime
import time
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import yfinance as yf
from src.logger import logger
from src.config import (
    API_KEY, API_SECRET, USE_TESTNET, HISTORICAL_DATA_FILE, KAGGLE_BOOTSTRAP_FILE, FORCE_OFFLINE_MODE, SYMBOL
)

class DataManager:
    """
    Gerencia todo o pipeline de dados: busca, combinação e salvamento de dados
    de cripto e macroeconômicos para operação online e offline.
    """
    def __init__(self):
        self.client = None
        if not FORCE_OFFLINE_MODE:
            try:
                self.client = Client(
                    API_KEY, API_SECRET, tld='com',
                    testnet=USE_TESTNET, requests_params={"timeout": 30}
                )
                self.client.ping()
                logger.info("Cliente Binance inicializado e conexão com a API confirmada.")
            except Exception as e:
                logger.warning(f"FALHA NA CONEXÃO: {e}. O bot operará em modo OFFLINE-FALLBACK, usando apenas dados locais.")
                self.client = None
        else:
            logger.info("MODO OFFLINE FORÇADO está ativo. Nenhuma conexão com a API da Binance será tentada.")

    def get_current_price(self, symbol):
        """Busca o preço de mercado mais recente para um símbolo."""
        if not self.client:
            logger.warning("Cliente Binance não disponível. Não é possível buscar o preço atual.")
            return None
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Erro ao buscar preço atual para {symbol}: {e}"); return None

    def _preprocess_kaggle_data(self, df_kaggle: pd.DataFrame) -> pd.DataFrame:
        logger.info("Pré-processando dados do Kaggle...")
        column_mapping = {'Timestamp': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
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
        
        df = df_kaggle[['open', 'high', 'low', 'close', 'volume']].copy()
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
            try:
                klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)
                if not klines: break
                df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','qav','nt','tbbav','tbqav','ignore'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.index = df.index.tz_localize('UTC')
                df = df[['open','high','low','close','volume']].astype(float)
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Erro ao baixar lote da Binance: {e}. Pulando este lote.")
            cursor = next_cursor
            time.sleep(0.5)
        return pd.concat(all_dfs) if all_dfs else pd.DataFrame()

    def _fetch_base_data(self, symbol, interval='1m'):
        if os.path.exists(HISTORICAL_DATA_FILE):
            logger.info(f"Arquivo de dados mestre encontrado: '{HISTORICAL_DATA_FILE}'. Carregando...")
            df = pd.read_csv(HISTORICAL_DATA_FILE, index_col=0, parse_dates=True)
            df.index = df.index.tz_convert('UTC')
            return df

        if os.path.exists(KAGGLE_BOOTSTRAP_FILE):
            logger.info(f"Iniciando a partir do arquivo Kaggle: '{KAGGLE_BOOTSTRAP_FILE}'")
            df_kaggle = pd.read_csv(KAGGLE_BOOTSTRAP_FILE, low_memory=False, on_bad_lines='skip')
            return self._preprocess_kaggle_data(df_kaggle)
        
        if self.client:
            logger.warning("Nenhum arquivo local encontrado. Baixando o último ano da Binance.")
            end_utc = datetime.datetime.now(datetime.timezone.utc)
            start_utc = end_utc - datetime.timedelta(days=365)
            return self.get_historical_data_by_batch(symbol, interval, start_utc, end_utc)
        
        logger.error("Nenhum dado local encontrado e o bot está em modo offline. Não é possível continuar.")
        return pd.DataFrame()

    def _fetch_macro_data_hybrid(self, ticker_symbol: str):
        logger.info(f"Buscando dados macro para: {ticker_symbol}")
        ticker = yf.Ticker(ticker_symbol)
        
        start_date_req = pd.to_datetime("2018-01-01", utc=True)
        end_date_req = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1) # Adiciona 1 dia para garantir que pegamos até hoje
        
        try:
            df_macro = ticker.history(start=start_date_req, end=end_date_req, interval="1d", auto_adjust=True, back_adjust=True)
            if df_macro.empty:
                logger.warning(f"Nenhum dado diário retornado para {ticker_symbol}.")
                return pd.DataFrame()
            
            df_macro.index = df_macro.index.tz_convert('UTC')
            clean_name = ticker_symbol.lower().replace('^', '').replace('=f', '').replace('-y.nyb', '')
            return df_macro[['Close']].rename(columns={'Close': f"{clean_name}_close"})
        except Exception as e:
            logger.warning(f"Não foi possível buscar dados para {ticker_symbol}: {e}")
            return pd.DataFrame()

    def update_and_load_data(self, symbol, interval='1m'):
        df_base = self._fetch_base_data(symbol, interval)
        if df_base.empty: return pd.DataFrame()

        if self.client:
            last_timestamp = df_base.index.max()
            end_utc = datetime.datetime.now(datetime.timezone.utc)
            if last_timestamp < end_utc - datetime.timedelta(minutes=5):
                logger.info(f"Tentando buscar novos dados do BTC ({len(df_base)} registros existentes)...")
                try:
                    df_new = self.get_historical_data_by_batch(symbol, interval, last_timestamp + datetime.timedelta(minutes=1), end_utc)
                    if not df_new.empty:
                        df_base = pd.concat([df_base, df_new]).loc[~df_base.index.duplicated(keep='last')].sort_index()
                except Exception as e: logger.warning(f"FALHA NA ATUALIZAÇÃO DO BTC: {e}. Usando dados locais.")
        
        start_date, end_date = df_base.index.min(), df_base.index.max()
        macro_tickers = {'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'GOLD': 'GC=F', 'TNX': '^TNX'}
        
        df_combined = df_base
        for name, ticker in macro_tickers.items():
            clean_name = ticker.lower().replace('^', '').replace('=f', '').replace('-y.nyb', '')
            col_name = f"{clean_name}_close"

            if col_name in df_combined.columns and FORCE_OFFLINE_MODE and not df_combined[col_name].isnull().all():
                logger.info(f"Usando dados macro locais para {name}.")
                continue
            
            df_macro = self._fetch_macro_data_hybrid(ticker)
            if not df_macro.empty:
                logger.info(f"Combinando dados do {name}...")
                full_range = pd.date_range(start=start_date, end=end_date, freq='min', tz='UTC')
                df_resampled = df_macro.reindex(full_range).ffill().bfill()
                # Remove a coluna antiga antes de juntar para evitar conflitos
                if col_name in df_combined.columns:
                    df_combined = df_combined.drop(columns=[col_name])
                df_combined = df_combined.join(df_resampled, how='left')
            else:
                logger.warning(f"Não foi possível obter dados para {name}. A coluna '{col_name}' será preenchida com 0 se não existir.")
                if col_name not in df_combined.columns:
                    df_combined[col_name] = 0
        
        df_combined.ffill(inplace=True)
        df_combined.bfill(inplace=True)

        logger.info(f"Salvando o arquivo de dados mestre combinado em '{HISTORICAL_DATA_FILE}'.")
        df_combined.to_csv(HISTORICAL_DATA_FILE)
        
        logger.info("Processo de coleta de dados concluído.")
        return df_combined