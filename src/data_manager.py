# src/data_manager.py

import os
import datetime
import time
import pandas as pd
import numpy as np
import gc
from binance.client import Client
from binance.exceptions import BinanceAPIException
import yfinance as yf
from src.logger import logger
from src.config import (
    API_KEY, API_SECRET, USE_TESTNET, SYMBOL, FORCE_OFFLINE_MODE
)

# --- Constantes de Configuração de Dados ---
# Define o diretório onde todos os dados serão salvos
DATA_DIR = "data"
# Define um subdiretório específico para os dados macroeconômicos
MACRO_DATA_DIR = os.path.join(DATA_DIR, "macro")
# Define a data de início para buscar todo o histórico, caso não haja cache
DATA_START_DATE = "2018-01-01"


def reduce_mem_usage(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Itera sobre todas as colunas de um dataframe e modifica seus tipos de dados
    para consumir menos memória, reduzindo o uso em 50-70%.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                # Converte float64 para float32, a maior fonte de economia de memória
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.debug(f'Uso de memória do DataFrame reduzido de {start_mem:.2f}MB para {end_mem:.2f}MB ({100*(start_mem-end_mem)/start_mem:.1f}% de redução).')
    return df


class DataManager:
    """
    Gerencia o pipeline de dados de forma robusta e eficiente.
    - Usa caches individuais para cada ativo (crypto e macro).
    - Atualiza os caches de forma inteligente, buscando apenas dados novos.
    - Otimiza o uso de memória para lidar com grandes datasets.
    """
    def __init__(self):
        self.client = None
        # Garante que os diretórios para o cache de dados existam
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MACRO_DATA_DIR, exist_ok=True)

        if not FORCE_OFFLINE_MODE:
            try:
                self.client = Client(
                    API_KEY, API_SECRET, tld='com', testnet=USE_TESTNET,
                    requests_params={"timeout": 30}
                )
                self.client.ping()
                logger.info("Cliente Binance inicializado e conexão com a API confirmada.")
            except Exception as e:
                logger.warning(f"FALHA NA CONEXÃO: {e}. O bot operará em modo OFFLINE-FALLBACK.")
                self.client = None
        else:
            logger.info("MODO OFFLINE FORÇADO está ativo. Nenhuma conexão com a internet será tentada.")

    def _fetch_crypto_data_with_cache(self, symbol: str, interval: str) -> pd.DataFrame:
        """Gerencia o cache de dados para o ativo de criptomoeda com paginação de API."""
        cache_file = os.path.join(DATA_DIR, f"{symbol.lower()}_{interval}.csv")
        df = pd.DataFrame()

        if os.path.exists(cache_file):
            logger.info(f"Cache local para {symbol} encontrado. Carregando...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            df = reduce_mem_usage(df)
            df.index = df.index.tz_convert('UTC')

        if self.client:
            start_update_dt = df.index.max() if not df.empty else pd.to_datetime(DATA_START_DATE, utc=True)
            end_update_dt = datetime.datetime.now(datetime.timezone.utc)

            if start_update_dt < end_update_dt - datetime.timedelta(minutes=5):
                logger.info(f"Cache de {symbol} desatualizado. Buscando novos dados desde {start_update_dt} em lotes...")
                all_new_klines = []
                cursor = start_update_dt + datetime.timedelta(minutes=1) # Começa do minuto seguinte ao último registro

                # Loop para buscar dados em lotes de 1000 velas, garantindo que tudo seja baixado
                while cursor < end_update_dt:
                    try:
                        klines = self.client.get_historical_klines(symbol, interval, cursor.strftime("%d %b %Y %H:%M:%S"))
                        if not klines:
                            break # Sai do loop se não houver mais dados
                        all_new_klines.extend(klines)
                        new_last_timestamp = pd.to_datetime(klines[-1][0], unit='ms', utc=True)
                        logger.debug(f"Lote de {len(klines)} velas baixado para {symbol}, até {new_last_timestamp}.")
                        cursor = new_last_timestamp + datetime.timedelta(minutes=1)
                        time.sleep(0.2) # Pausa para não sobrecarregar a API
                    except BinanceAPIException as e:
                        logger.error(f"Erro de API da Binance ao buscar lote para {symbol}: {e}. Tentando novamente em 10s.")
                        time.sleep(10)

                if all_new_klines:
                    df_new = pd.DataFrame(all_new_klines, columns=['timestamp','open','high','low','close','volume','close_time','qav','nt','tbbav','tbqav','ignore'])
                    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
                    df_new.set_index('timestamp', inplace=True)
                    df_new.index = df_new.index.tz_localize('UTC')
                    df_new = df_new[['open','high','low','close','volume']].astype(float)
                    
                    df = pd.concat([df, df_new]).loc[~df.index.duplicated(keep='last')].sort_index()
                    df.to_csv(cache_file)
                    logger.info(f"Cache de {symbol} atualizado com {len(df_new)} novos registros e salvo.")
            else:
                logger.info(f"Cache de {symbol} já está atualizado.")

        if df.empty:
            logger.error(f"Não foi possível carregar dados para {symbol}, nem do cache nem da API.")
        
        return df

    def _fetch_macro_data_with_cache(self, ticker_symbol: str) -> pd.DataFrame:
        """Gerencia o cache de dados para um único ativo macroeconômico."""
        clean_name = ticker_symbol.lower().replace('^', '').replace('=f', '').replace('-y.nyb', '')
        cache_file = os.path.join(MACRO_DATA_DIR, f"{clean_name}.csv")
        df = pd.DataFrame()

        # 1. Tenta carregar do cache local
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            df = reduce_mem_usage(df) # Otimiza memória
            df.index = df.index.tz_convert('UTC')

        # 2. Se online, tenta atualizar
        if not FORCE_OFFLINE_MODE:
            try:
                last_timestamp = df.index.max() if not df.empty else pd.to_datetime(DATA_START_DATE, utc=True)
                end_utc = datetime.datetime.now(datetime.timezone.utc)

                if last_timestamp < end_utc.replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=1):
                    logger.info(f"Buscando dados macro atualizados para {ticker_symbol}...")
                    ticker = yf.Ticker(ticker_symbol)
                    df_new = ticker.history(start=last_timestamp, end=end_utc, interval="1d")
                    
                    if not df_new.empty:
                        df_new.index = df_new.index.tz_convert('UTC')
                        df = pd.concat([df, df_new]).loc[~df.index.duplicated(keep='last')].sort_index()
                        df.to_csv(cache_file)
                        logger.info(f"Cache de {ticker_symbol} atualizado e salvo.")
            except Exception as e:
                logger.warning(f"Não foi possível buscar dados para {ticker_symbol}: {e}. Usando dados de cache, se disponíveis.")

        if df.empty:
            logger.error(f"FALHA TOTAL ao carregar dados para o ativo macro {ticker_symbol}.")
        
        return df[['Close']].rename(columns={'Close': f"{clean_name}_close"})

    def update_and_load_data(self) -> pd.DataFrame:
        """
        Orquestra todo o processo: carrega/atualiza o ativo principal e os dados macro,
        e os combina em um único DataFrame final para uso do bot.
        """
        logger.info("="*50)
        logger.info("INICIANDO PROCESSO DE CARGA E ATUALIZAÇÃO DE DADOS...")
        
        # 1. Carrega o ativo principal (BTC)
        df_crypto = self._fetch_crypto_data_with_cache(SYMBOL, '1m')
        if df_crypto.empty:
            return pd.DataFrame()

        # 2. Carrega todos os ativos macro
        macro_tickers = {'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'GOLD': 'GC=F', 'US10Y': '^TNX'}
        
        df_combined = df_crypto.copy()
        del df_crypto
        gc.collect()

        for name, ticker in macro_tickers.items():
            df_macro = self._fetch_macro_data_with_cache(ticker)
            if not df_macro.empty:
                logger.info(f"Combinando dados de {name} ({ticker})...")
                # Reamostra dados macro (diários) para a frequência de 1 minuto do BTC
                # ffill preenche para frente, garantindo que o valor do dia se aplique a todos os minutos
                df_resampled = df_macro.reindex(df_combined.index, method='ffill')
                df_combined = df_combined.join(df_resampled, how='left')
            else:
                # Garante que a coluna exista mesmo em caso de falha, para não quebrar o modelo
                clean_name = ticker.lower().replace('^', '').replace('=f', '').replace('-y.nyb', '')
                col_name = f"{clean_name}_close"
                if col_name not in df_combined.columns:
                    df_combined[col_name] = 0
                logger.warning(f"Coluna '{col_name}' preenchida com 0 devido à falha na obtenção dos dados.")

        # 3. Limpeza final
        # Preenche quaisquer lacunas no início do dataset
        df_combined.bfill(inplace=True)
        # Preenche lacunas que podem ter sobrado no meio (pouco provável, mas seguro)
        df_combined.ffill(inplace=True)

        logger.info(f"Processo de dados finalizado. DataFrame pronto com {len(df_combined)} linhas e {len(df_combined.columns)} colunas.")
        logger.info("="*50)

        return df_combined