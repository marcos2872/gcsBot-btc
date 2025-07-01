# src/data_manager.py (VERSÃO OTIMIZADA)

import os
import datetime
import time
import pandas as pd
import numpy as np # Necessário para o downcasting
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from src.logger import logger
from src.config import (
    API_KEY, API_SECRET, USE_TESTNET, HISTORICAL_DATA_FILE, KAGGLE_BOOTSTRAP_FILE,
    FORCE_OFFLINE_MODE, COMBINED_DATA_CACHE_FILE
)

# NOVO: Função auxiliar para otimização de memória
def _optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Itera sobre todas as colunas de um DataFrame e modifica o tipo de dado
    para reduzir o consumo de memória.
    """
    logger.debug("Otimizando uso de memória do DataFrame...")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and 'datetime' not in str(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    logger.debug("Otimização de memória concluída.")
    return df

class DataManager:
    # ... (o resto da classe permanece o mesmo, exceto o final de update_and_load_data)
    def __init__(self):
        self.client = None
        if not FORCE_OFFLINE_MODE:
            try:
                self.client = Client(
                    API_KEY, API_SECRET,
                    tld='com',
                    testnet=USE_TESTNET,
                    requests_params={"timeout": 30}
                )
                self.client.ping()
                logger.info("Cliente Binance inicializado e conexão com a API confirmada.")
            except (BinanceAPIException, BinanceRequestException, Exception) as e:
                logger.warning(f"FALHA NA CONEXÃO: {e}. O bot operará em modo OFFLINE-FALLBACK.")
                self.client = None
        else:
            logger.info("MODO OFFLINE FORÇADO está ativo. Nenhuma conexão com a API da Binance será tentada.")

    def get_current_price(self, symbol):
        if not self.client:
            logger.warning("Cliente Binance não está disponível. Não é possível buscar o preço atual.")
            return None
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Erro ao buscar preço atual para {symbol}: {e}")
            return None

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
            logger.info(f"Arquivo de dados local do BTC encontrado em '{HISTORICAL_DATA_FILE}'. Carregando...")
            df = pd.read_csv(HISTORICAL_DATA_FILE, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            if self.client:
                last_timestamp = df.index.max()
                if last_timestamp < end_utc:
                    logger.info("Tentando buscar novos dados da Binance para atualizar o arquivo do BTC...")
                    try:
                        df_new = self.get_historical_data_by_batch(symbol, interval, last_timestamp, end_utc)
                        if not df_new.empty:
                            df = pd.concat([df, df_new])
                            df = df.loc[~df.index.duplicated(keep='last')]
                            df.sort_index(inplace=True)
                            df.to_csv(HISTORICAL_DATA_FILE)
                            logger.info(f"SUCESSO: Arquivo de dados do BTC atualizado com {len(df_new)} novas velas.")
                    except Exception as e:
                        logger.warning(f"FALHA NA ATUALIZAÇÃO DO BTC: {e}. Continuando com dados locais.")
            return df
        if os.path.exists(KAGGLE_BOOTSTRAP_FILE):
            logger.info(f"Arquivo mestre do BTC não encontrado. Iniciando a partir do arquivo Kaggle: '{KAGGLE_BOOTSTRAP_FILE}'")
            df_kaggle = pd.read_csv(KAGGLE_BOOTSTRAP_FILE, low_memory=False, on_bad_lines='skip')
            df = self._preprocess_kaggle_data(df_kaggle)
            last_timestamp = df.index.max()
            if self.client and last_timestamp < end_utc:
                logger.info("Atualizando dados do Kaggle com os dados mais recentes da Binance...")
                try:
                    df_new = self.get_historical_data_by_batch(symbol, interval, last_timestamp, end_utc)
                    if not df_new.empty:
                        df = pd.concat([df, df_new])
                        df = df.loc[~df.index.duplicated(keep='last')]
                        df.sort_index(inplace=True)
                except Exception as e:
                    logger.warning(f"FALHA NA ATUALIZAÇÃO DO BTC: {e}. Continuando com dados do Kaggle.")
            logger.info(f"Salvando o novo arquivo de dados mestre do BTC em '{HISTORICAL_DATA_FILE}'.")
            df.to_csv(HISTORICAL_DATA_FILE)
            return df
        if self.client:
            logger.warning("Nenhum arquivo local do BTC encontrado. Baixando o último ano da Binance como fallback.")
            start_utc = end_utc - datetime.timedelta(days=365)
            df = self.get_historical_data_by_batch(symbol, interval, start_utc, end_utc)
            if not df.empty:
                df.to_csv(HISTORICAL_DATA_FILE)
            return df
        logger.error("Nenhum arquivo de dados local do BTC encontrado e o bot está em modo offline. Não é possível continuar.")
        return pd.DataFrame()

    def _load_and_unify_local_macro_data(self, caminho_dados: str = 'data/macro') -> pd.DataFrame:
        logger.info("Iniciando o processo de padronização de dados macro locais...")
        config_ativos = {
            'dxy':  {'arquivo': 'dx.csv', 'separador': ',', 'formato_data': '%m/%d/%y'},
            'gold': {'arquivo': 'gold.csv', 'separador': ';', 'formato_data': None},
            'tnx':  {'arquivo': 'tnx.csv', 'separador': ',', 'formato_data': '%m/%d/%y'},
            'vix':  {'arquivo': 'vix.csv', 'separador': ',', 'formato_data': '%m/%d/%y'}
        }
        lista_dataframes = []
        for nome_ativo, config in config_ativos.items():
            caminho_arquivo = os.path.join(caminho_dados, config['arquivo'])
            if not os.path.exists(caminho_arquivo):
                logger.warning(f"AVISO: Arquivo macro '{caminho_arquivo}' não encontrado. Pulando o ativo '{nome_ativo}'.")
                continue
            try:
                logger.debug(f"Processando arquivo macro '{config['arquivo']}'...")
                df = pd.read_csv(caminho_arquivo, sep=config['separador'])
                df.columns = [col.strip().lower() for col in df.columns]
                df['date'] = pd.to_datetime(df['date'], format=config['formato_data'], errors='coerce').dt.normalize()
                df.dropna(subset=['date'], inplace=True)
                df = df[['date', 'close']].copy()
                df.rename(columns={'close': f'{nome_ativo}_close'}, inplace=True)
                df.set_index('date', inplace=True)
                df.index = df.index.tz_localize('UTC')
                lista_dataframes.append(df)
            except Exception as e:
                logger.error(f"ERRO ao processar o arquivo macro '{config['arquivo']}': {e}")
        if not lista_dataframes:
            logger.warning("Nenhum dado macro foi processado.")
            return pd.DataFrame()
        df_final = pd.concat(lista_dataframes, axis=1, join='outer')
        df_final.sort_index(inplace=True)
        df_final.ffill(inplace=True)
        df_final.dropna(inplace=True)
        logger.info("Dados macro locais unificados com sucesso.")
        return df_final

    def update_and_load_data(self, symbol, interval='1m'):
        """
        Método orquestrador com lógica de cache e otimização de memória.
        """
        df_btc = self._fetch_and_manage_btc_data(symbol, interval)
        if df_btc.empty:
            return pd.DataFrame()
        last_btc_timestamp = df_btc.index.max()

        if os.path.exists(COMBINED_DATA_CACHE_FILE):
            logger.info(f"Arquivo de cache encontrado em '{COMBINED_DATA_CACHE_FILE}'. Verificando se está atualizado...")
            df_cache = pd.read_csv(COMBINED_DATA_CACHE_FILE, index_col=0, parse_dates=True)
            df_cache.index = pd.to_datetime(df_cache.index, utc=True)
            
            if not df_cache.empty and df_cache.index.max() == last_btc_timestamp:
                logger.info("✅ Cache está atualizado! Carregando dados unificados diretamente do cache.")
                # ATUALIZADO: Otimiza a memória mesmo ao carregar do cache
                return _optimize_memory_usage(df_cache)
            else:
                logger.info("Cache está desatualizado. Reconstruindo...")
        
        logger.info("Iniciando processo de unificação de dados (cache não disponível ou obsoleto).")
        df_macro = self._load_and_unify_local_macro_data()

        if not df_macro.empty:
            logger.info("Combinando dados do BTC com dados macro unificados...")
            df_combined = df_btc.join(df_macro, how='left')
        else:
            df_combined = df_btc

        df_combined.ffill(inplace=True)
        df_combined.bfill(inplace=True)
        
        # ATUALIZADO: Otimiza a memória antes de salvar no cache
        df_combined = _optimize_memory_usage(df_combined)
        
        logger.info(f"Salvando dados unificados e otimizados no arquivo de cache: '{COMBINED_DATA_CACHE_FILE}'")
        df_combined.to_csv(COMBINED_DATA_CACHE_FILE)

        logger.info("Processo de coleta e combinação de dados concluído.")
        return df_combined