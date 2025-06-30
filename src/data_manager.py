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
    API_KEY, API_SECRET, USE_TESTNET, HISTORICAL_DATA_FILE, KAGGLE_BOOTSTRAP_FILE, FORCE_OFFLINE_MODE
)

class DataManager:
    """
    Gerencia todo o pipeline de dados: busca de dados históricos do BTC,
    busca de múltiplos indicadores macroeconômicos e combinação de todas as fontes.
    Projetado para ser resiliente a falhas de rede com um modo offline inteligente.
    """
    def __init__(self):
        self.client = None
        if not FORCE_OFFLINE_MODE:
            try:
                # Tenta inicializar o cliente apenas se não estiver em modo offline forçado
                self.client = Client(
                    API_KEY, API_SECRET,
                    tld='com',
                    testnet=USE_TESTNET,
                    requests_params={"timeout": 30} # Timeout estendido para mais resiliência
                )
                self.client.ping() # Testa a conexão para falhar rápido se não houver rede
                logger.info("Cliente Binance inicializado e conexão com a API confirmada.")
            except (BinanceAPIException, BinanceRequestException, Exception) as e:
                logger.warning(f"FALHA NA CONEXÃO: {e}. O bot operará em modo OFFLINE-FALLBACK, usando apenas dados locais.")
                self.client = None # Garante que o cliente seja nulo se a conexão falhar
        else:
            logger.info("MODO OFFLINE FORÇADO está ativo. Nenhuma conexão com a API da Binance será tentada.")

    def get_current_price(self, symbol):
        """Busca o preço de mercado mais recente para um símbolo."""
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
            logger.info(f"Arquivo de dados local encontrado em '{HISTORICAL_DATA_FILE}'. Carregando...")
            df = pd.read_csv(HISTORICAL_DATA_FILE, index_col=0, parse_dates=True)
            df.index = df.index.tz_convert('UTC')
            
            if self.client:
                last_timestamp = df.index.max()
                if last_timestamp < end_utc:
                    logger.info("Tentando buscar novos dados da Binance para atualizar o arquivo local...")
                    try:
                        df_new = self.get_historical_data_by_batch(symbol, interval, last_timestamp, end_utc)
                        if not df_new.empty:
                            df = pd.concat([df, df_new])
                            df = df.loc[~df.index.duplicated(keep='last')]
                            df.sort_index(inplace=True)
                            df.to_csv(HISTORICAL_DATA_FILE)
                            logger.info(f"SUCESSO: Arquivo de dados atualizado com {len(df_new)} novas velas.")
                    except Exception as e:
                        logger.warning(f"FALHA NA ATUALIZAÇÃO: Não foi possível buscar novos dados da Binance ({e}). Continuando com os dados locais existentes.")
            return df

        if os.path.exists(KAGGLE_BOOTSTRAP_FILE):
            logger.info(f"Arquivo mestre não encontrado. Iniciando a partir do arquivo Kaggle: '{KAGGLE_BOOTSTRAP_FILE}'")
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
                    logger.warning(f"FALHA NA ATUALIZAÇÃO: Não foi possível buscar dados da Binance ({e}). Continuando com os dados do Kaggle.")
            
            logger.info(f"Salvando o novo arquivo de dados mestre em '{HISTORICAL_DATA_FILE}'.")
            df.to_csv(HISTORICAL_DATA_FILE)
            return df

        if self.client:
            logger.warning("Nenhum arquivo local encontrado. Baixando o último ano da Binance como fallback.")
            start_utc = end_utc - datetime.timedelta(days=365)
            df = self.get_historical_data_by_batch(symbol, interval, start_utc, end_utc)
            if not df.empty:
                df.to_csv(HISTORICAL_DATA_FILE)
            return df
        
        logger.error("Nenhum arquivo de dados local encontrado e o bot está em modo offline. Não é possível continuar.")
        return pd.DataFrame()

    def _fetch_macro_data_hybrid(self, ticker_symbol: str, start_date, end_date):
        logger.info(f"Iniciando busca híbrida para o ativo macroeconômico: {ticker_symbol}")
        ticker = yf.Ticker(ticker_symbol)
        today = datetime.datetime.now(datetime.timezone.utc)
        cutoff_date = today - datetime.timedelta(days=59)
        
        all_dfs = []
        try:
            logger.debug(f"Buscando dados históricos diários para {ticker_symbol}...")
            df_daily = ticker.history(start=start_date, end=cutoff_date, interval="1d")
            if not df_daily.empty: all_dfs.append(df_daily[['Close']])
        except Exception as e: logger.warning(f"Não foi possível buscar dados diários para {ticker_symbol}: {e}")
        try:
            logger.debug(f"Buscando dados recentes de hora em hora para {ticker_symbol}...")
            df_hourly = ticker.history(period="60d", interval="1h")
            if not df_hourly.empty: all_dfs.append(df_hourly[['Close']])
        except Exception as e: logger.warning(f"Não foi possível buscar dados de hora em hora para {ticker_symbol}: {e}")

        if not all_dfs:
            logger.error(f"Falha total ao buscar quaisquer dados para {ticker_symbol}.")
            return pd.DataFrame()

        combined = pd.concat(all_dfs)
        combined = combined.loc[~combined.index.duplicated(keep='last')].sort_index()
        combined.index = combined.index.tz_convert('UTC')
        
        clean_name = ticker_symbol.lower().replace('^', '').replace('=f', '').replace('-y.nyb', '')
        return combined.rename(columns={'Close': f"{clean_name}_close"})

    def update_and_load_data(self, symbol, interval='1m'):
        df_btc = self._fetch_and_manage_btc_data(symbol, interval)
        if df_btc.empty: return pd.DataFrame()

        start_date, end_date = df_btc.index.min(), df_btc.index.max()
        
        macro_tickers = {
            'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'GOLD': 'GC=F', 'TNX': '^TNX'
        }
        
        df_combined = df_btc
        for name, ticker in macro_tickers.items():
            df_macro = self._fetch_macro_data_hybrid(ticker, start_date, end_date)
            if not df_macro.empty:
                logger.info(f"Combinando dados do {name}...")
                full_range = pd.date_range(start=start_date, end=end_date, freq='min', tz='UTC')
                df_resampled = df_macro.reindex(full_range).ffill().bfill()
                df_combined = df_combined.join(df_resampled, how='left')
            else:
                clean_name = ticker.lower().replace('^', '').replace('=f', '').replace('-y.nyb', '')
                logger.warning(f"Não foi possível obter dados para {name}. A coluna '{clean_name}_close' será preenchida com 0.")
                df_combined[f"{clean_name}_close"] = 0
        
        df_combined.ffill(inplace=True)
        df_combined.bfill(inplace=True)
        
        logger.info("Processo de coleta de dados concluído.")
        return df_combined