import csv
import os
import time
import pandas as pd
from binance.client import Client
from .config import API_KEY, API_SECRET, TESTNET_API_KEY, TESTNET_API_SECRET, USE_TESTNET
from binance.exceptions import BinanceAPIException
from .logger import logger

class BinanceClient:
    # O método __init__ e os outros (get_current_price, get_balance, etc.) continuam os mesmos.
    def __init__(self):
        if USE_TESTNET:
            if not TESTNET_API_KEY or not TESTNET_API_SECRET:
                raise ValueError("Chaves da Testnet não configuradas no arquivo .env")
            self.client = Client(TESTNET_API_KEY, TESTNET_API_SECRET, testnet=True)
            logger.info("✅ Conectado à Binance Testnet")
        else:
            if not API_KEY or not API_SECRET:
                raise ValueError("Chaves da API real não configuradas no arquivo .env")
            self.client = Client(API_KEY, API_SECRET)
            logger.info("✅ Conectado à Binance Rede Real")
        try:
            self.client.ping()
            logger.info("✅ Ping bem-sucedido!")
            account_info = self.client.get_account()
            logger.info(f"✅ Acesso à conta confirmado - Tipo: {account_info.get('accountType', 'N/A')}")
        except Exception as e:
            logger.error(f"❌ Erro ao conectar: {e}")
            raise

    def get_current_price(self, symbol):
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Erro ao obter preço de {symbol}: {e}")
            raise

    def get_balance(self, asset='USDT'):
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            logger.error(f"Erro ao obter saldo de {asset}: {e}")
            raise

    def place_order(self, symbol, side, quantity):
        # Este método continua o mesmo
        pass

    # --- FUNÇÃO MODIFICADA PARA APRENDIZAGEM PERSISTENTE ---
    def update_historical_data(self, symbol, interval='1h', initial_limit=300):
        """
        Atualiza o arquivo de dados históricos com os dados mais recentes,
        em vez de sobrescrevê-lo.
        """
        data_filepath = 'data/historical_data.csv'
        os.makedirs('data', exist_ok=True)
        
        start_str = None
        # Verifica se o arquivo já existe e tem dados
        if os.path.exists(data_filepath) and os.path.getsize(data_filepath) > 0:
            try:
                df_existing = pd.read_csv(data_filepath)
                # Pega o timestamp do último registro salvo
                last_timestamp = df_existing['timestamp'].iloc[-1]
                # Converte para o formato que a API da Binance espera
                start_str = pd.to_datetime(last_timestamp, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"Arquivo de dados existente. Buscando novos dados a partir de {start_str}")
            except (pd.errors.EmptyDataError, IndexError):
                 logger.warning("Arquivo de dados corrompido ou vazio. Baixando histórico completo.")
                 start_str = f"{initial_limit} hours ago UTC"
        else:
            # Se não existe, busca o histórico inicial completo
            logger.info("Nenhum arquivo de dados encontrado. Baixando histórico inicial completo.")
            start_str = f"{initial_limit} hours ago UTC"

        try:
            klines = self.client.get_historical_klines(symbol, interval, start_str)
            
            if not klines:
                logger.info("Nenhum dado novo para adicionar.")
                return

            new_data = []
            for k in klines:
                new_data.append({
                    'timestamp': int(k[0]), 'open': float(k[1]), 'high': float(k[2]),
                    'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])
                })
            
            # Adiciona os novos dados ao arquivo CSV (append mode)
            # Se o arquivo não existia, 'a' (append) funciona como 'w' (write)
            is_new_file = not os.path.exists(data_filepath) or os.path.getsize(data_filepath) == 0
            with open(data_filepath, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                if is_new_file:
                    writer.writeheader()
                writer.writerows(new_data)
            
            logger.info(f"✅ {len(new_data)} novos registros adicionados a {data_filepath}")

        except Exception as e:
            logger.error(f"Erro ao atualizar dados históricos: {e}")
            raise